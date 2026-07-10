/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package validation

import (
	"context"
	"fmt"
	"os"
	"slices"
	"strings"
	"sync"
	"sync/atomic"
	"testing"

	nvidiacomv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/features"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/testing/operatorenv"
	authenticationv1 "k8s.io/api/authentication/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/client-go/dynamic"
	"k8s.io/client-go/rest"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/envtest"
	"sigs.k8s.io/controller-runtime/pkg/webhook/admission"
)

const (
	admissionDCDKind = "DynamoComponentDeployment"
	admissionDGDKind = "DynamoGraphDeployment"
)

var (
	// Slots vary handler dependencies per matrix row without restarting the shared API server.
	dcdAlphaValidation = newValidatorSlot()
	dcdBetaValidation  = newValidatorSlot()
	dgdAlphaValidation = newValidatorSlot()
	dgdBetaValidation  = newValidatorSlot()
	dgdValidationGate  = newFeatureGateSlot(features.Defaults())

	validationOperatorEnv = operatorenv.New(operatorenv.Options{
		Admission:     operatorenv.AdmissionWebhooks{Validating: true},
		Conversion:    true,
		SetupWebhooks: setupValidationAdmissionWebhooks,
	})
)

func TestMain(m *testing.M) {
	os.Exit(validationOperatorEnv.RunM(m))
}

func setupValidationAdmissionWebhooks(mgr ctrl.Manager, _ operatorenv.WebhookSetupOptions) error {
	for _, object := range []runtime.Object{
		&nvidiacomv1beta1.DynamoGraphDeployment{},
		&nvidiacomv1beta1.DynamoComponentDeployment{},
	} {
		if err := ctrl.NewWebhookManagedBy(mgr, object).Complete(); err != nil {
			return fmt.Errorf("register conversion webhook for %T: %w", object, err)
		}
	}

	// Serve both API endpoints so the Helm registration selects the intended compatibility route.
	dcdRegistration := NewDynamoComponentDeploymentHandler()
	dcdRegistration.registerWithManager(
		mgr,
		&nvidiacomv1beta1.DynamoComponentDeployment{},
		dynamoComponentDeploymentV1Beta1WebhookPath,
		dcdBetaValidation,
		features.Defaults(),
	)
	dcdRegistration.registerWithManager(
		mgr,
		&nvidiacomv1alpha1.DynamoComponentDeployment{},
		dynamoComponentDeploymentV1Alpha1WebhookPath,
		dcdAlphaValidation,
		features.Defaults(),
	)

	dgdRegistration := &DynamoGraphDeploymentHandler{}
	dgdRegistration.registerWithManager(
		mgr,
		&nvidiacomv1beta1.DynamoGraphDeployment{},
		dynamoGraphDeploymentV1Beta1WebhookPath,
		dgdBetaValidation,
		dgdValidationGate,
	)
	dgdRegistration.registerWithManager(
		mgr,
		&nvidiacomv1alpha1.DynamoGraphDeployment{},
		dynamoGraphDeploymentV1Alpha1WebhookPath,
		dgdAlphaValidation,
		dgdValidationGate,
	)
	return nil
}

type admissionTestEnvironment struct {
	environment *operatorenv.TestEnv
	client      dynamic.Interface
	warnings    *admissionWarningCollector
	userID      atomic.Uint64
}

func newAdmissionTestEnvironment(t *testing.T) *admissionTestEnvironment {
	t.Helper()
	environment := validationOperatorEnv.ForTest(t)
	warnings := &admissionWarningCollector{}
	config := environment.RESTConfig()
	config.WarningHandler = warnings
	client, err := dynamic.NewForConfig(config)
	if err != nil {
		t.Fatalf("create admission test client: %v", err)
	}
	return &admissionTestEnvironment{
		environment: environment,
		client:      client,
		warnings:    warnings,
	}
}

func (e *admissionTestEnvironment) useDynamoComponentDeploymentHandler(handler *DynamoComponentDeploymentHandler) {
	dcdBetaValidation.Set(handler)
	dcdAlphaValidation.Set(&dynamoComponentDeploymentV1Alpha1Handler{handler: handler})
}

func (e *admissionTestEnvironment) allowDynamoComponentDeployments() {
	dcdBetaValidation.Set(allowingValidator{})
	dcdAlphaValidation.Set(allowingValidator{})
}

func (e *admissionTestEnvironment) useDynamoGraphDeploymentHandler(handler *DynamoGraphDeploymentHandler, gate features.Gate) {
	dgdBetaValidation.Set(handler)
	dgdAlphaValidation.Set(&dynamoGraphDeploymentV1Alpha1Handler{handler: handler})
	dgdValidationGate.Set(gate)
}

func (e *admissionTestEnvironment) allowDynamoGraphDeployments() {
	dgdBetaValidation.Set(allowingValidator{})
	dgdAlphaValidation.Set(allowingValidator{})
}

func (e *admissionTestEnvironment) Admit(
	t *testing.T,
	oldObject runtime.Object,
	object runtime.Object,
	mutateRequest func(*testing.T, map[string]any),
	userInfo *authenticationv1.UserInfo,
	allow func(),
	configure func(),
) ([]string, error) {
	t.Helper()

	request := admissionUnstructured(t, object)
	if mutateRequest != nil {
		mutateRequest(t, request)
	}
	resourceInfo, err := admissionTarget(request)
	if err != nil {
		t.Fatal(err)
	}
	e.ensureNamespace(t, resourceInfo.namespace)
	resource := resourceInfo.ForClient(e.client)

	if oldObject != nil {
		if admissionSourceVersion(t, oldObject) != admissionSourceVersion(t, object) {
			t.Fatal("old and current source versions differ")
		}
		allow()
		oldRequest := admissionUnstructured(t, oldObject)
		seedRequest := (&unstructured.Unstructured{Object: oldRequest}).DeepCopy()

		// Restart is update-only, so reproduce its two-step API lifecycle when seeding old state.
		_, hasRestart, err := unstructured.NestedFieldNoCopy(seedRequest.Object, "spec", "restart")
		if err != nil {
			t.Fatalf("inspect old admission restart: %v", err)
		}
		if hasRestart {
			unstructured.RemoveNestedField(seedRequest.Object, "spec", "restart")
		}
		created, err := resource.Create(t.Context(), seedRequest, metav1.CreateOptions{})
		if err != nil {
			t.Fatalf("seed old admission object: %v", err)
		}
		e.cleanup(t, resource, created.GetName())
		if hasRestart {
			desiredOld := &unstructured.Unstructured{Object: oldRequest}
			desiredOld.SetResourceVersion(created.GetResourceVersion())
			created, err = resource.Update(t.Context(), desiredOld, metav1.UpdateOptions{})
			if err != nil {
				t.Fatalf("seed old admission restart: %v", err)
			}
		}
		created = e.updateStatus(t, resource, created, oldObject)
		if err := unstructured.SetNestedField(request, created.GetResourceVersion(), "metadata", "resourceVersion"); err != nil {
			t.Fatalf("set update resource version: %v", err)
		}
	}

	configure()
	client, err := e.clientForUser(userInfo)
	if err != nil {
		t.Fatalf("create admission client: %v", err)
	}
	resource = resourceInfo.ForClient(client)
	e.warnings.Reset()
	if oldObject == nil {
		created, err := resource.Create(t.Context(), &unstructured.Unstructured{Object: request}, metav1.CreateOptions{})
		if err == nil {
			e.cleanup(t, resource, created.GetName())
		}
		return e.warnings.List(), err
	}
	_, err = resource.Update(t.Context(), &unstructured.Unstructured{Object: request}, metav1.UpdateOptions{})
	return e.warnings.List(), err
}

type admissionResourceTarget struct {
	groupVersionResource schema.GroupVersionResource
	namespace            string
}

func (r admissionResourceTarget) ForClient(client dynamic.Interface) dynamic.ResourceInterface {
	return client.Resource(r.groupVersionResource).Namespace(r.namespace)
}

func admissionTarget(object map[string]any) (admissionResourceTarget, error) {
	apiVersion, _, err := unstructured.NestedString(object, "apiVersion")
	if err != nil || apiVersion == "" {
		return admissionResourceTarget{}, fmt.Errorf("admission object has no apiVersion")
	}
	groupVersion, err := schema.ParseGroupVersion(apiVersion)
	if err != nil {
		return admissionResourceTarget{}, fmt.Errorf("parse admission apiVersion %q: %w", apiVersion, err)
	}
	kind, _, err := unstructured.NestedString(object, "kind")
	if err != nil || kind == "" {
		return admissionResourceTarget{}, fmt.Errorf("admission object has no kind")
	}
	var resource string
	switch kind {
	case admissionDCDKind:
		resource = "dynamocomponentdeployments"
	case admissionDGDKind:
		resource = "dynamographdeployments"
	default:
		return admissionResourceTarget{}, fmt.Errorf("unsupported admission kind %q", kind)
	}
	namespace, _, err := unstructured.NestedString(object, "metadata", "namespace")
	if err != nil || namespace == "" {
		return admissionResourceTarget{}, fmt.Errorf("admission object has no namespace")
	}
	return admissionResourceTarget{
		groupVersionResource: groupVersion.WithResource(resource),
		namespace:            namespace,
	}, nil
}

func (e *admissionTestEnvironment) updateStatus(
	t *testing.T,
	resource dynamic.ResourceInterface,
	created *unstructured.Unstructured,
	oldObject runtime.Object,
) *unstructured.Unstructured {
	t.Helper()
	oldWithStatus, err := runtime.DefaultUnstructuredConverter.ToUnstructured(oldObject)
	if err != nil {
		t.Fatalf("convert old admission status: %v", err)
	}
	status, found := oldWithStatus["status"]
	if !found || !hasNonZeroAdmissionStatus(status) {
		return created
	}
	withStatus := created.DeepCopy()
	withStatus.Object["status"] = status
	updated, err := resource.UpdateStatus(t.Context(), withStatus, metav1.UpdateOptions{})
	if err != nil {
		t.Fatalf("seed old admission status: %v", err)
	}
	return updated
}

func hasNonZeroAdmissionStatus(value any) bool {
	switch value := value.(type) {
	case nil:
		return false
	case map[string]any:
		for _, child := range value {
			if hasNonZeroAdmissionStatus(child) {
				return true
			}
		}
		return false
	case []any:
		for _, child := range value {
			if hasNonZeroAdmissionStatus(child) {
				return true
			}
		}
		return false
	case string:
		return value != ""
	case bool:
		return value
	case int64:
		return value != 0
	case int32:
		return value != 0
	case int:
		return value != 0
	case float64:
		return value != 0
	case float32:
		return value != 0
	default:
		return true
	}
}

func (e *admissionTestEnvironment) ensureNamespace(t *testing.T, namespace string) {
	t.Helper()
	namespaces := e.client.Resource(schema.GroupVersionResource{Version: "v1", Resource: "namespaces"})
	if _, err := namespaces.Get(t.Context(), namespace, metav1.GetOptions{}); err == nil {
		return
	} else if !apierrors.IsNotFound(err) {
		t.Fatalf("get admission namespace %q: %v", namespace, err)
	}
	_, err := namespaces.Create(t.Context(), &unstructured.Unstructured{Object: map[string]any{
		"apiVersion": "v1",
		"kind":       "Namespace",
		"metadata": map[string]any{
			"name": namespace,
		},
	}}, metav1.CreateOptions{})
	if err != nil && !apierrors.IsAlreadyExists(err) {
		t.Fatalf("create admission namespace %q: %v", namespace, err)
	}
}

func (e *admissionTestEnvironment) cleanup(t *testing.T, resource dynamic.ResourceInterface, name string) {
	t.Helper()
	t.Cleanup(func() {
		if err := resource.Delete(context.Background(), name, metav1.DeleteOptions{}); err != nil && !apierrors.IsNotFound(err) {
			t.Errorf("delete admitted object %q: %v", name, err)
		}
	})
}

func assertAdmissionErrors(t *testing.T, err error, schemaErr, celErr string, webhookErrs []string) {
	t.Helper()

	want := webhookErrs
	if schemaErr != "" {
		if celErr != "" || len(webhookErrs) != 0 {
			t.Fatal("schema rejection cannot have downstream expectations")
		}
		want = []string{schemaErr}
	} else if celErr != "" {
		if len(webhookErrs) != 0 {
			t.Fatal("CEL rejection cannot have webhook expectations")
		}
		want = []string{celErr}
	}

	if len(want) == 0 {
		if err != nil {
			t.Fatalf("admission error = %v, want none", err)
		}
		return
	}
	if err == nil {
		t.Fatalf("admission errors = nil, want %v", want)
	}
	statusErr, ok := err.(*apierrors.StatusError)
	if !ok || !apierrors.IsInvalid(err) {
		t.Fatalf("error = %T %v, want typed Kubernetes invalid error", err, err)
	}
	if statusErr.ErrStatus.Details == nil {
		t.Fatalf("error = %v, want typed field causes", err)
	}

	causes := statusErr.ErrStatus.Details.Causes
	got := make([]string, 0, len(causes))
	for _, cause := range causes {
		// Ignore the API server's unfielded cascade notice after a primary schema error.
		if strings.Contains(cause.Message, "some validation rules were not checked because the object was invalid") {
			continue
		}
		if cause.Field == "" {
			t.Fatalf("error cause = %#v, want an exact field path", cause)
		}
		got = append(got, fmt.Sprintf("%s: %s", cause.Field, cause.Message))
	}
	if !slices.Equal(got, want) {
		t.Fatalf("admission errors = %v, want %v", got, want)
	}
}

func assertBetaValidationErrors(t *testing.T, err error, want []string) {
	t.Helper()
	assertAdmissionErrors(t, err, "", "", want)
}

func (e *admissionTestEnvironment) clientForUser(userInfo *authenticationv1.UserInfo) (dynamic.Interface, error) {
	if userInfo == nil {
		return e.client, nil
	}
	if userInfo.UID != "" || len(userInfo.Extra) != 0 {
		return nil, fmt.Errorf("envtest admission users support username and groups only")
	}
	config, err := e.environment.AddUser(envtest.User{Name: userInfo.Username, Groups: userInfo.Groups})
	if err != nil {
		return nil, err
	}
	if err := e.authorizeTestUser(userInfo.Username); err != nil {
		return nil, err
	}
	config.WarningHandler = e.warnings
	return dynamic.NewForConfig(config)
}

func (e *admissionTestEnvironment) authorizeTestUser(username string) error {
	// Bind by username so authorization does not add privileged groups to the admission identity.
	bindingName := fmt.Sprintf("admission-test-%s-cluster-admin-%d", e.environment.Namespace(), e.userID.Add(1))
	bindings := e.client.Resource(schema.GroupVersionResource{
		Group:    "rbac.authorization.k8s.io",
		Version:  "v1",
		Resource: "clusterrolebindings",
	})
	_, err := bindings.Create(context.Background(), &unstructured.Unstructured{Object: map[string]any{
		"apiVersion": "rbac.authorization.k8s.io/v1",
		"kind":       "ClusterRoleBinding",
		"metadata": map[string]any{
			"name": bindingName,
		},
		"roleRef": map[string]any{
			"apiGroup": "rbac.authorization.k8s.io",
			"kind":     "ClusterRole",
			"name":     "cluster-admin",
		},
		"subjects": []any{map[string]any{
			"apiGroup": "rbac.authorization.k8s.io",
			"kind":     "User",
			"name":     username,
		}},
	}}, metav1.CreateOptions{})
	if err != nil {
		return fmt.Errorf("authorize admission test user %q: %w", username, err)
	}
	return nil
}

type admissionWarningCollector struct {
	lock     sync.Mutex
	warnings []string
}

func (c *admissionWarningCollector) HandleWarningHeader(_ int, _ string, text string) {
	c.lock.Lock()
	defer c.lock.Unlock()
	c.warnings = append(c.warnings, text)
}

func (c *admissionWarningCollector) Reset() {
	c.lock.Lock()
	defer c.lock.Unlock()
	c.warnings = nil
}

func (c *admissionWarningCollector) List() []string {
	c.lock.Lock()
	defer c.lock.Unlock()
	warnings := make([]string, 0, len(c.warnings))
	for _, warning := range c.warnings {
		// The matrices assert webhook warnings, not Kubernetes' built-in API deprecation warning.
		if strings.HasPrefix(warning, "nvidia.com/v1alpha1 ") && strings.Contains(warning, " is deprecated; use nvidia.com/v1beta1 ") {
			continue
		}
		warnings = append(warnings, warning)
	}
	return warnings
}

type validatorSlot struct {
	lock      sync.RWMutex
	validator admission.CustomValidator
}

type featureGateSlot struct {
	lock sync.RWMutex
	gate features.Gate
}

func newFeatureGateSlot(gate features.Gate) *featureGateSlot {
	return &featureGateSlot{gate: gate}
}

func (s *featureGateSlot) Set(gate features.Gate) {
	s.lock.Lock()
	defer s.lock.Unlock()
	s.gate = gate
}

func (s *featureGateSlot) Enabled(name features.Name) bool {
	s.lock.RLock()
	defer s.lock.RUnlock()
	return s.gate.Enabled(name)
}

func newValidatorSlot() *validatorSlot {
	return &validatorSlot{validator: allowingValidator{}}
}

func (s *validatorSlot) Set(validator admission.CustomValidator) {
	s.lock.Lock()
	defer s.lock.Unlock()
	s.validator = validator
}

func (s *validatorSlot) ValidateCreate(ctx context.Context, object runtime.Object) (admission.Warnings, error) {
	s.lock.RLock()
	defer s.lock.RUnlock()
	return s.validator.ValidateCreate(ctx, object)
}

func (s *validatorSlot) ValidateUpdate(ctx context.Context, oldObject, object runtime.Object) (admission.Warnings, error) {
	s.lock.RLock()
	defer s.lock.RUnlock()
	return s.validator.ValidateUpdate(ctx, oldObject, object)
}

func (s *validatorSlot) ValidateDelete(ctx context.Context, object runtime.Object) (admission.Warnings, error) {
	s.lock.RLock()
	defer s.lock.RUnlock()
	return s.validator.ValidateDelete(ctx, object)
}

type allowingValidator struct{}

func (allowingValidator) ValidateCreate(context.Context, runtime.Object) (admission.Warnings, error) {
	return nil, nil
}

func (allowingValidator) ValidateUpdate(context.Context, runtime.Object, runtime.Object) (admission.Warnings, error) {
	return nil, nil
}

func (allowingValidator) ValidateDelete(context.Context, runtime.Object) (admission.Warnings, error) {
	return nil, nil
}

var _ admission.CustomValidator = &validatorSlot{}
var _ admission.CustomValidator = allowingValidator{}
var _ rest.WarningHandler = &admissionWarningCollector{}
