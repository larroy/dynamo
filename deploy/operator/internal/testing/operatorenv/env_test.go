/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package operatorenv

import (
	"strings"
	"testing"

	configv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/config/v1alpha1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/features"
	"k8s.io/client-go/rest"
	"k8s.io/utils/ptr"
)

func TestWebhookInstallOptionsSelectsValidatingConfiguration(t *testing.T) {
	t.Log("Render only the validating admission registration")
	install, err := webhookInstallOptions(Options{
		Admission: AdmissionWebhooks{Validating: true},
	})
	if err != nil {
		t.Fatalf("render webhook install options: %v", err)
	}

	t.Log("Assert mutating admission is omitted while validation remains installed")
	if len(install.MutatingWebhooks) != 0 {
		t.Fatalf("mutating webhook configurations = %d, want 0", len(install.MutatingWebhooks))
	}
	if len(install.ValidatingWebhooks) == 0 {
		t.Fatal("validating webhook configurations = 0, want at least 1")
	}
}

func TestWebhookSetupIsRequiredWhenWebhooksAreEnabled(t *testing.T) {
	t.Log("Start an environment with validation enabled but no handler setup")
	_, err := startRuntime(normalizeOptions(Options{
		Admission: AdmissionWebhooks{Validating: true},
	}))

	t.Log("Assert startup fails before creating the API server")
	if err == nil || !strings.Contains(err.Error(), "SetupWebhooks is required") {
		t.Fatalf("startRuntime() error = %v, want missing SetupWebhooks error", err)
	}
}

func TestRESTConfigReturnsCopy(t *testing.T) {
	t.Log("Create a test environment around a shared REST configuration")
	env := &TestEnv{rt: &runtimeEnv{config: &rest.Config{Host: "https://original.example"}}}

	t.Log("Mutate the returned configuration")
	config := env.RESTConfig()
	config.Host = "https://modified.example"

	t.Log("Assert the shared configuration remains unchanged")
	if env.rt.config.Host != "https://original.example" {
		t.Fatalf("shared REST config host = %q, want original value", env.rt.config.Host)
	}
}

func TestDefaultOperatorConfigPreservesExplicitGPUDiscovery(t *testing.T) {
	t.Log("Create an operator configuration with GPU discovery explicitly enabled")
	config := &configv1alpha1.OperatorConfiguration{
		GPU: configv1alpha1.GPUConfiguration{
			DiscoveryEnabled: ptr.To(true),
		},
	}

	t.Log("Build the envtest operator configuration")
	got := defaultOperatorConfig(config)

	t.Log("Assert the explicit GPU discovery setting is preserved")
	if got.GPU.DiscoveryEnabled == nil || !*got.GPU.DiscoveryEnabled {
		t.Fatalf("GPU discovery enabled = %v, want true", got.GPU.DiscoveryEnabled)
	}
}

func TestDefaultRuntimeConfigDerivesStaticFeatureGates(t *testing.T) {
	t.Log("Derive runtime gates for a cluster-wide environment with GPU discovery disabled in config")
	clusterWide := &configv1alpha1.OperatorConfiguration{
		GPU: configv1alpha1.GPUConfiguration{DiscoveryEnabled: ptr.To(false)},
	}
	clusterWideRuntime := defaultRuntimeConfig(clusterWide)

	t.Log("Assert cluster-wide mode still enables GPU discovery like production")
	if !clusterWideRuntime.Gate.Enabled(features.GPUDiscovery) {
		t.Fatal("cluster-wide GPU discovery gate = false, want true")
	}

	t.Log("Derive runtime gates for a namespace-restricted environment")
	restricted := clusterWide.DeepCopy()
	restricted.Namespace.Restricted = "test-namespace"
	restricted.Checkpoint.Enabled = true
	restrictedRuntime := defaultRuntimeConfig(restricted)

	t.Log("Assert restricted mode honors GPU discovery config and the checkpoint setting")
	if restrictedRuntime.Gate.Enabled(features.GPUDiscovery) {
		t.Fatal("namespace-restricted GPU discovery gate = true, want false")
	}
	if !restrictedRuntime.Gate.Enabled(features.Checkpoint) {
		t.Fatal("checkpoint gate = false, want true")
	}
}
