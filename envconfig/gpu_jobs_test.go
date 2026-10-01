package envconfig

import "testing"

func TestGPUJobsEnabledAlias(t *testing.T) {
	t.Setenv("ZEROLLAMA_GPU_JOBS", "")
	t.Setenv("OLLAMA_TRAINING", "")
	if !GPUJobsEnabled(true) {
		t.Fatal("expected default true")
	}
	if GPUJobsEnabled(false) {
		t.Fatal("expected default false when requested")
	}

	t.Setenv("OLLAMA_TRAINING", "false")
	if GPUJobsEnabled(true) {
		t.Fatal("OLLAMA_TRAINING=false should disable")
	}

	t.Setenv("ZEROLLAMA_GPU_JOBS", "true")
	t.Setenv("OLLAMA_TRAINING", "false")
	if !GPUJobsEnabled(false) {
		t.Fatal("ZEROLLAMA_GPU_JOBS should win over OLLAMA_TRAINING alias")
	}

	t.Setenv("ZEROLLAMA_GPU_JOBS", "false")
	t.Setenv("OLLAMA_TRAINING", "true")
	if GPUJobsEnabled(true) {
		t.Fatal("ZEROLLAMA_GPU_JOBS=false should win over alias true")
	}

	// Deprecated name remains wired to the same resolver.
	t.Setenv("ZEROLLAMA_GPU_JOBS", "")
	t.Setenv("OLLAMA_TRAINING", "true")
	if !TrainingEnabled(false) {
		t.Fatal("TrainingEnabled should accept OLLAMA_TRAINING alias")
	}
}
