package envconfig

import (
	"os"
	"testing"
)

func TestApplyInferenceProfileDefaults_Throughput(t *testing.T) {
	clearInferenceProfileEnv(t)
	t.Setenv("ZEROLLAMA_INFERENCE_PROFILE", "throughput")

	ApplyInferenceProfileDefaults(false)

	if got := os.Getenv("ZEROLLAMA_GPU_PROFILE"); got != "1" {
		t.Fatalf("GPU_PROFILE=%q want 1", got)
	}
	if got := os.Getenv("ZEROLLAMA_LLAMA_FORK"); got != "0" {
		t.Fatalf("LLAMA_FORK=%q want 0", got)
	}
	if got := os.Getenv("ZEROLLAMA_LLAMA_CACHE"); got != "1" {
		t.Fatalf("LLAMA_CACHE=%q want 1", got)
	}
	if got := os.Getenv("ZEROLLAMA_L3_PROFILE"); got != "" {
		t.Fatalf("L3_PROFILE should stay unset for throughput, got %q", got)
	}
	_, resolved, applied := InferenceProfileStatus()
	if resolved != InferenceProfileThroughput {
		t.Fatalf("resolved=%q", resolved)
	}
	if len(applied) == 0 {
		t.Fatal("expected applied env list")
	}
}

func TestApplyInferenceProfileDefaults_Agent(t *testing.T) {
	clearInferenceProfileEnv(t)
	t.Setenv("ZEROLLAMA_INFERENCE_PROFILE", "agent")

	ApplyInferenceProfileDefaults(false)

	if got := os.Getenv("ZEROLLAMA_L3_PROFILE"); got != "agent" {
		t.Fatalf("L3_PROFILE=%q want agent", got)
	}
	if got := os.Getenv("ZEROLLAMA_RADIX_PREFIX_SHARE"); got != "1" {
		t.Fatalf("RADIX=%q want 1", got)
	}
}

func TestApplyInferenceProfileDefaults_VRAM(t *testing.T) {
	clearInferenceProfileEnv(t)
	t.Setenv("ZEROLLAMA_INFERENCE_PROFILE", "vram")

	ApplyInferenceProfileDefaults(false)

	if got := os.Getenv("ZEROLLAMA_LLAMA_FORK_AUTO_VRAM"); got != "1" {
		t.Fatalf("AUTO_VRAM=%q want 1", got)
	}
	if got := os.Getenv("ZEROLLAMA_LLAMA_FORK"); got != "0" {
		t.Fatalf("FORK must stay 0 for tok/s, got %q", got)
	}
}

func TestApplyInferenceProfileDefaults_RespectsExplicit(t *testing.T) {
	clearInferenceProfileEnv(t)
	t.Setenv("ZEROLLAMA_INFERENCE_PROFILE", "throughput")
	t.Setenv("ZEROLLAMA_GPU_PROFILE", "0")
	t.Setenv("ZEROLLAMA_LLAMA_CACHE", "0")

	ApplyInferenceProfileDefaults(false)

	if got := os.Getenv("ZEROLLAMA_GPU_PROFILE"); got != "0" {
		t.Fatalf("explicit GPU_PROFILE overwritten: %q", got)
	}
	if got := os.Getenv("ZEROLLAMA_LLAMA_CACHE"); got != "0" {
		t.Fatalf("explicit LLAMA_CACHE overwritten: %q", got)
	}
}

func TestApplyInferenceProfileDefaults_Off(t *testing.T) {
	clearInferenceProfileEnv(t)
	t.Setenv("ZEROLLAMA_INFERENCE_PROFILE", "off")

	ApplyInferenceProfileDefaults(true)

	if got := os.Getenv("ZEROLLAMA_GPU_PROFILE"); got != "" {
		t.Fatalf("off should not set GPU_PROFILE, got %q", got)
	}
}

func TestApplyInferenceProfileDefaults_AutoDefault(t *testing.T) {
	clearInferenceProfileEnv(t)
	ApplyInferenceProfileDefaults(true)
	_, resolved, _ := InferenceProfileStatus()
	if resolved != InferenceProfileThroughput {
		t.Fatalf("auto should resolve to throughput, got %q", resolved)
	}
	if got := os.Getenv("ZEROLLAMA_GPU_PROFILE"); got != "1" {
		t.Fatalf("auto GPU_PROFILE=%q", got)
	}
}

func clearInferenceProfileEnv(t *testing.T) {
	t.Helper()
	for _, k := range []string{
		"ZEROLLAMA_INFERENCE_PROFILE",
		"ZEROLLAMA_GPU_PROFILE",
		"ZEROLLAMA_LLAMA_FORK",
		"ZEROLLAMA_LLAMA_CACHE",
		"ZEROLLAMA_L3_PROFILE",
		"ZEROLLAMA_RADIX_PREFIX_SHARE",
		"ZEROLLAMA_LLAMA_FORK_AUTO_VRAM",
		"ZEROLLAMA_LLAMA_FORK_PROFILE",
		"GGML_CUDA_USE_GRAPHS",
	} {
		_ = os.Unsetenv(k)
	}
	t.Setenv("ZEROLLAMA_INFERENCE_PROFILE", "off")
	ApplyInferenceProfileDefaults(false)
	_ = os.Unsetenv("ZEROLLAMA_INFERENCE_PROFILE")
}
