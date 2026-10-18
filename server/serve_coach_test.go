package server

import "testing"

func TestLocalManifestCountSmoke(t *testing.T) {
	// Must not panic when models dir is empty/missing in CI.
	_ = localManifestCount()
}

func TestLibllamaLikelyPresentNoEnv(t *testing.T) {
	t.Setenv("LLAMA_CPP_LIB", "")
	t.Setenv("LLAMA_CPP_ROOT", "/nonexistent")
	// Result depends on workspace; just ensure it returns without panic.
	_ = libllamaLikelyPresent()
}
