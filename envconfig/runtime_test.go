package envconfig

import (
	"runtime"
	"testing"
)

func TestRuntimeDefaultOn(t *testing.T) {
	t.Setenv("ZEROLLAMA_RUNTIME_URL", "")
	t.Setenv("ZEROLLAMA_RUNTIME", "")
	if RuntimeDefaultOn() {
		t.Fatal("no URL")
	}

	t.Setenv("ZEROLLAMA_RUNTIME_URL", "http://127.0.0.1:8081")
	t.Setenv("ZEROLLAMA_RUNTIME", "")
	if !RuntimeDefaultOn() {
		t.Fatal("implicit on")
	}

	t.Setenv("ZEROLLAMA_RUNTIME", "0")
	if RuntimeDefaultOn() {
		t.Fatal("explicit off")
	}

	t.Setenv("ZEROLLAMA_RUNTIME", "1")
	if !RuntimeDefaultOn() {
		t.Fatal("explicit on")
	}
}

func TestRuntimeEmbedDisabledOnDarwinByDefault(t *testing.T) {
	if runtime.GOOS != "darwin" {
		t.Skip("darwin-only default")
	}
	t.Setenv("ZEROLLAMA_RUNTIME_EMBED", "")
	t.Setenv("ZEROLLAMA_RUNTIME_URL", "")
	if RuntimeEmbedEnabled() {
		t.Fatal("embed should default off on darwin")
	}
	t.Setenv("ZEROLLAMA_RUNTIME_EMBED", "1")
	if !RuntimeEmbedEnabled() {
		t.Fatal("embed explicit on")
	}
}
