package envconfig

import "testing"

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
