package discover

import (
	"runtime"
	"strings"
	"testing"
)

func TestUpsertEnv(t *testing.T) {
	env := []string{"FOO=1", "ZEROLLAMA_ANE_DRAFT_DRIVE=shadow"}
	env = upsertEnv(env, "ZEROLLAMA_ANE_DRAFT_DRIVE", "force")
	env = upsertEnv(env, "BAR", "2")
	if len(env) != 3 {
		t.Fatalf("len=%d", len(env))
	}
	if v, ok := aneDraftEnvValue(env, "ZEROLLAMA_ANE_DRAFT_DRIVE"); !ok || v != "force" {
		t.Fatalf("drive=%q ok=%v", v, ok)
	}
	if v, ok := aneDraftEnvValue(env, "BAR"); !ok || v != "2" {
		t.Fatalf("bar=%q", v)
	}
}

func TestDedupeEnvLastWins(t *testing.T) {
	env := []string{
		"ZEROLLAMA_ANE_DRAFT_MATMUL_OC=512",
		"FOO=1",
		"ZEROLLAMA_ANE_DRAFT_MATMUL_OC=5120",
	}
	out := dedupeEnvLastWins(env)
	if v, ok := aneDraftEnvValue(out, "ZEROLLAMA_ANE_DRAFT_MATMUL_OC"); !ok || v != "5120" {
		t.Fatalf("matmul_oc=%q ok=%v env=%v", v, ok, out)
	}
	if len(out) != 2 {
		t.Fatalf("len=%d want 2: %v", len(out), out)
	}
}

func TestPrependLlamaServerLibPath(t *testing.T) {
	bin := "/tmp/llama/build/bin/llama-server"
	env := prependLlamaServerLibPath([]string{"HOME=/x"}, bin)
	key := "LD_LIBRARY_PATH"
	if runtime.GOOS == "darwin" {
		key = "DYLD_LIBRARY_PATH"
	} else if runtime.GOOS == "windows" {
		key = "PATH"
	}
	found := false
	for _, e := range env {
		k, v, ok := strings.Cut(e, "=")
		if ok && k == key {
			found = true
			if !strings.HasPrefix(v, "/tmp/llama/build/bin") {
				t.Fatalf("%s=%q", key, v)
			}
		}
	}
	if !found {
		t.Fatalf("missing %s in %#v", key, env)
	}
	env2 := prependLlamaServerLibPath([]string{key + "=/old"}, bin)
	_, v, _ := strings.Cut(env2[0], "=")
	if !strings.HasPrefix(v, "/tmp/llama/build/bin") || !strings.Contains(v, "/old") {
		t.Fatalf("prepend existing = %q", v)
	}
}
