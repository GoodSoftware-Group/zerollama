package envconfig

import (
	"testing"
	"time"
)

func TestMemoryReclaimThreshold(t *testing.T) {
	t.Setenv("ZEROLLAMA_MEMORY_RECLAIM_THRESHOLD", "0.95")
	if got := MemoryReclaimThreshold(); got != 0.95 {
		t.Fatalf("got %v", got)
	}
	t.Setenv("ZEROLLAMA_MEMORY_RECLAIM_THRESHOLD", "bad")
	if got := MemoryReclaimThreshold(); got != 0 {
		t.Fatalf("invalid should disable, got %v", got)
	}
}

func TestRunnerBusyTimeout(t *testing.T) {
	t.Setenv("ZEROLLAMA_RUNNER_BUSY_TIMEOUT", "10m")
	if got := RunnerBusyTimeout(); got != 10*time.Minute {
		t.Fatalf("got %v", got)
	}
}

func TestLoadCooldownInitial(t *testing.T) {
	t.Setenv("ZEROLLAMA_LOAD_COOLDOWN", "")
	if got := LoadCooldownInitial(); got != 10*time.Second {
		t.Fatalf("default got %v", got)
	}
	t.Setenv("ZEROLLAMA_LOAD_COOLDOWN", "0")
	if got := LoadCooldownInitial(); got != 0 {
		t.Fatalf("off got %v", got)
	}
	t.Setenv("ZEROLLAMA_LOAD_COOLDOWN", "30s")
	if got := LoadCooldownInitial(); got != 30*time.Second {
		t.Fatalf("got %v", got)
	}
}

func TestRouterConfigPathOff(t *testing.T) {
	t.Setenv("ZEROLLAMA_ROUTER_CONFIG", "0")
	if RouterConfigPath() != "" {
		t.Fatal("0 should disable")
	}
	t.Setenv("ZEROLLAMA_ROUTER_CONFIG", "/tmp/r.yaml")
	if RouterConfigPath() != "/tmp/r.yaml" {
		t.Fatalf("got %s", RouterConfigPath())
	}
}

func TestAliasesConfigPathOff(t *testing.T) {
	t.Setenv("ZEROLLAMA_ALIASES_CONFIG", "0")
	if AliasesConfigPath() != "" {
		t.Fatal("0 should disable")
	}
}

func TestChatCompressionEnv(t *testing.T) {
	t.Setenv("ZEROLLAMA_CHAT_COMPRESSION", "")
	if ChatCompression() {
		t.Fatal("default off")
	}
	t.Setenv("ZEROLLAMA_CHAT_COMPRESSION", "1")
	if !ChatCompression() {
		t.Fatal("1 should enable")
	}
	t.Setenv("ZEROLLAMA_CHAT_COMPRESSOR", "tiny")
	if ChatCompressor() != "tiny" {
		t.Fatal(ChatCompressor())
	}
	t.Setenv("ZEROLLAMA_CHAT_COMPRESSION_MODE", "placeholder")
	if ChatCompressionMode() != "placeholder" {
		t.Fatal(ChatCompressionMode())
	}
}

func TestBackendParentWatch(t *testing.T) {
	t.Setenv("ZEROLLAMA_BACKEND_PARENT_WATCH", "")
	if !BackendParentWatch() {
		t.Fatal("default on")
	}
	t.Setenv("ZEROLLAMA_BACKEND_PARENT_WATCH", "0")
	if BackendParentWatch() {
		t.Fatal("0 should disable")
	}
}
