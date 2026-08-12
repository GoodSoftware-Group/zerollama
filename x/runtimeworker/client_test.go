package runtimeworker

import (
	"context"
	"net"
	"testing"
	"time"
)

func TestCheckLoopbackPortFree(t *testing.T) {
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	port := ln.Addr().(*net.TCPAddr).Port
	if err := checkLoopbackPortFree(port); err == nil {
		t.Fatal("expected port busy while listener held")
	}
	ln.Close()
	if err := checkLoopbackPortFree(port); err != nil {
		t.Fatalf("expected port free after close: %v", err)
	}
}

func TestHealthEmbedBoot(t *testing.T) {
	if got := healthEmbedBoot([]byte(`{"embed_boot":"abc"}`)); got != "abc" {
		t.Fatalf("got %q", got)
	}
	if got := healthEmbedBoot([]byte(`{"status":"ok"}`)); got != "" {
		t.Fatalf("got %q", got)
	}
}

func TestEmbedReadySyncWaitDefault(t *testing.T) {
	t.Setenv("ZEROLLAMA_RUNTIME_EMBED_SYNC_WAIT", "")
	if got := embedReadySyncWait(); got != 3*time.Second {
		t.Fatalf("got %v", got)
	}
	t.Setenv("ZEROLLAMA_RUNTIME_EMBED_SYNC_WAIT", "500ms")
	if got := embedReadySyncWait(); got != 500*time.Millisecond {
		t.Fatalf("got %v", got)
	}
}

func TestWaitEmbedHealthyTimeout(t *testing.T) {
	ctx := context.Background()
	// Nothing listening — must return false quickly.
	ok := waitEmbedHealthy(ctx, "http://127.0.0.1:1", "boot", 300*time.Millisecond)
	if ok {
		t.Fatal("expected unhealthy")
	}
}
