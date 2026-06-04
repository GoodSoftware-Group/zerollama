package runtimeworker

import (
	"net"
	"testing"
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
