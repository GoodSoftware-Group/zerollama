package server

import (
	"testing"
	"time"

	"github.com/gin-gonic/gin"

	"github.com/ollama/ollama/api"
)

func TestEnqueueChatStreamErrorEmitsFinishBeforeError(t *testing.T) {
	ch := make(chan any, 4)
	var sentDone bool
	enqueueChatStreamError(ch, "m", &sentDone, "runner exited", 0)
	close(ch)

	var gotFinish, gotError bool
	for v := range ch {
		switch t := v.(type) {
		case api.ChatResponse:
			if t.Done && t.DoneReason == "stop" {
				gotFinish = true
			}
		case gin.H:
			if t["error"] == "runner exited" {
				gotError = true
			}
		}
	}
	if !gotFinish {
		t.Fatal("expected synthetic finish chunk before error")
	}
	if !gotError {
		t.Fatal("expected error chunk after finish")
	}
	if !sentDone {
		t.Fatal("expected sentDone to be set")
	}
}

func TestStreamKeepaliveInterval(t *testing.T) {
	t.Setenv("OLLAMA_STREAM_KEEPALIVE_INTERVAL", "")
	if got := streamKeepaliveInterval(); got != defaultStreamKeepaliveInterval {
		t.Fatalf("default = %v want %v", got, defaultStreamKeepaliveInterval)
	}
	t.Setenv("OLLAMA_STREAM_KEEPALIVE_INTERVAL", "0")
	if got := streamKeepaliveInterval(); got != 0 {
		t.Fatalf("disabled = %v want 0", got)
	}
	t.Setenv("OLLAMA_STREAM_KEEPALIVE_INTERVAL", "30")
	if got := streamKeepaliveInterval(); got != 30*time.Second {
		t.Fatalf("override = %v want 30s", got)
	}
	t.Setenv("OLLAMA_STREAM_KEEPALIVE_INTERVAL", "bogus")
	if got := streamKeepaliveInterval(); got != defaultStreamKeepaliveInterval {
		t.Fatalf("invalid = %v want default %v", got, defaultStreamKeepaliveInterval)
	}
}
