package llm

import (
	"context"
	"errors"
	"testing"
	"time"
)

func TestIsBenignLlamaServerStreamError(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	if !isBenignLlamaServerStreamError(ctx, errors.New("read tcp: use of closed network connection")) {
		t.Fatal("expected canceled parent context to classify drain error as benign")
	}
	if !isBenignLlamaServerStreamError(context.Background(), context.Canceled) {
		t.Fatal("expected context.Canceled to be benign")
	}
	if isBenignLlamaServerStreamError(context.Background(), errors.New("unexpected EOF")) {
		t.Fatal("unexpected EOF should not be benign")
	}
}

func TestIsBenignLlamaServerStreamErrorDeadline(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), time.Millisecond)
	defer cancel()
	time.Sleep(2 * time.Millisecond)

	if !isBenignLlamaServerStreamError(ctx, errors.New("context deadline exceeded")) {
		t.Fatal("expected deadline exceeded to be benign when ctx expired")
	}
}
