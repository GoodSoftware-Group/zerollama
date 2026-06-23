package fleet

import (
	"context"
	"sync/atomic"
	"testing"
	"time"

	"github.com/ollama/ollama/api"
)

func TestProbeCacheReusesFreshEntry(t *testing.T) {
	cache := NewProbeCache(time.Second)
	var calls atomic.Int32

	fetch := func(ctx context.Context) (*api.StatusResponse, error) {
		calls.Add(1)
		return &api.StatusResponse{}, nil
	}

	ctx := context.Background()
	if _, err := cache.Fetch(ctx, "http://a:11434", fetch); err != nil {
		t.Fatal(err)
	}
	if _, err := cache.Fetch(ctx, "http://a:11434", fetch); err != nil {
		t.Fatal(err)
	}
	if got := calls.Load(); got != 1 {
		t.Fatalf("fetch calls=%d want 1", got)
	}
}

func TestProbeCacheCoalescesInflight(t *testing.T) {
	cache := NewProbeCache(time.Second)
	started := make(chan struct{})
	var calls atomic.Int32

	fetch := func(ctx context.Context) (*api.StatusResponse, error) {
		calls.Add(1)
		close(started)
		time.Sleep(50 * time.Millisecond)
		return &api.StatusResponse{}, nil
	}

	ctx := context.Background()
	errCh := make(chan error, 2)
	go func() {
		_, err := cache.Fetch(ctx, "http://a:11434", fetch)
		errCh <- err
	}()
	<-started
	go func() {
		_, err := cache.Fetch(ctx, "http://a:11434", fetch)
		errCh <- err
	}()
	for i := 0; i < 2; i++ {
		if err := <-errCh; err != nil {
			t.Fatal(err)
		}
	}
	if got := calls.Load(); got != 1 {
		t.Fatalf("fetch calls=%d want 1", got)
	}
}
