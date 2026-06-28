package llamarunner

import "testing"

func TestGetPrecomputedChunks_sessionHit(t *testing.T) {
	c := &ImageContext{
		precomputed: make([]precomputedCache, 4),
		sessionEmbeds: make(map[string]sessionEmbedState),
	}
	rows := [][]float32{{1, 2}, {3, 4}}
	hash := hashPrecomputedRows(rows)

	c.mu.Lock()
	c.storeSessionPrecomputedLocked("agent-1", hash, visionChunksFromPrecomputed(rows))
	c.mu.Unlock()

	vc1, err := c.GetPrecomputedChunks(rows, "agent-1", true)
	if err != nil || len(vc1) != 2 {
		t.Fatalf("first get: vc=%v err=%v", vc1, err)
	}
	vc2, err := c.GetPrecomputedChunks(rows, "agent-1", true)
	if err != nil || len(vc2) != 2 {
		t.Fatalf("session hit: vc=%v err=%v", vc2, err)
	}
}

func TestGetPrecomputedChunks_overlayOffSkipsSession(t *testing.T) {
	c := &ImageContext{
		precomputed: make([]precomputedCache, 4),
		sessionEmbeds: make(map[string]sessionEmbedState),
	}
	rows := [][]float32{{9}}
	hash := hashPrecomputedRows(rows)

	c.mu.Lock()
	c.storeSessionPrecomputedLocked("agent-1", hash, visionChunksFromPrecomputed(rows))
	c.mu.Unlock()

	vc, err := c.GetPrecomputedChunks(rows, "agent-1", false)
	if err != nil || len(vc) != 1 {
		t.Fatalf("materialize: vc=%v err=%v", vc, err)
	}
}
