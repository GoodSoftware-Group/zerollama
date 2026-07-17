package llamarunner

import "testing"

func TestGetPrecomputedChunks_sessionHit(t *testing.T) {
	c := &ImageContext{
		precomputed:   make([]precomputedCache, 4),
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
		precomputed:   make([]precomputedCache, 4),
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

func TestSessionPrecomputed_hashCapAndShare(t *testing.T) {
	t.Setenv("OLLAMA_IMAGE_EMBED_CACHE_MAX", "2")
	c := &ImageContext{
		precomputed:   make([]precomputedCache, 4),
		sessionEmbeds: make(map[string]sessionEmbedState),
	}
	const session = "agent-1"
	a := visionChunksFromPrecomputed([][]float32{{1}})
	b := visionChunksFromPrecomputed([][]float32{{2}})
	d := visionChunksFromPrecomputed([][]float32{{3}})

	c.mu.Lock()
	c.storeSessionPrecomputedLocked(session, 1, a)
	c.storeSessionPrecomputedLocked(session, 2, b)
	// Same slice as store — no clone on write.
	if &c.sessionEmbeds[session].precomputedByHash[1][0].embed[0] != &a[0].embed[0] {
		t.Fatal("session precomputed should share with store arg")
	}
	c.storeSessionPrecomputedLocked(session, 3, d)
	_, ok1 := c.findSessionPrecomputedLocked(session, 1)
	_, ok2 := c.findSessionPrecomputedLocked(session, 2)
	_, ok3 := c.findSessionPrecomputedLocked(session, 3)
	c.mu.Unlock()

	if ok1 {
		t.Fatal("hash 1 should be evicted at precomputed session hash cap")
	}
	if !ok2 || !ok3 {
		t.Fatalf("expected hashes 2 and 3: ok2=%v ok3=%v", ok2, ok3)
	}
}
