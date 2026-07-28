package modality

import (
	"context"
	"sync/atomic"
	"testing"
	"time"

	"github.com/ollama/ollama/api"
)

func TestVideoExpandCache_preservesPaddedInputIDs(t *testing.T) {
	resetVideoExpandCache()
	resetSessionVideoExpandCache()

	policy := VideoSamplingPolicy{Mode: "fps", FPS: 1, MaxFrames: 4, MaxVideosPerMessage: 4, MaxImagesAfterExpand: 64, MaxBytes: 1 << 20}
	video := []byte{0x00, 0x00, 0x00, 0x18, 0x66, 0x74, 0x79, 0x70}
	padded := []int{101, 102, 103}

	var calls atomic.Int32
	orig := ExternalVideoDecodeHook
	defer func() { ExternalVideoDecodeHook = orig }()
	ExternalVideoDecodeHook = func(ctx context.Context, policy VideoSamplingPolicy, data []byte) ([]api.ImageData, error) {
		calls.Add(1)
		return []api.ImageData{{0x89, 0x50}}, nil
	}

	sessionKey := "layout-agent-1"
	req := &api.ChatRequest{
		Options: map[string]any{"prompt_cache_key": sessionKey},
		Messages: []api.Message{{
			Videos:         []api.VideoData{video},
			PaddedInputIDs: padded,
		}},
	}
	if err := ExpandVideosInChatRequest(context.Background(), policy, req); err != nil {
		t.Fatal(err)
	}
	if calls.Load() != 1 {
		t.Fatalf("first expand calls=%d, want 1", calls.Load())
	}

	entry, ok := lookupSessionVideoExpand(sessionKey, policy, video)
	if !ok {
		t.Fatal("expected session cache entry")
	}
	if len(entry.paddedInputIDs) != len(padded) {
		t.Fatalf("session padded len=%d want %d", len(entry.paddedInputIDs), len(padded))
	}
	for i := range padded {
		if entry.paddedInputIDs[i] != padded[i] {
			t.Fatalf("session padded[%d]=%d want %d", i, entry.paddedInputIDs[i], padded[i])
		}
	}

	global, ok := lookupVideoExpandCache(policy, video)
	if !ok {
		t.Fatal("expected global cache entry")
	}
	// Global cache intentionally does NOT store paddedInputIDs (it is client-specific).
	if len(global.paddedInputIDs) != 0 {
		t.Fatalf("global cache must not store padded_input_ids, got len=%d", len(global.paddedInputIDs))
	}

	// Turn 2: same clip, client omits padded_input_ids — restore from session cache.
	req2 := &api.ChatRequest{
		Options:  map[string]any{"prompt_cache_key": sessionKey},
		Messages: []api.Message{{Videos: []api.VideoData{video}}},
	}
	if err := ExpandVideosInChatRequest(context.Background(), policy, req2); err != nil {
		t.Fatal(err)
	}
	if calls.Load() != 1 {
		t.Fatalf("cached expand calls=%d, want still 1", calls.Load())
	}
	if len(req2.Messages[0].PaddedInputIDs) != len(padded) {
		t.Fatalf("restored padded len=%d want %d", len(req2.Messages[0].PaddedInputIDs), len(padded))
	}
	for i := range padded {
		if req2.Messages[0].PaddedInputIDs[i] != padded[i] {
			t.Fatalf("restored padded[%d]=%d want %d", i, req2.Messages[0].PaddedInputIDs[i], padded[i])
		}
	}
}

func TestGlobalVideoExpandCache_stitchesSessionLayoutAfterPromote(t *testing.T) {
	resetVideoExpandCache()
	resetSessionVideoExpandCache()

	policy := VideoSamplingPolicy{Mode: "fps", FPS: 1, MaxFrames: 4, MaxVideosPerMessage: 4, MaxImagesAfterExpand: 64, MaxBytes: 1 << 20}
	video := []byte{0x00, 0x00, 0x00, 0x18, 0x66, 0x74, 0x79, 0x70}
	padded := []int{201, 202}
	sessionKey := "layout-promote-1"

	rememberVideoExpandCache(policy, video, []api.ImageData{{0x89, 0x50}}, []int{1, 4, 4})
	rememberSessionVideoExpand(sessionKey, policy, video, []api.ImageData{{0x89, 0x50}}, []int{1, 4, 4}, padded)

	// Simulate session frame eviction while layout survives (LRU pressure).
	videoKey := videoExpandCacheKey(policy, video)
	globalSessionVideoExpandCache.mu.Lock()
	st := globalSessionVideoExpandCache.sessions[sessionKey]
	delete(st.videos, videoKey)
	globalSessionVideoExpandCache.sessions[sessionKey] = st
	globalSessionVideoExpandCache.mu.Unlock()

	sample, err := sampleVideoToPNGs(context.Background(), policy, sessionKey, video, nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(sample.frames) != 1 {
		t.Fatalf("frames=%d want 1 from global promote", len(sample.frames))
	}
	if len(sample.paddedInputIDs) != len(padded) {
		t.Fatalf("stitched padded len=%d want %d", len(sample.paddedInputIDs), len(padded))
	}
	for i := range padded {
		if sample.paddedInputIDs[i] != padded[i] {
			t.Fatalf("stitched padded[%d]=%d want %d", i, sample.paddedInputIDs[i], padded[i])
		}
	}
}

func TestSessionVideoExpandCache_evictionClearsLayout(t *testing.T) {
	resetSessionVideoExpandCache()
	policy := VideoSamplingPolicy{Mode: "fps", FPS: 1, MaxFrames: 4}
	video := []byte{0x01}
	frames := []api.ImageData{{0xaa}}
	padded := []int{9, 8, 7}

	rememberSessionVideoExpand("s1", policy, video, frames, []int{1, 4, 4}, padded)
	videoKey := videoExpandCacheKey(policy, video)

	st := globalSessionVideoExpandCache.sessions["s1"]
	evictOldestSessionVideoLocked(&st, time.Now().UTC())
	globalSessionVideoExpandCache.sessions["s1"] = st

	if _, ok := st.videos[videoKey]; ok {
		t.Fatal("expected video entry evicted")
	}
	if _, ok := st.layouts[videoKey]; ok {
		t.Fatal("expected layout evicted with video entry")
	}
}

func TestSessionVideoExpandCache_hitWhenGlobalEvicted(t *testing.T) {
	resetVideoExpandCache()
	resetSessionVideoExpandCache()

	policy := VideoSamplingPolicy{Mode: "fps", FPS: 1, MaxFrames: 8, MaxVideosPerMessage: 4, MaxImagesAfterExpand: 64, MaxBytes: 1 << 20}
	video := []byte{0x00, 0x00, 0x00, 0x18, 0x66, 0x74, 0x79, 0x70}

	var calls atomic.Int32
	orig := ExternalVideoDecodeHook
	defer func() { ExternalVideoDecodeHook = orig }()
	ExternalVideoDecodeHook = func(ctx context.Context, policy VideoSamplingPolicy, data []byte) ([]api.ImageData, error) {
		calls.Add(1)
		return []api.ImageData{{0x89, 0x50}}, nil
	}

	sessionKey := "agent-thread-1"
	req := &api.ChatRequest{
		Options:  map[string]any{"prompt_cache_key": sessionKey},
		Messages: []api.Message{{Videos: []api.VideoData{video}}},
	}
	if err := ExpandVideosInChatRequest(context.Background(), policy, req); err != nil {
		t.Fatal(err)
	}
	if calls.Load() != 1 {
		t.Fatalf("first expand calls=%d, want 1", calls.Load())
	}

	// Simulate global LRU eviction while session cache still holds the clip.
	resetVideoExpandCache()

	req2 := &api.ChatRequest{
		Options:  map[string]any{"prompt_cache_key": sessionKey},
		Messages: []api.Message{{Videos: []api.VideoData{video}}},
	}
	if err := ExpandVideosInChatRequest(context.Background(), policy, req2); err != nil {
		t.Fatal(err)
	}
	if calls.Load() != 1 {
		t.Fatalf("session cached expand calls=%d, want still 1", calls.Load())
	}
	if len(req2.Messages[0].Images) != 1 {
		t.Fatalf("cached frames=%d, want 1", len(req2.Messages[0].Images))
	}
}

func TestExtractPromptCacheKey_elizaAndFlat(t *testing.T) {
	if got := ExtractPromptCacheKey(map[string]any{
		"eliza": map[string]any{"promptCacheKey": "eliza-key"},
	}); got != "eliza-key" {
		t.Fatalf("eliza promptCacheKey=%q", got)
	}
	if got := ExtractPromptCacheKey(map[string]any{"prompt_cache_key": "flat-key"}); got != "flat-key" {
		t.Fatalf("flat key=%q", got)
	}
	if got := ExtractPromptCacheKey(map[string]any{"session_id": "sgl-sess"}); got != "sgl-sess" {
		t.Fatalf("session_id=%q", got)
	}
	if got := ExtractPromptCacheKey(map[string]any{
		"prompt_cache_key": "p",
		"session_id":       "s",
	}); got != "p" {
		t.Fatalf("prompt_cache_key should win over session_id, got %q", got)
	}
}

func TestExpandVideosInChatRequest_skipsPreprocessed(t *testing.T) {
	resetVideoExpandCache()
	resetSessionVideoExpandCache()

	var calls atomic.Int32
	orig := ExternalVideoDecodeHook
	defer func() { ExternalVideoDecodeHook = orig }()
	ExternalVideoDecodeHook = func(ctx context.Context, policy VideoSamplingPolicy, data []byte) ([]api.ImageData, error) {
		calls.Add(1)
		return nil, nil
	}

	req := &api.ChatRequest{
		Messages: []api.Message{{
			Images:     []api.ImageData{{1}, {2}},
			VideoSpans: []api.VideoSpan{{FrameCount: 2}},
		}},
	}
	if err := ExpandVideosInChatRequest(context.Background(), VideoSamplingPolicy{MaxVideosPerMessage: 4, MaxImagesAfterExpand: 64, MaxBytes: 1 << 20}, req); err != nil {
		t.Fatal(err)
	}
	if calls.Load() != 0 {
		t.Fatalf("preprocessed expand calls=%d, want 0", calls.Load())
	}
}

func TestExpandVideosInChatRequest_rejectsInvalidPreprocessedSpans(t *testing.T) {
	req := &api.ChatRequest{
		Messages: []api.Message{{
			Images:     []api.ImageData{{1}},
			VideoSpans: []api.VideoSpan{{FrameCount: 2}},
		}},
	}
	if err := ExpandVideosInChatRequest(context.Background(), VideoSamplingPolicy{MaxVideosPerMessage: 4, MaxImagesAfterExpand: 64, MaxBytes: 1 << 20}, req); err == nil {
		t.Fatal("expected error for inconsistent video_spans")
	}
}

func TestExpandVideosInChatRequest_agentSecondTurn(t *testing.T) {
	resetVideoExpandCache()
	resetSessionVideoExpandCache()

	var calls atomic.Int32
	orig := ExternalVideoDecodeHook
	defer func() { ExternalVideoDecodeHook = orig }()
	ExternalVideoDecodeHook = func(ctx context.Context, policy VideoSamplingPolicy, data []byte) ([]api.ImageData, error) {
		calls.Add(1)
		return []api.ImageData{{0x89, 0x50}, {0x89, 0x51}}, nil
	}

	policy := VideoSamplingPolicy{Mode: "fps", FPS: 1, MaxFrames: 8, MaxVideosPerMessage: 4, MaxImagesAfterExpand: 64, MaxBytes: 1 << 20}
	video := []byte{0x00, 0x00, 0x00, 0x18, 0x66, 0x74, 0x79, 0x70}
	opts := map[string]any{"prompt_cache_key": "video-agent-thread-1"}

	turn1 := &api.ChatRequest{
		Options:  opts,
		Messages: []api.Message{{Role: "user", Content: "describe", Videos: []api.VideoData{video}}},
	}
	if err := ExpandVideosInChatRequest(context.Background(), policy, turn1); err != nil {
		t.Fatal(err)
	}
	if calls.Load() != 1 {
		t.Fatalf("turn1 decode calls=%d want 1", calls.Load())
	}

	// Agent turn 2: history without raw video; user re-sends the same clip on the latest turn.
	turn2 := &api.ChatRequest{
		Options: opts,
		Messages: []api.Message{
			{Role: "user", Content: "describe"},
			{Role: "assistant", Content: "a test pattern"},
			{Role: "user", Content: "again", Videos: []api.VideoData{video}},
		},
	}
	if err := ExpandVideosInChatRequest(context.Background(), policy, turn2); err != nil {
		t.Fatal(err)
	}
	if calls.Load() != 1 {
		t.Fatalf("turn2 decode calls=%d want 1 (session cache on resend)", calls.Load())
	}
	if len(turn2.Messages[2].Images) != 2 {
		t.Fatalf("turn2 latest images=%d want 2", len(turn2.Messages[2].Images))
	}
}

func TestSessionVideoExpandCache_evictsOnlyWhenAddingNewSession(t *testing.T) {
	resetSessionVideoExpandCache()
	policy := VideoSamplingPolicy{Mode: "fps", FPS: 1, MaxFrames: 4}
	video := []byte{0x01}
	frames := []api.ImageData{{0xaa}}

	rememberSessionVideoExpand("existing", policy, video, frames, []int{1, 4, 4}, nil)
	if len(globalSessionVideoExpandCache.sessions) != 1 {
		t.Fatalf("sessions=%d, want 1", len(globalSessionVideoExpandCache.sessions))
	}

	rememberSessionVideoExpand("existing", policy, video, frames, []int{1, 4, 4}, nil)
	if len(globalSessionVideoExpandCache.sessions) != 1 {
		t.Fatalf("update should not add session, got %d", len(globalSessionVideoExpandCache.sessions))
	}
}
