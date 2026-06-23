package modality

import (
	"crypto/sha256"
	"fmt"
	"sync"
	"time"

	"github.com/ollama/ollama/api"
)

const (
	videoExpandCacheMaxEntries = 32
	videoExpandCacheTTL        = 30 * time.Minute
)

type videoExpandEntry struct {
	frames    []api.ImageData
	gridTHW   []int // layout metadata cache: [T,H,W] patch grid (SGLang video_grid_thw row)
	updatedAt time.Time
	// paddedInputIDs is NOT stored in the global cache (it is session/client-specific).
	// It is populated only by lookupSessionVideoExpand from sessionVideoExpandState.layouts,
	// and on fresh decode by sampleVideoToPNGs when the caller provided it.
	paddedInputIDs []int
}

// videoExpandCache memoizes ffmpeg sampling by (video digest, sampling policy).
// Repeat agent turns with the same clip skip subprocess work.
// Complemented by sessionVideoExpandCache when options carry prompt_cache_key:
// global LRU (32) can evict under fleet load while per-thread cache stays warm.
type videoExpandCache struct {
	mu      sync.Mutex
	entries map[string]videoExpandEntry
}

var globalVideoExpandCache = &videoExpandCache{entries: make(map[string]videoExpandEntry)}

func videoExpandCacheKey(policy VideoSamplingPolicy, data []byte) string {
	sum := sha256.Sum256(data)
	return fmt.Sprintf("%s:%x", policyFingerprint(policy), sum)
}

func policyFingerprint(policy VideoSamplingPolicy) string {
	return fmt.Sprintf("%s:%.6f:%d:%d:%d:%d",
		policy.Mode, policy.FPS, policy.Stride, policy.MaxFrames,
		policy.visionPatchSize(), policy.visionSpatialMergeSize())
}

func cloneVideoExpandEntry(e videoExpandEntry) videoExpandEntry {
	out := videoExpandEntry{updatedAt: e.updatedAt}
	if len(e.frames) > 0 {
		out.frames = make([]api.ImageData, len(e.frames))
		for i, f := range e.frames {
			out.frames[i] = append([]byte(nil), f...)
		}
	}
	if len(e.gridTHW) == 3 {
		out.gridTHW = append([]int(nil), e.gridTHW...)
	}
	return out
}

func lookupVideoExpandCache(policy VideoSamplingPolicy, data []byte) (videoExpandEntry, bool) {
	var miss videoExpandEntry
	key := videoExpandCacheKey(policy, data)
	now := time.Now().UTC()
	globalVideoExpandCache.mu.Lock()
	defer globalVideoExpandCache.mu.Unlock()
	e, ok := globalVideoExpandCache.entries[key]
	if !ok || len(e.frames) == 0 {
		return miss, false
	}
	if videoExpandCacheTTL > 0 && now.Sub(e.updatedAt) > videoExpandCacheTTL {
		delete(globalVideoExpandCache.entries, key)
		return miss, false
	}
	return cloneVideoExpandEntry(e), true
}

func rememberVideoExpandCache(policy VideoSamplingPolicy, data []byte, frames []api.ImageData, gridTHW []int) {
	if len(frames) == 0 {
		return
	}
	key := videoExpandCacheKey(policy, data)
	stored := make([]api.ImageData, len(frames))
	for i, f := range frames {
		stored[i] = append([]byte(nil), f...)
	}
	var storedGrid []int
	if len(gridTHW) == 3 {
		storedGrid = append([]int(nil), gridTHW...)
	}
	now := time.Now().UTC()
	globalVideoExpandCache.mu.Lock()
	defer globalVideoExpandCache.mu.Unlock()
	if len(globalVideoExpandCache.entries) >= videoExpandCacheMaxEntries {
		evictOldestVideoExpandEntryLocked(now)
	}
	globalVideoExpandCache.entries[key] = videoExpandEntry{frames: stored, gridTHW: storedGrid, updatedAt: now}
}

func evictOldestVideoExpandEntryLocked(now time.Time) {
	var oldestKey string
	var oldestAt time.Time
	first := true
	for k, e := range globalVideoExpandCache.entries {
		if videoExpandCacheTTL > 0 && now.Sub(e.updatedAt) > videoExpandCacheTTL {
			delete(globalVideoExpandCache.entries, k)
			continue
		}
		if first || e.updatedAt.Before(oldestAt) {
			oldestKey = k
			oldestAt = e.updatedAt
			first = false
		}
	}
	if oldestKey != "" {
		delete(globalVideoExpandCache.entries, oldestKey)
	}
}

// resetVideoExpandCache clears the process-global cache (tests only).
func resetVideoExpandCache() {
	globalVideoExpandCache.mu.Lock()
	defer globalVideoExpandCache.mu.Unlock()
	globalVideoExpandCache.entries = make(map[string]videoExpandEntry)
}

// ResetExpandCachesForTest clears global and session expansion LRU (integration tests).
//
// Why exported: server-level ChatHandler tests are expensive to compile; modality tests assert
// agent-loop cache behavior without pulling in the full scheduler package.
func ResetExpandCachesForTest() {
	resetVideoExpandCache()
	resetSessionVideoExpandCache()
}
