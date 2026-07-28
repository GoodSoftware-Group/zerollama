package modality

import (
	"sync"
	"time"

	"github.com/ollama/ollama/api"
)

const (
	sessionVideoExpandMaxVideos   = 16
	sessionVideoExpandMaxSessions = 256
	sessionVideoExpandTTL         = 30 * time.Minute
)

type sessionVideoExpandState struct {
	videos    map[string]videoExpandEntry
	layouts   map[string][]int // paddedInputIDs keyed by same videoKey; separate from global-sharable frames
	updatedAt time.Time
}

// sessionVideoExpandCache pins expanded frames per agent thread (prompt_cache_key).
// Complements the process-global LRU: a busy fleet can evict a clip from the global
// cache while the same eliza thread still hits here on the next turn.
//
// Bounds: 16 videos/session, 256 sessions, 30m TTL (sliding on session cache hit).
// Eviction of a session slot runs only when inserting a NEW session key, not on update.
type sessionVideoExpandCache struct {
	mu       sync.Mutex
	sessions map[string]sessionVideoExpandState
}

var globalSessionVideoExpandCache = &sessionVideoExpandCache{sessions: make(map[string]sessionVideoExpandState)}

// ExtractPromptCacheKey returns the L3 / agent session key from request options.
// Priority matches runtime/cache_bridge.py so Go expansion cache and Python L3
// pinning agree on the same thread id (eliza.promptCacheKey → conversationId → flat keys).
//
// Why one extractor: session video LRU and L3 must share the same string; OpenAI
// FromChatRequest maps prompt_cache_key / options into this map before expand runs.
func ExtractPromptCacheKey(opts map[string]any) string {
	if opts == nil {
		return ""
	}
	if eliza, ok := opts["eliza"].(map[string]any); ok {
		if v, ok := eliza["promptCacheKey"].(string); ok && v != "" {
			return v
		}
		if v, ok := eliza["conversationId"].(string); ok && v != "" {
			return v
		}
	}
	if v, ok := opts["prompt_cache_key"].(string); ok && v != "" {
		return v
	}
	// SGLang session_id (#29436) — same thread pin when clients use that field name.
	if v, ok := opts["session_id"].(string); ok && v != "" {
		return v
	}
	if v, ok := opts["conversation_id"].(string); ok && v != "" {
		return v
	}
	return ""
}

func lookupSessionVideoExpand(sessionKey string, policy VideoSamplingPolicy, data []byte) (videoExpandEntry, bool) {
	var miss videoExpandEntry
	if sessionKey == "" {
		return miss, false
	}
	videoKey := videoExpandCacheKey(policy, data)
	now := time.Now().UTC()
	globalSessionVideoExpandCache.mu.Lock()
	defer globalSessionVideoExpandCache.mu.Unlock()
	st, ok := globalSessionVideoExpandCache.sessions[sessionKey]
	if !ok {
		return miss, false
	}
	if sessionVideoExpandTTL > 0 && now.Sub(st.updatedAt) > sessionVideoExpandTTL {
		delete(globalSessionVideoExpandCache.sessions, sessionKey)
		return miss, false
	}
	e, ok := st.videos[videoKey]
	if !ok || len(e.frames) == 0 {
		return miss, false
	}
	if sessionVideoExpandTTL > 0 && now.Sub(e.updatedAt) > sessionVideoExpandTTL {
		delete(st.videos, videoKey)
		deleteSessionVideoLayoutLocked(&st, videoKey)
		globalSessionVideoExpandCache.sessions[sessionKey] = st
		return miss, false
	}
	st.updatedAt = now
	globalSessionVideoExpandCache.sessions[sessionKey] = st
	out := cloneVideoExpandEntry(e)
	if ids, ok := sessionVideoLayoutLocked(st, videoKey); ok {
		out.paddedInputIDs = ids
	}
	return out, true
}

// lookupSessionVideoLayout returns session-scoped padded_input_ids without requiring
// a session frame cache hit (used after global-hit promotion).
func lookupSessionVideoLayout(sessionKey string, policy VideoSamplingPolicy, data []byte) ([]int, bool) {
	if sessionKey == "" {
		return nil, false
	}
	videoKey := videoExpandCacheKey(policy, data)
	globalSessionVideoExpandCache.mu.Lock()
	defer globalSessionVideoExpandCache.mu.Unlock()
	st, ok := globalSessionVideoExpandCache.sessions[sessionKey]
	if !ok {
		return nil, false
	}
	ids, ok := sessionVideoLayoutLocked(st, videoKey)
	return ids, ok
}

func sessionVideoLayoutLocked(st sessionVideoExpandState, videoKey string) ([]int, bool) {
	if st.layouts == nil {
		return nil, false
	}
	ids, ok := st.layouts[videoKey]
	if !ok || len(ids) == 0 {
		return nil, false
	}
	return append([]int(nil), ids...), true
}

func deleteSessionVideoLayoutLocked(st *sessionVideoExpandState, videoKey string) {
	if st.layouts != nil {
		delete(st.layouts, videoKey)
	}
}

func ensureSessionLayouts(st *sessionVideoExpandState) {
	if st.layouts == nil {
		st.layouts = make(map[string][]int)
	}
}

func rememberSessionVideoExpand(sessionKey string, policy VideoSamplingPolicy, data []byte, frames []api.ImageData, gridTHW, paddedInputIDs []int) {
	if sessionKey == "" || len(frames) == 0 {
		return
	}
	videoKey := videoExpandCacheKey(policy, data)
	stored := make([]api.ImageData, len(frames))
	for i, f := range frames {
		stored[i] = append([]byte(nil), f...)
	}
	var storedGrid []int
	if len(gridTHW) == 3 {
		storedGrid = append([]int(nil), gridTHW...)
	}
	now := time.Now().UTC()
	globalSessionVideoExpandCache.mu.Lock()
	defer globalSessionVideoExpandCache.mu.Unlock()
	st, ok := globalSessionVideoExpandCache.sessions[sessionKey]
	if !ok {
		if len(globalSessionVideoExpandCache.sessions) >= sessionVideoExpandMaxSessions {
			evictOldestSessionLocked(now)
		}
		st = sessionVideoExpandState{videos: make(map[string]videoExpandEntry)}
	}
	ensureSessionLayouts(&st)
	if len(st.videos) >= sessionVideoExpandMaxVideos {
		evictOldestSessionVideoLocked(&st, now)
	}
	st.videos[videoKey] = videoExpandEntry{frames: stored, gridTHW: storedGrid, updatedAt: now}
	if len(paddedInputIDs) > 0 {
		st.layouts[videoKey] = append([]int(nil), paddedInputIDs...)
	}
	// When paddedInputIDs is nil we deliberately leave any existing layout entry for this
	// videoKey untouched. The caller (global-hit promotion path) passes nil to avoid
	// overwriting a session layout that was stored on the original decode turn.
	st.updatedAt = now
	globalSessionVideoExpandCache.sessions[sessionKey] = st
}

func evictOldestSessionLocked(now time.Time) {
	var oldestKey string
	var oldestAt time.Time
	first := true
	for k, st := range globalSessionVideoExpandCache.sessions {
		if sessionVideoExpandTTL > 0 && now.Sub(st.updatedAt) > sessionVideoExpandTTL {
			delete(globalSessionVideoExpandCache.sessions, k)
			continue
		}
		if first || st.updatedAt.Before(oldestAt) {
			oldestKey = k
			oldestAt = st.updatedAt
			first = false
		}
	}
	if oldestKey != "" {
		delete(globalSessionVideoExpandCache.sessions, oldestKey)
	}
}

func evictOldestSessionVideoLocked(st *sessionVideoExpandState, now time.Time) {
	var oldestKey string
	var oldestAt time.Time
	first := true
	for k, e := range st.videos {
		if sessionVideoExpandTTL > 0 && now.Sub(e.updatedAt) > sessionVideoExpandTTL {
			delete(st.videos, k)
			continue
		}
		if first || e.updatedAt.Before(oldestAt) {
			oldestKey = k
			oldestAt = e.updatedAt
			first = false
		}
	}
	if oldestKey != "" {
		delete(st.videos, oldestKey)
		deleteSessionVideoLayoutLocked(st, oldestKey)
	}
}

func resetSessionVideoExpandCache() {
	globalSessionVideoExpandCache.mu.Lock()
	defer globalSessionVideoExpandCache.mu.Unlock()
	globalSessionVideoExpandCache.sessions = make(map[string]sessionVideoExpandState)
}
