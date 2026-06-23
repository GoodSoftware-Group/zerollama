// Remote video body LRU: repeat agent turns often hit the same HTTPS URL.
// Keyed by full URL (after SSRF checks in fetchVideoURL). Why separate from
// expansion cache: this layer saves network + container bytes before ffmpeg runs.
package openai

import (
	"sync"
	"time"

	"github.com/ollama/ollama/api"
)

const (
	videoURLFetchCacheMaxEntries = 32
	videoURLFetchCacheTTL        = 30 * time.Minute
)

type videoURLFetchEntry struct {
	body      api.VideoData
	updatedAt time.Time
}

var videoURLFetchCache = struct {
	mu      sync.Mutex
	entries map[string]videoURLFetchEntry
}{entries: make(map[string]videoURLFetchEntry)}

func lookupVideoURLFetchCache(rawURL string) (api.VideoData, bool) {
	now := time.Now().UTC()
	videoURLFetchCache.mu.Lock()
	defer videoURLFetchCache.mu.Unlock()
	e, ok := videoURLFetchCache.entries[rawURL]
	if !ok || len(e.body) == 0 {
		return nil, false
	}
	if videoURLFetchCacheTTL > 0 && now.Sub(e.updatedAt) > videoURLFetchCacheTTL {
		delete(videoURLFetchCache.entries, rawURL)
		return nil, false
	}
	out := make(api.VideoData, len(e.body))
	copy(out, e.body)
	return out, true
}

func rememberVideoURLFetchCache(rawURL string, body api.VideoData) {
	if rawURL == "" || len(body) == 0 {
		return
	}
	stored := make(api.VideoData, len(body))
	copy(stored, body)
	now := time.Now().UTC()
	videoURLFetchCache.mu.Lock()
	defer videoURLFetchCache.mu.Unlock()
	if len(videoURLFetchCache.entries) >= videoURLFetchCacheMaxEntries {
		evictOldestVideoURLFetchEntryLocked(now)
	}
	videoURLFetchCache.entries[rawURL] = videoURLFetchEntry{body: stored, updatedAt: now}
}

func evictOldestVideoURLFetchEntryLocked(now time.Time) {
	var oldestKey string
	var oldestAt time.Time
	first := true
	for k, e := range videoURLFetchCache.entries {
		if videoURLFetchCacheTTL > 0 && now.Sub(e.updatedAt) > videoURLFetchCacheTTL {
			delete(videoURLFetchCache.entries, k)
			continue
		}
		if first || e.updatedAt.Before(oldestAt) {
			oldestKey = k
			oldestAt = e.updatedAt
			first = false
		}
	}
	if oldestKey != "" {
		delete(videoURLFetchCache.entries, oldestKey)
	}
}

func resetVideoURLFetchCache() {
	videoURLFetchCache.mu.Lock()
	defer videoURLFetchCache.mu.Unlock()
	videoURLFetchCache.entries = make(map[string]videoURLFetchEntry)
}
