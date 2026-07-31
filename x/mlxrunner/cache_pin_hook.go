package mlxrunner

// CacheKeyPinned, when set by server init, reports whether a prompt_cache_key
// holds an active /api/cache/pin lease. mlxrunner cannot import server (cycle).
// Eviction skips trie nodes whose session key is pinned.
var CacheKeyPinned func(promptCacheKey string) bool

func cacheKeyPinned(promptCacheKey string) bool {
	if promptCacheKey == "" || CacheKeyPinned == nil {
		return false
	}
	return CacheKeyPinned(promptCacheKey)
}
