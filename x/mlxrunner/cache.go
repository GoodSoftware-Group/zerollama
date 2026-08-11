// cache.go manages a shared KV cache across conversations using a compressed
// prefix trie. Each trie node stores a token sequence (edge) and optional
// per-layer snapshots that can be paged in/out of the live MLX cache arrays.
//
// Key properties:
//   - Only one path through the trie is "active" (backed by live MLX arrays)
//     at a time. Switching paths pages out the frontier node and pages in the
//     new path.
//   - Snapshots are only captured at the frontier (end) of the active path.
//     Intermediate node snapshots come from split prefill.
//   - All cache layers must stay at the same token offset.
//   - Sibling edges must not share a common token prefix (compressed trie
//     invariant).
//   - begin() always re-evaluates at least one token so the pipeline can seed
//     generation, even on a full prefix match.

package mlxrunner

import (
	"cmp"
	"fmt"
	"log/slog"
	"slices"
	"strings"
	"time"

	"github.com/ollama/ollama/logutil"
	"github.com/ollama/ollama/x/mlxrunner/cache"
	"github.com/ollama/ollama/x/mlxrunner/mlx"
	"github.com/ollama/ollama/x/mlxrunner/model/base"
)

const maxPagedOutBytes int64 = 8 << 30 // 8 GiB eviction threshold for paged-out snapshot memory

type kvCache struct {
	root               *trieNode   // root of the prefix trie
	activePath         []*trieNode // current root→leaf path with live MLX arrays
	caches             []cache.Cache
	pagedOutBytes      int64 // total bytes in paged-out snapshot memory across the trie
	lastPromptCacheKey string
	lastSessionInputs  []int32

	// draftLookahead is how far the draft caches' entries reference past
	// their own slot; trie keys pack each token with its look-ahead (see key).
	draftLookahead int
}

// pendingSnapshot is a snapshot scheduled to be taken during prefill.
type pendingSnapshot struct {
	offset int
	user   bool
}

// cacheSession manages caches for a single pipeline run.
// Callers should append generated tokens to outputs and
// defer close to save the cache state.
type cacheSession struct {
	cache   *kvCache
	inputs  []int32
	outputs []int32

	caches         []cache.Cache
	remaining      []int32
	cachedPrefix   int // tokens restored from trie before prefill
	promptCacheKey string
	fastPath       bool // live-session continuation; skip trie page-in/out
	sameBranch     bool // trie same-branch restore; skip page-out/in round trip

	// pendingSnapshots lists offsets where snapshots should be captured
	// during prefill, sorted by offset. Entries are scheduled on the caches
	// before prefill and drained or discarded after.
	pendingSnapshots []pendingSnapshot
}

func (c *kvCache) ensureCaches(m base.Model) {
	if len(c.caches) != 0 {
		return
	}
	if cacheFactory, ok := m.(interface{ NewCaches() []cache.Cache }); ok {
		c.caches = cacheFactory.NewCaches()
		return
	}
	c.caches = make([]cache.Cache, m.NumLayers())
	for i := range c.caches {
		c.caches[i] = cache.NewKVCache()
	}
}

func (c *kvCache) ensureRoot() {
	if c.root == nil {
		c.root = &trieNode{
			lastUsed: time.Now(),
		}
		c.activePath = []*trieNode{c.root}
	}
}

// begin prepares caches for a new request. It finds the nearest
// matching cache or creates new caches if none match.
//
// WHY cacheReset: harnesses ask for a miss under the same promptCacheKey
// (options.zerollama.cache_reset) without inventing a cold: key namespace.
// Skip live extend and trie hit for this turn only; trie branches may remain
// for later content matches after compaction — enough for "don't resume now."
func (c *kvCache) begin(m base.Model, inputs []int32, promptCacheKey string, cacheReset bool) *cacheSession {
	c.ensureCaches(m)
	c.ensureRoot()

	key := strings.TrimSpace(promptCacheKey)
	if cacheReset || (key != "" && key != c.lastPromptCacheKey) {
		c.lastSessionInputs = nil
	}
	if !cacheReset && key != "" && key == c.lastPromptCacheKey {
		if session, ok := c.tryExtendLiveSession(modelSlidingWindow(m), inputs); ok {
			c.lastPromptCacheKey = key
			return session
		}
	}

	var matchPath []*trieNode
	var matched int
	if cacheReset {
		matchPath, matched = []*trieNode{c.root}, 0
	} else {
		matchPath, matched = findBestMatch(c.root, c.key(inputs))
	}
	originalMatched := matched

	// Always keep at least one token to re-evaluate so the
	// pipeline can seed token generation from it.
	if matched == len(inputs) && matched > 0 {
		matchPath, matched = findBestMatch(c.root, c.key(inputs)[:matched-1])
	}

	// Switch to the matched path, paging in/out as needed.
	// Mid-edge cap applies only to rotating KV (Gemma4 OptiQ); full KV caches
	// can live-rewind to an interior offset.
	if modelSlidingWindow(m) > 0 {
		matchPath, matched = capTrieMatchForRestore(matchPath, matched)
	}
	sameBranch := c.trySameBranchRestore(matchPath, matched)
	if !sameBranch {
		c.switchToPath(matchPath, matched)
	}

	// switchToPath aligns caches to a common offset
	prefix := c.minCacheOffset()
	remaining := inputs[prefix:]

	session := &cacheSession{
		cache:          c,
		inputs:         inputs,
		caches:         c.caches,
		remaining:      remaining,
		cachedPrefix:   prefix,
		promptCacheKey: key,
		sameBranch:     sameBranch,
	}

	// Schedule a snapshot at the branch point during prefill so future
	// requests diverging here can restore instead of re-evaluating.
	if prefix < matched {
		session.pendingSnapshots = append(session.pendingSnapshots, pendingSnapshot{offset: matched, user: false})
	}

	msg := "cache hit"
	if prefix == 0 {
		msg = "cache miss"
	}
	utilPct := 0.0
	if len(inputs) > 0 {
		utilPct = float64(prefix) / float64(len(inputs)) * 100
	}
	attrs := []any{
		"total", len(inputs),
		"matched", originalMatched,
		"cached", prefix,
		"left", len(remaining),
		"utilization_pct", utilPct,
	}
	if sameBranch {
		attrs = append(attrs, "same_branch", true)
	}
	if originalMatched > 0 && prefix == 0 {
		attrs := []any{
			"matched", originalMatched,
			"restorable", matched,
			"total", len(inputs),
		}
		if key != "" {
			attrs = append(attrs, "prompt_cache_key", key)
		}
		// Expected on short sidecar /api/generate calls and rotating-KV edge caps;
		// only surface at warn for long agent prompts with a session key — otherwise
		// operators chase harmless sidecar noise while real misses use prefill debug.
		if key != "" && originalMatched >= 8192 {
			attrs = append(attrs, "hint", "rotating KV cannot restore mid-edge; ensure prompt_cache_key is stable across turns")
			slog.Warn("mlx prefix trie match but KV restore missed; full prefill required", attrs...)
		} else {
			slog.Debug("mlx prefix trie match but KV restore missed; full prefill required", attrs...)
		}
	}
	slog.Info(msg, attrs...)

	// Do not clear lastPromptCacheKey on unkeyed sidecar /api/generate — that
	// would make the next keyed agent turn wipe lastSessionInputs and miss fast_path.
	if key != "" {
		c.lastPromptCacheKey = key
	}
	return session
}

// key converts tokens to trie keys, one per restorable cache offset. A model
// that drafts through MTP-style draft caches pairs each cache slot with the
// token after it, so slot i is reusable only if token i+1 also matched. The
// key for offset i then packs (token i, token i+1): matching k keys verifies
// k+1 tokens, making every match a valid restore point.
func (c *kvCache) key(tokens []int32) []trieKey {
	keys := make([]trieKey, max(len(tokens)-c.draftLookahead, 0))
	switch c.draftLookahead {
	case 0:
		for i, t := range tokens {
			keys[i] = trieKey(t)
		}
	case 1:
		for i := range keys {
			keys[i] = trieKey(uint32(tokens[i]))<<32 | trieKey(uint32(tokens[i+1]))
		}
	default:
		panic(fmt.Sprintf("kvCache: unsupported draft look-ahead %d", c.draftLookahead))
	}
	return keys
}

// tryExtendLiveSession reuses resident MLX KV when the same agent thread extends
// its prompt without a trie page-in/out round trip. Generation tokens from the
// prior turn sit in KV and the trie but not in the new prompt prefix, so we
// match on the longest common prefix with the prior prompt and rewind past gen.
//
// Requires the same prompt_cache_key as the prior turn (lastSessionInputs set in
// close()). On rotating KV (Gemma4 OptiQ), live rewind uses rewindCachesViaSnapshots
// because Restore(nil, offset) fails once offset > sliding_window — wrapped buffers
// need trie snapshot page-in, not in-place index rewind.
func (c *kvCache) tryExtendLiveSession(slidingWindow int, inputs []int32) (*cacheSession, bool) {
	if len(c.activePath) == 0 || len(c.caches) == 0 || len(c.lastSessionInputs) == 0 {
		return nil, false
	}
	lcp := longestCommonPrefix(inputs, c.lastSessionInputs)
	minRetain := len(c.lastSessionInputs) - agentGenStubTokenSlack
	if minRetain < 0 {
		minRetain = 0
	}
	if lcp < minRetain {
		return nil, false
	}

	live := c.minCacheOffset()
	if live < lcp {
		// Runtime sidecar shares this runner and switches activePath/KV without a
		// prompt_cache_key; restore the agent prefix from trie before gen-token rewind.
		if !c.bootstrapLiveSessionFromTrie(inputs, lcp, minRetain, slidingWindow) {
			return nil, false
		}
		live = c.minCacheOffset()
		if live < lcp {
			return nil, false
		}
	}
	rotating := slidingWindow > 0
	branch := trimPathToOffset(c.activePath, lcp)
	rewindTo := lcp
	if rotating {
		var ok bool
		branch, rewindTo, ok = bestRestorableOffset(branch, lcp)
		if !ok || rewindTo < minRetain {
			return nil, false
		}
	}
	if live > rewindTo {
		c.snapshotActiveLeafBeforeRewind(rewindTo)
		if rotating {
			if !c.rewindCachesViaSnapshots(branch, rewindTo) {
				return nil, false
			}
		} else if !c.rewindCachesTo(rewindTo) {
			return nil, false
		}
	}

	prefix := rewindTo
	if prefix == len(inputs) && prefix > 0 {
		prefix--
	}
	remaining := inputs[prefix:]
	session := &cacheSession{
		cache:          c,
		inputs:         inputs,
		caches:         c.caches,
		remaining:      remaining,
		cachedPrefix:   prefix,
		promptCacheKey: c.lastPromptCacheKey,
		fastPath:       true,
	}
	utilPct := 0.0
	if len(inputs) > 0 {
		utilPct = float64(prefix) / float64(len(inputs)) * 100
	}
	slog.Info("cache hit",
		"total", len(inputs),
		"matched", lcp,
		"cached", prefix,
		"left", len(remaining),
		"utilization_pct", utilPct,
		"fast_path", true,
		"rewound_from", live,
		"rewound_to", rewindTo,
	)
	return session, true
}

const agentGenStubTokenSlack = 512

// bootstrapLiveSessionFromTrie reloads agent KV from trie snapshots when live
// caches were left on an unkeyed sidecar branch between keyed agent turns.
func (c *kvCache) bootstrapLiveSessionFromTrie(inputs []int32, lcp, minRetain, slidingWindow int) bool {
	matchPath, matched := findBestMatch(c.root, c.key(inputs))
	if matched < minRetain {
		return false
	}
	rotating := slidingWindow > 0
	if rotating {
		matchPath, matched = capTrieMatchForRestore(matchPath, matched)
		if matched < minRetain {
			return false
		}
	}
	branch := trimPathToOffset(matchPath, matched)
	rewindTo := matched
	if rotating {
		var ok bool
		branch, rewindTo, ok = bestRestorableOffset(branch, matched)
		if !ok || rewindTo < minRetain {
			return false
		}
	}
	if rewindTo < lcp && lcp-rewindTo > agentGenStubTokenSlack {
		return false
	}
	if rotating {
		if !c.rewindCachesViaSnapshots(branch, rewindTo) {
			return false
		}
	} else {
		c.switchToPath(branch, rewindTo)
	}
	c.activePath = branch
	if len(c.activePath) > 0 {
		c.activePath[len(c.activePath)-1].lastUsed = time.Now()
	}
	slog.Debug("live session bootstrap from trie after sidecar clobber",
		"rewind_to", rewindTo, "lcp", lcp, "min_retain", minRetain)
	return c.minCacheOffset() >= rewindTo
}

func longestCommonPrefix(a, b []int32) int {
	n := min(len(a), len(b))
	for i := 0; i < n; i++ {
		if a[i] != b[i] {
			return i
		}
	}
	return n
}

func (c *kvCache) rewindCachesTo(offset int) bool {
	rewound := false
	for _, kv := range c.caches {
		if kv == nil {
			continue
		}
		if kv.Offset() <= offset {
			continue
		}
		if !kv.Restore(nil, offset) {
			slog.Debug("mlx cache rewind failed; freeing caches", "target", offset, "had", kv.Offset())
			c.freeAll()
			return false
		}
		rewound = true
	}
	if !rewound {
		return true
	}
	for i := len(c.activePath) - 1; i >= 0; i-- {
		if c.activePath[i].endOffset <= offset {
			c.activePath = c.activePath[:i+1]
			break
		}
	}
	return true
}

// rewindCachesViaSnapshots restores KV to matched by paging in trie snapshots on
// path. Used for live-session and same-branch rewind on rotating KV where
// Restore(nil, target) fails once the window has wrapped.
func (c *kvCache) rewindCachesViaSnapshots(path []*trieNode, matched int) bool {
	if matched < 0 {
		return false
	}
	if matched == 0 {
		return len(path) <= 1
	}
	if len(path) == 0 {
		return false
	}

	// Cheap live rewind for layers that support it. When live rewind fails on a
	// wrapped rotating buffer, Free the layer so page-in can restore it from trie
	// snapshots — same pattern as switchToPath.
	for _, kv := range c.caches {
		if kv == nil {
			continue
		}
		if kv.Offset() <= matched {
			continue
		}
		if !kv.Restore(nil, matched) {
			kv.Free()
		}
	}

	for _, node := range path {
		if node.endOffset > matched {
			continue
		}
		if !node.hasSnapshots() {
			continue
		}
		nodeTarget := node.endOffset
		for j, kv := range c.caches {
			if kv == nil {
				continue
			}
			if j >= len(node.snapshots) || node.snapshots[j] == nil {
				continue
			}
			if kv.Offset() >= nodeTarget {
				continue
			}
			if !kv.Restore(node.snapshots[j], nodeTarget) {
				c.freeAll()
				return false
			}
		}
	}

	minOff := c.minCacheOffset()
	if minOff < matched {
		c.freeAll()
		return false
	}
	if minOff > matched {
		if !c.rewindCachesTo(matched) {
			return false
		}
	}
	for _, kv := range c.caches {
		if kv == nil {
			continue
		}
		if kv.Offset() == matched {
			continue
		}
		if !kv.Restore(nil, matched) {
			c.freeAll()
			return false
		}
	}
	c.activePath = trimPathToOffset(path, matched)
	if len(c.activePath) > 0 {
		c.activePath[len(c.activePath)-1].lastUsed = time.Now()
	}
	return c.minCacheOffset() == matched && c.maxCacheOffset() == matched
}

// trySameBranchRestore skips full trie page-out/in when the matched path extends
// the current active branch and live KV already covers the restore point.
func (c *kvCache) trySameBranchRestore(newPath []*trieNode, matched int) bool {
	if matched <= 0 || len(c.activePath) == 0 || len(newPath) < len(c.activePath) {
		return false
	}
	for i := range c.activePath {
		if c.activePath[i] != newPath[i] {
			return false
		}
	}
	cur := c.minCacheOffset()
	if cur < matched {
		return false
	}
	if cur > matched {
		c.snapshotActiveLeafBeforeRewind(matched)
	}
	if !c.rewindCachesTo(matched) && !c.rewindCachesViaSnapshots(newPath, matched) {
		return false
	}
	c.activePath = trimPathToOffset(newPath, matched)
	if len(c.activePath) > 0 {
		c.activePath[len(c.activePath)-1].lastUsed = time.Now()
	}
	slog.Debug("same-branch cache restore", "matched", matched, "rewound_from", cur)
	return true
}

// snapshotActiveLeafBeforeRewind pages out the active leaf when a same-branch
// rewind will discard KV state past matched (mirrors switchToPath leaf page-out).
func (c *kvCache) snapshotActiveLeafBeforeRewind(matched int) {
	if len(c.activePath) == 0 {
		return
	}
	leaf := c.activePath[len(c.activePath)-1]
	if matched >= leaf.endOffset || leaf.hasAllSnapshots() {
		return
	}
	fromOffset := leaf.startOffset()
	snaps := make([]cache.Snapshot, len(c.caches))
	for j, kv := range c.caches {
		if kv == nil {
			continue
		}
		snaps[j] = kv.Snapshot(fromOffset)
	}
	leaf.setSnapshots(snaps, &c.pagedOutBytes)
}

func trimPathToOffset(path []*trieNode, offset int) []*trieNode {
	for i := len(path) - 1; i >= 0; i-- {
		if path[i].endOffset <= offset {
			return path[:i+1]
		}
	}
	return path[:1]
}

// bestRestorableOffset returns the largest snapshot boundary on path that is
// <= target. Prefer interior trie nodes with snapshots over mid-edge caps.
//
// Why: capTrieMatchForRestore alone walks leaf edges and can rewind thousands of
// tokens short of LCP when snapshots exist at finer boundaries (production symptom:
// ~16k cached on long prompts after messages_dropped changed the prefix).
func bestRestorableOffset(path []*trieNode, target int) ([]*trieNode, int, bool) {
	if target <= 0 || len(path) <= 1 {
		return path, 0, false
	}
	path = trimPathToOffset(path, target)
	best := 0
	bestLen := 1
	for i := 1; i < len(path); i++ {
		node := path[i]
		if node.endOffset > target || !node.hasSnapshots() {
			continue
		}
		if node.endOffset > best {
			best = node.endOffset
			bestLen = i + 1
		}
	}
	if best > 0 {
		return path[:bestLen], best, true
	}
	path, matched := capTrieMatchForRestore(path, target)
	return path, matched, matched > 0
}

// capTrieMatchForRestore trims a trie match when the last traversed edge extends
// past matched (partial edge). RotatingKV snapshots cannot restore below their
// capture point, so page-in skips nodes with endOffset > matched.
func capTrieMatchForRestore(path []*trieNode, matched int) ([]*trieNode, int) {
	for len(path) > 1 {
		leaf := path[len(path)-1]
		if matched >= leaf.endOffset {
			return path, matched
		}
		matched = leaf.startOffset()
		path = path[:len(path)-1]
	}
	return path, matched
}

// switchToPath transitions from the current active path to a new path,
// paging out diverging segments and paging in the new path.
func (c *kvCache) switchToPath(newPath []*trieNode, matched int) {
	defer c.enforceEvictionPolicy()

	// Find common ancestor index.
	commonLen := 0
	for commonLen < len(c.activePath) && commonLen < len(newPath) {
		if c.activePath[commonLen] != newPath[commonLen] {
			break
		}
		commonLen++
	}

	ancestorOffset := 0
	if commonLen > 0 {
		ancestorOffset = c.activePath[commonLen-1].endOffset
	}

	var pageOutCount, pageInCount int

	// Page out the leaf of the old path. Only the leaf's live cache
	// state is correct — intermediate nodes already have snapshots
	// captured during their creation (splitNode + prefill). Snapshotting
	// non-leaf nodes here would produce wrong results for non-rewindable
	// caches (e.g. RecurrentCache) whose state reflects the leaf, not
	// the intermediate boundary.
	leaf := len(c.activePath) - 1
	leafDiverges := leaf >= commonLen
	leafNeedsRewind := matched < c.activePath[leaf].endOffset
	if leafDiverges || leafNeedsRewind {
		node := c.activePath[leaf]
		if !node.hasAllSnapshots() {
			fromOffset := node.startOffset()
			snaps := make([]cache.Snapshot, len(c.caches))
			for j, kv := range c.caches {
				if kv == nil {
					continue
				}
				snaps[j] = kv.Snapshot(fromOffset)
			}
			node.setSnapshots(snaps, &c.pagedOutBytes)
			pageOutCount++
			logutil.Trace(fmt.Sprintf("page out: [%d, %d)", fromOffset, node.endOffset))
		}
	}

	// Rewind each cache to the target offset or free it. When matched
	// falls within the ancestor's range (same-path case), we rewind
	// directly to the match point. Otherwise we rewind to the ancestor
	// and let page-in bring us forward to matched.
	rewindTarget := min(ancestorOffset, matched)
	for _, kv := range c.caches {
		if kv == nil {
			continue
		}
		if !kv.Restore(nil, rewindTarget) {
			kv.Free()
		}
	}

	// Page in — walk the full new path, restoring from snapshots.
	// Freed caches naturally pick up the first available snapshot.
	// Caches already past a node skip it via offset check.
pageIn:
	for _, node := range newPath {
		if !node.hasSnapshots() {
			continue
		}
		nodeTarget := min(node.endOffset, matched)
		for j, kv := range c.caches {
			if kv == nil {
				continue
			}
			if j >= len(node.snapshots) || node.snapshots[j] == nil {
				continue
			}
			if kv.Offset() >= nodeTarget {
				continue
			}
			if !kv.Restore(node.snapshots[j], nodeTarget) {
				// Restore failed — stop page-in and let alignment
				// bring all caches to a consistent offset.
				break pageIn
			}
		}
		if node.endOffset > ancestorOffset {
			pageInCount++
			logutil.Trace(fmt.Sprintf("page in: [%d, %d)", node.startOffset(), nodeTarget))
		}
	}

	// Align all caches to the minimum offset.
	c.activePath = newPath
	minOff := c.minCacheOffset()
	for _, kv := range c.caches {
		if kv != nil && kv.Offset() != minOff {
			if !kv.Restore(nil, minOff) {
				slog.Warn("failed to restore cache, freeing all caches", "offset", minOff)
				c.freeAll()
				break
			}
		}
	}
	for i := len(c.activePath) - 1; i >= 0; i-- {
		if c.activePath[i].endOffset <= minOff {
			c.activePath = c.activePath[:i+1]
			break
		}
	}

	// Update last-used time on only the final used node. For recurrent
	// caches we don't need the intermediate snapshots and for KV caches
	// we can reslice the data out of merged edges.
	if len(c.activePath) > 0 {
		c.activePath[len(c.activePath)-1].lastUsed = time.Now()
	}

	if pageOutCount > 0 || pageInCount > 0 {
		slog.Debug("switching cache path", "page_out", pageOutCount, "page_in", pageInCount)
	}
}

// schedulePrefillSnapshots schedules every cache to capture snapshots as the
// forward pass crosses the given absolute token offsets, so a single full-size
// prefill records interior states without the caller breaking the batch. The
// passed offsets are user-requested restore points; they are merged with any
// snapshots begin already scheduled (e.g. a branch point), with coinciding
// offsets upgraded to user so eviction preserves them.
//
// Offsets at or before the current cache position, or past the end of the
// prompt, are dropped: callers only request offsets ahead of the prefill base,
// so this is a defensive guard.
func (s *cacheSession) schedulePrefillSnapshots(offsets []int) {
	c := s.cache
	base := c.minCacheOffset()
	for _, offset := range offsets {
		offset -= c.draftLookahead
		if offset <= base || offset > len(s.inputs) {
			continue
		}
		// Deduplicate: if this offset already exists, upgrade to user.
		found := false
		for i := range s.pendingSnapshots {
			if s.pendingSnapshots[i].offset == offset {
				s.pendingSnapshots[i].user = true
				found = true
				break
			}
		}
		if !found {
			s.pendingSnapshots = append(s.pendingSnapshots, pendingSnapshot{offset: offset, user: true})
		}
	}
	slices.SortFunc(s.pendingSnapshots, func(a, b pendingSnapshot) int {
		return a.offset - b.offset
	})

	if len(s.pendingSnapshots) == 0 {
		return
	}

	prepared := make([]int, len(s.pendingSnapshots))
	for i, p := range s.pendingSnapshots {
		prepared[i] = p.offset
	}
	for _, kv := range c.caches {
		if kv == nil {
			continue
		}
		cur := kv.Offset()
		var forKV []int
		for _, off := range prepared {
			if off >= cur && off <= len(s.inputs) {
				forKV = append(forKV, off)
			}
		}
		if len(forKV) > 0 {
			kv.PrepareSnapshots(forKV)
		}
	}
}

// discardPrefillSnapshots drains and closes the snapshots scheduled by
// schedulePrefillSnapshots without attaching them to the trie, releasing their
// pinned/lazy state. It is a no-op once attachPrefillSnapshots has drained the
// schedule, so close can call it unconditionally to clean up an abandoned
// prefill.
func (s *cacheSession) discardPrefillSnapshots() {
	if len(s.pendingSnapshots) == 0 {
		return
	}
	s.pendingSnapshots = nil

	for _, kv := range s.cache.caches {
		if kv == nil {
			continue
		}
		for _, snap := range kv.TakeSnapshots() {
			if snap != nil {
				snap.Close()
			}
		}
	}
}

// attachPrefillSnapshots collects the snapshots captured during prefill and
// attaches them to the trie, materializing a node at each requested offset.
// Pending offsets are ascending and were scheduled in the same order, so the
// snapshots each cache returns line up with them. The trie frontier is
// advanced to each offset in turn, so its node edges [prev, offset) match the
// edge-local ranges the caches captured.
func (s *cacheSession) attachPrefillSnapshots() {
	if len(s.pendingSnapshots) == 0 {
		return
	}

	c := s.cache
	pending := s.pendingSnapshots
	s.pendingSnapshots = nil

	// Drain each cache's captures (one per pending offset, in order) into
	// per-offset rows.
	rows := make([][]cache.Snapshot, len(pending))
	for i := range rows {
		rows[i] = make([]cache.Snapshot, len(c.caches))
	}
	for j, kv := range c.caches {
		if kv == nil {
			continue
		}
		taken := kv.TakeSnapshots()
		for i := range pending {
			if i < len(taken) {
				rows[i][j] = taken[i]
			}
		}
	}

	// Prefill leaves one token unprocessed for decode seeding, so an offset
	// at or past the live cache position was never crossed by a write and has
	// no captured state. Skip it rather than materialize a node whose edge
	// claims tokens the cache never wrote. Closing its (nil) row is a no-op.
	reached := c.minCacheOffset()
	stored := c.key(append(s.inputs, s.outputs...))
	for i, p := range pending {
		if p.offset > reached {
			// Never crossed by a write, so the row is nil; close any entry
			// defensively in case a cache captured one anyway.
			for _, snap := range rows[i] {
				if snap != nil {
					snap.Close()
				}
			}
			continue
		}
		frontier := c.activePath[len(c.activePath)-1]
		if frontier.endOffset < p.offset {
			edgeTokens := stored[frontier.endOffset:p.offset]
			frontier = c.advancePath(frontier, edgeTokens, p.offset)
		}
		if p.user {
			frontier.user = true
		}
		s.attachCapturedSnapshots(frontier, rows[i])
	}
}

// attachCapturedSnapshots stores pre-captured snapshots on a trie node. Unlike
// taking a fresh Snapshot from the live cache, this works for an interior node
// whose offset the live cache has already advanced past: the snapshots come
// from the capture scheduled earlier, not from the cache's current state. The
// node takes ownership of the snapshots (TakeSnapshots already transferred it).
func (s *cacheSession) attachCapturedSnapshots(node *trieNode, snaps []cache.Snapshot) {
	c := s.cache
	node.setSnapshots(snaps, &c.pagedOutBytes)
	node.lastUsed = time.Now()
	slog.Debug("created snapshot", "offset", node.endOffset)
	c.enforceEvictionPolicy()
}

// advancePath advances the active path from the current frontier by matching
// tokens against existing trie children, splitting partial matches, and
// appending any remaining tokens as new nodes. Returns the new frontier.
func (c *kvCache) advancePath(frontier *trieNode, tokens []trieKey, endOffset int) *trieNode {
	// Check if existing children already cover some or all of tokens.
	// tokens may span multiple trie nodes when extending a previous run's
	// leaf and this snapshot now overlaps that same range.
	matchPath, matched := findBestMatch(frontier, tokens)
	// matchPath[0] is frontier itself; the rest are newly traversed nodes.
	remaining := tokens[matched:]

	// Check for a partial match within the last node's edge — if so, split it.
	if len(matchPath) > 1 {
		lastNode := matchPath[len(matchPath)-1]
		matchedInEdge := frontier.endOffset + matched - lastNode.startOffset()
		if matchedInEdge > 0 && matchedInEdge < len(lastNode.tokens) {
			matchPath[len(matchPath)-1] = splitNode(lastNode, matchedInEdge, c.caches, &c.pagedOutBytes)
		}
	}

	// Append traversed nodes (excluding frontier) to the active path.
	c.activePath = append(c.activePath, matchPath[1:]...)
	dest := matchPath[len(matchPath)-1]

	if len(remaining) > 0 {
		// Drop non-user snapshots so appendTokens can extend in-place
		// rather than creating a new child node.
		if len(dest.children) == 0 && !dest.user {
			dest.setSnapshots(nil, &c.pagedOutBytes)
		}
		newDest := dest.appendTokens(c.root, remaining, endOffset)
		if newDest != dest {
			c.activePath = append(c.activePath, newDest)
		}
		dest = newDest
	}
	return dest
}

// freeAll releases all cache layers.
func (c *kvCache) freeAll() {
	for _, kv := range c.caches {
		if kv != nil {
			kv.Free()
		}
	}
}

func (c *kvCache) minCacheOffset() int {
	offset := 0
	found := false
	for _, kv := range c.caches {
		if kv == nil {
			continue
		}
		if off := kv.Offset(); !found || off < offset {
			offset = off
			found = true
		}
	}
	return offset
}

func (c *kvCache) maxCacheOffset() int {
	offset := 0
	found := false
	for _, kv := range c.caches {
		if kv == nil {
			continue
		}
		if off := kv.Offset(); !found || off > offset {
			offset = off
			found = true
		}
	}
	return offset
}

// close saves the token state if the forward pass ran.
func (s *cacheSession) close() {
	// Release any prefill snapshots the session scheduled but never attached to
	// the trie. A successful prefill drains them in attachPrefillSnapshots (so
	// this is a no-op then); an abandoned one (e.g. cancellation between
	// schedule and attach) leaves them in the caches, where the next request's
	// PrepareSnapshots would overwrite the schedule without closing them,
	// leaking the pinned/lazy snapshots and their VRAM.
	s.discardPrefillSnapshots()

	offset := s.cache.minCacheOffset()
	if offset <= 0 {
		return
	}

	arrays := make([]*mlx.Array, 0, 2*len(s.caches))
	for _, kv := range s.caches {
		if kv == nil {
			continue
		}
		arrays = append(arrays, kv.State()...)
	}

	// Ensure that if we have run the forward pass and set the metadata
	// that we also actually have the data.
	mlx.AsyncEval(arrays...)

	// Advance the trie frontier with any newly generated tokens.
	c := s.cache
	stored := c.key(append(s.inputs, s.outputs...))
	if offset > len(stored) {
		panic(fmt.Sprintf("cache: offset %d exceeds %d stored keys", offset, len(stored)))
	}
	if len(c.activePath) > 0 {
		frontier := c.activePath[len(c.activePath)-1]

		if offset > frontier.endOffset {
			newTokens := stored[frontier.endOffset:offset]
			c.advancePath(frontier, newTokens, offset)
		}
		c.activePath[len(c.activePath)-1].lastUsed = time.Now()
		if key := strings.TrimSpace(s.promptCacheKey); key != "" {
			leaf := c.activePath[len(c.activePath)-1]
			leaf.user = true
			leaf.promptCacheKey = key
			c.lastPromptCacheKey = key
			c.lastSessionInputs = slices.Clone(s.inputs)
		}
	}
}

// enforceEvictionPolicy evicts eligible nodes until paged-out memory is within limits.
func (c *kvCache) enforceEvictionPolicy() {
	if c.pagedOutBytes <= maxPagedOutBytes {
		return
	}

	activeSet := make(map[*trieNode]bool, len(c.activePath))
	for _, n := range c.activePath {
		activeSet[n] = true
	}

	for c.pagedOutBytes > maxPagedOutBytes {
		var best *trieNode
		walkNodes(c.root, func(n *trieNode) bool {
			if n == c.root || activeSet[n] || len(n.children) > 1 || n.user {
				return true
			}
			// WHY: /api/cache/pin leases protect keyed branches beyond the user flag
			// (e.g. interior nodes that lost user during merge pressure).
			if cacheKeyPinned(n.promptCacheKey) {
				return true
			}
			// Evict: oldest, then deepest, then largest.
			if best == nil || cmp.Or(
				n.lastUsed.Compare(best.lastUsed),
				cmp.Compare(best.endOffset, n.endOffset),
				cmp.Compare(best.snapshotBytes(), n.snapshotBytes()),
			) < 0 {
				best = n
			}
			return true
		})
		if best == nil {
			break
		}
		c.evictNode(best)
	}
}

// evictNode evicts a single node from the trie, freeing its snapshot memory.
func (c *kvCache) evictNode(node *trieNode) {
	if len(node.children) == 0 {
		// Leaf: remove entirely.
		slog.Debug("evicting leaf", "offset", node.startOffset(), "tokens", len(node.tokens), "freed", mlx.PrettyBytes(int(node.snapshotBytes())))
		removeNode(node, &c.pagedOutBytes)
	} else if len(node.children) == 1 {
		// Interior node with one child: merge with child.
		before := c.pagedOutBytes
		tokens := len(node.tokens)
		mergeWithChild(node, c.caches, &c.pagedOutBytes)
		slog.Debug("evicting interior node", "offset", node.startOffset(), "tokens", tokens, "freed", mlx.PrettyBytes(int(before-c.pagedOutBytes)))
	} else {
		panic("evictNode called on multi-child branch point")
	}
}

func (c *kvCache) dumpTree() {
	// Summary stats
	var cacheBytes int
	for _, kv := range c.caches {
		if kv == nil {
			continue
		}
		for _, a := range kv.State() {
			if a != nil {
				cacheBytes += a.NumBytes()
			}
		}
	}

	// Build active path set for marking.
	active := make(map[*trieNode]bool, len(c.activePath))
	for _, n := range c.activePath {
		active[n] = true
	}

	var nodeCount, snapshotCount int
	var pagedBytes int64
	var lines []string
	var dump func(n *trieNode, prefix string, isLast bool)
	dump = func(n *trieNode, prefix string, isLast bool) {
		if n == nil {
			return
		}
		nodeCount++

		// Build connector
		var connector string
		if n.parent == nil {
			connector = ""
		} else if isLast {
			connector = prefix + "`-- "
		} else {
			connector = prefix + "|-- "
		}

		// Node label
		nodeBytes := n.snapshotBytes()
		pagedBytes += nodeBytes

		label := fmt.Sprintf("[%d,%d) %dt", n.startOffset(), n.endOffset, len(n.tokens))
		if nodeBytes > 0 {
			label += " " + mlx.PrettyBytes(int(nodeBytes)).String()
		}
		if !n.lastUsed.IsZero() {
			label += fmt.Sprintf(" %s ago", time.Since(n.lastUsed).Truncate(time.Millisecond))
		}
		var flags []string
		if n.user {
			flags = append(flags, "user")
		}
		if n.hasAllSnapshots() {
			snapshotCount++
			flags = append(flags, "snap")
		}
		if active[n] {
			flags = append(flags, "active")
		}
		if len(flags) > 0 {
			label += " (" + flags[0]
			for _, f := range flags[1:] {
				label += ", " + f
			}
			label += ")"
		}
		lines = append(lines, connector+label)

		// Recurse children
		childPrefix := prefix
		if n.parent != nil {
			if isLast {
				childPrefix += "    "
			} else {
				childPrefix += "|   "
			}
		}
		for i, child := range n.children {
			dump(child, childPrefix, i == len(n.children)-1)
		}
	}
	dump(c.root, "", true)

	offset := c.minCacheOffset()
	logutil.Trace(fmt.Sprintf("kv cache active_tokens: %d, active_size: %s, paged_out: %s, trie: nodes=%d, snapshots=%d",
		offset, mlx.PrettyBytes(cacheBytes), mlx.PrettyBytes(int(pagedBytes)), nodeCount, snapshotCount))
	for i, l := range lines {
		if i == 0 {
			logutil.Trace("cache trie: " + l)
		} else {
			logutil.Trace("  " + l)
		}
	}
}
