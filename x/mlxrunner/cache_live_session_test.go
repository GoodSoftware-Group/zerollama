package mlxrunner

import (
	"testing"
)

func TestLongestCommonPrefix(t *testing.T) {
	a := []int32{1, 2, 3, 4, 5}
	b := []int32{1, 2, 3, 9, 8}
	if got := longestCommonPrefix(a, b); got != 3 {
		t.Fatalf("lcp=%d want 3", got)
	}
	if got := longestCommonPrefix(a, a); got != 5 {
		t.Fatalf("identical lcp=%d want 5", got)
	}
}

func TestTryExtendLiveSessionRewindsGenTokens(t *testing.T) {
	env := newTransformerEnv()
	kvc := env.kvc
	kvc.lastPromptCacheKey = "hermes:main"

	turn1 := make([]int32, 100)
	for i := range turn1 {
		turn1[i] = int32(i + 1)
	}
	kvc.lastSessionInputs = append([]int32(nil), turn1...)

	// Turn 1 ended with prompt + 20 generated tokens in KV (live > lcp).
	gen := make([]int32, 20)
	for i := range gen {
		gen[i] = int32(9000 + i)
	}
	feedAll(kvc.caches, turn1)
	feedAll(kvc.caches, gen)
	kvc.activePath = []*trieNode{kvc.root, {tokens: turn1, endOffset: len(turn1)}}
	if got := kvc.minCacheOffset(); got != len(turn1)+len(gen) {
		t.Fatalf("setup offset=%d want %d", got, len(turn1)+len(gen))
	}

	turn2 := append(append([]int32(nil), turn1...), 200, 201, 202)
	session, ok := kvc.tryExtendLiveSession(0, turn2)
	if !ok {
		t.Fatal("expected live session extension")
	}
	if !session.fastPath {
		t.Fatal("expected fast_path session")
	}
	if session.cachedPrefix != 100 {
		t.Fatalf("cachedPrefix=%d want 100", session.cachedPrefix)
	}
	if len(session.remaining) != 3 {
		t.Fatalf("remaining=%d want 3", len(session.remaining))
	}
	if kvc.minCacheOffset() != 100 {
		t.Fatalf("kv offset=%d want 100 after rewind", kvc.minCacheOffset())
	}
}

func TestTryExtendLiveSessionRejectsStubRewrite(t *testing.T) {
	env := newTransformerEnv()
	kvc := env.kvc
	kvc.lastPromptCacheKey = "hermes:main"

	turn1 := make([]int32, 600)
	for i := range turn1 {
		turn1[i] = int32(i + 1)
	}
	kvc.lastSessionInputs = append([]int32(nil), turn1...)
	feedAll(kvc.caches, turn1)
	kvc.activePath = []*trieNode{kvc.root, {tokens: turn1, endOffset: len(turn1)}}

	turn2 := make([]int32, 600)
	for i := range turn2 {
		turn2[i] = int32(9000 + i)
	}
	if _, ok := kvc.tryExtendLiveSession(0, turn2); ok {
		t.Fatal("expected reject on non-append-only rewrite")
	}
}

func TestRewindCachesToFreesOnFailure(t *testing.T) {
	env := newSlidingWindowEnv()
	kvc := env.kvc
	kvc.ensureRoot()

	// Fill past maxSize so live rewind to an interior offset fails.
	many := make([]int32, 12)
	for i := range many {
		many[i] = int32(i + 1)
	}
	feedAll(kvc.caches, many)
	if kvc.minCacheOffset() != len(many) {
		t.Fatalf("offset=%d want %d", kvc.minCacheOffset(), len(many))
	}

	if kvc.rewindCachesTo(4) {
		t.Fatal("expected rewind failure past sliding window")
	}
	for _, kv := range kvc.caches {
		if kv != nil && kv.Offset() != 0 {
			t.Fatalf("expected freed cache, got offset=%d", kv.Offset())
		}
	}
}

func TestTryExtendLiveSessionRotatingViaSnapshots(t *testing.T) {
	env := newSlidingWindowEnv()
	kvc := env.kvc
	kvc.ensureRoot()

	turn1 := []int32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
	simulateRequest(t, kvc, turn1, []int32{900, 901, 902}, 4, 8, len(turn1))
	kvc.lastPromptCacheKey = "agent:test"
	kvc.lastSessionInputs = append([]int32(nil), turn1...)

	live := kvc.minCacheOffset()
	if live != len(turn1)+3 {
		t.Fatalf("setup offset=%d want %d", live, len(turn1)+3)
	}

	turn2 := append(append([]int32(nil), turn1...), 100, 101)
	session, ok := kvc.tryExtendLiveSession(4, turn2)
	if !ok {
		t.Fatal("expected live session via snapshots on rotating KV")
	}
	if !session.fastPath {
		t.Fatal("expected fast_path session")
	}
	if session.cachedPrefix != len(turn1) {
		t.Fatalf("cachedPrefix=%d want %d (snapshot at prompt end)", session.cachedPrefix, len(turn1))
	}
	if kvc.minCacheOffset() != len(turn1) {
		t.Fatalf("kv offset=%d want %d after rewind", kvc.minCacheOffset(), len(turn1))
	}
	assertCacheOffsetAlignment(t, kvc, "after fast_path rewind")

	// Regression: schedulePrefillSnapshots must not panic when trie snapshots
	// include promptLen-1 after a wrapped rotating rewind left layers misaligned.
	session.schedulePrefillSnapshots([]int{len(turn2) - 1})
	assertCacheOffsetAlignment(t, kvc, "after schedulePrefillSnapshots")
}

func TestTrySameBranchRestoreSkipsPageIn(t *testing.T) {
	env := newTransformerEnv()
	kvc := env.kvc
	kvc.ensureRoot()

	inputs := []int32{1, 2, 3, 4, 5, 6, 7, 8}
	simulateRequest(t, kvc, inputs, nil)

	node := kvc.activePath[len(kvc.activePath)-1]
	path := slicesCloneNodes(kvc.activePath)
	matched := kvc.minCacheOffset()
	if !kvc.trySameBranchRestore(path, matched) {
		t.Fatal("expected same-branch restore")
	}
	if kvc.minCacheOffset() != matched {
		t.Fatalf("offset=%d want %d", kvc.minCacheOffset(), matched)
	}
	if node != kvc.activePath[len(kvc.activePath)-1] {
		t.Fatal("active path leaf changed on same-branch restore")
	}
}

func slicesCloneNodes(path []*trieNode) []*trieNode {
	out := make([]*trieNode, len(path))
	copy(out, path)
	return out
}

func simulateKeyedRequest(t *testing.T, kvc *kvCache, key string, inputs, generated []int32, userSnapshotAt ...int) {
	t.Helper()
	session := kvc.begin(nil, inputs, key, false)
	var snapshotOffsets []int
	for _, at := range userSnapshotAt {
		if at > 0 {
			snapshotOffsets = append(snapshotOffsets, at)
		}
	}
	assertCacheOffsetAlignment(t, kvc, "after begin")
	baseOffset := kvc.minCacheOffset()
	remaining := inputs[baseOffset:]
	session.schedulePrefillSnapshots(snapshotOffsets)
	if len(remaining) > 0 {
		feedAll(kvc.caches, remaining)
	}
	session.attachPrefillSnapshots()
	assertCacheOffsetAlignment(t, kvc, "after prefill")
	if len(generated) > 0 {
		session.outputs = generated
		feedAll(kvc.caches, generated)
	}
	assertCacheOffsetAlignment(t, kvc, "before close")
	session.close()
}

func TestTryExtendLiveSessionAfterSidecarClobber(t *testing.T) {
	env := newTransformerEnv()
	kvc := env.kvc

	turn1 := make([]int32, 120)
	for i := range turn1 {
		turn1[i] = int32(i + 1)
	}
	simulateKeyedRequest(t, kvc, "hermes:main", turn1, []int32{800, 801, 802}, len(turn1))

	if kvc.lastPromptCacheKey != "hermes:main" {
		t.Fatalf("lastPromptCacheKey=%q want hermes:main", kvc.lastPromptCacheKey)
	}
	if len(kvc.lastSessionInputs) != len(turn1) {
		t.Fatalf("lastSessionInputs=%d want %d", len(kvc.lastSessionInputs), len(turn1))
	}
	turn2 := append(append([]int32(nil), turn1...), 900, 901)

	// Unkeyed runtime sidecar shares the runner and switches activePath.
	sidecar := make([]int32, 40)
	for i := range sidecar {
		sidecar[i] = int32(5000 + i)
	}
	simulateRequest(t, kvc, sidecar, nil)
	if kvc.lastPromptCacheKey != "hermes:main" {
		t.Fatalf("sidecar cleared lastPromptCacheKey=%q", kvc.lastPromptCacheKey)
	}

	session, ok := kvc.tryExtendLiveSession(0, turn2)
	if !ok {
		t.Fatal("expected live session after sidecar clobber bootstrap")
	}
	if !session.fastPath {
		t.Fatal("expected fast_path session")
	}
	if session.cachedPrefix != len(turn1) {
		t.Fatalf("cachedPrefix=%d want %d", session.cachedPrefix, len(turn1))
	}
}
