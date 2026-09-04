package mlxrunner

import "testing"

func TestCompletionPromptMetrics(t *testing.T) {
	t.Parallel()

	eval, cached := completionPromptMetrics(&cacheSession{
		inputs:       make([]int32, 100_000),
		cachedPrefix: 99_500,
	})
	if eval != 500 || cached != 99_500 {
		t.Fatalf("eval=%d cached=%d, want 500 / 99500", eval, cached)
	}

	eval, cached = completionPromptMetrics(&cacheSession{
		inputs:       make([]int32, 10),
		cachedPrefix: 0,
	})
	if eval != 10 || cached != 0 {
		t.Fatalf("eval=%d cached=%d, want 10 / 0", eval, cached)
	}

	eval, cached = completionPromptMetrics(nil)
	if eval != 0 || cached != 0 {
		t.Fatalf("nil session eval=%d cached=%d", eval, cached)
	}
}

func TestPrefillBodyLeavesSeed(t *testing.T) {
	if prefillBodyLen(0) != 0 || prefillBodyLen(prefillSeedCount) != 0 {
		t.Fatal("seed-only remaining must not enter the prefill loop")
	}
	if prefillBodyLen(5) != 4 {
		t.Fatalf("body=%d want 4 (leave the last token)", prefillBodyLen(5))
	}
	processed := 0
	total := 5
	chunk := 2
	for body := prefillBodyLen(total - processed); body > 0; body = prefillBodyLen(total - processed) {
		n := min(chunk, body)
		processed += n
	}
	if leftover := total - processed; leftover != prefillSeedCount {
		t.Fatalf("leftover=%d want %d", leftover, prefillSeedCount)
	}
}
