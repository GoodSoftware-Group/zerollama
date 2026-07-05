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
		t.Fatalf("miss eval=%d cached=%d, want 10 / 0", eval, cached)
	}

	eval, cached = completionPromptMetrics(nil)
	if eval != 0 || cached != 0 {
		t.Fatalf("nil session eval=%d cached=%d", eval, cached)
	}
}
