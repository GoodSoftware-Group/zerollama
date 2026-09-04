package server

import (
	"strings"
	"testing"
)

func TestChatCompressLabCompare(t *testing.T) {
	rows, err := ChatCompressLabCompare(4096)
	if err != nil {
		t.Fatal(err)
	}
	by := map[string]ChatCompressLabRow{}
	for _, r := range rows {
		by[r.Name] = r
	}
	none, ph, sum := by["none"], by["placeholder"], by["summary"]
	if none.Recompute < 1000 {
		t.Fatalf("fixture too small: %+v", none)
	}
	if ph.Recompute >= none.Recompute {
		t.Fatalf("placeholder recompute %d should beat none %d", ph.Recompute, none.Recompute)
	}
	if ph.Reuse <= sum.Reuse {
		t.Fatalf("placeholder reuse %d should beat summary %d (in-place elide)", ph.Reuse, sum.Reuse)
	}
	if ph.Reuse < 200 {
		t.Fatalf("newest-first elide should keep a long exact prefix, reuse=%d", ph.Reuse)
	}
	strip := by["suffix-strip+anchor"]
	if strip.Recompute != 400 {
		t.Fatalf("anchor should prefill only suffix, got %d", strip.Recompute)
	}
	sparse := by["suffix-strip+ckpt@4k"]
	if sparse.Recompute < strip.Recompute {
		t.Fatalf("sparse ckpt %d vs anchor %d", sparse.Recompute, strip.Recompute)
	}
	line := ChatCompressLabSummary()
	t.Log(line)
	if !strings.Contains(line, "placeholder reuse=") || !strings.Contains(line, "suffix-strip+anchor=400") || !strings.Contains(line, "sticky=prompt_cache_key") {
		t.Fatalf("summary %q", line)
	}
}
