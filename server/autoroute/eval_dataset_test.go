package autoroute

import (
	"bufio"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

// evalRow mirrors testdata/router_eval/prompts.jsonl (docs/auto-router.md AR7). Labels are
// model-agnostic so the same set scores any node's router.yaml.
type evalRow struct {
	ID         string   `json:"id"`
	Surface    string   `json:"surface"`
	Task       string   `json:"task"`
	Difficulty int      `json:"difficulty"`
	Tier       string   `json:"tier"`
	Output     string   `json:"output"`
	Prompt     string   `json:"prompt"`
	System     string   `json:"system"`
	Attach     []string `json:"attach"`
}

// TestEvalDatasetSchema keeps the AR7 dataset loadable and representative. It runs in plain
// CI (no models); scripts/eval/router_eval.py does the scoring against a live node.
func TestEvalDatasetSchema(t *testing.T) {
	path := filepath.Join("..", "..", "testdata", "router_eval", "prompts.jsonl")
	f, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()

	enum := func(vals ...string) map[string]bool {
		m := map[string]bool{}
		for _, v := range vals {
			m[v] = true
		}
		return m
	}
	surfaces := enum("chat", "generate", "image", "image_edit")
	tasks := enum("code", "math", "chat", "extract", "creative")
	tiers := enum("low", "mid", "high")
	outputs := enum("text", "image", "speech", "video", "music")
	attaches := enum("image", "video", "audio", "tools")

	seen := map[string]bool{}
	count := map[string]map[string]int{"surface": {}, "task": {}, "tier": {}, "output": {}, "attach": {}}
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 1<<20), 1<<20)
	line := 0
	for sc.Scan() {
		line++
		var r evalRow
		if err := json.Unmarshal(sc.Bytes(), &r); err != nil {
			t.Fatalf("line %d: %v", line, err)
		}
		switch {
		case r.ID == "" || seen[r.ID]:
			t.Fatalf("line %d: missing or duplicate id %q", line, r.ID)
		case r.Prompt == "":
			t.Fatalf("%s: empty prompt", r.ID)
		case !surfaces[r.Surface] || !tasks[r.Task] || !tiers[r.Tier] || !outputs[r.Output]:
			t.Fatalf("%s: bad enum in %+v", r.ID, r)
		case r.Difficulty < 1 || r.Difficulty > 5:
			t.Fatalf("%s: difficulty %d outside 1..5", r.ID, r.Difficulty)
		case r.Surface == "image_edit" && !contains(r.Attach, "image"):
			t.Fatalf("%s: image_edit rows need an image attachment", r.ID)
		}
		for _, a := range r.Attach {
			if !attaches[a] {
				t.Fatalf("%s: unknown attachment %q", r.ID, a)
			}
			count["attach"][a]++
		}
		seen[r.ID] = true
		count["surface"][r.Surface]++
		count["task"][r.Task]++
		count["tier"][r.Tier]++
		count["output"][r.Output]++
	}
	if err := sc.Err(); err != nil {
		t.Fatal(err)
	}

	if len(seen) < 100 {
		t.Fatalf("AR7 needs ≥100 labeled prompts, have %d", len(seen))
	}
	// Every label value must be represented so per-class metrics are never empty.
	for dim, vals := range map[string]map[string]bool{"surface": surfaces, "task": tasks, "tier": tiers, "output": outputs, "attach": attaches} {
		for v := range vals {
			if count[dim][v] == 0 {
				t.Errorf("no rows with %s=%s", dim, v)
			}
		}
	}
	// Capability filters must be exercised enough to make "100% correct" meaningful.
	for _, a := range []string{"tools", "image", "video", "audio"} {
		if count["attach"][a] < 3 {
			t.Errorf("only %d rows attach %s; want ≥3", count["attach"][a], a)
		}
	}
}

func contains(xs []string, want string) bool {
	for _, x := range xs {
		if x == want {
			return true
		}
	}
	return false
}
