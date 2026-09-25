package llm

import (
	"context"
	"encoding/json"
	"math"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"
)

func TestLoadCLMHeads(t *testing.T) {
	path := filepath.Join(os.Getenv("HOME"), ".cache/clm/CLM_v0.1-8B.gguf")
	if _, err := os.Stat(path); err != nil {
		t.Skip("CLM heads GGUF not present:", path)
	}
	h, err := LoadCLMHeads(path)
	if err != nil {
		t.Fatal(err)
	}
	if h.Hidden != 4096 || h.Width != 1536 || h.Proj != 512 || h.Depth != 3 {
		t.Fatalf("meta %+v", h)
	}
	if h.Scale < 50 || h.Scale > 100.5 {
		t.Fatalf("scale %v", h.Scale)
	}
	// Project a unit-ish vector through both heads (smoke).
	x := make([]float32, h.Hidden)
	x[0] = 1
	zs := h.ProjectStates([][]float32{x})
	za := h.ProjectActions([][]float32{x})
	if len(zs[0]) != h.Proj || len(za[0]) != h.Proj {
		t.Fatalf("proj dim")
	}
	var ns, na float64
	for i := range zs[0] {
		ns += float64(zs[0][i]) * float64(zs[0][i])
		na += float64(za[0][i]) * float64(za[0][i])
	}
	if math.Abs(ns-1) > 1e-3 || math.Abs(na-1) > 1e-3 {
		t.Fatalf("not normalized: %v %v", ns, na)
	}
}

func TestCLMSchemaChoiceOrder(t *testing.T) {
	q := DecisionQuestion{
		Type:         "choice",
		Instructions: "Which?",
		Criteria:     json.RawMessage(`{"billing":"Charges","technical":"Bugs"}`),
	}
	keys, texts, err := clmCandidates(q)
	if err != nil {
		t.Fatal(err)
	}
	if keys[0] != "billing" || keys[1] != "technical" {
		t.Fatalf("keys %v", keys)
	}
	if texts[0] != "Charges" || texts[1] != "Bugs" {
		t.Fatalf("texts %v", texts)
	}
}

func TestCLMAnswerMockEmbed(t *testing.T) {
	path := filepath.Join(os.Getenv("HOME"), ".cache/clm/CLM_v0.1-8B.gguf")
	if _, err := os.Stat(path); err != nil {
		t.Skip(path)
	}
	h, err := LoadCLMHeads(path)
	if err != nil {
		t.Fatal(err)
	}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body struct {
			Input []string `json:"input"`
		}
		_ = json.NewDecoder(r.Body).Decode(&body)
		data := make([]map[string]any, len(body.Input))
		for i := range body.Input {
			emb := make([]float64, h.Hidden)
			emb[i%h.Hidden] = 1
			data[i] = map[string]any{"index": i, "embedding": emb}
		}
		_ = json.NewEncoder(w).Encode(map[string]any{
			"data":  data,
			"usage": map[string]int{"prompt_tokens": 10},
		})
	}))
	t.Cleanup(srv.Close)

	resp, err := CLMAnswer(context.Background(), path, srv.URL+"/v1/embeddings", "qwen3-8b", DecisionsRequest{
		Model: "clm",
		State: "customer angry about double charge",
		Questions: map[string]DecisionQuestion{
			"dept": {
				Type:         "choice",
				Instructions: "Which team?",
				Criteria:     json.RawMessage(`{"billing":"refunds","tech":"bugs"}`),
			},
		},
	}, 1)
	if err != nil {
		t.Fatal(err)
	}
	if _, ok := resp.Answers["dept"]; !ok {
		t.Fatalf("answers %v", resp.Answers)
	}
	var ans map[string]any
	if err := json.Unmarshal(resp.Answers["dept"], &ans); err != nil {
		t.Fatal(err)
	}
	if ans["type"] != "choice" {
		t.Fatalf("%v", ans)
	}
	if resp.Usage.InputTokens != 10 {
		t.Fatalf("tokens %d", resp.Usage.InputTokens)
	}
}
