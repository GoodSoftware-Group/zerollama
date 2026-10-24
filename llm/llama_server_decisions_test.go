package llm

import (
	"encoding/json"
	"fmt"
	"hash/fnv"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"sync/atomic"
	"testing"
)

// TestLayaTokenCacheSkipsRepeatTokenize pins the AR latency fix: router questions are
// static, so after the first decision only a new state reaches llama-server /tokenize,
// and cached packing must be identical to uncached packing.
func TestLayaTokenCacheSkipsRepeatTokenize(t *testing.T) {
	var calls atomic.Int64
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/tokenize" {
			t.Errorf("unexpected path %s", r.URL.Path)
			return
		}
		calls.Add(1)
		var req struct {
			Content string `json:"content"`
		}
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			t.Errorf("decode: %v", err)
			return
		}
		ids := []int{}
		for _, w := range strings.Fields(req.Content) {
			h := fnv.New32a()
			h.Write([]byte(w))
			ids = append(ids, 1000+int(h.Sum32()%5000))
		}
		json.NewEncoder(w).Encode(map[string][]int{"tokens": ids})
	}))
	defer srv.Close()

	parts := strings.Split(srv.URL, ":")
	var port int
	fmt.Sscanf(parts[len(parts)-1], "%d", &port)

	questions := map[string]DecisionQuestion{
		"task": {Type: "choice", Instructions: "What kind of work is this?",
			Criteria: json.RawMessage(`{"code":"write, fix or explain code","chat":"conversation or advice","math":"calculation or proof"}`)},
		"reasoning": {Type: "noul", Instructions: "Does answering need multi-step reasoning?"},
	}
	newTok := func(s *llamaServerRunner) LayaTokenizer {
		return &layaServerTokenizer{s: s, cls: 101, sep: 102, mask: 103, maskTok: "[MASK]"}
	}

	cached := &llamaServerRunner{port: port}
	first, err := PackQuestions(newTok(cached), "fix my python loop", questions, defaultLayaMaxLen, defaultLayaHeadMaxLen)
	if err != nil {
		t.Fatal(err)
	}
	cold := calls.Load()
	if cold == 0 {
		t.Fatal("first pack made no /tokenize calls")
	}

	calls.Store(0)
	if _, err := PackQuestions(newTok(cached), "fix my python loop", questions, defaultLayaMaxLen, defaultLayaHeadMaxLen); err != nil {
		t.Fatal(err)
	}
	if n := calls.Load(); n != 0 {
		t.Fatalf("repeat request: %d /tokenize calls, want 0", n)
	}

	calls.Store(0)
	second, err := PackQuestions(newTok(cached), "prove sqrt 2 is irrational", questions, defaultLayaMaxLen, defaultLayaHeadMaxLen)
	if err != nil {
		t.Fatal(err)
	}
	if n := calls.Load(); n != 1 {
		t.Fatalf("new state: %d /tokenize calls, want 1 (cold was %d)", n, cold)
	}

	// Same inputs through a fresh (uncached) runner must pack identically.
	for state, want := range map[string][]LayaPackedInput{"fix my python loop": first, "prove sqrt 2 is irrational": second} {
		got, err := PackQuestions(newTok(&llamaServerRunner{port: port}), state, questions, defaultLayaMaxLen, defaultLayaHeadMaxLen)
		if err != nil {
			t.Fatal(err)
		}
		if !reflect.DeepEqual(got, want) {
			t.Fatalf("state %q: cached pack differs from uncached\n got %+v\nwant %+v", state, got, want)
		}
	}
}
