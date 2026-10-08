package mlxrunner

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"testing"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/llm"
)

func TestClientScore(t *testing.T) {
	input := llm.ScoreRequest{MaxTokens: 2048, Rows: []llm.ScoreRow{{Prompt: "prompt", Candidates: []string{"A", "B"}}}}
	for _, status := range []int{http.StatusOK, http.StatusBadRequest} {
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if r.URL.Path != "/v1/score" || r.Method != "POST" {
				t.Errorf("wrong endpoint: %s %s", r.Method, r.URL.Path)
			}
			var got llm.ScoreRequest
			if err := json.NewDecoder(r.Body).Decode(&got); err != nil || !reflect.DeepEqual(got, input) {
				t.Errorf("request changed: %+v, %v", got, err)
			}
			if status != http.StatusOK {
				http.Error(w, "prompt too long", status)
				return
			}
			_ = json.NewEncoder(w).Encode(llm.ScoreResponse{Logits: [][]float32{{-1, 2}}, InputTokens: 3})
		}))
		client := &Client{port: srv.Listener.Addr().(*net.TCPAddr).Port, client: srv.Client()}
		result, err := client.Score(context.Background(), input)
		srv.Close()
		if status == http.StatusOK {
			if err != nil || result.InputTokens != 3 || !reflect.DeepEqual(result.Logits, [][]float32{{-1, 2}}) {
				t.Fatalf("ok: result=%+v err=%v", result, err)
			}
			continue
		}
		var statusErr api.StatusError
		if !errors.As(err, &statusErr) || statusErr.StatusCode != status {
			t.Fatalf("bad status: err=%v", err)
		}
	}
}

func TestClientScoreRunnerError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "internal", http.StatusInternalServerError)
	}))
	defer srv.Close()
	client := &Client{port: srv.Listener.Addr().(*net.TCPAddr).Port, client: srv.Client()}
	_, err := client.Score(context.Background(), llm.ScoreRequest{MaxTokens: 1, Rows: []llm.ScoreRow{{Prompt: "x", Candidates: []string{"y"}}}})
	if err == nil {
		t.Fatal("expected error")
	}
	body, _ := io.ReadAll(strings.NewReader(""))
	_ = body
}
