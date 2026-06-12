package fleet

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/ollama/ollama/api"
)

func TestManagerPollsStatus(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		wait := 1
		run := 0
		_ = json.NewEncoder(w).Encode(api.StatusResponse{
			Inference: api.InferenceStatus{
				Ggml: api.GgmlStatus{
					Pending:      0,
					Active:       1,
					Loaded:       1,
					LoadedModels: []string{"llama3:latest"},
				},
				Runtime: api.RuntimeStatus{
					Enabled:   true,
					Available: true,
					Waiting:   &wait,
					Running:   &run,
				},
			},
		})
	}))
	defer srv.Close()

	m, err := NewManager(Config{
		Peers:        []string{srv.URL},
		PollInterval: time.Hour,
		ProbeTimeout: 2 * time.Second,
	})
	if err != nil {
		t.Fatal(err)
	}

	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	go m.Run(ctx)

	deadline := time.Now().Add(2 * time.Second)
	for {
		snap := m.Snapshot()
		if len(snap.Nodes) == 1 && snap.Nodes[0].Available {
			if snap.Nodes[0].QueueDepth != 2 {
				t.Fatalf("queue=%d", snap.Nodes[0].QueueDepth)
			}
			if len(snap.WarmModels) != 1 || snap.WarmModels[0].Model != "llama3:latest" {
				t.Fatalf("warm=%+v", snap.WarmModels)
			}
			return
		}
		if time.Now().After(deadline) {
			t.Fatal("timed out waiting for poll")
		}
		time.Sleep(20 * time.Millisecond)
	}
}

func TestServerAssignEndpoint(t *testing.T) {
	nodes := []NodeSnapshot{
		{ID: "a", URL: "http://a:11434", Available: true, LoadedModels: []string{"llama3"}, QueueDepth: 0},
	}
	m := &Manager{nodes: map[string]NodeSnapshot{"a": nodes[0]}}
	srv := httptest.NewServer(NewServer(m).Handler())
	defer srv.Close()

	req, err := http.NewRequest(http.MethodPost, srv.URL+"/api/fleet/assign", strings.NewReader(`{"model":"llama3"}`))
	if err != nil {
		t.Fatal(err)
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status=%d", resp.StatusCode)
	}
}
