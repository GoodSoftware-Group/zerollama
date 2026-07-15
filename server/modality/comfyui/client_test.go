package comfyui

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

// TestQueuePromptAndPollHistory exercises the client against a mock ComfyUI server that
// mimics the real /prompt, /history/{id}, /view HTTP surface without needing a real
// ComfyUI instance.
func TestQueuePromptAndPollHistory(t *testing.T) {
	const promptID = "abc123"
	const pngBody = "not-a-real-png-but-good-enough"

	pollCount := 0
	mux := http.NewServeMux()
	mux.HandleFunc("/prompt", func(w http.ResponseWriter, r *http.Request) {
		var body map[string]any
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Errorf("decode /prompt body: %v", err)
		}
		if body["prompt"] == nil {
			t.Errorf("expected prompt graph in body, got %v", body)
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{"prompt_id": promptID, "number": 1})
	})
	mux.HandleFunc("/history/"+promptID, func(w http.ResponseWriter, r *http.Request) {
		pollCount++
		w.Header().Set("Content-Type", "application/json")
		if pollCount < 2 {
			// Not complete yet on the first poll.
			json.NewEncoder(w).Encode(map[string]any{})
			return
		}
		json.NewEncoder(w).Encode(map[string]any{
			promptID: map[string]any{
				"status": map[string]any{"completed": true, "status_str": "success"},
				"outputs": map[string]any{
					"9": map[string]any{
						"images": []map[string]any{
							{"filename": "out.png", "subfolder": "", "type": "output"},
						},
					},
				},
			},
		})
	})
	mux.HandleFunc("/view", func(w http.ResponseWriter, r *http.Request) {
		if got := r.URL.Query().Get("filename"); got != "out.png" {
			t.Errorf("view filename: got %q", got)
		}
		w.Write([]byte(pngBody))
	})
	mux.HandleFunc("/upload/image", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{"name": "uploaded.png", "subfolder": "", "type": "input"})
	})

	srv := httptest.NewServer(mux)
	defer srv.Close()

	client := NewClient(srv.URL, 5*time.Second)
	ctx := context.Background()

	uploaded, err := client.UploadImage(ctx, "agent-input.png", []byte("fake-image-bytes"))
	if err != nil {
		t.Fatalf("UploadImage: %v", err)
	}
	if uploaded.Name != "uploaded.png" {
		t.Errorf("uploaded name: got %q", uploaded.Name)
	}

	gotPromptID, err := client.QueuePrompt(ctx, map[string]any{"1": map[string]any{"class_type": "X"}}, "test")
	if err != nil {
		t.Fatalf("QueuePrompt: %v", err)
	}
	if gotPromptID != promptID {
		t.Fatalf("prompt id: got %q want %q", gotPromptID, promptID)
	}

	img, err := client.PollHistory(ctx, gotPromptID, 5*time.Millisecond)
	if err != nil {
		t.Fatalf("PollHistory: %v", err)
	}
	if img.Filename != "out.png" {
		t.Fatalf("image filename: got %q", img.Filename)
	}

	data, err := client.FetchImage(ctx, img)
	if err != nil {
		t.Fatalf("FetchImage: %v", err)
	}
	if string(data) != pngBody {
		t.Fatalf("image bytes: got %q want %q", string(data), pngBody)
	}
}

func TestQueuePromptRejected(t *testing.T) {
	mux := http.NewServeMux()
	mux.HandleFunc("/prompt", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{"error": map[string]any{"type": "invalid_prompt"}})
	})
	srv := httptest.NewServer(mux)
	defer srv.Close()

	client := NewClient(srv.URL, 5*time.Second)
	if _, err := client.QueuePrompt(context.Background(), map[string]any{}, "test"); err == nil {
		t.Fatal("expected error for rejected workflow")
	}
}

func TestPollHistoryExecutionError(t *testing.T) {
	const promptID = "failed123"
	mux := http.NewServeMux()
	mux.HandleFunc("/history/"+promptID, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{
			promptID: map[string]any{
				"status": map[string]any{
					"completed":  false,
					"status_str": "error",
					"messages": []any{
						[]any{"execution_error", map[string]any{
							"node_id":           "3",
							"node_type":         "KSampler",
							"exception_message": "checkpoint file not found",
						}},
					},
				},
				"outputs": map[string]any{},
			},
		})
	})
	srv := httptest.NewServer(mux)
	defer srv.Close()

	client := NewClient(srv.URL, 5*time.Second)
	_, err := client.PollHistory(context.Background(), promptID, 5*time.Millisecond)
	if err == nil {
		t.Fatal("expected execution error")
	}
	if got := err.Error(); !containsAll(got, "KSampler", "checkpoint file not found") {
		t.Fatalf("error missing execution detail: %q", got)
	}
}

func containsAll(s string, substrs ...string) bool {
	for _, sub := range substrs {
		if !strings.Contains(s, sub) {
			return false
		}
	}
	return true
}

func TestPollHistoryContextTimeout(t *testing.T) {
	mux := http.NewServeMux()
	mux.HandleFunc("/history/", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{}) // never completes
	})
	srv := httptest.NewServer(mux)
	defer srv.Close()

	client := NewClient(srv.URL, 5*time.Second)
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Millisecond)
	defer cancel()
	if _, err := client.PollHistory(ctx, "never", 5*time.Millisecond); err == nil {
		t.Fatal("expected context deadline error")
	}
}
