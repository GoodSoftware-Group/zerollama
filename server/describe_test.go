package server

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/gin-gonic/gin"

	"github.com/ollama/ollama/describe"
)

type stubDescribeBackend struct {
	embedCalls int
}

func (s *stubDescribeBackend) EmbedTexts(_ context.Context, _ string, texts []string) ([][]float32, error) {
	s.embedCalls++
	out := make([][]float32, len(texts))
	for i := range texts {
		out[i] = []float32{1, 0}
	}
	return out, nil
}

func (s *stubDescribeBackend) EmbedImage(_ context.Context, _ string, _ []byte) ([]float32, error) {
	s.embedCalls++
	return []float32{1, 0}, nil
}

func (s *stubDescribeBackend) Caption(context.Context, string, []byte) (string, []string, string, error) {
	return "hi", nil, `{"caption":"hi","text":[]}`, nil
}

func (s *stubDescribeBackend) Detect(context.Context, []byte, []string) (*describe.ObjectsResult, error) {
	return &describe.ObjectsResult{Status: "ok", Detections: []describe.Detection{}}, nil
}

func TestDescribeHandlerProcessedAndKept(t *testing.T) {
	gin.SetMode(gin.TestMode)
	stub := &stubDescribeBackend{}
	describeBackendForTest = stub
	t.Cleanup(func() { describeBackendForTest = nil })

	s := &Server{}
	r := gin.New()
	r.POST("/api/describe", s.DescribeHandler)

	body := map[string]any{
		"fields":    []string{"tags"},
		"labels":    []string{"dog"},
		"tag_model": "siglip",
		"min_score": 0.2,
		"images": []map[string]any{
			{"id": "a.jpg", "data": base64.StdEncoding.EncodeToString([]byte("pixels"))},
		},
	}
	raw, _ := json.Marshal(body)
	req := httptest.NewRequest(http.MethodPost, "/api/describe", bytes.NewReader(raw))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	r.ServeHTTP(w, req)
	if w.Code != 200 {
		t.Fatalf("status %d: %s", w.Code, w.Body.String())
	}
	var resp describe.Response
	if err := json.Unmarshal(w.Body.Bytes(), &resp); err != nil {
		t.Fatal(err)
	}
	if len(resp.Images) != 1 || resp.Images[0].Action != describe.ActionProcessed {
		t.Fatalf("%#v", resp)
	}
	if stub.embedCalls < 2 {
		t.Fatalf("expected label+image embeds, got %d", stub.embedCalls)
	}

	// second call with catalog + sha only → kept
	entry := resp.Images[0].Entry
	body2 := map[string]any{
		"fields":    []string{"tags"},
		"labels":    []string{"dog"},
		"tag_model": "siglip",
		"min_score": 0.2,
		"catalog":   []describe.Entry{entry},
		"images": []map[string]any{
			{"id": "a.jpg", "sha256": entry.SHA256},
		},
	}
	raw2, _ := json.Marshal(body2)
	callsBefore := stub.embedCalls
	req2 := httptest.NewRequest(http.MethodPost, "/api/describe", bytes.NewReader(raw2))
	req2.Header.Set("Content-Type", "application/json")
	w2 := httptest.NewRecorder()
	r.ServeHTTP(w2, req2)
	if w2.Code != 200 {
		t.Fatalf("status %d: %s", w2.Code, w2.Body.String())
	}
	var resp2 describe.Response
	_ = json.Unmarshal(w2.Body.Bytes(), &resp2)
	if resp2.Images[0].Action != describe.ActionKept {
		t.Fatalf("want kept, got %#v", resp2.Images[0])
	}
	if stub.embedCalls != callsBefore {
		t.Fatalf("kept should not embed, calls %d → %d", callsBefore, stub.embedCalls)
	}
}

func TestDescribeHandlerNeedsImage(t *testing.T) {
	gin.SetMode(gin.TestMode)
	describeBackendForTest = &stubDescribeBackend{}
	t.Cleanup(func() { describeBackendForTest = nil })

	s := &Server{}
	r := gin.New()
	r.POST("/api/describe", s.DescribeHandler)

	body := map[string]any{
		"fields":    []string{"tags"},
		"tag_model": "siglip",
		"images":    []map[string]any{{"id": "new.jpg", "sha256": "abc"}},
	}
	raw, _ := json.Marshal(body)
	req := httptest.NewRequest(http.MethodPost, "/api/describe", bytes.NewReader(raw))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	r.ServeHTTP(w, req)
	if w.Code != 200 {
		t.Fatalf("%d %s", w.Code, w.Body.String())
	}
	var resp describe.Response
	_ = json.Unmarshal(w.Body.Bytes(), &resp)
	if resp.Images[0].Action != describe.ActionNeedsImage {
		t.Fatalf("%#v", resp.Images[0])
	}
}
