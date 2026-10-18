package storaged

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"path/filepath"
	"strings"
	"sync"

	"github.com/ollama/ollama/types/model"
)

// PullOptions are registry options for a remote pull into ModelsDir.
type PullOptions struct {
	Insecure bool
	Source   string // optional huggingface:// URI; Model is the local tag
}

// ProgressEvent is one NDJSON line on POST /v1/pull (stream=true).
// Shape matches api.ProgressResponse so CLI progress bars stay shared.
type ProgressEvent struct {
	Status    string `json:"status"`
	Digest    string `json:"digest,omitempty"`
	Total     int64  `json:"total,omitempty"`
	Completed int64  `json:"completed,omitempty"`
	Error     string `json:"error,omitempty"`
}

// PullFunc downloads a model into the process ModelsDir (OLLAMA_MODELS).
// Why injected: storaged must not import package server (inference) — cmd wires
// this to server.PullModel when starting storage serve.
type PullFunc func(ctx context.Context, modelName string, opts PullOptions, fn func(ProgressEvent)) error

// PullRequest is the JSON body for POST /v1/pull.
type PullRequest struct {
	Model    string `json:"model"`
	Name     string `json:"name"` // deprecated alias for Model
	Source   string `json:"source,omitempty"`
	Insecure bool   `json:"insecure,omitempty"`
	Stream   *bool  `json:"stream,omitempty"`
}

func (s *Server) handlePull(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	body, err := io.ReadAll(http.MaxBytesReader(w, r.Body, 1<<20))
	if err != nil {
		http.Error(w, "read body: "+err.Error(), http.StatusBadRequest)
		return
	}
	if err := s.Auth.VerifyRequest(r, body); err != nil {
		http.Error(w, err.Error(), http.StatusUnauthorized)
		return
	}
	if s.Pull == nil {
		http.Error(w, "remote pull not available on this storage server", http.StatusNotImplemented)
		return
	}

	var req PullRequest
	if len(body) > 0 {
		if err := json.Unmarshal(body, &req); err != nil {
			http.Error(w, "invalid JSON: "+err.Error(), http.StatusBadRequest)
			return
		}
	}
	modelName := strings.TrimSpace(req.Model)
	if modelName == "" {
		modelName = strings.TrimSpace(req.Name)
	}
	if modelName == "" {
		http.Error(w, "model is required", http.StatusBadRequest)
		return
	}
	if strings.Contains(modelName, "..") || strings.ContainsAny(modelName, "\x00\n\r") {
		http.Error(w, "invalid model name", http.StatusBadRequest)
		return
	}

	stream := true
	if req.Stream != nil {
		stream = *req.Stream
	}
	opts := PullOptions{Insecure: req.Insecure, Source: strings.TrimSpace(req.Source)}
	ctx := r.Context()
	if stream {
		s.streamPull(w, ctx, modelName, opts)
		return
	}
	s.syncPull(w, ctx, modelName, opts)
}

func (s *Server) streamPull(w http.ResponseWriter, ctx context.Context, modelName string, opts PullOptions) {
	w.Header().Set("Content-Type", "application/x-ndjson")
	w.Header().Set("Cache-Control", "no-cache")
	w.WriteHeader(http.StatusOK)
	flusher, _ := w.(http.Flusher)
	enc := json.NewEncoder(w)
	var writeMu sync.Mutex
	write := func(ev ProgressEvent) {
		writeMu.Lock()
		defer writeMu.Unlock()
		_ = enc.Encode(ev)
		if flusher != nil {
			flusher.Flush()
		}
	}

	err := s.Pull(ctx, modelName, opts, write)
	if err != nil {
		write(ProgressEvent{Status: "error", Error: err.Error()})
		return
	}
	s.enqueuePullReplicate(modelName)
	write(ProgressEvent{Status: "success"})
}

func (s *Server) syncPull(w http.ResponseWriter, ctx context.Context, modelName string, opts PullOptions) {
	err := s.Pull(ctx, modelName, opts, func(ProgressEvent) {})
	if err != nil {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusBadGateway)
		_ = json.NewEncoder(w).Encode(map[string]string{"error": err.Error()})
		return
	}
	s.enqueuePullReplicate(modelName)
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(ProgressEvent{Status: "success"})
}

func (s *Server) enqueuePullReplicate(modelName string) {
	n := model.ParseName(modelName)
	if !n.IsFullyQualified() {
		return
	}
	rel := filepath.ToSlash(n.Filepath())
	s.enqueueReplicateManifest(rel)
}
