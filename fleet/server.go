package fleet

import (
	"encoding/json"
	"errors"
	"log/slog"
	"net/http"
)

// Server exposes fleet management HTTP endpoints for agents.
// Why separate from zerollama serve: management is optional, stateless-ish, and may run on a non-GPU host.
type Server struct {
	manager *Manager
	mux     *http.ServeMux
}

func NewServer(manager *Manager) *Server {
	s := &Server{
		manager: manager,
		mux:     http.NewServeMux(),
	}
	s.mux.HandleFunc("GET /health", s.handleHealth)
	s.mux.HandleFunc("GET /api/fleet/status", s.handleStatus)
	s.mux.HandleFunc("POST /api/fleet/assign", s.handleAssign)
	return s
}

func (s *Server) Handler() http.Handler {
	return s.mux
}

func (s *Server) handleHealth(w http.ResponseWriter, _ *http.Request) {
	writeJSON(w, http.StatusOK, map[string]string{"status": "ok"})
}

func (s *Server) handleStatus(w http.ResponseWriter, _ *http.Request) {
	writeJSON(w, http.StatusOK, s.manager.Snapshot())
}

func (s *Server) handleAssign(w http.ResponseWriter, r *http.Request) {
	var req AssignRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": "invalid JSON body"})
		return
	}

	resp, err := s.manager.Assign(req)
	if err != nil {
		status := http.StatusServiceUnavailable
		switch {
		case errors.Is(err, ErrModelRequired):
			status = http.StatusBadRequest
		case errors.Is(err, ErrNoWarmNode):
			status = http.StatusNotFound
		}
		writeJSON(w, status, map[string]string{"error": err.Error()})
		return
	}
	writeJSON(w, http.StatusOK, resp)
}

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	w.WriteHeader(status)
	if err := json.NewEncoder(w).Encode(v); err != nil {
		slog.Error("fleet: write JSON response", "error", err)
	}
}
