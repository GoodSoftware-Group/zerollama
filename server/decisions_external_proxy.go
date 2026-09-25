package server

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/server/modality"
	"github.com/ollama/ollama/types/model"
)

// decisionsExternalClient bounds time-to-first-byte for CLM/Laya sidecars.
var decisionsExternalClient = &http.Client{
	Timeout: 120 * time.Second,
}

// isCLMModelName reports models that should route to ZEROLLAMA_CLM_URL when set.
// Matches: clm, clm:tag, clm-*, contrastive-lm/*, Contrastive-LM/*.
func isCLMModelName(name string) bool {
	n := strings.ToLower(strings.TrimSpace(name))
	if n == "" {
		return false
	}
	if n == "clm" || strings.HasPrefix(n, "clm:") || strings.HasPrefix(n, "clm-") {
		return true
	}
	if strings.HasPrefix(n, "contrastive-lm/") {
		return true
	}
	return false
}

// decisionsBackend returns modality_backends.decisions, else inference, else "".
func decisionsBackend(cfg model.ConfigV2) string {
	if b := modality.BackendFor(cfg, model.ModalityDecisions); b != "" {
		return b
	}
	return modality.BackendFor(cfg, model.ModalityInference)
}

// decisionsExternalResolve picks an external Decider base URL.
// Prefer per-model modality backend, then CLM name heuristics.
// Returns ("", nil) to use native CLM (heads GGUF) or local Laya Decider.
// Returns ("", err) when the model clearly needs an external URL that is unset
// and native CLM is not configured.
func decisionsExternalResolve(modelName string, m *Model) (base string, err error) {
	nativeCLM := envconfig.CLMHeads() != "" && envconfig.CLMEmbURL() != ""
	if m != nil {
		switch decisionsBackend(m.Config) {
		case model.BackendCLM:
			if u := envconfig.CLMURL(); u != "" {
				return u, nil
			}
			if nativeCLM {
				return "", nil
			}
			return "", fmt.Errorf("modality_backends.decisions=clm requires ZEROLLAMA_CLM_HEADS+ZEROLLAMA_CLM_EMB_URL (native) or ZEROLLAMA_CLM_URL (clm-serve fallback); see docs/clm.md")
		case model.BackendLaya:
			if u := envconfig.LayaURL(); u != "" {
				return u, nil
			}
			// Empty LayaURL: fall through to local GGUF Decider when available.
		}
	}
	if isCLMModelName(modelName) {
		if u := envconfig.CLMURL(); u != "" {
			return u, nil
		}
		if nativeCLM {
			return "", nil
		}
		return "", fmt.Errorf("model %q requires ZEROLLAMA_CLM_HEADS+ZEROLLAMA_CLM_EMB_URL (native Go) or ZEROLLAMA_CLM_URL; see docs/clm.md", modelName)
	}
	return "", nil
}

// clmWantsNative reports CLM requests that should use heads GGUF + embeddings (no Python).
func clmWantsNative(modelName string, m *Model) bool {
	if envconfig.CLMHeads() == "" || envconfig.CLMEmbURL() == "" {
		return false
	}
	// Prefer native when configured; external URL still wins if set (explicit sidecar).
	if envconfig.CLMURL() != "" {
		return false
	}
	if isCLMModelName(modelName) {
		return true
	}
	if m != nil && decisionsBackend(m.Config) == model.BackendCLM {
		return true
	}
	return false
}

// decisionsExternalPath is the upstream path for a given backend.
// CLM speaks /v1/systemone; external Laya uses the same public /v1/decisions wire.
func decisionsExternalPath(base string, modelName string, m *Model) string {
	base = strings.TrimSuffix(base, "/")
	backend := ""
	if m != nil {
		backend = decisionsBackend(m.Config)
	}
	if backend == model.BackendCLM || isCLMModelName(modelName) {
		return base + "/v1/systemone"
	}
	return base + "/v1/decisions"
}

// proxyDecisionsExternal POSTs the public DecisionsRequest JSON to an external Decider.
func proxyDecisionsExternal(ctx context.Context, base string, modelName string, m *Model, req api.DecisionsRequest) (api.DecisionsResponse, error) {
	target := decisionsExternalPath(base, modelName, m)
	body, err := json.Marshal(req)
	if err != nil {
		return api.DecisionsResponse{}, err
	}
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, target, bytes.NewReader(body))
	if err != nil {
		return api.DecisionsResponse{}, err
	}
	httpReq.Header.Set("Content-Type", "application/json")
	resp, err := decisionsExternalClient.Do(httpReq)
	if err != nil {
		return api.DecisionsResponse{}, fmt.Errorf("decisions external proxy: %w", err)
	}
	defer resp.Body.Close()
	raw, err := io.ReadAll(resp.Body)
	if err != nil {
		return api.DecisionsResponse{}, fmt.Errorf("decisions external proxy: read body: %w", err)
	}
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		msg := strings.TrimSpace(string(raw))
		if msg == "" {
			msg = resp.Status
		}
		// Prefer upstream JSON error field when present.
		var ej struct {
			Error string `json:"error"`
		}
		if json.Unmarshal(raw, &ej) == nil && ej.Error != "" {
			msg = ej.Error
		}
		return api.DecisionsResponse{}, api.StatusError{
			StatusCode:   resp.StatusCode,
			Status:       resp.Status,
			ErrorMessage: msg,
		}
	}
	var out api.DecisionsResponse
	if err := json.Unmarshal(raw, &out); err != nil {
		return api.DecisionsResponse{}, fmt.Errorf("decisions external proxy: decode: %w", err)
	}
	if out.Model == "" {
		out.Model = modelName
	}
	if out.Answers == nil {
		out.Answers = map[string]json.RawMessage{}
	}
	return out, nil
}
