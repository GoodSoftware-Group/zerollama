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
	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/server/modality"
	"github.com/ollama/ollama/types/model"
)

// decisionsExternalClient bounds time-to-first-byte for CLM/Laya sidecars.
var decisionsExternalClient = &http.Client{
	Timeout: 120 * time.Second,
}

// openJevExternalClient allows longer cold-load + denoise on 16GB (partial -ngl).
// WHY 10m (not 120s): Q4_K_M load with -ngl 25 + denoise routinely exceeds two minutes
// on first request; the CLM/Laya timeout would false-fail the spike path.
var openJevExternalClient = &http.Client{
	Timeout: 10 * time.Minute,
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

// isOpenJevModelName reports DiffusionGemma / OpenJev Decider names for
// ZEROLLAMA_OPENJEV_URL. Matches: openjev, openjev:tag, openjev-*, diffusiongemma*.
//
// WHY keep the "openjev" name while the engine is DiffusionGemma: public wire is
// Jev-shaped (/v1/systemone); agents already say model=openjev. Docs must say
// DG2 v0 is uncalibrated — see docs/diffusion-gemma-llama-cpp-findings.md Finding 6.
func isOpenJevModelName(name string) bool {
	n := strings.ToLower(strings.TrimSpace(name))
	if n == "" {
		return false
	}
	if n == "openjev" || strings.HasPrefix(n, "openjev:") || strings.HasPrefix(n, "openjev-") {
		return true
	}
	if strings.HasPrefix(n, "diffusiongemma") {
		return true
	}
	return false
}

// isGlinerDecideModelName reports Fastino GLiNER2.5-Decide Decider names for
// ZEROLLAMA_GLINER_DECIDE_URL. Matches: gliner-decide, gliner-decide:*, gliner2.5-decide*.
//
// WHY not reuse isGlinerModelName: extract gliner-* is NER; Decide is triage (GD*).
func isGlinerDecideModelName(name string) bool {
	n := strings.ToLower(strings.TrimSpace(name))
	if n == "" {
		return false
	}
	if n == "gliner-decide" || strings.HasPrefix(n, "gliner-decide:") || strings.HasPrefix(n, "gliner-decide-") {
		return true
	}
	if strings.HasPrefix(n, "gliner2.5-decide") || strings.HasPrefix(n, "gliner2-5-decide") {
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
		case model.BackendOpenJev:
			if u := envconfig.OpenJevURL(); u != "" {
				return u, nil
			}
			return "", fmt.Errorf("modality_backends.decisions=openjev requires ZEROLLAMA_OPENJEV_URL (llama-diffusion-gemma-server); see docs/diffusion-gemma-llama-cpp.md")
		case model.BackendGlinerDecide:
			if u := envconfig.GlinerDecideURL(); u != "" {
				return u, nil
			}
			return "", fmt.Errorf("modality_backends.decisions=gliner-decide requires ZEROLLAMA_GLINER_DECIDE_URL; see docs/gliner-decide.md")
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
	if isOpenJevModelName(modelName) {
		if u := envconfig.OpenJevURL(); u != "" {
			return u, nil
		}
		return "", fmt.Errorf("model %q requires ZEROLLAMA_OPENJEV_URL (DiffusionGemma server); see docs/diffusion-gemma-llama-cpp.md", modelName)
	}
	if isGlinerDecideModelName(modelName) {
		if u := envconfig.GlinerDecideURL(); u != "" {
			return u, nil
		}
		return "", fmt.Errorf("model %q requires ZEROLLAMA_GLINER_DECIDE_URL; see docs/gliner-decide.md", modelName)
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
// CLM, OpenJev, and GLiNER-Decide speak /v1/systemone; external Laya uses /v1/decisions.
func decisionsExternalPath(base string, modelName string, m *Model) string {
	base = strings.TrimSuffix(base, "/")
	backend := ""
	if m != nil {
		backend = decisionsBackend(m.Config)
	}
	if backend == model.BackendCLM || isCLMModelName(modelName) {
		return base + "/v1/systemone"
	}
	if backend == model.BackendOpenJev || isOpenJevModelName(modelName) {
		return base + "/v1/systemone"
	}
	if backend == model.BackendGlinerDecide || isGlinerDecideModelName(modelName) {
		return base + "/v1/systemone"
	}
	return base + "/v1/decisions"
}

// proxyDecisionsExternal POSTs to an external Decider.
//
// WHY branch OpenJev: CLM/Laya sidecars speak public DecisionsRequest JSON;
// DiffusionGemma C++ speaks prompt-only {prompt}→{answer}. Blind passthrough
// would either 400 on C++ or reintroduce C++ packing (LA16 inversion).
func proxyDecisionsExternal(ctx context.Context, base string, modelName string, m *Model, req api.DecisionsRequest) (api.DecisionsResponse, error) {
	backend := ""
	if m != nil {
		backend = decisionsBackend(m.Config)
	}
	if backend == model.BackendOpenJev || isOpenJevModelName(modelName) {
		return proxyOpenJevDecisions(ctx, base, modelName, req)
	}
	return proxyDecisionsPassthrough(ctx, base, modelName, m, req)
}

// proxyOpenJevDecisions packs state/questions in Go, POSTs prompt (+ DG3a slots /
// DG2c gather token ids) to llama-diffusion-gemma-server /v1/systemone, then
// parses answer text and merges canvas logit gathers.
// WHY not passthrough: see proxyDecisionsExternal; LA16 + calibrated:false.
func proxyOpenJevDecisions(ctx context.Context, base, modelName string, req api.DecisionsRequest) (api.DecisionsResponse, error) {
	prompt, err := llm.PackOpenJevPrompt(req.State, req.Questions)
	if err != nil {
		return api.DecisionsResponse{}, err
	}
	base = strings.TrimSuffix(base, "/")
	tokenizeFn := func(text string) ([]int, error) {
		return openJevTokenize(ctx, base, text)
	}
	slots, err := llm.BuildOpenJevSlotSpecs(req.Questions, tokenizeFn)
	if err != nil {
		return api.DecisionsResponse{}, err
	}
	gather, err := llm.BuildOpenJevGatherSpecs(req.Questions, tokenizeFn)
	if err != nil {
		return api.DecisionsResponse{}, err
	}
	payload := map[string]any{
		"model":  modelName,
		"prompt": prompt,
	}
	openJevApplyOptions(payload, req.Options)
	if len(slots) > 0 || len(gather) > 0 {
		ro := map[string]any{}
		if len(slots) > 0 {
			ro["slots"] = slots
		}
		if len(gather) > 0 {
			ro["gather"] = gather
		}
		payload["readout"] = ro
	}
	body, err := json.Marshal(payload)
	if err != nil {
		return api.DecisionsResponse{}, err
	}
	target := base + "/v1/systemone"
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, target, bytes.NewReader(body))
	if err != nil {
		return api.DecisionsResponse{}, err
	}
	httpReq.Header.Set("Content-Type", "application/json")
	resp, err := openJevExternalClient.Do(httpReq)
	if err != nil {
		return api.DecisionsResponse{}, fmt.Errorf("openjev proxy: %w", err)
	}
	defer resp.Body.Close()
	raw, err := io.ReadAll(resp.Body)
	if err != nil {
		return api.DecisionsResponse{}, fmt.Errorf("openjev proxy: read body: %w", err)
	}
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		msg := strings.TrimSpace(string(raw))
		if msg == "" {
			msg = resp.Status
		}
		var ej struct {
			Error any `json:"error"`
		}
		if json.Unmarshal(raw, &ej) == nil {
			switch e := ej.Error.(type) {
			case string:
				if e != "" {
					msg = e
				}
			case map[string]any:
				if m, _ := e["message"].(string); m != "" {
					msg = m
				}
			}
		}
		return api.DecisionsResponse{}, api.StatusError{
			StatusCode:   resp.StatusCode,
			Status:       resp.Status,
			ErrorMessage: msg,
		}
	}
	var upstream struct {
		Model       string `json:"model"`
		Answer      string `json:"answer"`
		ReadoutMode string `json:"readout_mode"`
		Usage       struct {
			InputTokens  int `json:"input_tokens"`
			OutputTokens int `json:"output_tokens"`
		} `json:"usage"`
		Gather []llm.OpenJevGatherHit `json:"gather"`
	}
	if err := json.Unmarshal(raw, &upstream); err != nil {
		return api.DecisionsResponse{}, fmt.Errorf("openjev proxy: decode: %w", err)
	}
	answers, err := llm.ParseOpenJevAnswers(upstream.Answer, req.Questions)
	if err != nil {
		return api.DecisionsResponse{}, err
	}
	scoreSrc := upstream.ReadoutMode
	if scoreSrc == "" {
		scoreSrc = "final_answer_logit_gather"
	}
	answers, err = llm.MergeOpenJevGather(answers, req.Questions, upstream.Gather, scoreSrc, envconfig.OpenJevTemperature(), envconfig.OpenJevCalibrated(), envconfig.OpenJevNoulBias(), envconfig.OpenJevCalibrated() && envconfig.OpenJevNoulCalibrated())
	if err != nil {
		return api.DecisionsResponse{}, err
	}
	out := api.DecisionsResponse{
		Model:   upstream.Model,
		Answers: answers,
	}
	if out.Model == "" {
		out.Model = modelName
	}
	out.Usage.InputTokens = upstream.Usage.InputTokens
	out.Usage.OutputTokens = upstream.Usage.OutputTokens
	return out, nil
}

// openJevApplyOptions forwards Decider options to the sibling diffusion body.
// WHY: DG4a oracle needs a pinned seed + step count; without this Go dropped them.
func openJevApplyOptions(payload map[string]any, opts map[string]any) {
	if len(opts) == 0 {
		return
	}
	if v, ok := opts["seed"]; ok {
		payload["seed"] = v
	}
	if v, ok := opts["n_steps"]; ok {
		payload["n_steps"] = v
	} else if v, ok := opts["diffusion_steps"]; ok {
		payload["n_steps"] = v
	}
	if v, ok := opts["max_tokens"]; ok {
		payload["max_tokens"] = v
	}
}

func openJevTokenize(ctx context.Context, base, content string) ([]int, error) {
	body, err := json.Marshal(map[string]any{"content": content, "add_special": false})
	if err != nil {
		return nil, err
	}
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, base+"/tokenize", bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	httpReq.Header.Set("Content-Type", "application/json")
	resp, err := openJevExternalClient.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("openjev tokenize: %w", err)
	}
	defer resp.Body.Close()
	raw, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return nil, fmt.Errorf("openjev tokenize: %s: %s", resp.Status, strings.TrimSpace(string(raw)))
	}
	var out struct {
		Tokens []int `json:"tokens"`
	}
	if err := json.Unmarshal(raw, &out); err != nil {
		return nil, err
	}
	return out.Tokens, nil
}

func proxyDecisionsPassthrough(ctx context.Context, base string, modelName string, m *Model, req api.DecisionsRequest) (api.DecisionsResponse, error) {
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
