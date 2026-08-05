package modelrepair

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"
)

// HTTPAPI talks to a running zerollama/ollama Go API.
type HTTPAPI struct {
	Base   string
	Client *http.Client
}

func NewHTTPAPI(base string) *HTTPAPI {
	base = strings.TrimRight(base, "/")
	if !strings.HasPrefix(base, "http") {
		base = "http://" + base
	}
	return &HTTPAPI{
		Base: base,
		Client: &http.Client{
			Timeout: 300 * time.Second,
		},
	}
}

func (h *HTTPAPI) Show(ctx context.Context, name string) (*ShowInfo, error) {
	var resp struct {
		Template     string         `json:"template"`
		Parser       string         `json:"parser"`
		Parameters   string         `json:"parameters"`
		Modelfile    string         `json:"modelfile"`
		Capabilities []string       `json:"capabilities"`
		ModelInfo    map[string]any `json:"model_info"`
	}
	if err := h.postJSON(ctx, "/api/show", map[string]any{"name": name, "model": name}, &resp); err != nil {
		return nil, err
	}
	arch := ""
	if resp.ModelInfo != nil {
		if v, ok := resp.ModelInfo["general.architecture"]; ok {
			arch = fmt.Sprint(v)
		}
	}
	if arch == "" {
		arch = architectureFromModelfile(resp.Modelfile)
	}
	return &ShowInfo{
		Name:         name,
		Template:     resp.Template,
		Parser:       resp.Parser,
		Parameters:   resp.Parameters,
		Modelfile:    resp.Modelfile,
		Capabilities: resp.Capabilities,
		Architecture: arch,
	}, nil
}

func architectureFromModelfile(mf string) string {
	// Best-effort: some shows embed arch only in model_info; leave empty if unknown.
	_ = mf
	return ""
}

func (h *HTTPAPI) Generate(ctx context.Context, name string, prompt string, think *bool, opts map[string]any) (*GenerateResult, error) {
	body := map[string]any{
		"model":      name,
		"prompt":     prompt,
		"stream":     false,
		"options":    opts,
		"keep_alive": "30s",
	}
	if think != nil {
		body["think"] = *think
	}
	var resp struct {
		Response   string `json:"response"`
		Thinking   string `json:"thinking"`
		EvalCount  int    `json:"eval_count"`
		DoneReason string `json:"done_reason"`
	}
	if err := h.postJSON(ctx, "/api/generate", body, &resp); err != nil {
		return nil, err
	}
	return &GenerateResult{
		Response:   resp.Response,
		Thinking:   resp.Thinking,
		EvalCount:  resp.EvalCount,
		DoneReason: resp.DoneReason,
	}, nil
}

func (h *HTTPAPI) Chat(ctx context.Context, name string, messages []map[string]string, opts map[string]any) (*ChatResult, error) {
	body := map[string]any{
		"model":      name,
		"messages":   messages,
		"stream":     false,
		"options":    opts,
		"keep_alive": "30s",
	}
	var resp struct {
		Message struct {
			Content  string `json:"content"`
			Thinking string `json:"thinking"`
		} `json:"message"`
		EvalCount  int    `json:"eval_count"`
		DoneReason string `json:"done_reason"`
	}
	if err := h.postJSON(ctx, "/api/chat", body, &resp); err != nil {
		return nil, err
	}
	return &ChatResult{
		Content:    resp.Message.Content,
		Thinking:   resp.Message.Thinking,
		EvalCount:  resp.EvalCount,
		DoneReason: resp.DoneReason,
	}, nil
}

func (h *HTTPAPI) Create(ctx context.Context, name, from, template, parser string, params map[string]any) error {
	stream := true
	body := map[string]any{
		"model":      name,
		"name":       name,
		"from":       from,
		"stream":     stream,
		"template":   template,
		"parser":     parser,
		"parameters": params,
	}
	reqBody, err := json.Marshal(body)
	if err != nil {
		return err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, h.Base+"/api/create", bytes.NewReader(reqBody))
	if err != nil {
		return err
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := h.Client.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		raw, _ := io.ReadAll(io.LimitReader(resp.Body, 1<<16))
		return fmt.Errorf("create HTTP %d: %s", resp.StatusCode, strings.TrimSpace(string(raw)))
	}
	dec := json.NewDecoder(resp.Body)
	for {
		var line map[string]any
		if err := dec.Decode(&line); err != nil {
			if err == io.EOF {
				break
			}
			return err
		}
		if errStr, _ := line["error"].(string); errStr != "" {
			return fmt.Errorf("create: %s", errStr)
		}
	}
	return nil
}

func (h *HTTPAPI) ListRunning(ctx context.Context) ([]string, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, h.Base+"/api/ps", nil)
	if err != nil {
		return nil, err
	}
	resp, err := h.Client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("ps HTTP %d", resp.StatusCode)
	}
	var body struct {
		Models []struct {
			Name  string `json:"name"`
			Model string `json:"model"`
		} `json:"models"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&body); err != nil {
		return nil, err
	}
	out := make([]string, 0, len(body.Models))
	seen := map[string]bool{}
	for _, m := range body.Models {
		n := m.Name
		if n == "" {
			n = m.Model
		}
		if n == "" || seen[n] {
			continue
		}
		seen[n] = true
		out = append(out, n)
	}
	return out, nil
}

func (h *HTTPAPI) Unload(ctx context.Context, name string) error {
	// Why unload between probes: after a slash generation, llama-server prefix
	// KV can poison later turns so a clean prompt still collapses. keep_alive:0
	// expires the runner without scheduling a new load for an empty prompt.
	body := map[string]any{
		"model":      name,
		"prompt":     "",
		"keep_alive": 0,
	}
	var discard map[string]any
	_ = h.postJSON(ctx, "/api/generate", body, &discard)
	return nil
}

func (h *HTTPAPI) postJSON(ctx context.Context, path string, payload any, out any) error {
	raw, err := json.Marshal(payload)
	if err != nil {
		return err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, h.Base+path, bytes.NewReader(raw))
	if err != nil {
		return err
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := h.Client.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	body, err := io.ReadAll(io.LimitReader(resp.Body, 4<<20))
	if err != nil {
		return err
	}
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("%s HTTP %d: %s", path, resp.StatusCode, truncate(string(body), 200))
	}
	if out == nil {
		return nil
	}
	return json.Unmarshal(body, out)
}
