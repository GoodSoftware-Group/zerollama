// Package comfyui implements a thin HTTP client for a ComfyUI server, used as the
// "comfyui" modality_backends.image driver. ComfyUI already ships node graphs for
// Qwen-Image(-Edit), FLUX.1/2, and GLM-Image with LoRA/ControlNet/upscale support —
// porting those DiTs into the MLX Go runner (x/imagegen) would take months per family.
// Zerollama orchestrates a running ComfyUI instance instead: queue a workflow, poll
// history, download the PNG output. See docs/comfyui-image-backend.md.
package comfyui

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"mime/multipart"
	"net/http"
	"net/url"
	"time"
)

// Client talks to a running ComfyUI server's HTTP API.
type Client struct {
	BaseURL    string
	HTTPClient *http.Client
}

// NewClient returns a Client for baseURL (e.g. http://127.0.0.1:8188).
func NewClient(baseURL string, timeout time.Duration) *Client {
	return &Client{
		BaseURL:    baseURL,
		HTTPClient: &http.Client{Timeout: timeout},
	}
}

// UploadedImage identifies an image ComfyUI has staged in its input directory,
// suitable for a LoadImage node's "image" widget value.
type UploadedImage struct {
	Name      string `json:"name"`
	Subfolder string `json:"subfolder"`
	Type      string `json:"type"`
}

// UploadImage sends raw image bytes to ComfyUI's /upload/image endpoint so a
// workflow's LoadImage node can reference it by filename.
func (c *Client) UploadImage(ctx context.Context, filename string, data []byte) (UploadedImage, error) {
	var body bytes.Buffer
	w := multipart.NewWriter(&body)
	part, err := w.CreateFormFile("image", filename)
	if err != nil {
		return UploadedImage{}, fmt.Errorf("comfyui: create form file: %w", err)
	}
	if _, err := part.Write(data); err != nil {
		return UploadedImage{}, fmt.Errorf("comfyui: write image data: %w", err)
	}
	// WHY overwrite=true: Comfy appends " (1)" on name collision; agents always
	// upload as agent-input.png / agent-control.png and must bind that exact name.
	_ = w.WriteField("overwrite", "true")
	if err := w.Close(); err != nil {
		return UploadedImage{}, fmt.Errorf("comfyui: close multipart writer: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.BaseURL+"/upload/image", &body)
	if err != nil {
		return UploadedImage{}, err
	}
	req.Header.Set("Content-Type", w.FormDataContentType())

	resp, err := c.HTTPClient.Do(req)
	if err != nil {
		return UploadedImage{}, fmt.Errorf("comfyui: upload image: %w", err)
	}
	defer resp.Body.Close()
	respBody, _ := io.ReadAll(resp.Body)
	if resp.StatusCode != http.StatusOK {
		return UploadedImage{}, fmt.Errorf("comfyui: upload image: status %d: %s", resp.StatusCode, string(respBody))
	}
	var out UploadedImage
	if err := json.Unmarshal(respBody, &out); err != nil {
		return UploadedImage{}, fmt.Errorf("comfyui: parse upload response: %w", err)
	}
	return out, nil
}

// queuePromptRequest is the body for POST /prompt.
type queuePromptRequest struct {
	Prompt   map[string]any `json:"prompt"`
	ClientID string         `json:"client_id,omitempty"`
}

type queuePromptResponse struct {
	PromptID string         `json:"prompt_id"`
	Number   int            `json:"number"`
	Error    map[string]any `json:"error,omitempty"`
}

// QueuePrompt submits a workflow graph (API format: node id -> {class_type, inputs})
// and returns the prompt id used to poll /history.
func (c *Client) QueuePrompt(ctx context.Context, graph map[string]any, clientID string) (string, error) {
	body, err := json.Marshal(queuePromptRequest{Prompt: graph, ClientID: clientID})
	if err != nil {
		return "", fmt.Errorf("comfyui: marshal prompt: %w", err)
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.BaseURL+"/prompt", bytes.NewReader(body))
	if err != nil {
		return "", err
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := c.HTTPClient.Do(req)
	if err != nil {
		return "", fmt.Errorf("comfyui: queue prompt: %w", err)
	}
	defer resp.Body.Close()
	respBody, _ := io.ReadAll(resp.Body)
	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("comfyui: queue prompt: status %d: %s", resp.StatusCode, string(respBody))
	}
	var out queuePromptResponse
	if err := json.Unmarshal(respBody, &out); err != nil {
		return "", fmt.Errorf("comfyui: parse queue response: %w: %s", err, string(respBody))
	}
	if len(out.Error) > 0 {
		return "", fmt.Errorf("comfyui: workflow rejected: %v", out.Error)
	}
	if out.PromptID == "" {
		return "", fmt.Errorf("comfyui: queue prompt returned no prompt_id: %s", string(respBody))
	}
	return out.PromptID, nil
}

// HistoryOutputImage describes one SaveImage/PreviewImage output entry.
type HistoryOutputImage struct {
	Filename  string `json:"filename"`
	Subfolder string `json:"subfolder"`
	Type      string `json:"type"`
}

type historyEntry struct {
	Status struct {
		StatusStr string  `json:"status_str"`
		Completed bool    `json:"completed"`
		Messages  [][]any `json:"messages"`
	} `json:"status"`
	Outputs map[string]struct {
		Images []HistoryOutputImage `json:"images"`
	} `json:"outputs"`
}

// executionErrorMessage extracts a human-readable error from a "execution_error"
// entry in history[promptID].status.messages, e.g.
// ["execution_error", {"node_id": "3", "node_type": "KSampler", "exception_message": "..."}].
func (h historyEntry) executionErrorMessage() string {
	for _, msg := range h.Status.Messages {
		if len(msg) < 2 {
			continue
		}
		kind, _ := msg[0].(string)
		if kind != "execution_error" {
			continue
		}
		detail, ok := msg[1].(map[string]any)
		if !ok {
			continue
		}
		nodeType, _ := detail["node_type"].(string)
		nodeID, _ := detail["node_id"].(string)
		exc, _ := detail["exception_message"].(string)
		if exc == "" {
			exc = "unknown error"
		}
		if nodeType != "" {
			return fmt.Sprintf("node %s (%s): %s", nodeID, nodeType, exc)
		}
		return exc
	}
	return ""
}

// PollHistory polls GET /history/{promptID} until ComfyUI reports the job complete
// (or interval*maxAttempts is exhausted) and returns the first output image found
// across all nodes. Utility workflows (Qwen-Image GGUF, GLM-Image) can legitimately
// take minutes on 16GB cards, so callers pass a generous ctx deadline instead of a
// fixed attempt count; this loop only bounds the polling cadence.
func (c *Client) PollHistory(ctx context.Context, promptID string, interval time.Duration) (HistoryOutputImage, error) {
	if interval <= 0 {
		interval = time.Second
	}
	ticker := time.NewTicker(interval)
	defer ticker.Stop()

	for {
		img, done, execErr, err := c.historySnapshot(ctx, promptID)
		if err != nil {
			return HistoryOutputImage{}, err
		}
		if execErr != "" {
			return HistoryOutputImage{}, fmt.Errorf("comfyui: prompt %s failed: %s", promptID, execErr)
		}
		if done {
			if img.Filename == "" {
				return HistoryOutputImage{}, fmt.Errorf("comfyui: prompt %s completed without an image output", promptID)
			}
			return img, nil
		}
		select {
		case <-ctx.Done():
			return HistoryOutputImage{}, fmt.Errorf("comfyui: waiting for prompt %s: %w", promptID, ctx.Err())
		case <-ticker.C:
		}
	}
}

// historySnapshot returns (image, done, executionErrorMessage, err). executionErrorMessage
// is non-empty when Comfy reported a node execution failure for this prompt (status_str
// "error" with an "execution_error" message) — Comfy does not always set status.completed
// on failure, so this must be checked independently of done.
func (c *Client) historySnapshot(ctx context.Context, promptID string) (HistoryOutputImage, bool, string, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, c.BaseURL+"/history/"+url.PathEscape(promptID), nil)
	if err != nil {
		return HistoryOutputImage{}, false, "", err
	}
	resp, err := c.HTTPClient.Do(req)
	if err != nil {
		return HistoryOutputImage{}, false, "", fmt.Errorf("comfyui: poll history: %w", err)
	}
	defer resp.Body.Close()
	body, _ := io.ReadAll(resp.Body)
	if resp.StatusCode != http.StatusOK {
		return HistoryOutputImage{}, false, "", fmt.Errorf("comfyui: poll history: status %d: %s", resp.StatusCode, string(body))
	}

	var entries map[string]historyEntry
	if err := json.Unmarshal(body, &entries); err != nil {
		return HistoryOutputImage{}, false, "", fmt.Errorf("comfyui: parse history: %w", err)
	}
	entry, ok := entries[promptID]
	if !ok {
		// Not in history yet — still queued or running.
		return HistoryOutputImage{}, false, "", nil
	}
	if entry.Status.StatusStr == "error" {
		msg := entry.executionErrorMessage()
		if msg == "" {
			msg = "unknown error (no execution_error detail in history)"
		}
		return HistoryOutputImage{}, true, msg, nil
	}
	if !entry.Status.Completed {
		return HistoryOutputImage{}, false, "", nil
	}
	for _, out := range entry.Outputs {
		for _, img := range out.Images {
			if img.Filename != "" {
				return img, true, "", nil
			}
		}
	}
	// Completed but no image output — surface as done so caller can error immediately.
	return HistoryOutputImage{}, true, "", nil
}

// FetchImage downloads a completed output via GET /view.
func (c *Client) FetchImage(ctx context.Context, img HistoryOutputImage) ([]byte, error) {
	q := url.Values{}
	q.Set("filename", img.Filename)
	if img.Subfolder != "" {
		q.Set("subfolder", img.Subfolder)
	}
	q.Set("type", cmpOr(img.Type, "output"))

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, c.BaseURL+"/view?"+q.Encode(), nil)
	if err != nil {
		return nil, err
	}
	resp, err := c.HTTPClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("comfyui: fetch image: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("comfyui: fetch image: status %d: %s", resp.StatusCode, string(body))
	}
	data, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("comfyui: read image body: %w", err)
	}
	if len(data) == 0 {
		return nil, fmt.Errorf("comfyui: fetched empty image for %s", img.Filename)
	}
	return data, nil
}

func cmpOr(v, def string) string {
	if v == "" {
		return def
	}
	return v
}
