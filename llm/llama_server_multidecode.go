package llm

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"

	"github.com/ollama/ollama/api"
)

// MultiDecode implements MultiDecoder against llama-server POST /v1/multidecode.
func (s *llamaServerRunner) MultiDecode(ctx context.Context, req MultiDecodeRequest) (MultiDecodeResponse, error) {
	if len(req.Tokens) == 0 {
		return MultiDecodeResponse{}, fmt.Errorf("multidecode: tokens required")
	}
	if len(req.Leaves) == 0 {
		return MultiDecodeResponse{}, fmt.Errorf("multidecode: leaves required")
	}
	if err := s.sem.Acquire(ctx, 1); err != nil {
		return MultiDecodeResponse{}, err
	}
	defer s.sem.Release(1)

	status, err := s.getServerStatusRetry(ctx)
	if err != nil {
		return MultiDecodeResponse{}, err
	}
	if status != ServerStatusReady {
		return MultiDecodeResponse{}, fmt.Errorf("unexpected server status: %s", status)
	}

	body := map[string]any{
		"tokens": req.Tokens,
		"pos":    req.Pos,
		"leaves": req.Leaves,
	}
	if len(req.Parent) > 0 {
		body["parent"] = req.Parent
	}
	if len(req.NodeIDs) > 0 {
		body["node_ids"] = req.NodeIDs
	}
	if len(req.ParentNodeIDs) > 0 {
		body["parent_node_ids"] = req.ParentNodeIDs
	}
	if req.ReturnLogits {
		body["return_logits"] = true
	}
	if req.Clear != nil {
		body["clear"] = *req.Clear
	}
	if req.NPredict > 0 {
		body["n_predict"] = req.NPredict
	}
	data, err := json.Marshal(body)
	if err != nil {
		return MultiDecodeResponse{}, err
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, fmt.Sprintf("http://127.0.0.1:%d/v1/multidecode", s.port), bytes.NewReader(data))
	if err != nil {
		return MultiDecodeResponse{}, err
	}
	httpReq.Header.Set("Content-Type", "application/json")

	resp, err := s.httpClient().Do(httpReq)
	if err != nil {
		return MultiDecodeResponse{}, err
	}
	defer resp.Body.Close()
	raw, err := io.ReadAll(resp.Body)
	if err != nil {
		return MultiDecodeResponse{}, err
	}
	if resp.StatusCode >= 400 {
		return MultiDecodeResponse{}, api.StatusError{StatusCode: resp.StatusCode, ErrorMessage: string(raw)}
	}

	var out MultiDecodeResponse
	if err := json.Unmarshal(raw, &out); err != nil {
		return MultiDecodeResponse{}, fmt.Errorf("decode multidecode: %w", err)
	}
	return out, nil
}
