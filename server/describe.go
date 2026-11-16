package server

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"github.com/gin-gonic/gin"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/describe"
	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/types/model"
)

// describeBackendForTest overrides the live embed/chat/detect backend in tests.
var describeBackendForTest describe.Backend

const maxDescribeBody = 24 << 20

// DescribeHandler serves POST /api/describe. The caller keeps the catalog; this
// route never opens a filesystem path.
func (s *Server) DescribeHandler(c *gin.Context) {
	body, err := io.ReadAll(io.LimitReader(c.Request.Body, maxDescribeBody+1))
	if err != nil {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": "read body: " + err.Error()})
		return
	}
	if len(body) > maxDescribeBody {
		c.AbortWithStatusJSON(http.StatusRequestEntityTooLarge, gin.H{"error": "describe body exceeds 24MB"})
		return
	}
	if len(bytes.TrimSpace(body)) == 0 {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": "missing request body"})
		return
	}

	var wire describeWireRequest
	if err := json.Unmarshal(body, &wire); err != nil {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}
	req, err := wire.toRequest()
	if err != nil {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	want := describe.ParseFields(req.Fields)
	if req.WantObjects {
		want.Objects = true
	}
	if want.Objects && strings.TrimSpace(envconfig.DetectURL()) == "" && describeBackendForTest == nil {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{
			"error": "objects requested but ZEROLLAMA_DETECT_URL is unset — see docs/detect.md",
		})
		return
	}

	backend := describeBackendForTest
	if backend == nil {
		backend = &serverDescribeBackend{s: s}
	}
	resp, err := describe.ProcessRequest(c.Request.Context(), req, backend)
	if err != nil {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}
	c.JSON(http.StatusOK, resp)
}

type describeWireRequest struct {
	Fields       []string            `json:"fields,omitempty"`
	Labels       []string            `json:"labels,omitempty"`
	MinScore     float64             `json:"min_score,omitempty"`
	TagModel     string              `json:"tag_model,omitempty"`
	CaptionModel string              `json:"caption_model,omitempty"`
	WantObjects  bool                `json:"objects,omitempty"`
	Catalog      []describe.Entry    `json:"catalog,omitempty"`
	Images       []describeWireImage `json:"images"`
}

type describeWireImage struct {
	ID     string `json:"id"`
	SHA256 string `json:"sha256,omitempty"`
	Data   string `json:"data,omitempty"`
}

func (w describeWireRequest) toRequest() (describe.Request, error) {
	req := describe.Request{
		Fields:       w.Fields,
		Labels:       w.Labels,
		MinScore:     w.MinScore,
		TagModel:     w.TagModel,
		CaptionModel: w.CaptionModel,
		WantObjects:  w.WantObjects,
		Catalog:      w.Catalog,
	}
	for _, img := range w.Images {
		in := describe.ImageInput{ID: img.ID, SHA256: img.SHA256}
		if strings.TrimSpace(img.Data) != "" {
			raw, err := base64.StdEncoding.DecodeString(img.Data)
			if err != nil {
				raw, err = base64.RawStdEncoding.DecodeString(img.Data)
				if err != nil {
					return req, fmt.Errorf("image %q: invalid base64 data", img.ID)
				}
			}
			in.Data = raw
		}
		req.Images = append(req.Images, in)
	}
	return req, nil
}

type serverDescribeBackend struct {
	s *Server
}

func (b *serverDescribeBackend) EmbedTexts(ctx context.Context, modelName string, texts []string) ([][]float32, error) {
	r, release, err := b.scheduleEmbed(ctx, modelName)
	if err != nil {
		return nil, err
	}
	defer release()
	out := make([][]float32, len(texts))
	for i, text := range texts {
		emb, _, err := r.Embedding(ctx, text)
		if err != nil {
			return nil, err
		}
		emb, err = normalize(emb)
		if err != nil {
			return nil, err
		}
		out[i] = emb
	}
	return out, nil
}

func (b *serverDescribeBackend) EmbedImage(ctx context.Context, modelName string, data []byte) ([]float32, error) {
	r, release, err := b.scheduleEmbed(ctx, modelName)
	if err != nil {
		return nil, err
	}
	defer release()
	me, ok := r.(interface {
		EmbedWithMedia(ctx context.Context, input string, media [][]byte) ([]float32, int, error)
	})
	if !ok {
		return nil, api.StatusError{
			StatusCode:   http.StatusNotImplemented,
			ErrorMessage: "model does not support media embeddings",
		}
	}
	emb, _, err := me.EmbedWithMedia(ctx, "", [][]byte{data})
	if err != nil {
		return nil, err
	}
	return normalize(emb)
}

func (b *serverDescribeBackend) Caption(ctx context.Context, modelName string, data []byte) (string, []string, string, error) {
	name, err := getExistingName(model.ParseName(modelName))
	if err != nil {
		return "", nil, "", fmt.Errorf("model '%s' not found", modelName)
	}
	m, err := GetModel(name.String())
	if err != nil {
		return "", nil, "", fmt.Errorf("model '%s' not found", modelName)
	}

	think := &api.ThinkValue{Value: false}
	optsMap := map[string]any{"num_predict": 128}
	r, modelRef, opts, _, releaseQoS, err := b.s.scheduleRunner(ctx, name.String(), []model.Capability{model.CapabilityCompletion}, optsMap, nil, nil, nil, nil)
	if err != nil {
		return "", nil, "", err
	}
	defer releaseQoS()
	if modelRef != nil {
		m = modelRef
	}

	msgs := []api.Message{{
		Role:    "user",
		Content: describe.CaptionPrompt,
		Images:  []api.ImageData{data},
	}}
	if m.System != "" {
		msgs = append([]api.Message{{Role: "system", Content: m.System}}, msgs...)
	}
	msgs = append(append([]api.Message{}, m.Messages...), msgs...)

	prompt, images, _, _, _, err := chatPrompt(ctx, m, r.Tokenize, opts, msgs, nil, think, false, 0, r.Detokenize)
	if err != nil {
		return "", nil, "", err
	}

	format, _ := json.Marshal("json")
	var buf strings.Builder
	err = r.Completion(ctx, llm.CompletionRequest{
		Prompt:  prompt,
		Format:  format,
		Images:  images,
		Options: opts,
	}, func(cr llm.CompletionResponse) {
		buf.WriteString(cr.Content)
	})
	if err != nil {
		return "", nil, "", err
	}
	raw := strings.TrimSpace(buf.String())
	if caption, text, ok := describe.ParseCaptionJSON(raw); ok {
		return caption, text, raw, nil
	}
	return "", nil, raw, nil
}

func (b *serverDescribeBackend) Detect(ctx context.Context, data []byte, queries []string) (*describe.ObjectsResult, error) {
	base := strings.TrimSpace(envconfig.DetectURL())
	body, err := json.Marshal(map[string]any{
		"image":     base64.StdEncoding.EncodeToString(data),
		"queries":   queries,
		"threshold": 0.1,
	})
	if err != nil {
		return nil, err
	}
	if base == "" {
		return &describe.ObjectsResult{
			Status:     "wire_only",
			Detections: []describe.Detection{},
			Reason:     "ZEROLLAMA_DETECT_URL is unset",
		}, nil
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, base+"/v1/detect", bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := perceptionClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("detect upstream: %w", err)
	}
	defer resp.Body.Close()
	out, err := io.ReadAll(io.LimitReader(resp.Body, maxPerceptionBody))
	if err != nil {
		return nil, err
	}
	if resp.StatusCode >= 300 {
		return nil, fmt.Errorf("detect status %d: %s", resp.StatusCode, strings.TrimSpace(string(out)))
	}
	var parsed struct {
		Status     string `json:"status"`
		Reason     string `json:"reason"`
		Detections []struct {
			Label string    `json:"label"`
			Score float64   `json:"score"`
			Box   []float64 `json:"box"`
		} `json:"detections"`
	}
	if err := json.Unmarshal(out, &parsed); err != nil {
		return nil, err
	}
	res := &describe.ObjectsResult{
		Status:     parsed.Status,
		Reason:     parsed.Reason,
		Detections: make([]describe.Detection, 0, len(parsed.Detections)),
	}
	if res.Status == "" {
		res.Status = "ok"
	}
	for _, d := range parsed.Detections {
		res.Detections = append(res.Detections, describe.Detection{
			Label: d.Label,
			Score: d.Score,
			Box:   d.Box,
		})
	}
	return res, nil
}

func (b *serverDescribeBackend) scheduleEmbed(ctx context.Context, modelName string) (llm.LlamaServer, func(), error) {
	name, err := getExistingName(model.ParseName(modelName))
	if err != nil {
		return nil, nil, fmt.Errorf("model '%s' not found", modelName)
	}
	// Match EmbedHandler: do not require CapabilityEmbedding. SigLIP MLX
	// manifests may only declare what create wrote; empty caps still load.
	keep := &api.Duration{Duration: 5 * time.Minute}
	r, _, _, _, releaseQoS, err := b.s.scheduleRunner(ctx, name.String(), []model.Capability{}, nil, keep, nil, nil, nil)
	if err != nil {
		return nil, nil, err
	}
	return r, releaseQoS, nil
}
