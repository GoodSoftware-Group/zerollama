package describe

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"image"
	_ "image/gif"
	_ "image/jpeg"
	_ "image/png"
	"strings"
	"time"

	_ "golang.org/x/image/webp"
)

// Backend runs the model steps for one describe request.
type Backend interface {
	EmbedTexts(ctx context.Context, model string, texts []string) ([][]float32, error)
	EmbedImage(ctx context.Context, model string, data []byte) ([]float32, error)
	Caption(ctx context.Context, model string, data []byte) (caption string, visible []string, raw string, err error)
	Detect(ctx context.Context, data []byte, queries []string) (*ObjectsResult, error)
}

// ProcessRequest runs keep/needs_image/process for every image in the request.
// Work is phased: all tags, then all captions, then all object detects, so two
// models are not resident for the same picture at once.
func ProcessRequest(ctx context.Context, req Request, backend Backend) (*Response, error) {
	if len(req.Images) == 0 {
		return &Response{Images: []ResultEntry{}}, nil
	}
	withBytes := 0
	for _, img := range req.Images {
		if len(img.Data) > 0 {
			withBytes++
		}
	}
	if withBytes > MaxImagesPerCall {
		return nil, fmt.Errorf("at most %d images with data per call (got %d)", MaxImagesPerCall, withBytes)
	}

	want := ParseFields(req.Fields)
	if req.WantObjects {
		want.Objects = true
	}
	labels, phrases := ResolveLabels(req.Labels)
	minScore := req.MinScore
	if minScore <= 0 {
		minScore = DefaultMinScore
	}
	settings := Settings{
		TagModel:     strings.TrimSpace(req.TagModel),
		CaptionModel: strings.TrimSpace(req.CaptionModel),
		Labels:       labels,
		MinScore:     minScore,
	}
	if want.Tags && settings.TagModel == "" {
		return nil, fmt.Errorf("tag_model is required when tags are requested")
	}
	if want.Caption && settings.CaptionModel == "" {
		return nil, fmt.Errorf("caption_model is required when caption is requested")
	}

	index := IndexByID(req.Catalog)
	type item struct {
		img      ImageInput
		sha      string
		existing *Entry
		decision Decision
		entry    Entry
		action   string
		work     bool
	}
	items := make([]item, 0, len(req.Images))
	needTagEmbed := false
	for _, img := range req.Images {
		it := item{img: img}
		if strings.TrimSpace(img.ID) == "" {
			it.entry = Entry{Error: "id is required"}
			it.action = ActionError
			items = append(items, it)
			continue
		}
		it.existing = index[img.ID]
		it.sha = img.SHA256
		if len(img.Data) > 0 {
			it.sha = HashBytes(img.Data)
		}
		it.decision = Decide(it.existing, it.sha, len(img.Data) > 0, want, settings)
		switch it.decision {
		case DecisionNeedsImg:
			e := Entry{ID: img.ID, SHA256: it.sha}
			if it.existing != nil {
				e = *it.existing
			}
			it.entry = e
			it.action = ActionNeedsImage
		case DecisionKept:
			it.entry = *it.existing
			it.action = ActionKept
		default:
			it.work = true
			it.action = ActionProcessed
			it.entry = Entry{ID: img.ID, SHA256: it.sha}
			if it.existing != nil {
				it.entry = *it.existing
				it.entry.ID = img.ID
				it.entry.SHA256 = it.sha
				it.entry.Error = ""
			}
			it.entry.File = fileFacts(img.Data, it.existing)
			needTagEmbed = needTagEmbed || want.Tags
		}
		items = append(items, it)
	}

	var labelVecs [][]float32
	if needTagEmbed && backend != nil {
		vecs, err := backend.EmbedTexts(ctx, settings.TagModel, phrases)
		if err != nil {
			return nil, fmt.Errorf("embed labels: %w", err)
		}
		labelVecs = vecs
	}

	if want.Tags && backend != nil {
		for i := range items {
			if !items[i].work || items[i].action == ActionError {
				continue
			}
			vec, err := backend.EmbedImage(ctx, settings.TagModel, items[i].img.Data)
			if err != nil {
				items[i].entry.Error = err.Error()
				items[i].action = ActionError
				items[i].work = false
				continue
			}
			tags := KeepTags(labels, labelVecs, vec, settings.MinScore)
			items[i].entry.SetTags(tags)
			items[i].entry.TagModel = settings.TagModel
			items[i].entry.Labels = append([]string(nil), labels...)
			items[i].entry.MinScore = settings.MinScore
		}
	}

	if want.Caption && backend != nil {
		for i := range items {
			if !items[i].work || items[i].action == ActionError {
				continue
			}
			caption, visible, raw, err := backend.Caption(ctx, settings.CaptionModel, items[i].img.Data)
			if err != nil {
				items[i].entry.Error = err.Error()
				items[i].action = ActionError
				items[i].work = false
				continue
			}
			if caption != "" {
				items[i].entry.Caption = caption
				items[i].entry.CaptionError = ""
				items[i].entry.Text = visible
			} else {
				items[i].entry.Caption = raw
				items[i].entry.CaptionError = "model did not return JSON caption"
				items[i].entry.Text = nil
			}
			items[i].entry.CaptionModel = settings.CaptionModel
		}
	}

	if want.Objects && backend != nil {
		for i := range items {
			if !items[i].work || items[i].action == ActionError {
				continue
			}
			obj, err := backend.Detect(ctx, items[i].img.Data, labels)
			if err != nil {
				items[i].entry.Error = err.Error()
				items[i].action = ActionError
				items[i].work = false
				continue
			}
			if obj == nil || obj.Status != "ok" {
				reason := "detect did not run"
				if obj != nil && obj.Reason != "" {
					reason = obj.Reason
				} else if obj != nil && obj.Status != "" {
					reason = "detect status=" + obj.Status
				}
				items[i].entry.Error = reason
				items[i].entry.Objects = nil
				items[i].action = ActionError
				items[i].work = false
				continue
			}
			items[i].entry.Objects = obj
			// Object queries share the label list; store it so a later --labels change redoes boxes.
			if len(items[i].entry.Labels) == 0 {
				items[i].entry.Labels = append([]string(nil), labels...)
			}
		}
	}

	out := make([]ResultEntry, 0, len(items))
	for _, it := range items {
		out = append(out, ResultEntry{Entry: it.entry, Action: it.action})
	}
	return &Response{Images: out}, nil
}

func fileFacts(data []byte, existing *Entry) *FileFacts {
	ff := &FileFacts{Bytes: int64(len(data))}
	if existing != nil && existing.File != nil && !existing.File.Mtime.IsZero() {
		ff.Mtime = existing.File.Mtime
	}
	if len(data) == 0 {
		return ff
	}
	cfg, _, err := image.DecodeConfig(bytes.NewReader(data))
	if err == nil {
		ff.Width = cfg.Width
		ff.Height = cfg.Height
	}
	if ff.Mtime.IsZero() {
		ff.Mtime = time.Now().UTC()
	}
	return ff
}

// ParseCaptionJSON extracts caption and text from a model JSON object.
func ParseCaptionJSON(raw string) (caption string, text []string, ok bool) {
	raw = strings.TrimSpace(raw)
	if raw == "" {
		return "", nil, false
	}
	var obj struct {
		Caption string   `json:"caption"`
		Text    []string `json:"text"`
	}
	if err := json.Unmarshal([]byte(raw), &obj); err != nil {
		return "", nil, false
	}
	if obj.Caption == "" {
		return "", nil, false
	}
	if obj.Text == nil {
		obj.Text = []string{}
	}
	return obj.Caption, obj.Text, true
}

// CaptionPrompt is the user message for vision captioning.
const CaptionPrompt = `Describe this image. Reply with one JSON object only: {"caption":"<one sentence>","text":["words visible in the image"]}. If no text is visible, use "text":[].`
