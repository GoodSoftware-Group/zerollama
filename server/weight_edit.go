package server

import (
	"bytes"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"unicode"

	"github.com/gin-gonic/gin"

	"github.com/ollama/ollama/manifest"
	"github.com/ollama/ollama/types/model"
	"github.com/ollama/ollama/x/weightedit"
	_ "github.com/ollama/ollama/x/weightedit/gpatch"
)

// HTTP body cap for weight-edit patches (32 MiB decoded). Library/CLI stream without this cap.
const weightEditMaxBody = 32 << 20

// JSON envelopes carry base64 patch bytes (~4/3 expansion) plus model/destination fields.
const weightEditMaxJSONBody = weightEditMaxBody*4/3 + 64<<10

type weightEditApplyRequest struct {
	Model       string `json:"model"`
	Destination string `json:"destination"`
}

type weightEditInspectResponse struct {
	Format         string                    `json:"format"`
	Name           string                    `json:"name"`
	ID             string                    `json:"id"`
	ChangedWeights int                       `json:"changed_weights"`
	Tensors        []weightEditTensorSummary `json:"tensors"`
}

type weightEditTensorSummary struct {
	Name    string  `json:"name"`
	Dtype   string  `json:"dtype"`
	Shape   []int64 `json:"shape"`
	Changed int     `json:"changed"`
	Touched []int64 `json:"touched_rows,omitempty"`
}

type weightEditApplyResponse struct {
	Model             string `json:"model"`
	Destination       string `json:"destination"`
	Out               string `json:"out"`
	Format            string `json:"format"`
	ID                string `json:"id"`
	ChangedWeights    int    `json:"changed_weights"`
	OutputFingerprint string `json:"output_fingerprint,omitempty"`
}

func (s *Server) registerWeightEditRoutes(r *gin.Engine) {
	r.POST("/api/weights/edits/inspect", s.WeightEditInspectHandler)
	r.POST("/api/weights/edits", s.WeightEditApplyHandler)
	r.POST("/api/weights/edits/restore", s.WeightEditRestoreHandler)
}

func (s *Server) WeightEditInspectHandler(c *gin.Context) {
	data, err := readWeightEditBody(c)
	if err != nil {
		abortWeightEdit(c, err)
		return
	}
	edit, codec, err := weightedit.Decode(bytes.NewReader(data))
	if err != nil {
		abortWeightEdit(c, err)
		return
	}
	c.JSON(http.StatusOK, inspectResponse(edit, codec))
}

func (s *Server) WeightEditApplyHandler(c *gin.Context) {
	s.handleWeightEditWrite(c, "source")
}

func (s *Server) WeightEditRestoreHandler(c *gin.Context) {
	s.handleWeightEditWrite(c, "result")
}

func (s *Server) handleWeightEditWrite(c *gin.Context, side string) {
	req, patch, err := parseWeightEditWriteRequest(c)
	if err != nil {
		abortWeightEdit(c, err)
		return
	}
	srcModel, srcDir, err := resolveLinkedSourceDir(req.Model)
	if err != nil {
		abortWeightEdit(c, err)
		return
	}
	dstName, err := parseDestinationName(req.Destination)
	if err != nil {
		abortWeightEdit(c, err)
		return
	}
	edit, codec, err := weightedit.Decode(bytes.NewReader(patch))
	if err != nil {
		abortWeightEdit(c, err)
		return
	}
	if destExists(dstName) {
		abortWeightEdit(c, errConflict(fmt.Sprintf("destination %q already exists", req.Destination)))
		return
	}

	out := siblingOutDir(srcDir, dstName)
	if fi, err := os.Stat(out); err == nil {
		if !fi.IsDir() {
			abortWeightEdit(c, errConflict(fmt.Sprintf("%s already exists", out)))
			return
		}
		entries, err := os.ReadDir(out)
		if err != nil {
			abortWeightEdit(c, err)
			return
		}
		if len(entries) > 0 {
			abortWeightEdit(c, errConflict(fmt.Sprintf("%s already exists and is not empty", out)))
			return
		}
	} else if !os.IsNotExist(err) {
		abortWeightEdit(c, err)
		return
	}

	store, err := weightedit.OpenDirectory(srcDir)
	if err != nil {
		abortWeightEdit(c, err)
		return
	}
	bound, err := weightedit.Bind(store, edit, side)
	if err != nil {
		abortWeightEdit(c, err)
		return
	}
	res, err := weightedit.Apply(bound, out)
	if err != nil {
		abortWeightEdit(c, err)
		return
	}
	if err := registerLinkedDestination(srcModel, dstName, out); err != nil {
		_ = os.RemoveAll(out)
		abortWeightEdit(c, err)
		return
	}
	c.JSON(http.StatusOK, weightEditApplyResponse{
		Model:             req.Model,
		Destination:       dstName.String(),
		Out:               res.Out,
		Format:            codec.Format(),
		ID:                edit.ID,
		ChangedWeights:    res.ChangedWeights,
		OutputFingerprint: res.OutputFingerprint,
	})
}

func inspectResponse(edit *weightedit.Edit, codec weightedit.Codec) weightEditInspectResponse {
	tensors := make([]weightEditTensorSummary, 0, len(edit.Changes))
	for _, ch := range edit.Changes {
		changed := len(ch.Indices)
		if changed == 0 {
			changed = int(ch.Numel())
		}
		tensors = append(tensors, weightEditTensorSummary{
			Name:    ch.Name,
			Dtype:   ch.Dtype,
			Shape:   ch.Shape,
			Changed: changed,
			Touched: ch.TouchedRows(),
		})
	}
	format := edit.Format
	if format == "" && codec != nil {
		format = codec.Format()
	}
	return weightEditInspectResponse{
		Format:         format,
		Name:           edit.Name,
		ID:             edit.ID,
		ChangedWeights: edit.ChangedWeights(),
		Tensors:        tensors,
	}
}

func readWeightEditBody(c *gin.Context) ([]byte, error) {
	ct := c.ContentType()
	if strings.HasPrefix(ct, "multipart/") {
		file, err := c.FormFile("patch")
		if err != nil {
			file, err = c.FormFile("file")
		}
		if err != nil {
			return nil, errBadRequest("multipart field \"patch\" (or \"file\") required")
		}
		if file.Size > weightEditMaxBody {
			return nil, errTooLarge(fmt.Sprintf("patch exceeds %d byte limit", weightEditMaxBody))
		}
		f, err := file.Open()
		if err != nil {
			return nil, err
		}
		defer f.Close()
		data, err := io.ReadAll(io.LimitReader(f, weightEditMaxBody+1))
		if err != nil {
			return nil, err
		}
		if len(data) > weightEditMaxBody {
			return nil, errTooLarge(fmt.Sprintf("patch exceeds %d byte limit", weightEditMaxBody))
		}
		return data, nil
	}
	c.Request.Body = http.MaxBytesReader(c.Writer, c.Request.Body, weightEditMaxBody)
	data, err := io.ReadAll(c.Request.Body)
	if err != nil {
		var maxErr *http.MaxBytesError
		if errors.As(err, &maxErr) {
			return nil, errTooLarge(fmt.Sprintf("patch exceeds %d byte limit", weightEditMaxBody))
		}
		return nil, err
	}
	if len(data) == 0 {
		return nil, errBadRequest("empty patch body")
	}
	return data, nil
}

func parseWeightEditWriteRequest(c *gin.Context) (weightEditApplyRequest, []byte, error) {
	ct := c.ContentType()
	if strings.HasPrefix(ct, "multipart/") {
		var req weightEditApplyRequest
		req.Model = c.PostForm("model")
		req.Destination = c.PostForm("destination")
		if req.Model == "" || req.Destination == "" {
			return req, nil, errBadRequest("model and destination are required")
		}
		data, err := readWeightEditBody(c)
		return req, data, err
	}

	// JSON envelope with base64 is not used; prefer raw body + query, or
	// application/json with model/destination and a following convention:
	// For JSON requests, fields are in the body and the patch is base64 under "patch".
	if strings.Contains(ct, "application/json") {
		var envelope struct {
			Model       string `json:"model"`
			Destination string `json:"destination"`
			Patch       []byte `json:"patch"`
		}
		c.Request.Body = http.MaxBytesReader(c.Writer, c.Request.Body, weightEditMaxJSONBody)
		if err := c.ShouldBindJSON(&envelope); err != nil {
			var maxErr *http.MaxBytesError
			if errors.As(err, &maxErr) {
				return weightEditApplyRequest{}, nil, errTooLarge(fmt.Sprintf("patch exceeds %d byte limit", weightEditMaxBody))
			}
			return weightEditApplyRequest{}, nil, errBadRequest(err.Error())
		}
		if envelope.Model == "" || envelope.Destination == "" {
			return weightEditApplyRequest{}, nil, errBadRequest("model and destination are required")
		}
		if len(envelope.Patch) == 0 {
			return weightEditApplyRequest{}, nil, errBadRequest("patch bytes required")
		}
		if len(envelope.Patch) > weightEditMaxBody {
			return weightEditApplyRequest{}, nil, errTooLarge(fmt.Sprintf("patch exceeds %d byte limit", weightEditMaxBody))
		}
		return weightEditApplyRequest{Model: envelope.Model, Destination: envelope.Destination}, envelope.Patch, nil
	}

	req := weightEditApplyRequest{
		Model:       c.Query("model"),
		Destination: c.Query("destination"),
	}
	if req.Model == "" || req.Destination == "" {
		return req, nil, errBadRequest("model and destination query parameters required for raw patch bodies")
	}
	data, err := readWeightEditBody(c)
	return req, data, err
}

func resolveLinkedSourceDir(modelName string) (*Model, string, error) {
	m, err := GetModel(modelName)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, "", errBadRequest(fmt.Sprintf("model %q not found", modelName))
		}
		return nil, "", err
	}
	if m.Config.RemoteHost != "" {
		return nil, "", errBadRequest(fmt.Sprintf("model %q is remote", modelName))
	}
	if m.Config.ModelFormat != "safetensors" {
		return nil, "", errBadRequest(fmt.Sprintf("model %q is not a safetensors checkpoint (format=%q)", modelName, m.Config.ModelFormat))
	}
	src := strings.TrimSpace(m.Config.SourceDir)
	if src == "" {
		return nil, "", errBadRequest(fmt.Sprintf("model %q has no source_dir (only linked safetensors directories can receive weight edits)", modelName))
	}
	if fi, err := os.Stat(src); err != nil || !fi.IsDir() {
		return nil, "", errBadRequest(fmt.Sprintf("model %q source_dir %q is missing", modelName, src))
	}
	return m, src, nil
}

func parseDestinationName(dest string) (model.Name, error) {
	n := model.ParseName(dest)
	if !n.IsValid() {
		return model.Name{}, errBadRequest(fmt.Sprintf("destination %q is invalid", dest))
	}
	n, err := getExistingName(n)
	if err != nil {
		return model.Name{}, err
	}
	return n, nil
}

func destExists(n model.Name) bool {
	_, err := manifest.ParseNamedManifest(n)
	return err == nil
}

func siblingOutDir(srcDir string, dst model.Name) string {
	base := dst.DisplayShortest()
	if base == "" {
		base = dst.String()
	}
	var b strings.Builder
	for _, r := range base {
		switch {
		case r == '/' || r == ':' || r == ' ':
			b.WriteByte('-')
		case unicode.IsLetter(r) || unicode.IsDigit(r) || r == '-' || r == '_' || r == '.':
			b.WriteRune(r)
		default:
			b.WriteByte('-')
		}
	}
	name := b.String()
	if name == "" {
		name = "edited"
	}
	return filepath.Join(filepath.Dir(srcDir), name)
}

func registerLinkedDestination(src *Model, dst model.Name, outDir string) error {
	cfg := src.Config
	abs, err := filepath.Abs(outDir)
	if err != nil {
		return err
	}
	cfg.SourceDir = abs
	if cfg.FileType == "" {
		cfg.FileType = "linked"
	}
	cfg.ModelFormat = "safetensors"

	configLayer, err := createConfigLayer(cfg)
	if err != nil {
		return err
	}

	mf, err := manifest.ParseNamedManifest(model.ParseName(src.Name))
	if err != nil {
		return err
	}
	layers := append([]manifest.Layer(nil), mf.Layers...)
	return manifest.WriteManifest(dst, *configLayer, layers)
}

type weightEditHTTPError struct {
	status int
	msg    string
}

func (e *weightEditHTTPError) Error() string { return e.msg }

func errBadRequest(msg string) error {
	return &weightEditHTTPError{status: http.StatusBadRequest, msg: msg}
}
func errConflict(msg string) error {
	return &weightEditHTTPError{status: http.StatusConflict, msg: msg}
}
func errTooLarge(msg string) error {
	return &weightEditHTTPError{status: http.StatusRequestEntityTooLarge, msg: msg}
}

func abortWeightEdit(c *gin.Context, err error) {
	var he *weightEditHTTPError
	if errors.As(err, &he) {
		c.AbortWithStatusJSON(he.status, gin.H{"error": he.msg})
		return
	}
	if errors.Is(err, weightedit.ErrUnknownFormat) {
		c.AbortWithStatusJSON(http.StatusUnsupportedMediaType, gin.H{"error": err.Error()})
		return
	}
	if errors.Is(err, weightedit.ErrIncompatible) {
		c.AbortWithStatusJSON(http.StatusUnprocessableEntity, gin.H{"error": err.Error()})
		return
	}
	if errors.Is(err, weightedit.ErrInvalidPatch) {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}
	c.AbortWithStatusJSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
}
