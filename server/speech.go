package server

import (
	"errors"
	"fmt"
	"io/fs"
	"log/slog"
	"net/http"
	"os"
	"strings"

	"github.com/gin-gonic/gin"

	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/manifest"
	"github.com/ollama/ollama/middleware"
	"github.com/ollama/ollama/openai"
	"github.com/ollama/ollama/server/modality"
	"github.com/ollama/ollama/types/errtypes"
	"github.com/ollama/ollama/types/model"
)

// SpeechHandler serves POST /v1/audio/speech for Piper and remote-tts speech models.
func (s *Server) SpeechHandler(c *gin.Context) {
	v, ok := c.Get(middleware.CtxKeySpeechRequest)
	if !ok {
		c.AbortWithStatusJSON(http.StatusInternalServerError, gin.H{"error": "missing speech request"})
		return
	}
	req := v.(openai.SpeechCreateRequest)

	modelRef, err := parseAndValidateModelRef(req.Model)
	if err != nil {
		writeModelRefParseError(c, err, http.StatusBadRequest, "model is required")
		return
	}
	if modelRef.Source == modelSourceCloud {
		c.JSON(http.StatusBadRequest, gin.H{"error": "cloud speech is not supported on this endpoint"})
		return
	}

	name := modelRef.Name
	name, err = getExistingName(name)
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "model is required"})
		return
	}

	m, err := GetModel(name.String())
	if err != nil {
		switch {
		case errors.Is(err, fs.ErrNotExist), os.IsNotExist(err):
			c.JSON(http.StatusNotFound, gin.H{"error": fmt.Sprintf("model '%s' not found", req.Model)})
		case err.Error() == errtypes.InvalidModelNameErrMsg:
			c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		default:
			c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		}
		return
	}

	if m.Config.RemoteHost != "" && m.Config.RemoteModel != "" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "remote models are not supported on this endpoint"})
		return
	}

	if err := m.CheckCapabilities(model.CapabilitySpeech); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": fmt.Sprintf("%q does not support speech (set modality_backends.speech=piper|remote-tts and capability speech)", req.Model)})
		return
	}

	backend := modality.BackendFor(m.Config, model.ModalitySpeech)
	if backend == "" && modality.PathFor(m.Config, "piper_model") != "" {
		backend = model.BackendPiper
	}
	if backend == "" && (modality.PathFor(m.Config, "tts_url") != "" || envconfig.TTSURL() != "") {
		backend = model.BackendRemoteTTS
	}

	var data []byte
	var contentType string
	switch backend {
	case model.BackendPiper:
		data, contentType, err = modality.SpeechPiper(c.Request.Context(), m.Config, req.Input, req.Voice, req.Speed)
	case model.BackendRemoteTTS:
		data, contentType, err = modality.SpeechRemote(c.Request.Context(), m.Config, req.Model, req.Input, req.Voice, req.ResponseFormat, req.Emotion, req.Speed)
	default:
		c.JSON(http.StatusBadRequest, gin.H{"error": fmt.Sprintf("unsupported speech backend %q (want piper or remote-tts)", backend)})
		return
	}
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}
	if req.ResponseFormat != "" && req.ResponseFormat != "wav" && !strings.Contains(contentType, req.ResponseFormat) {
		slog.Debug("speech response_format may differ from upstream content-type", "format", req.ResponseFormat, "content_type", contentType)
	}
	ext := "wav"
	if strings.Contains(contentType, "mpeg") || strings.Contains(contentType, "mp3") {
		ext = "mp3"
	} else if strings.Contains(contentType, "ogg") || strings.Contains(contentType, "opus") {
		ext = "ogg"
	}
	c.Header("Content-Disposition", fmt.Sprintf(`attachment; filename="speech.%s"`, ext))
	c.Data(http.StatusOK, contentType, data)
}

// VoicesHandler serves GET /v1/audio/voices — lists selectable voice ids for speech models.
// Query model= optional; when set, returns voices for that tag only.
func (s *Server) VoicesHandler(c *gin.Context) {
	q := strings.TrimSpace(c.Query("model"))
	type modelVoices struct {
		Model   string                 `json:"model"`
		Backend string                 `json:"backend,omitempty"`
		Voices  []modality.SpeechVoice `json:"voices"`
	}
	var models []modelVoices

	if q != "" {
		modelRef, err := parseAndValidateModelRef(q)
		if err != nil {
			c.JSON(http.StatusBadRequest, gin.H{"error": "model is required"})
			return
		}
		name, err := getExistingName(modelRef.Name)
		if err != nil {
			c.JSON(http.StatusNotFound, gin.H{"error": fmt.Sprintf("model '%s' not found", q)})
			return
		}
		m, err := GetModel(name.String())
		if err != nil {
			c.JSON(http.StatusNotFound, gin.H{"error": fmt.Sprintf("model '%s' not found", q)})
			return
		}
		if err := m.CheckCapabilities(model.CapabilitySpeech); err != nil {
			c.JSON(http.StatusBadRequest, gin.H{"error": fmt.Sprintf("%q is not a speech model", q)})
			return
		}
		backend := modality.BackendFor(m.Config, model.ModalitySpeech)
		models = append(models, modelVoices{
			Model:   name.DisplayShortest(),
			Backend: backend,
			Voices:  modality.ListSpeechVoices(m.Config),
		})
		c.JSON(http.StatusOK, gin.H{"object": "list", "data": models})
		return
	}

	ms, err := manifest.Manifests(true)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}
	for name := range ms {
		m, err := GetModel(name.String())
		if err != nil {
			continue
		}
		if err := m.CheckCapabilities(model.CapabilitySpeech); err != nil {
			continue
		}
		backend := modality.BackendFor(m.Config, model.ModalitySpeech)
		models = append(models, modelVoices{
			Model:   name.DisplayShortest(),
			Backend: backend,
			Voices:  modality.ListSpeechVoices(m.Config),
		})
	}
	c.JSON(http.StatusOK, gin.H{"object": "list", "data": models})
}
