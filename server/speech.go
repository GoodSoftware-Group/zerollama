package server

import (
	"errors"
	"fmt"
	"io/fs"
	"log/slog"
	"net/http"
	"os"

	"github.com/gin-gonic/gin"

	"github.com/ollama/ollama/middleware"
	"github.com/ollama/ollama/openai"
	"github.com/ollama/ollama/server/modality"
	"github.com/ollama/ollama/types/errtypes"
	"github.com/ollama/ollama/types/model"
)

// SpeechHandler serves POST /v1/audio/speech for Piper-backed models (modality_backends.speech=piper).
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
		c.JSON(http.StatusBadRequest, gin.H{"error": fmt.Sprintf("%q does not support speech (set modality_backends.speech=piper and backend_paths.piper_model, or capability speech in config)", req.Model)})
		return
	}

	backend := modality.BackendFor(m.Config, model.ModalitySpeech)
	if backend == "" && modality.PathFor(m.Config, "piper_model") != "" {
		backend = model.BackendPiper
	}
	if backend != model.BackendPiper {
		c.JSON(http.StatusBadRequest, gin.H{"error": "speech backend must be piper (set modality_backends.speech=piper and backend_paths.piper_model)"})
		return
	}

	data, contentType, err := modality.SpeechPiper(c.Request.Context(), m.Config, req.Input, req.Voice, req.Speed)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}
	if req.ResponseFormat != "" && req.ResponseFormat != "wav" {
		slog.Debug("piper returns WAV; response_format ignored", "format", req.ResponseFormat)
	}
	c.Header("Content-Disposition", `attachment; filename="speech.wav"`)
	c.Data(http.StatusOK, contentType, data)
}
