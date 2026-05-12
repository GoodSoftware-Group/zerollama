package server

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"net/http"
	"os"

	"github.com/gin-gonic/gin"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/middleware"
	"github.com/ollama/ollama/openai"
	"github.com/ollama/ollama/server/modality"
	"github.com/ollama/ollama/types/errtypes"
	"github.com/ollama/ollama/types/model"
)

// TranscriptionHandler serves POST /v1/audio/transcriptions: optional Whisper subprocess
// when modality_backends.transcribe=whisper; otherwise delegates to ChatHandler (multimodal audio models).
func (s *Server) TranscriptionHandler(c *gin.Context) {
	body, err := io.ReadAll(c.Request.Body)
	if err != nil {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}
	c.Request.Body = io.NopCloser(bytes.NewReader(body))

	var chatReq api.ChatRequest
	if err := json.Unmarshal(body, &chatReq); err != nil {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	modelRef, err := parseAndValidateModelRef(chatReq.Model)
	if err != nil {
		writeModelRefParseError(c, err, http.StatusBadRequest, "model is required")
		return
	}
	if modelRef.Source == modelSourceCloud {
		c.Request.Body = io.NopCloser(bytes.NewReader(body))
		s.ChatHandler(c)
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
			c.JSON(http.StatusNotFound, gin.H{"error": fmt.Sprintf("model '%s' not found", chatReq.Model)})
		case err.Error() == errtypes.InvalidModelNameErrMsg:
			c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		default:
			c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		}
		return
	}

	if m.Config.RemoteHost != "" && m.Config.RemoteModel != "" {
		c.Request.Body = io.NopCloser(bytes.NewReader(body))
		s.ChatHandler(c)
		return
	}

	if modality.BackendFor(m.Config, model.ModalityTranscribe) == model.BackendWhisper {
		audio, err := audioBytesFromChat(&chatReq)
		if err != nil {
			c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": err.Error()})
			return
		}
		orig, _ := c.Get(middleware.CtxKeyTranscriptionOriginalFilename)
		origName, _ := orig.(string)
		langVal, _ := c.Get(middleware.CtxKeyTranscriptionLanguage)
		language, _ := langVal.(string)

		text, err := modality.TranscribeWhisper(c.Request.Context(), m.Config, audio, origName, language)
		if err != nil {
			c.AbortWithStatusJSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
			return
		}
		rf, _ := c.Get(middleware.CtxKeyTranscriptionResponseFormat)
		format, _ := rf.(string)
		contentType, body, err := openai.TranscriptionResult(text, format, language)
		if err != nil {
			c.AbortWithStatusJSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
			return
		}
		c.Data(http.StatusOK, contentType, body)
		return
	}

	c.Request.Body = io.NopCloser(bytes.NewReader(body))
	s.ChatHandler(c)
}

func audioBytesFromChat(req *api.ChatRequest) ([]byte, error) {
	for i := len(req.Messages) - 1; i >= 0; i-- {
		if req.Messages[i].Role == "user" && len(req.Messages[i].Images) > 0 {
			return append([]byte(nil), req.Messages[i].Images[0]...), nil
		}
	}
	return nil, errors.New("no audio attachment in transcription request")
}
