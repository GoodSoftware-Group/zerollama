package server

import (
	"errors"
	"fmt"
	"io"
	"io/fs"
	"net/http"
	"strings"

	"github.com/gin-gonic/gin"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/types/errtypes"
	"github.com/ollama/ollama/types/model"
)

type namedModelBody struct {
	Model   string `json:"model"`
	ModelID string `json:"model_id"`
	ID      string `json:"id"`
}

func wrongModelKeyError(b namedModelBody) string {
	if strings.TrimSpace(b.Model) != "" {
		return ""
	}
	var got []string
	if strings.TrimSpace(b.ModelID) != "" {
		got = append(got, `"model_id"`)
	}
	if strings.TrimSpace(b.ID) != "" {
		got = append(got, `"id"`)
	}
	if len(got) == 0 {
		return ""
	}
	return fmt.Sprintf("use %q (not %s)", "model", strings.Join(got, " or "))
}

type loadJSON struct {
	namedModelBody
	KeepAlive *api.Duration  `json:"keep_alive,omitempty"`
	Options   map[string]any `json:"options,omitempty"`
}

// LoadHandler implements POST /api/load (LA21): wait until the model is resident
// without a generate/chat round-trip. Inverse of keep_alive:0 unload.
func (s *Server) LoadHandler(c *gin.Context) {
	var req loadJSON
	if err := c.ShouldBindJSON(&req); errors.Is(err, io.EOF) {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": "missing request body"})
		return
	} else if err != nil {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}
	if msg := wrongModelKeyError(req.namedModelBody); msg != "" {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": msg})
		return
	}
	req.Model = strings.TrimSpace(req.Model)
	if req.Model == "" {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": "model is required"})
		return
	}

	if served, err := applyModelAlias(c, req.Model); err != nil {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	} else {
		req.Model = served
	}

	modelRef, err := parseAndValidateModelRef(req.Model)
	if err != nil {
		writeModelRefParseError(c, err, http.StatusBadRequest, "model is required")
		return
	}
	if modelRef.Source == modelSourceCloud {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": "load is not available for cloud models"})
		return
	}

	name, err := getExistingName(modelRef.Name)
	if err != nil {
		c.JSON(http.StatusNotFound, gin.H{"error": fmt.Sprintf("model '%s' not found", req.Model)})
		return
	}
	served := name.String()

	already := false
	if s.sched != nil {
		already = snapshotHasLoadedModel(s.sched.ProcessSnapshot(), served)
	}

	_, _, _, _, releaseQoS, err := s.scheduleRunner(c.Request.Context(), served, []model.Capability{}, req.Options, req.KeepAlive, nil, nil, nil)
	if err != nil {
		handleScheduleError(c, served, err)
		return
	}
	releaseQoS()

	c.JSON(http.StatusOK, api.LoadResponse{
		Model:         served,
		Loaded:        []string{served},
		Message:       "model loaded",
		AlreadyLoaded: already,
	})
}

// UnloadHandler implements POST /api/unload, the inverse of POST /api/load.
// A body with model_id/id and no model is 400.
func (s *Server) UnloadHandler(c *gin.Context) {
	var req namedModelBody
	if err := c.ShouldBindJSON(&req); errors.Is(err, io.EOF) {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": "missing request body"})
		return
	} else if err != nil {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}
	if msg := wrongModelKeyError(req); msg != "" {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": msg})
		return
	}
	req.Model = strings.TrimSpace(req.Model)
	if req.Model == "" {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": "model is required"})
		return
	}

	if served, err := applyModelAlias(c, req.Model); err != nil {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	} else {
		req.Model = served
	}

	modelRef, err := parseAndValidateModelRef(req.Model)
	if err != nil {
		writeModelRefParseError(c, err, http.StatusBadRequest, "model is required")
		return
	}
	if modelRef.Source == modelSourceCloud {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": "unload is not available for cloud models"})
		return
	}

	name, err := getExistingName(modelRef.Name)
	if err != nil {
		c.JSON(http.StatusNotFound, gin.H{"error": fmt.Sprintf("model '%s' not found", req.Model)})
		return
	}

	m, err := GetModel(name.String())
	if err != nil {
		switch {
		case errors.Is(err, fs.ErrNotExist):
			c.JSON(http.StatusNotFound, gin.H{"error": fmt.Sprintf("model '%s' not found", req.Model)})
		case err.Error() == errtypes.InvalidModelNameErrMsg:
			c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		default:
			c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		}
		return
	}

	if s.sched != nil {
		s.sched.expireRunner(m)
	}

	c.JSON(http.StatusOK, api.UnloadResponse{
		Model:   name.String(),
		Message: "model unloaded",
	})
}

func snapshotHasLoadedModel(ps api.ProcessResponse, name string) bool {
	name = strings.TrimSpace(name)
	if name == "" {
		return false
	}
	for _, m := range ps.Models {
		if strings.EqualFold(m.Name, name) || strings.EqualFold(m.Model, name) {
			return true
		}
	}
	return false
}
