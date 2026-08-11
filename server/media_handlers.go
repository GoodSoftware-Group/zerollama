package server

import (
	"errors"
	"io"
	"net/http"
	"strconv"
	"strings"
	"sync"

	"github.com/gin-gonic/gin"

	"github.com/ollama/ollama/openai"
	"github.com/ollama/ollama/server/media"
)

// mediaStoreMu guards lazy init of mediaStore (OLLAMA_MODELS must be read at first use).
// WHY lazy: package init runs before some tests/wrappers set OLLAMA_MODELS; first
// request builds the store against the live env (docs/media-uploads.md).
var (
	mediaStoreMu sync.Mutex
	mediaStore   *media.Store
)

func getMediaStore() *media.Store {
	mediaStoreMu.Lock()
	defer mediaStoreMu.Unlock()
	if mediaStore == nil {
		mediaStore = media.Default()
	}
	return mediaStore
}

// setMediaStore replaces the process store (tests only).
func setMediaStore(s *media.Store) {
	mediaStoreMu.Lock()
	defer mediaStoreMu.Unlock()
	mediaStore = s
}

// MediaPutHandler handles PUT /v1/media/:session/:label.
// WHY raw body (not multipart JSON): agents stream large PNGs/JPEGs without base64.
func (s *Server) MediaPutHandler(c *gin.Context) {
	session := c.Param("session")
	label := c.Param("label")
	ct := c.GetHeader("Content-Type")
	body := http.MaxBytesReader(c.Writer, c.Request.Body, media.MaxObjectBytes+1)
	res, err := getMediaStore().Put(session, label, ct, body)
	if err != nil {
		writeMediaErr(c, err)
		return
	}
	c.JSON(http.StatusCreated, res)
}

// MediaHeadHandler handles HEAD /v1/media/:session/:label.
func (s *Server) MediaHeadHandler(c *gin.Context) {
	meta, err := getMediaStore().Head(c.Param("session"), c.Param("label"))
	if err != nil {
		writeMediaErr(c, err)
		return
	}
	c.Header("Content-Type", meta.ContentType)
	c.Header("X-Media-Digest", meta.Digest)
	c.Header("X-Media-Kind", string(meta.Kind))
	c.Header("Content-Length", strconv.FormatInt(meta.Size, 10))
	c.Status(http.StatusOK)
}

// MediaGetLabelHandler handles GET /v1/media/:session/:label (raw bytes).
func (s *Server) MediaGetLabelHandler(c *gin.Context) {
	path, meta, err := getMediaStore().GetPath(c.Param("session"), c.Param("label"))
	if err != nil {
		writeMediaErr(c, err)
		return
	}
	c.Header("Content-Type", meta.ContentType)
	c.Header("X-Media-Digest", meta.Digest)
	c.Header("X-Media-Kind", string(meta.Kind))
	c.File(path)
}

// MediaDeleteHandler handles DELETE /v1/media/:session/:label.
func (s *Server) MediaDeleteHandler(c *gin.Context) {
	if err := getMediaStore().Delete(c.Param("session"), c.Param("label")); err != nil {
		writeMediaErr(c, err)
		return
	}
	c.Status(http.StatusNoContent)
}

// MediaListHandler handles GET /v1/media/:session.
func (s *Server) MediaListHandler(c *gin.Context) {
	labels, err := getMediaStore().List(c.Param("session"))
	if err != nil {
		writeMediaErr(c, err)
		return
	}
	c.JSON(http.StatusOK, gin.H{
		"session": c.Param("session"),
		"labels":  labels,
	})
}

func writeMediaErr(c *gin.Context, err error) {
	switch {
	case errors.Is(err, media.ErrInvalidSession), errors.Is(err, media.ErrInvalidLabel), errors.Is(err, media.ErrEmptyBody):
		c.JSON(http.StatusBadRequest, openai.NewError(http.StatusBadRequest, err.Error()))
	case errors.Is(err, media.ErrTooLarge):
		c.JSON(http.StatusRequestEntityTooLarge, openai.NewError(http.StatusRequestEntityTooLarge, err.Error()))
	case errors.Is(err, media.ErrNotFound):
		c.JSON(http.StatusNotFound, openai.NewError(http.StatusNotFound, err.Error()))
	case errors.As(err, new(*http.MaxBytesError)) || strings.Contains(err.Error(), "http: request body too large"):
		c.JSON(http.StatusRequestEntityTooLarge, openai.NewError(http.StatusRequestEntityTooLarge, media.ErrTooLarge.Error()))
	default:
		if errors.Is(err, io.EOF) {
			c.JSON(http.StatusBadRequest, openai.NewError(http.StatusBadRequest, media.ErrEmptyBody.Error()))
			return
		}
		c.JSON(http.StatusInternalServerError, openai.NewError(http.StatusInternalServerError, err.Error()))
	}
}
