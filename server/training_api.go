// Training HTTP API (/api/train/*): thin handlers over the embedded Python training worker.
//
// Why handlers live in server/ and not in x/trainingworker: same Gin auth/middleware and
// lifecycle as the rest of the API; trainingworker stays CGO + wire protocol without importing gin.
package server

import (
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"strings"
	"time"

	"github.com/gin-gonic/gin"

	"github.com/ollama/ollama/x/trainingworker"
)

func (s *Server) registerTrainingRoutes(r *gin.Engine) {
	if s.training == nil {
		return
	}
	g := r.Group("/api/train")
	g.POST("/jobs", s.trainHTTPSubmitJob)
	g.GET("/jobs", s.trainHTTPListJobs)
	g.GET("/jobs/:id/events", s.trainHTTPJobEvents)
	g.GET("/jobs/:id", s.trainHTTPJobStatus)
	g.DELETE("/jobs/:id", s.trainHTTPCancelJob)
	g.POST("/unload", s.trainHTTPUnload)
	g.GET("/status", s.trainHTTPHealth)
}

func (s *Server) trainHTTPSubmitJob(c *gin.Context) {
	var req struct {
		Kind        string          `json:"kind"`
		Payload     json.RawMessage `json:"payload"`
		Priority    string          `json:"priority"`
		QueueOnBusy *bool           `json:"queue_on_busy"`
	}
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}
	kind := "train"
	if strings.EqualFold(req.Kind, "run_script") {
		kind = "run_script"
	}
	payload := req.Payload
	if len(payload) == 0 || string(payload) == "null" {
		payload = []byte("{}")
	}
	res, err := s.submitTrainingJob(c.Request.Context(), kind, payload, TrainingSubmitOptions{
		Priority:    parseTrainingPriority(req.Priority),
		QueueOnBusy: req.QueueOnBusy,
	})
	if err != nil {
		if TrainingSubmitMisconfigured(err) {
			c.JSON(http.StatusServiceUnavailable, gin.H{"error": err.Error()})
			return
		}
		if TrainingSubmitUnsupported(err) {
			c.JSON(http.StatusServiceUnavailable, gin.H{"error": err.Error()})
			return
		}
		if TrainingSubmitConflict(err) {
			c.JSON(http.StatusConflict, gin.H{"error": err.Error()})
			return
		}
		if strings.Contains(err.Error(), "defer queue full") {
			c.JSON(http.StatusServiceUnavailable, gin.H{"error": err.Error()})
			return
		}
		c.JSON(http.StatusBadGateway, gin.H{"error": err.Error()})
		return
	}
	out := gin.H{"job_id": res.JobID}
	if res.Queued {
		out["queued"] = true
		out["state"] = "waiting_for_inference_idle"
	}
	c.JSON(http.StatusAccepted, out)
}

func (s *Server) trainHTTPListJobs(c *gin.Context) {
	b, err := s.training.ListTrainingJobsJSON(c.Request.Context())
	if err != nil {
		c.JSON(http.StatusBadGateway, gin.H{"error": err.Error()})
		return
	}
	if merged, merr := s.mergeDeferredJobsListJSON(b); merr == nil {
		b = merged
	}
	c.Data(http.StatusOK, "application/json", b)
}

func (s *Server) trainHTTPJobStatus(c *gin.Context) {
	id := c.Param("id")
	b, err := s.trainingJobStatusJSON(c, id)
	if err != nil {
		if errors.Is(err, trainingworker.ErrJobNotFound) {
			c.JSON(http.StatusNotFound, gin.H{"error": err.Error()})
			return
		}
		c.JSON(http.StatusBadGateway, gin.H{"error": err.Error()})
		return
	}
	c.Data(http.StatusOK, "application/json", b)
}

func (s *Server) trainingJobStatusJSON(c *gin.Context, id string) ([]byte, error) {
	if isDeferredTrainingJobID(id) {
		return s.deferredTrainingJobStatusJSON(c.Request.Context(), id)
	}
	return s.training.JobTrainingStatusJSON(c.Request.Context(), id)
}

// trainHTTPJobEvents is T3 SSE: poll job status and push progress/metrics until terminal.
func (s *Server) trainHTTPJobEvents(c *gin.Context) {
	id := c.Param("id")
	flusher, ok := c.Writer.(http.Flusher)
	if !ok {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "streaming unsupported"})
		return
	}

	c.Writer.Header().Set("Content-Type", "text/event-stream")
	c.Writer.Header().Set("Cache-Control", "no-cache")
	c.Writer.Header().Set("Connection", "keep-alive")
	c.Writer.WriteHeader(http.StatusOK)
	flusher.Flush()

	poll := time.NewTicker(500 * time.Millisecond)
	defer poll.Stop()
	keepalive := time.NewTicker(15 * time.Second)
	defer keepalive.Stop()

	var lastSig string
	writeSSE := func(event string, payload any) error {
		raw, err := json.Marshal(payload)
		if err != nil {
			return err
		}
		if _, err := fmt.Fprintf(c.Writer, "event: %s\ndata: %s\n\n", event, raw); err != nil {
			return err
		}
		flusher.Flush()
		return nil
	}

	for {
		select {
		case <-c.Request.Context().Done():
			return
		case <-keepalive.C:
			if _, err := c.Writer.Write([]byte(": keepalive\n\n")); err != nil {
				return
			}
			flusher.Flush()
		case <-poll.C:
			b, err := s.trainingJobStatusJSON(c, id)
			if err != nil {
				_ = writeSSE("error", gin.H{"error": err.Error()})
				return
			}
			var wrap struct {
				Job map[string]any `json:"job"`
			}
			if err := json.Unmarshal(b, &wrap); err != nil || wrap.Job == nil {
				_ = writeSSE("error", gin.H{"error": "invalid job status"})
				return
			}
			job := wrap.Job
			sig := fmt.Sprintf("%v|%v|%v|%v",
				job["status"], job["progress"], job["progressMessage"], job["progressMetrics"])
			if sig != lastSig {
				lastSig = sig
				if err := writeSSE("progress", job); err != nil {
					return
				}
			}
			st, _ := job["status"].(string)
			if trainingJobTerminal(st) {
				_ = writeSSE("done", job)
				return
			}
		}
	}
}

func trainingJobTerminal(status string) bool {
	switch strings.ToLower(status) {
	case "completed", "failed", "cancelled":
		return true
	default:
		return false
	}
}

func (s *Server) trainHTTPCancelJob(c *gin.Context) {
	id := c.Param("id")
	if isDeferredTrainingJobID(id) {
		ok, err := s.cancelDeferredTrainingJob(id)
		if err != nil {
			c.JSON(http.StatusConflict, gin.H{"error": err.Error()})
			return
		}
		c.JSON(http.StatusOK, gin.H{"cancelled": ok})
		return
	}
	ok, err := s.training.CancelTrainingJob(c.Request.Context(), id)
	if err != nil {
		c.JSON(http.StatusBadGateway, gin.H{"error": err.Error()})
		return
	}
	c.JSON(http.StatusOK, gin.H{"cancelled": ok})
}

func (s *Server) trainHTTPUnload(c *gin.Context) {
	if err := s.training.UnloadTrainingModel(c.Request.Context()); err != nil {
		c.JSON(http.StatusBadGateway, gin.H{"error": err.Error()})
		return
	}
	c.JSON(http.StatusOK, gin.H{"status": "ok"})
}

func (s *Server) trainHTTPHealth(c *gin.Context) {
	raw, err := s.training.TrainingHealthJSON(c.Request.Context())
	if err != nil {
		c.JSON(http.StatusBadGateway, gin.H{"error": err.Error()})
		return
	}
	var h struct {
		Status     string `json:"status"`
		ExtrasJSON string `json:"extrasJson"`
	}
	if err := json.Unmarshal(raw, &h); err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}
	var extras any
	if h.ExtrasJSON != "" {
		_ = json.Unmarshal([]byte(h.ExtrasJSON), &extras)
	}
	c.JSON(http.StatusOK, gin.H{"status": h.Status, "extras": extras})
}
