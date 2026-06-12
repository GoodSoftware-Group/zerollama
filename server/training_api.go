// Training HTTP API (/api/train/*): thin handlers over the embedded Python training worker.
//
// Why handlers live in server/ and not in x/trainingworker: same Gin auth/middleware and
// lifecycle as the rest of the API; trainingworker stays CGO + wire protocol without importing gin.
package server

import (
	"encoding/json"
	"errors"
	"net/http"
	"strings"

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
	if isDeferredTrainingJobID(id) {
		b, err := s.deferredTrainingJobStatusJSON(c.Request.Context(), id)
		if err != nil {
			if errors.Is(err, trainingworker.ErrJobNotFound) {
				c.JSON(http.StatusNotFound, gin.H{"error": err.Error()})
			} else {
				c.JSON(http.StatusBadGateway, gin.H{"error": err.Error()})
			}
			return
		}
		c.Data(http.StatusOK, "application/json", b)
		return
	}
	b, err := s.training.JobTrainingStatusJSON(c.Request.Context(), id)
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
