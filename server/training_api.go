// Training HTTP API (/api/train/*): thin handlers over the trainingworker gRPC client when the
// Python sidecar started successfully. Why here: same Gin router and auth story as the rest of Ollama;
// why not call Python from every handler directly: one client, shared lifecycle with Serve().
package server

import (
	"encoding/json"
	"net/http"
	"strings"

	"github.com/gin-gonic/gin"
	"google.golang.org/protobuf/encoding/protojson"

	"github.com/ollama/ollama/x/trainingworker/trainingpb"
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
		Kind    string          `json:"kind"`
		Payload json.RawMessage `json:"payload"`
	}
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}
	kind := trainingpb.JobKind_JOB_KIND_TRAIN
	if strings.EqualFold(req.Kind, "run_script") {
		kind = trainingpb.JobKind_JOB_KIND_RUN_SCRIPT
	}
	payload := string(req.Payload)
	if payload == "" || payload == "null" {
		payload = "{}"
	}
	out, err := s.training.GRPC().SubmitJob(c.Request.Context(), &trainingpb.SubmitJobRequest{
		Kind:        kind,
		PayloadJson: payload,
	})
	if err != nil {
		c.JSON(http.StatusBadGateway, gin.H{"error": err.Error()})
		return
	}
	c.JSON(http.StatusAccepted, gin.H{"job_id": out.JobId})
}

func (s *Server) trainHTTPListJobs(c *gin.Context) {
	out, err := s.training.GRPC().ListJobs(c.Request.Context(), &trainingpb.ListJobsRequest{})
	if err != nil {
		c.JSON(http.StatusBadGateway, gin.H{"error": err.Error()})
		return
	}
	b, err := protojson.Marshal(out)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}
	c.Data(http.StatusOK, "application/json", b)
}

func (s *Server) trainHTTPJobStatus(c *gin.Context) {
	id := c.Param("id")
	out, err := s.training.GRPC().JobStatus(c.Request.Context(), &trainingpb.JobStatusRequest{JobId: id})
	if err != nil {
		c.JSON(http.StatusNotFound, gin.H{"error": err.Error()})
		return
	}
	b, err := protojson.Marshal(out)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}
	c.Data(http.StatusOK, "application/json", b)
}

func (s *Server) trainHTTPCancelJob(c *gin.Context) {
	id := c.Param("id")
	out, err := s.training.GRPC().CancelJob(c.Request.Context(), &trainingpb.CancelJobRequest{JobId: id})
	if err != nil {
		c.JSON(http.StatusBadGateway, gin.H{"error": err.Error()})
		return
	}
	c.JSON(http.StatusOK, gin.H{"cancelled": out.Cancelled})
}

func (s *Server) trainHTTPUnload(c *gin.Context) {
	_, err := s.training.GRPC().Unload(c.Request.Context(), &trainingpb.UnloadRequest{})
	if err != nil {
		c.JSON(http.StatusBadGateway, gin.H{"error": err.Error()})
		return
	}
	c.JSON(http.StatusOK, gin.H{"status": "ok"})
}

func (s *Server) trainHTTPHealth(c *gin.Context) {
	out, err := s.training.GRPC().Health(c.Request.Context(), &trainingpb.HealthRequest{})
	if err != nil {
		c.JSON(http.StatusBadGateway, gin.H{"error": err.Error()})
		return
	}
	var extras any
	if out.ExtrasJson != "" {
		_ = json.Unmarshal([]byte(out.ExtrasJson), &extras)
	}
	c.JSON(http.StatusOK, gin.H{"status": out.Status, "extras": extras})
}
