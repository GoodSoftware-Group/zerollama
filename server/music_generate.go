package server

// MiniMax Music 3 HTTP: async training run_script, same shape as Wan /v1/videos.
//
// Why not sync POST /v1/audio/speech bytes: songs are minutes of Metal/CUDA.
// Why not Comfy: GPL. Why python_bin → .venv-music: embed CPython has no mlx-audio.
// Why exclusive GPU: UMA chat reloading llama-server mid-DiT is the Wan OOM class.

import (
	"encoding/json"
	"errors"
	"fmt"
	"io/fs"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/gin-gonic/gin"

	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/middleware"
	"github.com/ollama/ollama/openai"
	"github.com/ollama/ollama/server/modality"
	"github.com/ollama/ollama/types/model"
	"github.com/ollama/ollama/x/trainingworker"
)

const defaultMusic3TimeoutSec = 3600

type music3JobPayload struct {
	ScriptPath  string            `json:"script_path"`
	PythonBin   string            `json:"python_bin,omitempty"`
	WorkingDir  string            `json:"working_dir,omitempty"`
	Env         map[string]string `json:"env,omitempty"`
	Timeout     int               `json:"timeout"`
	OutputPath  string            `json:"output_path"`
	SubmittedAt string            `json:"submitted_at,omitempty"`
	VideoModel  string            `json:"videoModel,omitempty"`
}

// MusicCreateHandler queues MiniMax Music 3 (mlx-audio wrapper) on the training run_script path.
func (s *Server) MusicCreateHandler(c *gin.Context) {
	v, ok := c.Get(middleware.CtxKeySpeechRequest)
	if !ok {
		c.AbortWithStatusJSON(http.StatusInternalServerError, gin.H{"error": "missing music request"})
		return
	}
	s.queueMusic3(c, v.(openai.SpeechCreateRequest))
}

func (s *Server) queueMusic3(c *gin.Context, req openai.SpeechCreateRequest) {
	if s.training == nil {
		c.JSON(http.StatusServiceUnavailable, openai.NewError(http.StatusServiceUnavailable, "music generation requires OLLAMA_TRAINING=true (embedded training worker)"))
		return
	}

	modelRef, err := parseAndValidateModelRef(req.Model)
	if err != nil {
		writeModelRefParseError(c, err, http.StatusBadRequest, "model is required")
		return
	}
	if modelRef.Source == modelSourceCloud {
		c.JSON(http.StatusBadRequest, openai.NewError(http.StatusBadRequest, "cloud models are not supported for local music generation"))
		return
	}
	name, err := getExistingName(modelRef.Name)
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "model is required"})
		return
	}
	m, err := GetModel(name.String())
	if err != nil {
		switch {
		case errors.Is(err, fs.ErrNotExist), os.IsNotExist(err):
			c.JSON(http.StatusNotFound, gin.H{"error": fmt.Sprintf("model '%s' not found", req.Model)})
		default:
			c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		}
		return
	}
	if err := m.CheckCapabilities(model.CapabilitySpeech); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": fmt.Sprintf("%q does not support speech/music", req.Model)})
		return
	}
	if modality.BackendFor(m.Config, model.ModalitySpeech) != model.BackendMusic3 {
		c.JSON(http.StatusBadRequest, gin.H{"error": "speech backend must be music3"})
		return
	}

	createdAt := time.Now().UTC()
	payload, err := buildMusic3JobPayload(m.Config, req, createdAt)
	if err != nil {
		c.JSON(http.StatusBadRequest, openai.NewError(http.StatusBadRequest, err.Error()))
		return
	}
	raw, err := json.Marshal(payload)
	if err != nil {
		c.JSON(http.StatusInternalServerError, openai.NewError(http.StatusInternalServerError, err.Error()))
		return
	}
	queueOnBusy := true
	res, err := s.submitTrainingJob(c.Request.Context(), "run_script", raw, TrainingSubmitOptions{
		QueueOnBusy: &queueOnBusy,
	})
	if err != nil {
		if TrainingSubmitMisconfigured(err) {
			c.JSON(http.StatusServiceUnavailable, openai.NewError(http.StatusServiceUnavailable, err.Error()))
			return
		}
		if TrainingSubmitConflict(err) {
			c.JSON(http.StatusConflict, openai.NewError(http.StatusConflict, err.Error()))
			return
		}
		c.JSON(http.StatusBadGateway, openai.NewError(http.StatusBadGateway, err.Error()))
		return
	}
	// Same as Wan: Metal/UMA music must not share the GPU with chat reloads.
	if !res.Queued {
		s.acquireVideoExclusiveGPU(c.Request.Context(), res.JobID)
	}
	c.JSON(http.StatusAccepted, openai.AudioGenerationFromSubmit(res.JobID, req.Model, res.Queued, createdAt.Unix()))
}

// MusicGetHandler returns GET /v1/audio/generations/:id
func (s *Server) MusicGetHandler(c *gin.Context) {
	if s.training == nil {
		c.JSON(http.StatusServiceUnavailable, openai.NewError(http.StatusServiceUnavailable, "music generation requires OLLAMA_TRAINING=true"))
		return
	}
	id := c.Param("id")
	raw, err := s.videoJobStatusJSON(c.Request.Context(), id)
	if err != nil {
		if errors.Is(err, trainingworker.ErrJobNotFound) {
			c.JSON(http.StatusNotFound, openai.NewError(http.StatusNotFound, "music job not found"))
		} else {
			c.JSON(http.StatusBadGateway, openai.NewError(http.StatusBadGateway, err.Error()))
		}
		return
	}
	var wrap struct {
		Job json.RawMessage `json:"job"`
	}
	if err := json.Unmarshal(raw, &wrap); err != nil || len(wrap.Job) == 0 {
		c.JSON(http.StatusBadGateway, openai.NewError(http.StatusBadGateway, "invalid job status"))
		return
	}
	audio, err := openai.AudioGenerationFromTrainingJob(wrap.Job)
	if err != nil {
		c.JSON(http.StatusBadGateway, openai.NewError(http.StatusBadGateway, err.Error()))
		return
	}
	if isDeferredTrainingJobID(id) {
		audio.ID = id
	}
	c.JSON(http.StatusOK, audio)
}

// MusicContentHandler streams GET /v1/audio/generations/:id/content (audio/wav).
func (s *Server) MusicContentHandler(c *gin.Context) {
	if s.training == nil {
		c.JSON(http.StatusServiceUnavailable, openai.NewError(http.StatusServiceUnavailable, "music generation requires OLLAMA_TRAINING=true"))
		return
	}
	id := c.Param("id")
	raw, err := s.videoJobStatusJSON(c.Request.Context(), id)
	if err != nil {
		if errors.Is(err, trainingworker.ErrJobNotFound) {
			c.JSON(http.StatusNotFound, openai.NewError(http.StatusNotFound, "music job not found"))
		} else {
			c.JSON(http.StatusBadGateway, openai.NewError(http.StatusBadGateway, err.Error()))
		}
		return
	}
	var wrap struct {
		Job json.RawMessage `json:"job"`
	}
	if err := json.Unmarshal(raw, &wrap); err != nil || len(wrap.Job) == 0 {
		c.JSON(http.StatusBadGateway, openai.NewError(http.StatusBadGateway, "invalid job status"))
		return
	}
	path, statusCode, err := completedVideoOutputFromJob(wrap.Job)
	if err != nil {
		msg := err.Error()
		msg = strings.ReplaceAll(msg, "video", "music")
		c.JSON(statusCode, openai.NewError(statusCode, msg))
		return
	}
	safe, err := safeMusicArtifactPath(path)
	if err != nil {
		c.JSON(http.StatusInternalServerError, openai.NewError(http.StatusInternalServerError, err.Error()))
		return
	}
	c.Header("Content-Type", "audio/wav")
	c.Header("Content-Disposition", fmt.Sprintf(`attachment; filename="%s.wav"`, id))
	c.File(safe)
}

func buildMusic3JobPayload(cfg model.ConfigV2, req openai.SpeechCreateRequest, submittedAt time.Time) (music3JobPayload, error) {
	lyrics := strings.TrimSpace(req.Input)
	if lyrics == "" {
		return music3JobPayload{}, errors.New("input (lyrics) is required")
	}
	repo, err := trainingworker.RepoRoot()
	if err != nil || repo == "" {
		return music3JobPayload{}, errors.New("cannot locate repository root (set ZEROLLAMA_REPO or OLLAMA_TRAINING_PYTHONPATH)")
	}
	scriptPath := filepath.Join(repo, "scripts", "audio", "music3_mlx_generate.py")
	if _, err := os.Stat(scriptPath); err != nil {
		return music3JobPayload{}, fmt.Errorf("music wrapper script not found at %s", scriptPath)
	}

	modelPath := expandUserPath(modality.PathFor(cfg, "music3_mlx_model"))
	if modelPath == "" {
		modelPath = os.Getenv("MUSIC3_MLX_MODEL")
	}
	if modelPath == "" {
		modelPath = "mlx-community/MiniMax-Music3-8bit"
	}

	caption := strings.TrimSpace(req.Instructions)
	if caption == "" {
		caption = "Warm acoustic pop, 96 BPM, intimate female vocal"
	}

	duration := 10.0
	if cfg.MusicGeneration != nil && cfg.MusicGeneration.DurationSec > 0 {
		duration = cfg.MusicGeneration.DurationSec
	}
	// Explicit duration wins. max_new_tokens is AR frames at 25 fps only when duration is omitted.
	// Why: Omni clients send max_new_tokens; a leftover 250 must not shrink an explicit 30s clip.
	if req.Duration != nil && *req.Duration > 0 {
		duration = *req.Duration
	} else if req.MaxNewTokens != nil && *req.MaxNewTokens > 0 {
		duration = float64(*req.MaxNewTokens) / 25.0
	}

	steps := 30
	if cfg.MusicGeneration != nil && cfg.MusicGeneration.Steps > 0 {
		steps = cfg.MusicGeneration.Steps
	}
	if req.Steps != nil && *req.Steps > 0 {
		steps = *req.Steps
	}

	timeout := defaultMusic3TimeoutSec
	if cfg.MusicGeneration != nil && cfg.MusicGeneration.TimeoutSec > 0 {
		timeout = cfg.MusicGeneration.TimeoutSec
	}

	pythonBin := resolveMusic3Python(repo)

	outputPath := filepath.Join(videoArtifactRoot(), "{job_id}.wav")
	env := map[string]string{
		"MUSIC3_MLX_MODEL":   modelPath,
		"MUSIC3_CAPTION":     caption,
		"MUSIC3_LYRICS":      lyrics,
		"MUSIC3_DURATION":    strconv.FormatFloat(duration, 'f', -1, 64),
		"MUSIC3_STEPS":       strconv.Itoa(steps),
		"MUSIC3_OUTPUT_PATH": outputPath,
	}
	if cli := strings.TrimSpace(envconfig.Var("ZEROLLAMA_MUSIC_CLI")); cli != "" {
		env["ZEROLLAMA_MUSIC_CLI"] = cli
	}
	if req.Seed != nil {
		env["MUSIC3_SEED"] = strconv.FormatInt(*req.Seed, 10)
	}

	return music3JobPayload{
		ScriptPath:  scriptPath,
		PythonBin:   pythonBin,
		WorkingDir:  repo,
		Env:         env,
		Timeout:     timeout,
		OutputPath:  outputPath,
		SubmittedAt: submittedAt.UTC().Format(time.RFC3339Nano),
		VideoModel:  req.Model,
	}, nil
}

func safeMusicArtifactPath(path string) (string, error) {
	return safeVideoArtifactPath(path)
}

func resolveMusic3Python(repo string) string {
	// Why not LookPath("python3") first: macOS /usr/bin/python3 is often 3.9 without mlx-audio.
	if p := strings.TrimSpace(envconfig.Var("ZEROLLAMA_MUSIC_PYTHON")); p != "" {
		return p
	}
	for _, rel := range []string{
		filepath.Join(".venv-music", "bin", "python"),
		filepath.Join(".venv-music", "bin", "python3"),
	} {
		cand := filepath.Join(repo, rel)
		if st, err := os.Stat(cand); err == nil && !st.IsDir() {
			return cand
		}
	}
	if p, err := exec.LookPath("python3"); err == nil {
		return p
	}
	return "/usr/bin/python3"
}
