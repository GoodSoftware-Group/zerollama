// Wan text-to-video HTTP handlers (OpenAI /v1/videos).
//
// Why not the Python runtime or ggml runner: Wan needs a long PyTorch subprocess and
// multi-GB checkpoints outside Ollama blobs. We queue run_script on the embedded training
// worker so VRAM handoff, defer-when-busy, and PROGRESS logging match training T6.
// Go owns registry validation, payload/env assembly, artifact paths, and OpenAI status mapping.
package server

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io/fs"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/gin-gonic/gin"

	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/middleware"
	"github.com/ollama/ollama/openai"
	"github.com/ollama/ollama/server/media"
	"github.com/ollama/ollama/server/modality"
	"github.com/ollama/ollama/types/model"
	"github.com/ollama/ollama/x/trainingworker"
)

const (
	wanProfile21T2V13B = "wan2.1-t2v-1.3b"
	wanProfile22TI2V5B = "wan2.2-ti2v-5b"
	wan22MaxFrames16g  = 81
)

type wanVideoJobPayload struct {
	ScriptPath  string            `json:"script_path"`
	PythonBin   string            `json:"python_bin,omitempty"`
	WorkingDir  string            `json:"working_dir,omitempty"`
	Env         map[string]string `json:"env,omitempty"`
	Timeout     int               `json:"timeout"`
	OutputPath  string            `json:"output_path"`
	SubmittedAt string            `json:"submitted_at,omitempty"`
	VideoModel  string            `json:"video_model,omitempty"`
	VideoSize   string            `json:"video_size,omitempty"`
}

// VideoCreateHandler accepts POST /v1/videos and queues a Wan run_script job.
// QueueOnBusy is always true so inference-first hosts get defer-* ids instead of 409 storms.
// PrepareForTraining (inside submitTrainingJob) unloads ggml/runtime before Wan grabs the GPU.
func (s *Server) VideoCreateHandler(c *gin.Context) {
	if s.training == nil {
		c.JSON(http.StatusServiceUnavailable, openai.NewError(http.StatusServiceUnavailable, "video generation requires OLLAMA_TRAINING=true (embedded training worker)"))
		return
	}

	v, ok := c.Get(middleware.CtxKeyVideoCreateRequest)
	if !ok {
		c.JSON(http.StatusInternalServerError, openai.NewError(http.StatusInternalServerError, "missing video request"))
		return
	}
	req := v.(openai.VideoCreateRequest)

	modelRef, err := parseAndValidateModelRef(req.Model)
	if err != nil {
		writeModelRefParseError(c, err, http.StatusBadRequest, "model is required")
		return
	}
	if modelRef.Source == modelSourceCloud {
		c.JSON(http.StatusBadRequest, openai.NewError(http.StatusBadRequest, "cloud models are not supported for local video generation"))
		return
	}

	name, err := getExistingName(modelRef.Name)
	if err != nil {
		c.JSON(http.StatusInternalServerError, openai.NewError(http.StatusInternalServerError, fmt.Sprintf("could not resolve model %q: %v", req.Model, err)))
		return
	}

	m, err := GetModel(name.String())
	if err != nil {
		if errors.Is(err, fs.ErrNotExist) || os.IsNotExist(err) {
			c.JSON(http.StatusNotFound, openai.NewError(http.StatusNotFound, fmt.Sprintf("model '%s' not found", req.Model)))
			return
		}
		c.JSON(http.StatusInternalServerError, openai.NewError(http.StatusInternalServerError, err.Error()))
		return
	}

	if err := m.CheckCapabilities(model.CapabilityVideoGen); err != nil {
		c.JSON(http.StatusBadRequest, openai.NewError(http.StatusBadRequest, fmt.Sprintf("%q does not support video generation", req.Model)))
		return
	}

	backend := modality.BackendFor(m.Config, model.ModalityVideoGeneration)
	switch backend {
	case model.BackendWan:
		// ok
	case model.BackendRIFE:
		c.JSON(http.StatusBadRequest, openai.NewError(http.StatusBadRequest, "video_generation backend \"rife\" is reserved but not implemented yet"))
		return
	default:
		c.JSON(http.StatusBadRequest, openai.NewError(http.StatusBadRequest, fmt.Sprintf("video_generation backend must be %q (got %q)", model.BackendWan, backend)))
		return
	}

	cfg, err := resolveVideoGenerationConfig(m, req)
	if err != nil {
		c.JSON(http.StatusBadRequest, openai.NewError(http.StatusBadRequest, err.Error()))
		return
	}

	mediaSession, keyframeLabels, err := media.ParseKeyframeRefs(req.Options)
	if err != nil {
		c.JSON(http.StatusBadRequest, openai.NewError(http.StatusBadRequest, err.Error()))
		return
	}
	if err := validateWanKeyframes(cfg.Profile, keyframeLabels); err != nil {
		c.JSON(http.StatusBadRequest, openai.NewError(http.StatusBadRequest, err.Error()))
		return
	}

	var keyframeDir string
	if len(keyframeLabels) > 0 {
		store := getMediaStore()
		paths, metas, missing, err := store.ResolveMany(mediaSession, keyframeLabels)
		if err != nil {
			c.JSON(http.StatusBadRequest, openai.NewError(http.StatusBadRequest, err.Error()))
			return
		}
		if len(missing) > 0 {
			// WHY structured media_missing: agents re-PUT listed labels without scraping messages.
			c.JSON(http.StatusBadRequest, openai.NewMediaMissingError(mediaSession, missing))
			return
		}
		_ = paths
		var bad []string
		for i, meta := range metas {
			if meta.Kind != media.KindImage {
				bad = append(bad, keyframeLabels[i])
			}
		}
		if len(bad) > 0 {
			// WHY reject video here: Wan TI2V keyframes are stills; kind=video is reserved for morph.
			c.JSON(http.StatusBadRequest, openai.NewMediaTypeMismatchError(mediaSession, bad, "wan keyframes must be images; upload video clips only for future video-morph backends"))
			return
		}
		stagingID := fmt.Sprintf("kf-%d", time.Now().UnixNano())
		keyframeDir = filepath.Join(videoArtifactRoot(), "keyframes", stagingID)
		// WHY materialize: freeze CAS into staging so LRU eviction cannot race a long Wan job.
		missing, err = store.Materialize(mediaSession, keyframeLabels, keyframeDir)
		if err != nil {
			c.JSON(http.StatusInternalServerError, openai.NewError(http.StatusInternalServerError, err.Error()))
			return
		}
		if len(missing) > 0 {
			_ = os.RemoveAll(keyframeDir)
			c.JSON(http.StatusBadRequest, openai.NewMediaMissingError(mediaSession, missing))
			return
		}
	}

	opts := req.Options
	if opts == nil {
		opts = map[string]any{}
	}
	videoHints := mlxScheduleHints{
		Route:    "video_generation",
		Modality: mlxModalityVideoGeneration,
		Stream:   false,
	}
	if err := s.waitRequestQoS(c.Request.Context(), nil, opts, videoHints); err != nil {
		if keyframeDir != "" {
			// WHY cleanup on QoS fail: materialize already wrote staging; do not orphan kf-* dirs.
			_ = os.RemoveAll(keyframeDir)
		}
		if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
			c.JSON(http.StatusRequestTimeout, openai.NewError(http.StatusRequestTimeout, err.Error()))
			return
		}
		c.JSON(http.StatusServiceUnavailable, openai.NewError(http.StatusServiceUnavailable, err.Error()))
		return
	}

	seed := req.Seed
	if seed == nil && req.Options != nil {
		if v, ok := req.Options["seed"]; ok {
			s := int64FromAny(v)
			seed = &s
		}
	}

	createdAt := time.Now().UTC()
	payload, err := buildVideoJobPayload(backend, m.Config, cfg, req.Model, req.Prompt, seed, createdAt, keyframeDir)
	if err != nil {
		if keyframeDir != "" {
			_ = os.RemoveAll(keyframeDir)
		}
		c.JSON(http.StatusBadRequest, openai.NewError(http.StatusBadRequest, err.Error()))
		return
	}

	raw, err := json.Marshal(payload)
	if err != nil {
		if keyframeDir != "" {
			_ = os.RemoveAll(keyframeDir)
		}
		c.JSON(http.StatusInternalServerError, openai.NewError(http.StatusInternalServerError, err.Error()))
		return
	}

	queueOnBusy := true
	res, err := s.submitTrainingJob(c.Request.Context(), "run_script", raw, TrainingSubmitOptions{
		QueueOnBusy: &queueOnBusy,
	})
	if err != nil {
		if keyframeDir != "" {
			_ = os.RemoveAll(keyframeDir)
		}
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

	video := openai.VideoFromSubmit(res.JobID, req.Model, cfg.Size, res.Queued, createdAt.Unix())
	c.JSON(http.StatusAccepted, video)
}

func validateWanKeyframes(profile string, labels []string) error {
	if len(labels) == 0 {
		return nil
	}
	p := strings.ToLower(strings.TrimSpace(profile))
	if strings.Contains(p, "ti2v") || strings.HasPrefix(p, "wan2.2") {
		return nil
	}
	// WHY: wan2.1-t2v has no start-image path; keyframes would be silently ignored otherwise.
	return fmt.Errorf("keyframes require a TI2V profile (e.g. wan2.2-ti2v-5b); %q is text-to-video only", profile)
}

// buildVideoJobPayload dispatches by video_generation backend. Wan is the only shipped runner.
func buildVideoJobPayload(backend string, cfg model.ConfigV2, vcfg model.VideoGenerationConfig, modelName, prompt string, seed *int64, submittedAt time.Time, keyframeDir string) (wanVideoJobPayload, error) {
	switch backend {
	case model.BackendWan:
		return buildWanVideoPayload(cfg, vcfg, modelName, prompt, seed, submittedAt, keyframeDir)
	case model.BackendRIFE:
		return wanVideoJobPayload{}, errors.New("rife backend is not implemented yet")
	default:
		return wanVideoJobPayload{}, fmt.Errorf("unsupported video_generation backend %q", backend)
	}
}

// VideoGetHandler returns GET /v1/videos/:id job status.
func (s *Server) VideoGetHandler(c *gin.Context) {
	if s.training == nil {
		c.JSON(http.StatusServiceUnavailable, openai.NewError(http.StatusServiceUnavailable, "video generation requires OLLAMA_TRAINING=true"))
		return
	}

	id := c.Param("id")
	raw, err := s.videoJobStatusJSON(c.Request.Context(), id)
	if err != nil {
		if errors.Is(err, trainingworker.ErrJobNotFound) {
			c.JSON(http.StatusNotFound, openai.NewError(http.StatusNotFound, "video job not found"))
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

	video, err := openai.VideoFromTrainingJob(wrap.Job)
	if err != nil {
		c.JSON(http.StatusBadGateway, openai.NewError(http.StatusBadGateway, err.Error()))
		return
	}
	// Keep defer-* ids stable in responses when the client polled the deferred job id.
	if isDeferredTrainingJobID(id) {
		video.ID = id
	}
	c.JSON(http.StatusOK, video)
}

// VideoContentHandler streams GET /v1/videos/:id/content (video/mp4).
func (s *Server) VideoContentHandler(c *gin.Context) {
	if s.training == nil {
		c.JSON(http.StatusServiceUnavailable, openai.NewError(http.StatusServiceUnavailable, "video generation requires OLLAMA_TRAINING=true"))
		return
	}

	id := c.Param("id")
	raw, err := s.videoJobStatusJSON(c.Request.Context(), id)
	if err != nil {
		if errors.Is(err, trainingworker.ErrJobNotFound) {
			c.JSON(http.StatusNotFound, openai.NewError(http.StatusNotFound, "video job not found"))
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
		c.JSON(statusCode, openai.NewError(statusCode, err.Error()))
		return
	}

	safe, err := safeVideoArtifactPath(path)
	if err != nil {
		c.JSON(http.StatusInternalServerError, openai.NewError(http.StatusInternalServerError, err.Error()))
		return
	}

	c.Header("Content-Type", "video/mp4")
	c.Header("Content-Disposition", fmt.Sprintf(`attachment; filename="%s.mp4"`, id))
	c.File(safe)
}

func (s *Server) videoJobStatusJSON(ctx context.Context, id string) ([]byte, error) {
	if isDeferredTrainingJobID(id) {
		return s.deferredTrainingJobStatusJSON(ctx, id)
	}
	b, err := s.training.JobTrainingStatusJSON(ctx, id)
	if err != nil {
		return nil, err
	}
	return b, nil
}

func resolveVideoGenerationConfig(m *Model, req openai.VideoCreateRequest) (model.VideoGenerationConfig, error) {
	if m == nil || m.Config.VideoGeneration == nil {
		return model.VideoGenerationConfig{}, errors.New("model manifest missing video_generation config")
	}
	cfg := *m.Config.VideoGeneration
	if cfg.Profile == "" {
		return model.VideoGenerationConfig{}, errors.New("video_generation.profile is required")
	}
	manifestFrames := cfg.Frames

	if req.Size != "" {
		cfg.Size = req.Size
	}
	if req.Options != nil {
		if v, ok := req.Options["frames"]; ok {
			cfg.Frames = intFromAny(v, cfg.Frames)
		}
		if v, ok := req.Options["steps"]; ok {
			cfg.Steps = intFromAny(v, cfg.Steps)
		}
	}
	cfg.Frames = clampWanFrames(cfg, cfg.Frames, manifestFrames)
	if cfg.Frames <= 0 {
		cfg.Frames = 49
	}
	if cfg.Steps <= 0 {
		cfg.Steps = 25
	}
	if cfg.Size == "" {
		cfg.Size = "832x480"
	}
	applyWan16gVRAMDefaults(&cfg)
	return cfg, nil
}

// applyWan16gVRAMDefaults forces T5-on-CPU and weight offload on 16g GPUs.
// Without t5_cpu, Wan moves T5-XXL to CUDA after loading DiT and OOMs on ~16GB cards.
func applyWan16gVRAMDefaults(cfg *model.VideoGenerationConfig) {
	if cfg.VRAMTier != "16g" {
		return
	}
	cfg.T5CPU = true
	cfg.OffloadModel = true
}

// wanForceSDPA selects PyTorch SDPA over flash_attn kernels. Source-built flash_attn on
// SM120 (5080) can abort at runtime; SDPA is slower but stable on 16g consumer GPUs.
func wanForceSDPA(cfg model.VideoGenerationConfig) string {
	if v := strings.TrimSpace(envconfig.Var("ZEROLLAMA_WAN_FORCE_SDPA")); v != "" {
		return v
	}
	if cfg.VRAMTier == "16g" {
		return "1"
	}
	return "0"
}

// wanUnloadT5 releases T5-XXL from host RAM after prompt encode (~11G). Default on for 16g tier.
func wanUnloadT5(cfg model.VideoGenerationConfig) string {
	if v := strings.TrimSpace(envconfig.Var("ZEROLLAMA_WAN_UNLOAD_T5")); v != "" {
		return v
	}
	if cfg.VRAMTier == "16g" {
		return "1"
	}
	return "0"
}

// wanVAECPU decodes latents on CPU (extra host RAM). Default on for 16g — GPU VAE needs ~15G contiguous VRAM.
func wanVAECPU(cfg model.VideoGenerationConfig) string {
	if v := strings.TrimSpace(envconfig.Var("ZEROLLAMA_WAN_VAE_CPU")); v != "" {
		return v
	}
	if cfg.VRAMTier == "16g" {
		return "1"
	}
	return "0"
}

// clampWanFrames caps request options on 16g tiers—Wan OOMs are expensive (30+ min lost).
func clampWanFrames(cfg model.VideoGenerationConfig, frames, manifestFrames int) int {
	// frames <= 0 means "unset"; caller applies manifest/default after clamping.
	if frames <= 0 {
		return frames
	}
	if cfg.VRAMTier != "16g" {
		return frames
	}
	max := wan16gDefaultMaxFrames(cfg.Profile)
	if manifestFrames > max {
		max = manifestFrames
	}
	if frames > max {
		return max
	}
	return frames
}

func wan16gDefaultMaxFrames(profile string) int {
	if profile == wanProfile22TI2V5B {
		return wan22MaxFrames16g
	}
	return 49
}

func intFromAny(v any, def int) int {
	switch t := v.(type) {
	case float64:
		return int(t)
	case int:
		return t
	case int64:
		return int(t)
	case json.Number:
		n, _ := t.Int64()
		return int(n)
	case string:
		n, err := strconv.Atoi(t)
		if err != nil {
			return def
		}
		return n
	default:
		return def
	}
}

func int64FromAny(v any) int64 {
	switch t := v.(type) {
	case float64:
		return int64(t)
	case int:
		return int64(t)
	case int64:
		return t
	case json.Number:
		n, _ := t.Int64()
		return n
	case string:
		n, _ := strconv.ParseInt(t, 10, 64)
		return n
	default:
		return 0
	}
}

// wanVideoJobMeta reads model/size metadata stored on run_script payloads (defer queue polling).
func wanVideoJobMeta(payload json.RawMessage) (modelName, size string) {
	var p struct {
		VideoModel string `json:"video_model"`
		VideoSize  string `json:"video_size"`
	}
	if err := json.Unmarshal(payload, &p); err != nil {
		return "", ""
	}
	return p.VideoModel, p.VideoSize
}

// buildWanVideoPayload turns a manifest + request into run_script data.
// {job_id} in output paths is expanded in Python when the job starts—Go does not know the id yet.
// PythonBin/WAN_* point at the Wan venv because the embedded training interpreter lacks Wan deps.
func buildWanVideoPayload(cfg model.ConfigV2, vcfg model.VideoGenerationConfig, modelName, prompt string, seed *int64, submittedAt time.Time, keyframeDir string) (wanVideoJobPayload, error) {
	if strings.TrimSpace(prompt) == "" {
		return wanVideoJobPayload{}, errors.New("prompt is required")
	}

	repo, err := trainingworker.RepoRoot()
	if err != nil || repo == "" {
		return wanVideoJobPayload{}, errors.New("cannot locate repository root (set ZEROLLAMA_REPO or OLLAMA_TRAINING_PYTHONPATH)")
	}

	scriptPath := filepath.Join(repo, "scripts", "video", "wan_video_generate.py")
	wanCLI := strings.TrimSpace(envconfig.Var("ZEROLLAMA_WAN_CLI"))
	if wanCLI == "" {
		wanCLI = expandUserPath(modality.PathFor(cfg, "wan_cli"))
	}
	useWanC := wanCLI != ""
	if useWanC {
		if st, err := os.Stat(wanCLI); err != nil || st.IsDir() {
			return wanVideoJobPayload{}, fmt.Errorf("ZEROLLAMA_WAN_CLI / backend_paths.wan_cli not found: %s", wanCLI)
		}
		scriptPath = filepath.Join(repo, "scripts", "video", "wan_c_generate.py")
		if _, err := os.Stat(scriptPath); err != nil {
			return wanVideoJobPayload{}, fmt.Errorf("wan-c wrapper script not found at %s", scriptPath)
		}
	} else if _, err := os.Stat(scriptPath); err != nil {
		return wanVideoJobPayload{}, fmt.Errorf("wan wrapper script not found at %s", scriptPath)
	}

	wanRepo := expandUserPath(modality.PathFor(cfg, "wan_repo"))
	wanCkpt := expandUserPath(modality.PathFor(cfg, "wan_ckpt_dir"))
	if wanRepo == "" || wanCkpt == "" {
		return wanVideoJobPayload{}, errors.New("backend_paths.wan_repo and backend_paths.wan_ckpt_dir are required")
	}
	if st, err := os.Stat(wanCkpt); err != nil || !st.IsDir() {
		return wanVideoJobPayload{}, fmt.Errorf("Wan checkpoint dir missing at %s — reinstall with: ./scripts/video/install_wan_video.sh --profile %s", wanCkpt, wanInstallProfile(vcfg.Profile))
	}
	if !wanCkptLooksPopulated(wanCkpt) {
		return wanVideoJobPayload{}, fmt.Errorf("Wan checkpoint dir at %s has no weight files (.pth/.safetensors) — reinstall with: ./scripts/video/install_wan_video.sh --profile %s", wanCkpt, wanInstallProfile(vcfg.Profile))
	}
	wanVenv := expandUserPath(modality.PathFor(cfg, "wan_venv"))
	if wanVenv == "" {
		// install_wan_video.sh uses $WAN_ROOT/venv with repos under $WAN_ROOT/Wan2.1
		wanVenv = filepath.Join(filepath.Dir(wanRepo), "venv")
	}
	pythonBin := filepath.Join(wanVenv, "bin", "python3")
	if !useWanC {
		if _, err := os.Stat(pythonBin); err != nil {
			return wanVideoJobPayload{}, fmt.Errorf("Wan venv python missing at %s — reinstall with: ./scripts/video/install_wan_video.sh --profile %s", pythonBin, wanInstallProfile(vcfg.Profile))
		}
	} else {
		// wan_c_generate.sh is bash; training run_script still needs a python_bin field — use system python3.
		if p, err := exec.LookPath("python3"); err == nil {
			pythonBin = p
		} else {
			pythonBin = "/usr/bin/python3"
		}
	}

	outputPath := videoArtifactPath("{job_id}")

	timeout := vcfg.TimeoutSec
	if t := envconfig.WanVideoTimeoutSec(); t > 0 {
		timeout = t
	}
	if timeout <= 0 {
		timeout = defaultWanTimeout(vcfg.Profile)
	}

	env := map[string]string{
		"WAN_PROFILE":            vcfg.Profile,
		"WAN_REPO":               wanRepo,
		"WAN_CKPT_DIR":           wanCkpt,
		"WAN_PROMPT":             prompt,
		"WAN_SIZE":               vcfg.Size,
		"WAN_FRAMES":             strconv.Itoa(vcfg.Frames),
		"WAN_STEPS":              strconv.Itoa(vcfg.Steps),
		"WAN_OFFLOAD_MODEL":      strconv.FormatBool(vcfg.OffloadModel),
		"WAN_T5_CPU":             strconv.FormatBool(vcfg.T5CPU),
		"WAN_FORCE_SDPA":         wanForceSDPA(vcfg),
		"WAN_UNLOAD_T5":          wanUnloadT5(vcfg),
		"WAN_VAE_CPU":            wanVAECPU(vcfg),
		"WAN_OUTPUT_PATH":        outputPath,
		"WAN_VENV":               wanVenv,
		"WAN_PYTHON":             pythonBin,
		"WAN_SUBPROCESS_TIMEOUT": strconv.Itoa(timeout),
		// Generic VIDEO_* aliases so future runners (RIFE) share the same contract.
		"VIDEO_OUTPUT_PATH": outputPath,
		"VIDEO_FRAMES":      strconv.Itoa(vcfg.Frames),
		"VIDEO_SIZE":        vcfg.Size,
	}
	if useWanC {
		env["WAN_CLI"] = wanCLI
		if v := expandUserPath(modality.PathFor(cfg, "wan_c_vocab")); v != "" {
			env["WAN_C_VOCAB"] = v
		}
		if v := strings.TrimSpace(envconfig.Var("UMA_SOCK")); v != "" {
			env["UMA_SOCK"] = v
		}
	}
	if seed != nil {
		env["WAN_SEED"] = strconv.FormatInt(*seed, 10)
		env["VIDEO_SEED"] = strconv.FormatInt(*seed, 10)
	}
	if gguf := expandUserPath(modality.PathFor(cfg, "wan_gguf_path")); gguf != "" {
		env["WAN_GGUF_PATH"] = gguf
	}
	if strings.HasPrefix(strings.ToLower(vcfg.Profile), "wan2.2") {
		env["WAN_CONVERT_MODEL_DTYPE"] = "true"
	}
	if keyframeDir != "" {
		env["VIDEO_KEYFRAME_DIR"] = keyframeDir
		env["WAN_KEYFRAME_DIR"] = keyframeDir
		// Wrapper removes the staging dir after success or failure.
		env["VIDEO_CLEANUP_KEYFRAME_DIR"] = "1"
		if entries, err := os.ReadDir(keyframeDir); err == nil {
			names := make([]string, 0, len(entries))
			for _, e := range entries {
				if !e.IsDir() {
					names = append(names, e.Name())
				}
			}
			sort.Strings(names)
			if len(names) > 0 {
				first := filepath.Join(keyframeDir, names[0])
				env["WAN_IMAGE"] = first
				env["VIDEO_IMAGE"] = first
			}
		}
	}

	payload := wanVideoJobPayload{
		ScriptPath:  scriptPath,
		PythonBin:   pythonBin,
		WorkingDir:  repo,
		Env:         env,
		Timeout:     timeout,
		OutputPath:  outputPath,
		SubmittedAt: submittedAt.UTC().Format(time.RFC3339),
		VideoModel:  modelName,
		VideoSize:   vcfg.Size,
	}
	return payload, nil
}

func defaultWanTimeout(profile string) int {
	switch profile {
	case wanProfile22TI2V5B:
		return 3600
	case wanProfile21T2V13B:
		return 2700
	default:
		return 2700
	}
}

// wanInstallProfile maps video_generation.profile to install_wan_video.sh --profile.
func wanInstallProfile(profile string) string {
	switch profile {
	case wanProfile22TI2V5B:
		return "2.2"
	case wanProfile21T2V13B:
		return "1.3b"
	default:
		return "all"
	}
}

// wanCkptLooksPopulated is true when the checkpoint dir has at least one weight file.
// Empty dirs (clone-only / deleted weights) otherwise pass os.Stat and fail deep in generate.py.
func wanCkptLooksPopulated(dir string) bool {
	ents, err := os.ReadDir(dir)
	if err != nil {
		return false
	}
	for _, e := range ents {
		if e.IsDir() {
			continue
		}
		n := strings.ToLower(e.Name())
		if strings.HasSuffix(n, ".pth") || strings.HasSuffix(n, ".safetensors") || strings.HasSuffix(n, ".pt") {
			return true
		}
	}
	return false
}

func videoArtifactRoot() string {
	return filepath.Join(envconfig.Models(), "generated")
}

// videoArtifactPath is under $OLLAMA_MODELS/generated so operators can prune one directory.
func videoArtifactPath(jobID string) string {
	return filepath.Join(videoArtifactRoot(), jobID+".mp4")
}

func expandUserPath(p string) string {
	p = strings.TrimSpace(p)
	if p == "" {
		return ""
	}
	if strings.HasPrefix(p, "~/") {
		home, err := os.UserHomeDir()
		if err != nil {
			return p
		}
		return filepath.Join(home, p[2:])
	}
	return p
}

// safeVideoArtifactPath rejects paths outside generated/ so a malicious resultJson cannot
// serve arbitrary files via GET /v1/videos/:id/content.
func safeVideoArtifactPath(path string) (string, error) {
	root := videoArtifactRoot()
	absRoot, err := filepath.Abs(root)
	if err != nil {
		return "", err
	}
	if err := os.MkdirAll(absRoot, 0o755); err != nil {
		return "", err
	}
	absPath, err := filepath.Abs(path)
	if err != nil {
		return "", err
	}
	rel, err := filepath.Rel(absRoot, absPath)
	if err != nil || strings.HasPrefix(rel, "..") {
		return "", errors.New("output path outside generated video directory")
	}
	if _, err := os.Stat(absPath); err != nil {
		return "", err
	}
	return absPath, nil
}

type trainingJobWire struct {
	Status     string `json:"status"`
	ResultJSON string `json:"resultJson"`
	Error      string `json:"error"`
}

// completedVideoOutputFromJob inspects a training job object and returns the output
// file path when the job has completed successfully. It also returns an HTTP status
// code for error responses:
//   - 202 when the job is still queued or running
//   - 410 when cancelled
//   - 502 when the result is malformed
//   - 500 when the job failed
func completedVideoOutputFromJob(jobJSON json.RawMessage) (path string, httpStatus int, err error) {
	var job trainingJobWire
	if err := json.Unmarshal(jobJSON, &job); err != nil {
		return "", http.StatusBadGateway, err
	}
	switch job.Status {
	case "completed":
		if job.ResultJSON != "" {
			var result map[string]any
			if err := json.Unmarshal([]byte(job.ResultJSON), &result); err == nil {
				if p, ok := result["output_path"].(string); ok && p != "" {
					return p, 0, nil
				}
			}
		}
		return "", http.StatusBadGateway, errors.New("job completed but no output_path in result")
	case "failed":
		msg := job.Error
		if msg == "" {
			msg = "video generation failed"
		}
		return "", http.StatusInternalServerError, errors.New(msg)
	case "pending", "running", "queued", "promoted":
		return "", http.StatusAccepted, errors.New("video is not ready yet; poll GET /v1/videos/:id until status is completed")
	case "cancelled":
		return "", http.StatusGone, errors.New("video job was cancelled")
	default:
		return "", http.StatusBadRequest, fmt.Errorf("video job status %q", job.Status)
	}
}

