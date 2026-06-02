// Package trainingworker bridges the Ollama Go daemon to embedded CPython for GPU training.
//
// Why embedded Python (CGO) instead of a subprocess + gRPC:
//   - One process: no UDS socket, no grpcio, no python3 on PATH for the daemon.
//   - Same VRAM / OOM coordination via a C callback from torch into Go.
//
// Why Go still fronts public TCP :9500 and HTTP /api/train:
//   - Single listener for policy, logging, and future auth.
//
// Start order matters: check InitAborted/IsInitialized before RegisterOOMHandler so a failed
// second Start returns an error without replacing the OOM handler for an already-live interpreter.
// RegisterOOMHandler runs immediately before InitEmbeddedPython; on init error we clear the handler
// so the next process Start does not inherit a closure tied to a failed attempt.
package trainingworker

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"net"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/ollama/ollama/server/vram"
	"github.com/ollama/ollama/x/trainingworker/pyembed"
)

// VRAMEvictor is implemented by server.Scheduler for inference-first VRAM coordination.
type VRAMEvictor interface {
	PauseNewLoads()
	UnloadAllRunners()
	ResumeLoads()
}

// InferenceSubmitGuard rejects training job submit when inference is still busy (T6).
type InferenceSubmitGuard func(ctx context.Context) error

// SubmitRequest is passed to SubmitHandler (HTTP/TCP training submit).
type SubmitRequest struct {
	Kind        string
	Payload     json.RawMessage
	Priority    string
	QueueOnBusy *bool
}

// SubmitResponse is returned from SubmitHandler.
type SubmitResponse struct {
	JobID  string
	Queued bool
}

// SubmitHandler implements server-side idle-wait and defer-queue policy.
type SubmitHandler func(ctx context.Context, req SubmitRequest) (SubmitResponse, error)

// DeferredJobStatusFn returns JSON job status for Go-side deferred training IDs.
type DeferredJobStatusFn func(ctx context.Context, id string) ([]byte, error)

// DeferredJobCancelFn cancels a waiting defer-* training job.
type DeferredJobCancelFn func(id string) (bool, error)

// DeferredListMergeFn appends defer-queue jobs to a Python list_jobs JSON blob.
type DeferredListMergeFn func(pythonList []byte) ([]byte, error)

// Client owns the embedded Python training interpreter lifecycle.
type Client struct {
	closeOnce sync.Once
	evictor   VRAMEvictor

	submitGuard      InferenceSubmitGuard
	submitFn         SubmitHandler
	deferStatusFn    DeferredJobStatusFn
	deferCancelFn    DeferredJobCancelFn
	deferListMergeFn DeferredListMergeFn

	gpuMu        sync.Mutex
	gpuAt        time.Time
	gpuStatus    TrainingGPUStatus
	gpuBusy      bool
	gpuHealthErr error
}

// Start initializes embedded CPython and repo-root training.py.
// On init failure, clears the OOM handler so a dangling closure never replaces a later successful Start's handler.
func Start(ctx context.Context, evictor VRAMEvictor) (*Client, error) {
	_ = ctx
	repo, err := resolveTrainingRepoRoot()
	if err != nil {
		return nil, err
	}
	if repo == "" {
		return nil, errors.New("training worker: set OLLAMA_TRAINING_PYTHONPATH (or ZEROLLAMA_REPO) to the repository root containing training.py, or use ~/zerollama")
	}
	if pyembed.InitAborted() {
		return nil, errors.New("training worker: embedded Python failed to start earlier; restart the process")
	}
	if pyembed.IsInitialized() {
		return nil, errors.New("training worker: already started")
	}
	pyembed.RegisterOOMHandler(func(jobID, msg string) {
		_ = msg
		if evictor == nil {
			return
		}
		vram.PrepareForTraining(context.Background(), evictor)
		pyembed.AckVRAMHeadroom(jobID)
	})
	if err := pyembed.InitEmbeddedPython(repo); err != nil {
		pyembed.RegisterOOMHandler(nil)
		return nil, fmt.Errorf("training worker: %w", err)
	}
	slog.Info("training worker started", "training_py_root", repo)
	return &Client{evictor: evictor}, nil
}

// SetInferenceSubmitGuard registers a callback from server (ggml + runtime backlog).
func (c *Client) SetInferenceSubmitGuard(fn InferenceSubmitGuard) {
	if c == nil {
		return
	}
	c.submitGuard = fn
}

// SetSubmitHandler registers server policy for training submit (priority, defer queue).
func (c *Client) SetSubmitHandler(fn SubmitHandler) {
	if c == nil {
		return
	}
	c.submitFn = fn
}

// SetDeferredJobStatusFn resolves defer-* job IDs for TCP/HTTP status.
func (c *Client) SetDeferredJobStatusFn(fn DeferredJobStatusFn) {
	if c == nil {
		return
	}
	c.deferStatusFn = fn
}

// SetDeferredJobCancelFn cancels defer-* jobs for TCP/HTTP.
func (c *Client) SetDeferredJobCancelFn(fn DeferredJobCancelFn) {
	if c == nil {
		return
	}
	c.deferCancelFn = fn
}

// SetDeferredListMergeFn merges defer jobs into list_jobs responses.
func (c *Client) SetDeferredListMergeFn(fn DeferredListMergeFn) {
	if c == nil {
		return
	}
	c.deferListMergeFn = fn
}

// Close stops the Python job processor (no Py_Finalize — unsafe with torch).
// Clears the OOM handler after Shutdown so a late native callback does not touch a stale evictor.
func (c *Client) Close() {
	c.closeOnce.Do(func() {
		pyembed.Shutdown()
		pyembed.RegisterOOMHandler(nil)
	})
}

// --- HTTP / legacy TCP helpers (JSON shapes compatible with former trainingpb protojson) ---

func (c *Client) trainSubmitJob(ctx context.Context, kind string, payload json.RawMessage) (jobID string, err error) {
	res, err := c.trainSubmitJobWithOpts(ctx, kind, payload, SubmitRequest{})
	if err != nil {
		return "", err
	}
	return res.JobID, nil
}

func (c *Client) trainSubmitJobWithOpts(
	ctx context.Context,
	kind string,
	payload json.RawMessage,
	opts SubmitRequest,
) (SubmitResponse, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	opts.Kind = kind
	opts.Payload = payload
	if c.submitFn != nil {
		return c.submitFn(ctx, opts)
	}
	if c.submitGuard != nil {
		if err := c.submitGuard(ctx); err != nil {
			return SubmitResponse{}, err
		}
	}
	id, err := c.submitTrainingDirect(ctx, kind, payload)
	if err != nil {
		return SubmitResponse{}, err
	}
	return SubmitResponse{JobID: id}, nil
}

// SubmitTrainingJobDirect submits to Python without idle-wait (server defer worker only).
func (c *Client) SubmitTrainingJobDirect(ctx context.Context, kind string, payload json.RawMessage) (string, error) {
	return c.submitTrainingDirect(ctx, kind, payload)
}

func (c *Client) submitTrainingDirect(ctx context.Context, kind string, payload json.RawMessage) (string, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	vram.PrepareForTraining(ctx, c.evictor)
	pl := string(payload)
	if pl == "" || pl == "null" {
		pl = "{}"
	}
	s, err := pyembed.SubmitJobJSON(kind, pl)
	if err != nil {
		return "", err
	}
	var out struct {
		JobID string `json:"jobId"`
		Error string `json:"error"`
	}
	if e := json.Unmarshal([]byte(s), &out); e != nil {
		return "", e
	}
	if out.Error != "" {
		return "", errors.New(out.Error)
	}
	return out.JobID, nil
}

func (c *Client) trainListJobsJSON(_ context.Context) ([]byte, error) {
	s, err := pyembed.ListJobsJSON()
	if err != nil {
		return nil, err
	}
	return []byte(s), nil
}

func (c *Client) trainJobStatusJSON(_ context.Context, id string) ([]byte, error) {
	s, err := pyembed.JobStatusJSON(id)
	if err != nil {
		return nil, err
	}
	var wrap struct {
		Error string          `json:"error"`
		Job   json.RawMessage `json:"job"`
	}
	if err := json.Unmarshal([]byte(s), &wrap); err != nil {
		return nil, err
	}
	if wrap.Error == "not_found" || wrap.Job == nil {
		return nil, ErrJobNotFound
	}
	return []byte(s), nil
}

var ErrJobNotFound = errors.New("job not found")

func (c *Client) trainCancelJob(_ context.Context, id string) (bool, error) {
	_ = c
	return pyembed.CancelJob(id)
}

func (c *Client) trainUnload(_ context.Context) error {
	_ = c
	return pyembed.Unload()
}

func (c *Client) trainHealthJSON(_ context.Context) ([]byte, error) {
	s, err := pyembed.HealthJSON()
	if err != nil {
		return nil, err
	}
	return []byte(s), nil
}

// OccupiesGPU reports whether training is running or still holds weights on the GPU.
func (c *Client) OccupiesGPU(ctx context.Context) (TrainingGPUStatus, bool) {
	if c == nil {
		return TrainingGPUStatus{}, false
	}
	ttl := trainingGPUStatusTTL()
	c.gpuMu.Lock()
	if !c.gpuAt.IsZero() && time.Since(c.gpuAt) < ttl {
		st, busy := c.gpuStatus, c.gpuBusy
		c.gpuMu.Unlock()
		return st, busy
	}
	c.gpuMu.Unlock()

	st, busy, healthErr := c.refreshGPUStatus(ctx)

	c.gpuMu.Lock()
	c.gpuAt = time.Now()
	c.gpuStatus = st
	c.gpuBusy = busy
	c.gpuHealthErr = healthErr
	c.gpuMu.Unlock()

	return st, busy
}

func (c *Client) refreshGPUStatus(ctx context.Context) (TrainingGPUStatus, bool, error) {
	raw, err := c.trainHealthJSON(ctx)
	if err != nil {
		return TrainingGPUStatus{}, false, err
	}
	var h struct {
		ExtrasJSON string `json:"extrasJson"`
	}
	if err := json.Unmarshal(raw, &h); err != nil {
		return TrainingGPUStatus{}, false, err
	}
	if h.ExtrasJSON == "" {
		return TrainingGPUStatus{}, false, errors.New("training health: missing extrasJson")
	}
	st, busy := trainingOccupiesGPU(h.ExtrasJSON)
	return st, busy, nil
}

type jsonJobInfo struct {
	JobID           string  `json:"jobId"`
	Status          string  `json:"status"`
	ResultJSON      string  `json:"resultJson"`
	Error           string  `json:"error"`
	Progress        float64 `json:"progress"`
	ProgressMessage string  `json:"progressMessage"`
}

type jsonJobStatusWrap struct {
	Job *jsonJobInfo `json:"job"`
}

func (c *Client) runSyncTrainJob(ctx context.Context, kind string, data map[string]any) map[string]any {
	payload, _ := json.Marshal(data)
	jobID, err := c.trainSubmitJob(ctx, kind, payload)
	if err != nil {
		return map[string]any{"status": "error", "error": err.Error()}
	}
	deadline := time.After(24 * time.Hour)
	for {
		select {
		case <-deadline:
			return map[string]any{"status": "error", "error": "timeout waiting for job"}
		case <-ctx.Done():
			return map[string]any{"status": "error", "error": ctx.Err().Error()}
		case <-time.After(300 * time.Millisecond):
		}
		raw, err := pyembed.JobStatusJSON(jobID)
		if err != nil {
			return map[string]any{"status": "error", "error": err.Error()}
		}
		var wrap jsonJobStatusWrap
		if err := json.Unmarshal([]byte(raw), &wrap); err != nil {
			return map[string]any{"status": "error", "error": err.Error()}
		}
		if wrap.Job == nil {
			return map[string]any{"status": "error", "error": "job not found"}
		}
		switch wrap.Job.Status {
		case "completed":
			var result map[string]any
			if wrap.Job.ResultJSON != "" {
				_ = json.Unmarshal([]byte(wrap.Job.ResultJSON), &result)
			}
			if result == nil {
				return map[string]any{"status": "ok"}
			}
			if _, ok := result["status"]; !ok {
				result["status"] = "ok"
			}
			return result
		case "failed", "cancelled":
			return map[string]any{"status": "error", "error": wrap.Job.Error}
		}
	}
}

// resolveTrainingRepoRoot finds the directory containing training.py.
// Explicit env (OLLAMA_TRAINING_PYTHONPATH, ZEROLLAMA_REPO) must contain training.py or Start fails.
// Otherwise: next to/walk from binary → walk cwd → ~/zerollama ~/ollama.
// Why: a typo in env must not silently fall through to a different checkout.
func resolveTrainingRepoRoot() (string, error) {
	for _, key := range []string{"OLLAMA_TRAINING_PYTHONPATH", "ZEROLLAMA_REPO"} {
		if p := strings.TrimSpace(os.Getenv(key)); p != "" {
			c := filepath.Clean(p)
			if hasTrainingPy(c) {
				return c, nil
			}
			return "", fmt.Errorf("training worker: %s=%q does not contain training.py", key, p)
		}
	}
	if exe, err := os.Executable(); err == nil {
		dir := filepath.Dir(exe)
		if hasTrainingPy(dir) {
			return dir, nil
		}
		for _, rel := range []string{"..", "../..", "../../..", "../../../.."} {
			cand := filepath.Clean(filepath.Join(dir, rel))
			if hasTrainingPy(cand) {
				return cand, nil
			}
		}
	}
	if wd, err := os.Getwd(); err == nil {
		for dir := wd; ; dir = filepath.Dir(dir) {
			if hasTrainingPy(dir) {
				return dir, nil
			}
			parent := filepath.Dir(dir)
			if parent == dir {
				break
			}
		}
	}
	if home, err := os.UserHomeDir(); err == nil {
		for _, name := range []string{"zerollama", "ollama"} {
			cand := filepath.Join(home, name)
			if hasTrainingPy(cand) {
				return cand, nil
			}
		}
	}
	return "", nil
}

func hasTrainingPy(dir string) bool {
	st, err := os.Stat(filepath.Join(dir, "training.py"))
	return err == nil && !st.IsDir()
}

// RepoRoot returns the directory containing training.py (same resolution as Start).
func RepoRoot() (string, error) {
	return resolveTrainingRepoRoot()
}

// SubmitTrainingJob queues a training job (HTTP /api/train/jobs).
func (c *Client) SubmitTrainingJob(ctx context.Context, kind string, payload json.RawMessage) (string, error) {
	return c.trainSubmitJob(ctx, kind, payload)
}

// SubmitTrainingJobWithPolicy submits with priority / queue_on_busy (HTTP/TCP).
func (c *Client) SubmitTrainingJobWithPolicy(ctx context.Context, req SubmitRequest) (SubmitResponse, error) {
	return c.trainSubmitJobWithOpts(ctx, req.Kind, req.Payload, req)
}

// ListTrainingJobsJSON returns JSON {"jobs":[...]} (proto-compatible).
func (c *Client) ListTrainingJobsJSON(ctx context.Context) ([]byte, error) {
	return c.trainListJobsJSON(ctx)
}

// JobTrainingStatusJSON returns JSON for one job or errJobNotFound.
func (c *Client) JobTrainingStatusJSON(ctx context.Context, id string) ([]byte, error) {
	return c.trainJobStatusJSON(ctx, id)
}

// CancelTrainingJob cancels a pending job.
func (c *Client) CancelTrainingJob(ctx context.Context, id string) (bool, error) {
	return c.trainCancelJob(ctx, id)
}

// UnloadTrainingModel frees GPU memory on the Python side.
func (c *Client) UnloadTrainingModel(ctx context.Context) error {
	return c.trainUnload(ctx)
}

// TrainingHealthJSON returns JSON {"status","extrasJson"}.
func (c *Client) TrainingHealthJSON(ctx context.Context) ([]byte, error) {
	return c.trainHealthJSON(ctx)
}

// ServePublicTCP accepts legacy newline-delimited JSON (same commands as historical training.py).
func (c *Client) ServePublicTCP(ctx context.Context, addr string) error {
	ln, err := net.Listen("tcp", addr)
	if err != nil {
		return err
	}
	slog.Info("training public TCP listening", "addr", ln.Addr().String())

	go func() {
		<-ctx.Done()
		_ = ln.Close()
	}()

	for {
		conn, err := ln.Accept()
		if err != nil {
			if ctx.Err() != nil {
				return ctx.Err()
			}
			continue
		}
		go c.handlePublicConn(ctx, conn)
	}
}

func (c *Client) handlePublicConn(ctx context.Context, conn net.Conn) {
	defer conn.Close()
	br := bufio.NewReader(conn)
	for {
		_ = conn.SetReadDeadline(time.Now().Add(10 * time.Minute))
		line, err := br.ReadString('\n')
		if err != nil {
			return
		}
		_ = conn.SetReadDeadline(time.Time{})
		_ = conn.SetWriteDeadline(time.Time{})

		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		var req map[string]any
		if err := json.Unmarshal([]byte(line), &req); err != nil {
			resp := map[string]any{"type": "result", "status": "error", "error": fmt.Sprintf("invalid JSON: %v", err)}
			b, _ := json.Marshal(resp)
			if _, werr := conn.Write(append(b, '\n')); werr != nil {
				return
			}
			continue
		}
		resp := c.handlePublicRequest(ctx, req)
		resp["type"] = "result"
		b, err := json.Marshal(resp)
		if err != nil {
			return
		}
		if _, err := conn.Write(append(b, '\n')); err != nil {
			return
		}
	}
}

func (c *Client) handlePublicRequest(ctx context.Context, req map[string]any) map[string]any {
	cmd, _ := req["cmd"].(string)
	switch cmd {
	case "ping":
		raw, err := c.trainHealthJSON(ctx)
		if err != nil {
			return map[string]any{"status": "error", "error": err.Error()}
		}
		var h struct {
			Status     string          `json:"status"`
			ExtrasJSON string          `json:"extrasJson"`
			Extras     json.RawMessage `json:"extras"`
		}
		if err := json.Unmarshal(raw, &h); err != nil {
			return map[string]any{"status": "error", "error": err.Error()}
		}
		out := map[string]any{"status": h.Status, "message": "pong"}
		if h.ExtrasJSON != "" {
			var extras map[string]any
			if json.Unmarshal([]byte(h.ExtrasJSON), &extras) == nil {
				for k, v := range extras {
					out[k] = v
				}
			}
		}
		return out
	case "submit_job":
		jobCmd, _ := req["job_cmd"].(string)
		if jobCmd == "" {
			jobCmd = "train"
		}
		data, _ := req["data"].(map[string]any)
		if data == nil {
			data = map[string]any{}
		}
		payload, _ := json.Marshal(data)
		priority, _ := req["priority"].(string)
		var queueOnBusy *bool
		if v, ok := req["queue_on_busy"].(bool); ok {
			queueOnBusy = &v
		}
		sub, err := c.trainSubmitJobWithOpts(ctx, jobCmd, payload, SubmitRequest{
			Priority:    priority,
			QueueOnBusy: queueOnBusy,
		})
		if err != nil {
			return map[string]any{"status": "error", "error": err.Error()}
		}
		out := map[string]any{"status": "ok", "job_id": sub.JobID, "message": "queued"}
		if sub.Queued {
			out["defer"] = true
		}
		if raw, herr := c.trainHealthJSON(ctx); herr == nil {
			var h struct {
				ExtrasJSON string `json:"extrasJson"`
			}
			if json.Unmarshal(raw, &h) == nil && h.ExtrasJSON != "" {
				var extras map[string]any
				if json.Unmarshal([]byte(h.ExtrasJSON), &extras) == nil {
					out["queue"] = extras["queue"]
				}
			}
		}
		return out
	case "job_status":
		jid, _ := req["job_id"].(string)
		if jid == "" {
			return map[string]any{"status": "error", "error": "job_id required"}
		}
		if c.deferStatusFn != nil {
			if raw, err := c.deferStatusFn(ctx, jid); err == nil {
				var wrap struct {
					Job map[string]any `json:"job"`
				}
				if json.Unmarshal(raw, &wrap) == nil && wrap.Job != nil {
					return map[string]any{"status": "ok", "job": wrap.Job}
				}
			}
		}
		raw, err := c.trainJobStatusJSON(ctx, jid)
		if err != nil {
			if errors.Is(err, ErrJobNotFound) {
				return map[string]any{"status": "error", "error": "job not found"}
			}
			return map[string]any{"status": "error", "error": err.Error()}
		}
		var wrap struct {
			Job map[string]any `json:"job"`
		}
		if err := json.Unmarshal(raw, &wrap); err != nil {
			return map[string]any{"status": "error", "error": err.Error()}
		}
		return map[string]any{"status": "ok", "job": wrap.Job}
	case "list_jobs":
		raw, err := c.trainListJobsJSON(ctx)
		if err != nil {
			return map[string]any{"status": "error", "error": err.Error()}
		}
		if c.deferListMergeFn != nil {
			if merged, merr := c.deferListMergeFn(raw); merr == nil {
				raw = merged
			}
		}
		var lj struct {
			Jobs []map[string]any `json:"jobs"`
		}
		if err := json.Unmarshal(raw, &lj); err != nil {
			return map[string]any{"status": "error", "error": err.Error()}
		}
		var queue any
		if hraw, herr := c.trainHealthJSON(ctx); herr == nil {
			var h struct {
				ExtrasJSON string `json:"extrasJson"`
			}
			if json.Unmarshal(hraw, &h) == nil && h.ExtrasJSON != "" {
				var extras map[string]any
				if json.Unmarshal([]byte(h.ExtrasJSON), &extras) == nil {
					queue = extras["queue"]
				}
			}
		}
		return map[string]any{"status": "ok", "jobs": lj.Jobs, "queue": queue}
	case "cancel_job":
		jid, _ := req["job_id"].(string)
		if jid == "" {
			return map[string]any{"status": "error", "error": "job_id required"}
		}
		if c.deferCancelFn != nil && strings.HasPrefix(jid, "defer-") {
			cr, err := c.deferCancelFn(jid)
			if err != nil {
				return map[string]any{"status": "error", "error": err.Error()}
			}
			if cr {
				return map[string]any{"status": "ok", "message": "cancelled"}
			}
			return map[string]any{"status": "error", "error": "cannot cancel job"}
		}
		cr, err := c.trainCancelJob(ctx, jid)
		if err != nil {
			return map[string]any{"status": "error", "error": err.Error()}
		}
		if cr {
			return map[string]any{"status": "ok", "message": "cancelled"}
		}
		return map[string]any{"status": "error", "error": "cannot cancel job"}
	case "queue_status":
		raw, err := c.trainHealthJSON(ctx)
		if err != nil {
			return map[string]any{"status": "error", "error": err.Error()}
		}
		var h struct {
			ExtrasJSON string `json:"extrasJson"`
		}
		_ = json.Unmarshal(raw, &h)
		var extras map[string]any
		_ = json.Unmarshal([]byte(h.ExtrasJSON), &extras)
		q, _ := extras["queue"].(map[string]any)
		return map[string]any{"status": "ok", "queue": q}
	case "train":
		data, _ := req["data"].(map[string]any)
		if data == nil {
			data = map[string]any{}
		}
		return c.runSyncTrainJob(ctx, "train", data)
	case "run_script":
		data, _ := req["data"].(map[string]any)
		if data == nil {
			data = map[string]any{}
		}
		return c.runSyncTrainJob(ctx, "run_script", data)
	case "unload":
		if err := c.trainUnload(ctx); err != nil {
			return map[string]any{"status": "error", "error": err.Error()}
		}
		return map[string]any{"status": "ok", "message": "Model unloaded"}
	case "shutdown":
		pyembed.Shutdown()
		return map[string]any{"status": "ok", "message": "Shutting down"}
	default:
		return map[string]any{"status": "error", "error": fmt.Sprintf("unknown command: %q", cmd)}
	}
}
