// Package trainingworker bridges the Ollama Go daemon to the Python GPU training process.
//
// Why Go fronts public TCP :9500 and HTTP /api/train (instead of Python):
//   - Single listener for policy, logging, and future auth; avoids port fights with inference.
// Why gRPC over a private Unix socket to Python:
//   - Strongly typed IPC, streaming events, and clear failure modes vs newline-JSON between processes.
// Why VRAMEvictor / runOOMBridge:
//   - On single-GPU machines, training OOM is coordinated inference-first: pause new loads, evict
//     llama runners, ack Python; defer ResumeLoads so inference never stays paused after an error path.
package trainingworker

import (
	"bufio"
	"context"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"

	"github.com/ollama/ollama/x/trainingworker/trainingpb"
)

// VRAMEvictor is implemented by server.Scheduler for inference-first VRAM coordination.
type VRAMEvictor interface {
	PauseNewLoads()
	UnloadAllRunners()
	ResumeLoads()
}

// Client wraps the gRPC connection to the training daemon subprocess.
type Client struct {
	closeOnce sync.Once
	grpc      trainingpb.TrainingDaemonClient
	conn     *grpc.ClientConn
	cmd      *exec.Cmd
	sockPath string
	evictor  VRAMEvictor

	oomCancel context.CancelFunc
	oomWG     sync.WaitGroup
}

// Start spawns python3 -m trainingdaemon on a private Unix socket and dials gRPC until Health succeeds.
//
// Why poll dial + Health instead of a single fixed sleep:
//   - Python import time (torch) is variable; we want fast success without brittle timeouts.
// Why a separate OOM stream goroutine when evictor is non-nil:
//   - OOM handling must not block every gRPC caller; the bridge serializes eviction + ack per event.
func Start(ctx context.Context, evictor VRAMEvictor) (*Client, error) {
	py, err := exec.LookPath("python3")
	if err != nil {
		return nil, fmt.Errorf("training worker: python3 not found: %w", err)
	}
	pythonPath := resolveTrainingPythonPath()
	if pythonPath == "" {
		return nil, errors.New("training worker: set OLLAMA_TRAINING_PYTHONPATH to the directory containing the trainingdaemon package (…/x/trainingdaemon)")
	}

	sock := filepath.Join(os.TempDir(), "ollama-training-"+randomHex(8)+".sock")
	_ = os.Remove(sock)

	cmd := exec.Command(py, "-m", "trainingdaemon", "--uds-path", sock)
	cmd.Env = append(os.Environ(), "PYTHONPATH="+pythonPath)
	cmd.Stderr = os.Stderr
	cmd.Stdout = os.Stdout
	if err := cmd.Start(); err != nil {
		return nil, fmt.Errorf("training worker: start python: %w", err)
	}

	dialCtx, cancel := context.WithTimeout(ctx, 45*time.Second)
	defer cancel()

	var conn *grpc.ClientConn
	var api trainingpb.TrainingDaemonClient
	for dialCtx.Err() == nil {
		var err error
		conn, err = grpc.NewClient(
			"unix://"+sock,
			grpc.WithTransportCredentials(insecure.NewCredentials()),
		)
		if err != nil {
			select {
			case <-dialCtx.Done():
				_ = cmd.Process.Kill()
				return nil, fmt.Errorf("training worker: dial: %w", err)
			case <-time.After(150 * time.Millisecond):
			}
			continue
		}
		api = trainingpb.NewTrainingDaemonClient(conn)
		_, err = api.Health(dialCtx, &trainingpb.HealthRequest{})
		if err == nil {
			break
		}
		_ = conn.Close()
		conn = nil
		api = nil
		select {
		case <-dialCtx.Done():
			_ = cmd.Process.Kill()
			return nil, fmt.Errorf("training worker: daemon not ready: %w", err)
		case <-time.After(150 * time.Millisecond):
		}
	}
	if conn == nil || api == nil {
		_ = cmd.Process.Kill()
		return nil, errors.New("training worker: daemon not ready (no connection)")
	}

	c := &Client{
		grpc:     api,
		conn:     conn,
		cmd:      cmd,
		sockPath: sock,
		evictor:  evictor,
	}
	if evictor != nil {
		oomCtx, oomCancel := context.WithCancel(context.Background())
		c.oomCancel = oomCancel
		c.oomWG.Add(1)
		go c.runOOMBridge(oomCtx)
	}
	return c, nil
}

// runOOMBridge consumes StreamEvents from Python. On OOM it asks the scheduler to free VRAM
// for training retries, then AckVRAMHeadroom. The inner func + defer ResumeLoads guarantee we
// never leave the scheduler paused if UnloadAllRunners panics or returns early.
func (c *Client) runOOMBridge(ctx context.Context) {
	defer c.oomWG.Done()
	stream, err := c.grpc.StreamEvents(ctx, &trainingpb.StreamEventsRequest{})
	if err != nil {
		slog.Debug("training OOM bridge stream failed", "error", err)
		return
	}
	for {
		ev, err := stream.Recv()
		if err != nil {
			if !errors.Is(err, io.EOF) && ctx.Err() == nil {
				slog.Debug("training stream recv ended", "error", err)
			}
			return
		}
		if oom := ev.GetOom(); oom != nil && c.evictor != nil {
			func() {
				c.evictor.PauseNewLoads()
				defer c.evictor.ResumeLoads()
				c.evictor.UnloadAllRunners()
				_, _ = c.grpc.AckVRAMHeadroom(context.Background(), &trainingpb.AckVRAMHeadroomRequest{JobId: ev.JobId})
			}()
		}
	}
}

// Close shuts down the daemon and removes the socket.
func (c *Client) Close() {
	c.closeOnce.Do(func() {
		if c.oomCancel != nil {
			c.oomCancel()
			c.oomWG.Wait()
		}
		if c.conn != nil {
			_, _ = c.grpc.Shutdown(context.Background(), &trainingpb.ShutdownRequest{})
			_ = c.conn.Close()
		}
		if c.cmd != nil && c.cmd.Process != nil {
			done := make(chan struct{})
			go func() {
				_ = c.cmd.Wait()
				close(done)
			}()
			select {
			case <-done:
			case <-time.After(25 * time.Second):
				_ = c.cmd.Process.Kill()
			}
		}
		if c.sockPath != "" {
			_ = os.Remove(c.sockPath)
		}
	})
}

// GRPC returns the underlying client for HTTP handlers.
func (c *Client) GRPC() trainingpb.TrainingDaemonClient { return c.grpc }

func randomHex(n int) string {
	b := make([]byte, n)
	_, _ = rand.Read(b)
	return hex.EncodeToString(b)
}

func resolveTrainingPythonPath() string {
	if p := strings.TrimSpace(os.Getenv("OLLAMA_TRAINING_PYTHONPATH")); p != "" {
		return filepath.Clean(p)
	}
	if exe, err := os.Executable(); err == nil {
		dir := filepath.Dir(exe)
		for _, rel := range []string{
			"../x/trainingdaemon",
			"../../x/trainingdaemon",
			"../../../x/trainingdaemon",
			"../../../../x/trainingdaemon",
		} {
			cand := filepath.Clean(filepath.Join(dir, rel))
			if hasDaemonMain(cand) {
				return cand
			}
		}
	}
	if wd, err := os.Getwd(); err == nil {
		cand := filepath.Join(wd, "x", "trainingdaemon")
		if hasDaemonMain(cand) {
			return cand
		}
	}
	return ""
}

func hasDaemonMain(dir string) bool {
	st, err := os.Stat(filepath.Join(dir, "trainingdaemon", "__main__.py"))
	return err == nil && !st.IsDir()
}

// ServePublicTCP accepts legacy newline-delimited JSON (same commands as historical training.py).
// ctx cancellation closes the listener. Why in Go: one public port, same process as inference scheduler.
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
		// Read idle deadline only while waiting for the next line. Clear before handlePublicRequest
		// so a synchronous multi-hour "train" command is not cut off by a connection-wide deadline.
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
		h, err := c.grpc.Health(ctx, &trainingpb.HealthRequest{})
		if err != nil {
			return map[string]any{"status": "error", "error": err.Error()}
		}
		out := map[string]any{"status": h.Status, "message": "pong"}
		if h.ExtrasJson != "" {
			var extras map[string]any
			if json.Unmarshal([]byte(h.ExtrasJson), &extras) == nil {
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
		kind := trainingpb.JobKind_JOB_KIND_TRAIN
		if jobCmd == "run_script" {
			kind = trainingpb.JobKind_JOB_KIND_RUN_SCRIPT
		}
		sj, err := c.grpc.SubmitJob(ctx, &trainingpb.SubmitJobRequest{Kind: kind, PayloadJson: string(payload)})
		if err != nil {
			return map[string]any{"status": "error", "error": err.Error()}
		}
		out := map[string]any{"status": "ok", "job_id": sj.JobId, "message": "queued"}
		if h, herr := c.grpc.Health(ctx, &trainingpb.HealthRequest{}); herr == nil && h.ExtrasJson != "" {
			var extras map[string]any
			if json.Unmarshal([]byte(h.ExtrasJson), &extras) == nil {
				out["queue"] = extras["queue"]
			}
		}
		return out
	case "job_status":
		jid, _ := req["job_id"].(string)
		if jid == "" {
			return map[string]any{"status": "error", "error": "job_id required"}
		}
		st, err := c.grpc.JobStatus(ctx, &trainingpb.JobStatusRequest{JobId: jid})
		if err != nil {
			return map[string]any{"status": "error", "error": err.Error()}
		}
		if st.Job == nil {
			return map[string]any{"status": "error", "error": "job not found"}
		}
		return map[string]any{"status": "ok", "job": jobInfoToLegacy(st.Job)}
	case "list_jobs":
		lj, err := c.grpc.ListJobs(ctx, &trainingpb.ListJobsRequest{})
		if err != nil {
			return map[string]any{"status": "error", "error": err.Error()}
		}
		jobs := make([]any, 0, len(lj.Jobs))
		for _, j := range lj.Jobs {
			jobs = append(jobs, jobInfoToLegacy(j))
		}
		h, _ := c.grpc.Health(ctx, &trainingpb.HealthRequest{})
		var queue any
		if h != nil && h.ExtrasJson != "" {
			var extras map[string]any
			if json.Unmarshal([]byte(h.ExtrasJson), &extras) == nil {
				queue = extras["queue"]
			}
		}
		return map[string]any{"status": "ok", "jobs": jobs, "queue": queue}
	case "cancel_job":
		jid, _ := req["job_id"].(string)
		if jid == "" {
			return map[string]any{"status": "error", "error": "job_id required"}
		}
		cr, err := c.grpc.CancelJob(ctx, &trainingpb.CancelJobRequest{JobId: jid})
		if err != nil {
			return map[string]any{"status": "error", "error": err.Error()}
		}
		if cr.Cancelled {
			return map[string]any{"status": "ok", "message": "cancelled"}
		}
		return map[string]any{"status": "error", "error": "cannot cancel job"}
	case "queue_status":
		h, err := c.grpc.Health(ctx, &trainingpb.HealthRequest{})
		if err != nil {
			return map[string]any{"status": "error", "error": err.Error()}
		}
		var extras map[string]any
		_ = json.Unmarshal([]byte(h.ExtrasJson), &extras)
		q, _ := extras["queue"].(map[string]any)
		return map[string]any{"status": "ok", "queue": q}
	case "train":
		data, _ := req["data"].(map[string]any)
		if data == nil {
			data = map[string]any{}
		}
		return c.runSyncTrainJob(ctx, trainingpb.JobKind_JOB_KIND_TRAIN, data)
	case "run_script":
		data, _ := req["data"].(map[string]any)
		if data == nil {
			data = map[string]any{}
		}
		return c.runSyncTrainJob(ctx, trainingpb.JobKind_JOB_KIND_RUN_SCRIPT, data)
	case "unload":
		_, err := c.grpc.Unload(ctx, &trainingpb.UnloadRequest{})
		if err != nil {
			return map[string]any{"status": "error", "error": err.Error()}
		}
		return map[string]any{"status": "ok", "message": "Model unloaded"}
	case "shutdown":
		_, _ = c.grpc.Shutdown(ctx, &trainingpb.ShutdownRequest{})
		return map[string]any{"status": "ok", "message": "Shutting down"}
	default:
		return map[string]any{"status": "error", "error": fmt.Sprintf("unknown command: %q", cmd)}
	}
}

func (c *Client) runSyncTrainJob(ctx context.Context, kind trainingpb.JobKind, data map[string]any) map[string]any {
	payload, _ := json.Marshal(data)
	sj, err := c.grpc.SubmitJob(ctx, &trainingpb.SubmitJobRequest{Kind: kind, PayloadJson: string(payload)})
	if err != nil {
		return map[string]any{"status": "error", "error": err.Error()}
	}
	jid := sj.JobId
	deadline := time.After(24 * time.Hour)
	for {
		select {
		case <-deadline:
			return map[string]any{"status": "error", "error": "timeout waiting for job"}
		case <-ctx.Done():
			return map[string]any{"status": "error", "error": ctx.Err().Error()}
		case <-time.After(300 * time.Millisecond):
		}
		st, err := c.grpc.JobStatus(ctx, &trainingpb.JobStatusRequest{JobId: jid})
		if err != nil {
			return map[string]any{"status": "error", "error": err.Error()}
		}
		switch st.Job.Status {
		case "completed":
			var result map[string]any
			if st.Job.ResultJson != "" {
				_ = json.Unmarshal([]byte(st.Job.ResultJson), &result)
			}
			if result == nil {
				return map[string]any{"status": "ok"}
			}
			if _, ok := result["status"]; !ok {
				result["status"] = "ok"
			}
			return result
		case "failed", "cancelled":
			return map[string]any{"status": "error", "error": st.Job.Error}
		}
	}
}

func jobInfoToLegacy(j *trainingpb.JobInfo) map[string]any {
	if j == nil {
		return map[string]any{}
	}
	cmd := "train"
	if j.Kind == trainingpb.JobKind_JOB_KIND_RUN_SCRIPT {
		cmd = "run_script"
	}
	var result any
	if j.ResultJson != "" {
		_ = json.Unmarshal([]byte(j.ResultJson), &result)
	}
	return map[string]any{
		"id":                 j.JobId,
		"cmd":                cmd,
		"status":             j.Status,
		"progress":           j.Progress,
		"progress_message":   j.ProgressMessage,
		"result":             result,
		"error":              j.Error,
		"submitted_at":       j.SubmittedAt,
		"started_at":         j.StartedAt,
		"completed_at":       j.CompletedAt,
	}
}
