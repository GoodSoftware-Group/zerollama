package server

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/gin-gonic/gin"

	"github.com/ollama/ollama/internal/slurm"
	"github.com/ollama/ollama/x/trainingworker"
)

type fakeSlurmCLI struct {
	out  map[string]string
	fail map[string]error
	cmds [][]string
}

func (f *fakeSlurmCLI) Run(_ context.Context, name string, args ...string) (string, string, error) {
	f.cmds = append(f.cmds, append([]string{name}, args...))
	if err := f.fail[name]; err != nil {
		msg := "batch job submission failed: Invalid partition name specified"
		return "", msg, err
	}
	return f.out[name], "", nil
}

func slurmTestRepo(t *testing.T) {
	t.Helper()
	repo := t.TempDir()
	if err := os.WriteFile(filepath.Join(repo, "training.py"), []byte("# test\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	t.Setenv("ZEROLLAMA_REPO", repo)
	t.Setenv("OLLAMA_TRAINING_PYTHONPATH", repo)
	t.Setenv("ZEROLLAMA_SLURM_SPOOL", t.TempDir())
	t.Setenv("ZEROLLAMA_SLURM_SUBMIT", "0")
	t.Setenv("ZEROLLAMA_SLURM_GUEST", "0")
	t.Setenv("SLURM_JOB_ID", "")
	t.Setenv("CUDA_VISIBLE_DEVICES", "")
	t.Setenv("ZEROLLAMA_RUNTIME_URL", "")
}

func TestTrainHTTPSubmitsToSlurm(t *testing.T) {
	gin.SetMode(gin.TestMode)
	slurmTestRepo(t)
	t.Setenv("ZEROLLAMA_SLURM_SUBMIT", "1")
	fake := &fakeSlurmCLI{out: map[string]string{
		"sbatch":   "48212\n",
		"scontrol": `{"jobs":[{"job_id":48212,"job_state":["RUNNING"],"partition":"gpu","time_left":{"number":90}}]}`,
	}}
	s := &Server{training: &trainingworker.Client{}, slurmCLI: &slurm.Client{Runner: fake}}
	r := gin.New()
	s.registerTrainingRoutes(r)

	body := []byte(`{"kind":"train","payload":{"model_name":"m"},"slurm":{"partition":"gpu","time":"01:00:00","gpus":1}}`)
	w := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodPost, "/api/train/jobs", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	r.ServeHTTP(w, req)
	if w.Code != http.StatusAccepted {
		t.Fatalf("status=%d body=%s", w.Code, w.Body.String())
	}
	var got map[string]any
	if err := json.Unmarshal(w.Body.Bytes(), &got); err != nil {
		t.Fatal(err)
	}
	if got["slurm_job_id"] != "48212" || got["state"] != "pending" || got["scheduler"] != "slurm" {
		t.Fatalf("%s", w.Body.String())
	}
	jobID, _ := got["job_id"].(string)
	if !isSlurmTrainingJobID(jobID) {
		t.Fatalf("job_id=%q", jobID)
	}
	joined := strings.Join(fake.cmds[0], " ")
	if !strings.Contains(joined, "--parsable") || !strings.Contains(joined, "--partition=gpu") || !strings.Contains(joined, "--gpus=1") {
		t.Fatalf("sbatch args: %s", joined)
	}
	if strings.Contains(joined, "--uid") {
		t.Fatalf("uid leaked: %s", joined)
	}
	script := ""
	for _, a := range fake.cmds[0][1:] {
		if strings.HasSuffix(a, "job.sh") {
			script = a
		}
	}
	b, err := os.ReadFile(script)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(b), "training.process_training_request") || strings.Contains(string(b), "zerollama serve") {
		t.Fatalf("script:\n%s", b)
	}
	info, err := os.Stat(filepath.Join(filepath.Dir(script), "payload.json"))
	if err != nil {
		t.Fatal(err)
	}
	if info.Mode().Perm() != 0o600 {
		t.Fatalf("payload mode %o", info.Mode().Perm())
	}

	w = httptest.NewRecorder()
	req = httptest.NewRequest(http.MethodGet, "/api/train/jobs/"+jobID, nil)
	r.ServeHTTP(w, req)
	if w.Code != http.StatusOK {
		t.Fatalf("status get %d %s", w.Code, w.Body.String())
	}
	if !strings.Contains(w.Body.String(), `"status":"running"`) || !strings.Contains(w.Body.String(), `"slurm_state":"RUNNING"`) {
		t.Fatalf("body %s", w.Body.String())
	}

	w = httptest.NewRecorder()
	req = httptest.NewRequest(http.MethodDelete, "/api/train/jobs/"+jobID, nil)
	r.ServeHTTP(w, req)
	if w.Code != http.StatusOK || !strings.Contains(w.Body.String(), `"cancelled":true`) {
		t.Fatalf("cancel %d %s", w.Code, w.Body.String())
	}
	if len(fake.cmds) == 0 || fake.cmds[len(fake.cmds)-1][0] != "scancel" {
		t.Fatalf("cmds %#v", fake.cmds)
	}
}

func TestTrainHTTPSlurmObjectNeedsSubmitFlag(t *testing.T) {
	gin.SetMode(gin.TestMode)
	slurmTestRepo(t)
	fake := &fakeSlurmCLI{out: map[string]string{"sbatch": "1\n"}}
	s := &Server{training: &trainingworker.Client{}, slurmCLI: &slurm.Client{Runner: fake}}
	r := gin.New()
	s.registerTrainingRoutes(r)
	w := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodPost, "/api/train/jobs", strings.NewReader(`{"kind":"train","slurm":{"time":"01:00:00"}}`))
	req.Header.Set("Content-Type", "application/json")
	r.ServeHTTP(w, req)
	if w.Code != http.StatusBadRequest || !strings.Contains(w.Body.String(), "ZEROLLAMA_SLURM_SUBMIT") {
		t.Fatalf("%d %s", w.Code, w.Body.String())
	}
	if len(fake.cmds) != 0 {
		t.Fatalf("sbatch ran: %#v", fake.cmds)
	}
}

func TestTrainHTTPSlurmRejectsMissingTime(t *testing.T) {
	gin.SetMode(gin.TestMode)
	slurmTestRepo(t)
	t.Setenv("ZEROLLAMA_SLURM_SUBMIT", "1")
	t.Setenv("ZEROLLAMA_SLURM_TIME", "")
	fake := &fakeSlurmCLI{}
	s := &Server{training: &trainingworker.Client{}, slurmCLI: &slurm.Client{Runner: fake}}
	r := gin.New()
	s.registerTrainingRoutes(r)
	w := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodPost, "/api/train/jobs", strings.NewReader(`{"kind":"train","slurm":{"partition":"gpu"}}`))
	req.Header.Set("Content-Type", "application/json")
	r.ServeHTTP(w, req)
	if w.Code != http.StatusBadRequest || !strings.Contains(w.Body.String(), "time") {
		t.Fatalf("%d %s", w.Code, w.Body.String())
	}
	if len(fake.cmds) != 0 {
		t.Fatalf("sbatch ran: %#v", fake.cmds)
	}
}

func TestTrainHTTPSlurmGuestGPUConflict(t *testing.T) {
	gin.SetMode(gin.TestMode)
	slurmTestRepo(t)
	t.Setenv("ZEROLLAMA_SLURM_SUBMIT", "1")
	t.Setenv("ZEROLLAMA_SLURM_GUEST", "auto")
	t.Setenv("SLURM_JOB_ID", "9")
	t.Setenv("CUDA_VISIBLE_DEVICES", "0")
	fake := &fakeSlurmCLI{out: map[string]string{"sbatch": "1\n"}}
	s := &Server{training: &trainingworker.Client{}, slurmCLI: &slurm.Client{Runner: fake}}
	r := gin.New()
	s.registerTrainingRoutes(r)
	w := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodPost, "/api/train/jobs", strings.NewReader(`{"kind":"train","slurm":{"time":"01:00:00"}}`))
	req.Header.Set("Content-Type", "application/json")
	r.ServeHTTP(w, req)
	if w.Code != http.StatusConflict {
		t.Fatalf("%d %s", w.Code, w.Body.String())
	}
	if len(fake.cmds) != 0 {
		t.Fatalf("sbatch ran: %#v", fake.cmds)
	}
}

func TestTrainHTTPSlurmRejectionIs400(t *testing.T) {
	gin.SetMode(gin.TestMode)
	slurmTestRepo(t)
	t.Setenv("ZEROLLAMA_SLURM_SUBMIT", "1")
	fake := &fakeSlurmCLI{fail: map[string]error{"sbatch": errors.New("exit status 1")}}
	s := &Server{training: &trainingworker.Client{}, slurmCLI: &slurm.Client{Runner: fake}}
	r := gin.New()
	s.registerTrainingRoutes(r)
	w := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodPost, "/api/train/jobs", strings.NewReader(`{"kind":"train","slurm":{"time":"00:10:00","partition":"nope"}}`))
	req.Header.Set("Content-Type", "application/json")
	r.ServeHTTP(w, req)
	if w.Code != http.StatusBadRequest || !strings.Contains(w.Body.String(), "Invalid partition") {
		t.Fatalf("%d %s", w.Code, w.Body.String())
	}
}

func TestDeferredSlurmPromoteDoesNotEmbed(t *testing.T) {
	slurmTestRepo(t)
	fake := &fakeSlurmCLI{out: map[string]string{"sbatch": "99\n"}}
	s := &Server{training: &trainingworker.Client{}, slurmCLI: &slurm.Client{Runner: fake}}
	s.trainingDefer = newTrainingDeferQueue(s)
	spec := slurm.Request{Time: "00:30:00", Partition: "gpu"}
	id, err := s.trainingDefer.enqueueSpec("train", []byte(`{"model_name":"m"}`), TrainingPriorityLow, &spec)
	if err != nil {
		t.Fatal(err)
	}
	s.trainingDefer.drainOnce(context.Background())
	j, ok := s.trainingDefer.lookupSnapshot(id)
	if !ok || j.state != deferredStatePromoted || !isSlurmTrainingJobID(j.promotedID) {
		t.Fatalf("job %+v ok=%v", j, ok)
	}
	if len(fake.cmds) != 1 || fake.cmds[0][0] != "sbatch" {
		t.Fatalf("cmds %#v", fake.cmds)
	}
}

func TestSlurmJobsReloadAfterRestart(t *testing.T) {
	slurmTestRepo(t)
	t.Setenv("ZEROLLAMA_SLURM_SUBMIT", "1")
	fake := &fakeSlurmCLI{out: map[string]string{"sbatch": "7\n"}}
	s := &Server{training: &trainingworker.Client{}, slurmCLI: &slurm.Client{Runner: fake}}
	res, err := s.submitTrainingJob(context.Background(), "train", []byte(`{"model_name":"m"}`), TrainingSubmitOptions{
		Slurm: &slurm.Request{Time: "00:10:00"},
	})
	if err != nil {
		t.Fatal(err)
	}
	again := &Server{slurmCLI: &slurm.Client{Runner: fake}}
	raw, err := again.slurmTrainingJobStatusJSON(context.Background(), res.JobID)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(raw), `"slurm_job_id":"7"`) {
		t.Fatalf("reloaded %s", raw)
	}
}

func TestSlurmStatusOnAPIStatus(t *testing.T) {
	slurmTestRepo(t)
	t.Setenv("ZEROLLAMA_SLURM_SUBMIT", "1")
	s := &Server{}
	st := s.statusResponse(context.Background())
	if st.Slurm == nil || !st.Slurm.Submit || st.Slurm.Transport != "sbatch" || st.Slurm.Guest {
		t.Fatalf("%+v", st.Slurm)
	}
}
