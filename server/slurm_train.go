// Slurm queue client (SL2): POST /api/train/jobs with a slurm object, or
// ZEROLLAMA_SLURM_SUBMIT=1, runs sbatch on this host and does not start the
// embedded trainer. State is polled from scontrol / squeue / sacct. DELETE
// calls scancel while the cluster job is still pending or running.
package server

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/internal/slurm"
	"github.com/ollama/ollama/x/trainingworker"
)

const slurmJobIDPrefix = "slurm-"

// errSlurmGuestGPU refuses a second cluster submit from an allocation that
// already has GPUs. That job should run the work itself.
var errSlurmGuestGPU = errors.New("refusing slurm submit from a guest allocation that already has GPUs")

func isSlurmTrainingJobID(id string) bool {
	return strings.HasPrefix(id, slurmJobIDPrefix)
}

type slurmJobRecord struct {
	LocalID     string
	SlurmID     string
	Kind        string
	Priority    string
	Partition   string
	Status      string
	State       string
	SlurmState  string
	TimeLeft    string
	Message     string
	SubmittedAt time.Time
}

type slurmJobStore struct {
	mu          sync.Mutex
	jobs        map[string]*slurmJobRecord
	order       []string
	path        string
	refreshedAt time.Time
}

func (st *slurmJobStore) add(rec *slurmJobRecord) {
	st.mu.Lock()
	defer st.mu.Unlock()
	if st.jobs == nil {
		st.jobs = make(map[string]*slurmJobRecord)
	}
	st.jobs[rec.LocalID] = rec
	st.order = append(st.order, rec.LocalID)
	st.saveLocked()
}

func (st *slurmJobStore) get(id string) (slurmJobRecord, bool) {
	st.mu.Lock()
	defer st.mu.Unlock()
	rec, ok := st.jobs[id]
	if !ok {
		return slurmJobRecord{}, false
	}
	return *rec, true
}

func (st *slurmJobStore) nonTerminal() []slurmJobRecord {
	st.mu.Lock()
	defer st.mu.Unlock()
	var out []slurmJobRecord
	for _, id := range st.order {
		rec := st.jobs[id]
		if rec == nil || slurmStatusTerminal(rec.Status) {
			continue
		}
		out = append(out, *rec)
	}
	return out
}

func (st *slurmJobStore) apply(id string, v slurm.View) {
	mapped := slurm.MapState(v.State)
	st.mu.Lock()
	defer st.mu.Unlock()
	rec := st.jobs[id]
	if rec == nil {
		return
	}
	rec.SlurmState = v.State
	rec.Status = mapped.Status
	rec.State = mapped.State
	if v.Partition != "" {
		rec.Partition = v.Partition
	}
	if v.TimeLeft != "" {
		rec.TimeLeft = v.TimeLeft
	}
	if v.Reason != "" {
		rec.Message = v.Reason
	}
	if mapped.Status == "failed" && rec.Message == "" {
		rec.Message = v.State
	}
	st.saveLocked()
}

func (st *slurmJobStore) markCancelled(id string) {
	st.mu.Lock()
	defer st.mu.Unlock()
	rec := st.jobs[id]
	if rec == nil {
		return
	}
	rec.Status = "cancelled"
	rec.State = "cancelled"
	rec.SlurmState = "CANCELLED"
	st.saveLocked()
}

func (st *slurmJobStore) fresh(d time.Duration) bool {
	st.mu.Lock()
	defer st.mu.Unlock()
	return !st.refreshedAt.IsZero() && time.Since(st.refreshedAt) < d
}

func (st *slurmJobStore) markRefreshed() {
	st.mu.Lock()
	st.refreshedAt = time.Now()
	st.mu.Unlock()
}

func (st *slurmJobStore) saveLocked() {
	if st == nil || st.path == "" {
		return
	}
	recs := make([]slurmJobRecord, 0, len(st.order))
	for _, id := range st.order {
		if rec := st.jobs[id]; rec != nil {
			recs = append(recs, *rec)
		}
	}
	b, err := json.Marshal(recs)
	if err != nil {
		return
	}
	if err := os.MkdirAll(filepath.Dir(st.path), 0o700); err != nil {
		slog.Warn("slurm job store not saved", "error", err)
		return
	}
	tmp := st.path + ".tmp"
	if err := os.WriteFile(tmp, b, 0o600); err != nil {
		slog.Warn("slurm job store not saved", "error", err)
		return
	}
	if err := os.Rename(tmp, st.path); err != nil {
		slog.Warn("slurm job store not saved", "error", err)
	}
}

func slurmJobsPath() string {
	spool := envconfig.SlurmSpool()
	if spool == "" {
		spool = filepath.Join(os.TempDir(), "zerollama-slurm")
	}
	return filepath.Join(spool, "jobs.json")
}

func loadSlurmJobStore(path string) *slurmJobStore {
	st := &slurmJobStore{jobs: make(map[string]*slurmJobRecord), path: path}
	b, err := os.ReadFile(path)
	if err != nil {
		return st
	}
	var recs []slurmJobRecord
	if json.Unmarshal(b, &recs) != nil {
		return st
	}
	for i := range recs {
		rec := recs[i]
		if rec.LocalID == "" {
			continue
		}
		cp := rec
		st.jobs[rec.LocalID] = &cp
		st.order = append(st.order, rec.LocalID)
	}
	return st
}

func (st *slurmJobStore) list() []slurmJobRecord {
	st.mu.Lock()
	defer st.mu.Unlock()
	out := make([]slurmJobRecord, 0, len(st.order))
	for _, id := range st.order {
		if rec := st.jobs[id]; rec != nil {
			out = append(out, *rec)
		}
	}
	return out
}

func (st *slurmJobStore) counts() (pending, running int) {
	st.mu.Lock()
	defer st.mu.Unlock()
	for _, rec := range st.jobs {
		switch rec.Status {
		case "pending":
			pending++
		case "running":
			running++
		}
	}
	return pending, running
}

func slurmStatusTerminal(status string) bool {
	switch status {
	case "completed", "failed", "cancelled":
		return true
	default:
		return false
	}
}

func (s *Server) ensureSlurmJobs() *slurmJobStore {
	s.slurmMu.Lock()
	defer s.slurmMu.Unlock()
	if s.slurmJobs == nil {
		s.slurmJobs = loadSlurmJobStore(slurmJobsPath())
	}
	return s.slurmJobs
}

// slurmStore returns the job store. A restart loads jobs.json so a slurm id
// stays queryable. Nil means this process has never submitted and no file exists.
func (s *Server) slurmStore() *slurmJobStore {
	if s == nil {
		return nil
	}
	s.slurmMu.Lock()
	defer s.slurmMu.Unlock()
	if s.slurmJobs != nil {
		return s.slurmJobs
	}
	path := slurmJobsPath()
	if _, err := os.Stat(path); err != nil {
		return nil
	}
	s.slurmJobs = loadSlurmJobStore(path)
	return s.slurmJobs
}

func (s *Server) slurmClient() *slurm.Client {
	s.slurmMu.Lock()
	defer s.slurmMu.Unlock()
	if s.slurmCLI == nil {
		s.slurmCLI = slurm.NewExecClient()
	}
	return s.slurmCLI
}

func (s *Server) submitTrainingJobViaSlurm(
	ctx context.Context,
	kind string,
	payload json.RawMessage,
	opts TrainingSubmitOptions,
) (TrainingSubmitResult, error) {
	spec, err := resolveSlurmRequest(opts.Slurm, kind)
	if err != nil {
		return TrainingSubmitResult{}, err
	}
	if envconfig.SlurmGuestHasGPU() {
		return TrainingSubmitResult{}, errSlurmGuestGPU
	}
	if !opts.Priority.bypassesIdleWait() {
		if err := checkTrainingAllowedWindow(); err != nil {
			if res, ok, derr := s.tryDeferSlurm(ctx, kind, payload, opts, spec, err); derr != nil {
				return TrainingSubmitResult{}, derr
			} else if ok {
				return res, nil
			}
			return TrainingSubmitResult{}, err
		}
		if err := s.checkTrainingSubmitAllowed(ctx); err != nil {
			if res, ok, derr := s.tryDeferSlurm(ctx, kind, payload, opts, spec, err); derr != nil {
				return TrainingSubmitResult{}, derr
			} else if ok {
				return res, nil
			}
			return TrainingSubmitResult{}, err
		}
	}
	return s.submitSlurmNow(ctx, kind, payload, opts.Priority, spec)
}

func (s *Server) tryDeferSlurm(
	ctx context.Context,
	kind string,
	payload json.RawMessage,
	opts TrainingSubmitOptions,
	spec slurm.Request,
	cause error,
) (TrainingSubmitResult, bool, error) {
	if !shouldDeferTrainingSubmit(cause, opts) {
		return TrainingSubmitResult{}, false, nil
	}
	if s.trainingDefer == nil {
		return TrainingSubmitResult{}, false, nil
	}
	id, err := s.trainingDefer.enqueueSpec(kind, payload, opts.Priority, &spec)
	if err != nil {
		return TrainingSubmitResult{}, false, err
	}
	s.syncRuntimeCoordinationAndPauses(ctx)
	return TrainingSubmitResult{JobID: id, Queued: true, Priority: opts.Priority, Scheduler: "slurm"}, true, nil
}

func (s *Server) submitSlurmNow(
	ctx context.Context,
	kind string,
	payload json.RawMessage,
	priority TrainingPriority,
	spec slurm.Request,
) (TrainingSubmitResult, error) {
	if envconfig.SlurmGuestHasGPU() {
		return TrainingSubmitResult{}, errSlurmGuestGPU
	}
	repo, err := trainingworker.RepoRoot()
	if err != nil {
		return TrainingSubmitResult{}, fmt.Errorf("%w: %s", slurm.ErrBadRequest, err.Error())
	}
	if repo == "" {
		return TrainingSubmitResult{}, fmt.Errorf("%w: set ZEROLLAMA_REPO to the directory containing training.py", slurm.ErrBadRequest)
	}
	localID, err := newSlurmJobID()
	if err != nil {
		return TrainingSubmitResult{}, err
	}
	spool := envconfig.SlurmSpool()
	if spool == "" {
		spool = filepath.Join(os.TempDir(), "zerollama-slurm")
	}
	dir := filepath.Join(spool, localID)
	python := envconfig.SlurmPython()
	script, err := slurm.WriteBundle(dir, repo, python, kind, payload)
	if err != nil {
		return TrainingSubmitResult{}, err
	}
	spec.ScriptPath = script
	spec.WorkDir = dir
	spec.Output = filepath.Join(dir, "slurm-%j.out")
	spec.Error = filepath.Join(dir, "slurm-%j.err")
	spec.JobName = "zerollama-" + kind
	slurmID, err := s.slurmClient().Submit(ctx, spec)
	if err != nil {
		return TrainingSubmitResult{}, err
	}
	mapped := slurm.MapState("PENDING")
	rec := &slurmJobRecord{
		LocalID:     localID,
		SlurmID:     slurmID,
		Kind:        kind,
		Priority:    priority.String(),
		Partition:   spec.Partition,
		Status:      mapped.Status,
		State:       mapped.State,
		SlurmState:  "PENDING",
		SubmittedAt: time.Now(),
	}
	s.ensureSlurmJobs().add(rec)
	slog.Info("submitted training job to slurm", "job_id", localID, "slurm_job_id", slurmID, "partition", spec.Partition)
	return TrainingSubmitResult{
		JobID:      localID,
		Priority:   priority,
		SlurmJobID: slurmID,
		State:      mapped.Status,
		Scheduler:  "slurm",
	}, nil
}

func resolveSlurmRequest(in *slurm.Request, kind string) (slurm.Request, error) {
	var spec slurm.Request
	if in != nil {
		spec = *in
	}
	if spec.Partition == "" {
		spec.Partition = envconfig.SlurmPartition()
	}
	if spec.Account == "" {
		spec.Account = envconfig.SlurmAccount()
	}
	if spec.Time == "" {
		spec.Time = envconfig.SlurmTime()
	}
	if spec.UID != "" && !envconfig.SlurmAllowUID() {
		return slurm.Request{}, fmt.Errorf("%w: uid requires ZEROLLAMA_SLURM_ALLOW_UID=1", slurm.ErrBadRequest)
	}
	if err := slurm.Validate(spec); err != nil {
		return slurm.Request{}, err
	}
	if err := slurm.ValidatePython(envconfig.SlurmPython()); err != nil {
		return slurm.Request{}, err
	}
	_ = kind
	return spec, nil
}

func newSlurmJobID() (string, error) {
	var b [8]byte
	if _, err := rand.Read(b[:]); err != nil {
		return "", err
	}
	return slurmJobIDPrefix + hex.EncodeToString(b[:]), nil
}

func (s *Server) refreshSlurmJobs(ctx context.Context) {
	st := s.slurmStore()
	if st == nil || st.fresh(2*time.Second) {
		return
	}
	cli := s.slurmClient()
	for _, rec := range st.nonTerminal() {
		view, err := cli.Job(ctx, rec.SlurmID)
		if err != nil {
			continue
		}
		st.apply(rec.LocalID, view)
	}
	st.markRefreshed()
}

func (s *Server) slurmTrainingJobStatusJSON(ctx context.Context, id string) ([]byte, error) {
	if s.slurmStore() == nil {
		return nil, trainingworker.ErrJobNotFound
	}
	s.refreshSlurmJobs(ctx)
	rec, ok := s.slurmStore().get(id)
	if !ok {
		return nil, trainingworker.ErrJobNotFound
	}
	return json.Marshal(map[string]any{"job": slurmJobBody(rec)})
}

func slurmJobBody(rec slurmJobRecord) map[string]any {
	msg := rec.Message
	if msg == "" {
		msg = "slurm " + rec.SlurmState
	}
	progress := 0.0
	if rec.Status == "completed" {
		progress = 1
	}
	body := map[string]any{
		"id":              rec.LocalID,
		"jobId":           rec.LocalID,
		"status":          rec.Status,
		"state":           rec.State,
		"progress":        progress,
		"progressMessage": msg,
		"priority":        rec.Priority,
		"scheduler":       "slurm",
		"slurm_job_id":    rec.SlurmID,
		"slurm_state":     rec.SlurmState,
		"submittedAt":     rec.SubmittedAt.UTC().Format(time.RFC3339),
	}
	if rec.Partition != "" {
		body["partition"] = rec.Partition
	}
	if rec.TimeLeft != "" {
		body["time_left"] = rec.TimeLeft
	}
	if rec.Status == "failed" && rec.Message != "" {
		body["error"] = rec.Message
	}
	return body
}

func (s *Server) cancelSlurmTrainingJob(ctx context.Context, id string) (bool, error) {
	st := s.slurmStore()
	if st == nil {
		return false, nil
	}
	rec, ok := st.get(id)
	if !ok {
		return false, nil
	}
	if slurmStatusTerminal(rec.Status) {
		// Completed jobs stay queryable. scancel is not sent.
		return false, nil
	}
	if err := s.slurmClient().Cancel(ctx, rec.SlurmID); err != nil {
		return false, err
	}
	st.markCancelled(id)
	return true, nil
}

func (s *Server) mergeSlurmJobsListJSON(pythonList []byte) ([]byte, error) {
	st := s.slurmStore()
	if st == nil {
		return pythonList, nil
	}
	s.refreshSlurmJobs(context.Background())
	var root map[string]any
	if err := json.Unmarshal(pythonList, &root); err != nil {
		return pythonList, nil
	}
	jobs, _ := root["jobs"].([]any)
	if jobs == nil {
		jobs = []any{}
	}
	for _, rec := range st.list() {
		jobs = append(jobs, slurmJobBody(rec))
	}
	root["jobs"] = jobs
	return json.Marshal(root)
}

func (s *Server) slurmStatus(ctx context.Context) *api.SlurmStatus {
	guest := envconfig.SlurmGuest()
	submit := envconfig.SlurmSubmit()
	if st := s.slurmStore(); st != nil && len(st.list()) > 0 {
		submit = true
	}
	if !guest && !submit {
		return nil
	}
	st := &api.SlurmStatus{
		Guest:  guest,
		Submit: submit,
	}
	if submit {
		st.Transport = "sbatch"
		st.RestURL = ""
	}
	if guest {
		st.JobID = strings.TrimSpace(os.Getenv("SLURM_JOB_ID"))
	}
	if jobs := s.slurmStore(); jobs != nil {
		s.refreshSlurmJobs(ctx)
		st.Pending, st.Running = jobs.counts()
	}
	return st
}
