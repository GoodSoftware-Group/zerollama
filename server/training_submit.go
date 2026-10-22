// training_submit applies priority and defer-queue policy before embedded Python.
// high bypasses idle-wait (operator override); low prefers defer over HTTP 409 spam.
package server

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"time"

	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/x/trainingworker"
)

// TrainingSubmitOptions controls idle-wait bypass and defer-queue behavior.
type TrainingSubmitOptions struct {
	Priority    TrainingPriority
	QueueOnBusy *bool
}

// TrainingSubmitResult is returned from a training job submit attempt.
type TrainingSubmitResult struct {
	JobID    string
	Queued   bool
	Priority TrainingPriority // resolved tier (T6b echo for operators / automation)
}

func (s *Server) submitTrainingJob(
	ctx context.Context,
	kind string,
	payload json.RawMessage,
	opts TrainingSubmitOptions,
) (TrainingSubmitResult, error) {
	if s == nil || s.training == nil {
		return TrainingSubmitResult{}, errors.New("training worker not available")
	}
	if err := checkTrainingQloraPayload(kind, payload); err != nil {
		return TrainingSubmitResult{}, err
	}
	if !opts.Priority.bypassesIdleWait() {
		if err := checkTrainingAllowedWindow(); err != nil {
			if res, ok, derr := s.tryDeferTrainingSubmit(ctx, kind, payload, opts, err); derr != nil {
				return TrainingSubmitResult{}, derr
			} else if ok {
				return res, nil
			}
			return TrainingSubmitResult{}, err
		}
		if err := s.checkTrainingSubmitAllowed(ctx); err != nil {
			if res, ok, derr := s.tryDeferTrainingSubmit(ctx, kind, payload, opts, err); derr != nil {
				return TrainingSubmitResult{}, derr
			} else if ok {
				return res, nil
			}
			return TrainingSubmitResult{}, err
		}
	}
	jobID, err := s.training.SubmitTrainingJobDirect(ctx, kind, payload)
	if err != nil {
		return TrainingSubmitResult{}, err
	}
	return TrainingSubmitResult{JobID: jobID, Priority: opts.Priority}, nil
}

func (s *Server) tryDeferTrainingSubmit(
	ctx context.Context,
	kind string,
	payload json.RawMessage,
	opts TrainingSubmitOptions,
	err error,
) (TrainingSubmitResult, bool, error) {
	if !shouldDeferTrainingSubmit(err, opts) {
		return TrainingSubmitResult{}, false, nil
	}
	if s.trainingDefer == nil {
		return TrainingSubmitResult{}, false, nil
	}
	id, qerr := s.trainingDefer.enqueue(kind, payload, opts.Priority)
	if qerr != nil {
		return TrainingSubmitResult{}, false, qerr
	}
	// WHY sync (not push-only): T6c must PauseNewLoads immediately when defer waits,
	// not wait for the ~400ms training GPU policy ticker.
	s.syncRuntimeCoordinationAndPauses(ctx)
	return TrainingSubmitResult{JobID: id, Queued: true, Priority: opts.Priority}, true, nil
}

func shouldDeferTrainingSubmit(err error, opts TrainingSubmitOptions) bool {
	if errors.Is(err, ErrTrainingWindowMisconfigured) {
		return false
	}
	if errors.Is(err, ErrTrainingOutsideWindow) {
		if !envconfig.TrainingAllowedWindowEnabled() {
			return false
		}
		if opts.QueueOnBusy != nil {
			return *opts.QueueOnBusy
		}
		if envconfig.TrainingQueueOnBusy() {
			return true
		}
		return opts.Priority.prefersDeferQueue()
	}
	if !errors.Is(err, ErrInferenceBacklogActive) {
		return false
	}
	if !envconfig.TrainingWaitInferenceIdle() {
		return false
	}
	if opts.QueueOnBusy != nil {
		return *opts.QueueOnBusy
	}
	if envconfig.TrainingQueueOnBusy() {
		return true
	}
	return opts.Priority.prefersDeferQueue()
}

func (s *Server) deferredTrainingJobStatusJSON(ctx context.Context, id string) ([]byte, error) {
	if s == nil || s.trainingDefer == nil {
		return nil, trainingworker.ErrJobNotFound
	}
	j, ok := s.trainingDefer.lookupSnapshot(id)
	if !ok {
		return nil, trainingworker.ErrJobNotFound
	}

	if j.state == deferredStatePromoted && j.promotedID != "" && s.training != nil {
		if raw, err := s.training.JobTrainingStatusJSON(ctx, j.promotedID); err == nil {
			var wrap map[string]any
			if json.Unmarshal(raw, &wrap) == nil {
				if job, ok := wrap["job"].(map[string]any); ok {
					job["deferJobId"] = id
					job["promotedJobId"] = j.promotedID
					job["defer"] = true
					job["priority"] = j.priority.String()
					if j.kind == "run_script" {
						modelName, size := wanVideoJobMeta(j.payload)
						if _, has := job["videoModel"]; !has && modelName != "" {
							job["videoModel"] = modelName
						}
						if _, has := job["videoSize"]; !has && size != "" {
							job["videoSize"] = size
						}
					}
					wrap["job"] = job
					return json.Marshal(wrap)
				}
			}
		}
	}

	status := "pending"
	msg := "waiting for inference idle"
	switch j.state {
	case deferredStatePromoted:
		status = "promoted"
		msg = "submitted to training worker; poll promotedJobId for progress"
	case deferredStateFailed:
		status = "failed"
		msg = j.errMsg
	case deferredStateCancelled:
		status = "cancelled"
		msg = "cancelled before promotion"
	case deferredStateWaiting:
		if j.retryCount > 0 {
			status = "pending"
			msg = fmt.Sprintf("retry %d scheduled after %s", j.retryCount, j.nextRetryAt.UTC().Format(time.RFC3339))
		}
	}

	job := map[string]any{
		"jobId":           id,
		"status":          status,
		"progress":        0.0,
		"progressMessage": msg,
		"promotedJobId":   j.promotedID,
		"defer":           true,
		"priority":        j.priority.String(),
		"enqueuedAt":      j.enqueuedAt.UTC().Format(time.RFC3339),
		"submittedAt":     j.enqueuedAt.UTC().Format(time.RFC3339),
	}
	if j.kind == "run_script" {
		modelName, size := wanVideoJobMeta(j.payload)
		if modelName != "" {
			job["videoModel"] = modelName
		}
		if size != "" {
			job["videoSize"] = size
		}
	}
	if j.retryCount > 0 {
		job["retryCount"] = j.retryCount
	}
	if !j.nextRetryAt.IsZero() {
		job["nextRetryAt"] = j.nextRetryAt.UTC().Format(time.RFC3339)
	}
	if j.errMsg != "" && j.state == deferredStateFailed {
		job["error"] = j.errMsg
	} else if j.errMsg != "" && j.state == deferredStateWaiting && j.retryCount > 0 {
		job["lastError"] = j.errMsg
	}
	return json.Marshal(map[string]any{"job": job})
}

func (s *Server) cancelDeferredTrainingJob(id string) (bool, error) {
	if s == nil || s.trainingDefer == nil {
		return false, nil
	}
	ok, err := s.trainingDefer.cancel(id)
	if ok {
		// Resume ggml loads if this was the last waiting defer (T6c).
		s.syncRuntimeCoordinationAndPauses(context.Background())
	}
	return ok, err
}

func (s *Server) mergeDeferredJobsListJSON(pythonList []byte) ([]byte, error) {
	if s == nil || s.trainingDefer == nil {
		return pythonList, nil
	}
	var root map[string]any
	if err := json.Unmarshal(pythonList, &root); err != nil {
		return pythonList, nil
	}
	jobs, _ := root["jobs"].([]any)
	if jobs == nil {
		jobs = []any{}
	}
	for _, j := range s.trainingDefer.listEntries() {
		entry := map[string]any{
			"jobId":      j.id,
			"status":     string(j.state),
			"defer":      true,
			"priority":   j.priority.String(),
			"enqueuedAt": j.enqueuedAt.UTC().Format(time.RFC3339),
		}
		if j.promotedID != "" {
			entry["promotedJobId"] = j.promotedID
		}
		if j.errMsg != "" {
			entry["error"] = j.errMsg
		}
		jobs = append(jobs, entry)
	}
	root["jobs"] = jobs
	return json.Marshal(root)
}

func (s *Server) handleTrainingSubmitRequest(
	ctx context.Context,
	req trainingworker.SubmitRequest,
) (trainingworker.SubmitResponse, error) {
	prio, perr := resolveTrainingSubmitPriority(req.Priority, req.SloClass)
	if perr != nil {
		return trainingworker.SubmitResponse{}, perr
	}
	res, err := s.submitTrainingJob(ctx, req.Kind, req.Payload, TrainingSubmitOptions{
		Priority:    prio,
		QueueOnBusy: req.QueueOnBusy,
	})
	if err != nil {
		return trainingworker.SubmitResponse{}, err
	}
	return trainingworker.SubmitResponse{
		JobID:    res.JobID,
		Queued:   res.Queued,
		Priority: res.Priority.String(),
	}, nil
}
