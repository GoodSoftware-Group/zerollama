// training_defer_queue: in-memory queue for jobs that cannot start while inference
// holds the GPU. Python cannot see ggml residency or scheduler backlog — defer extends
// the Go idle-wait gate (ZEROLLAMA_TRAINING_WAIT_INFERENCE_IDLE), not a second FIFO.
// defer-* IDs stay queryable after promotion (tombstone TTL) so status polling keeps
// working; records are not deleted on successful promote.
package server

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"slices"
	"strings"
	"sync"
	"time"

	"github.com/ollama/ollama/envconfig"
)

const deferredJobIDPrefix = "defer-"

type deferredJobState string

const (
	deferredStateWaiting   deferredJobState = "waiting"
	deferredStatePromoted  deferredJobState = "promoted"
	deferredStateFailed    deferredJobState = "failed"
	deferredStateCancelled deferredJobState = "cancelled"
)

type deferredTrainingJob struct {
	id         string
	kind       string
	payload    json.RawMessage
	enqueuedAt time.Time
	finishedAt time.Time
	nextRetryAt time.Time
	state      deferredJobState
	promotedID string
	errMsg     string
	retryCount int
	fifoSeq    uint64
}

type trainingDeferQueue struct {
	mu    sync.Mutex
	items map[string]*deferredTrainingJob
	order []string
	srv   *Server
}

func newTrainingDeferQueue(s *Server) *trainingDeferQueue {
	return &trainingDeferQueue{
		items: make(map[string]*deferredTrainingJob),
		srv:   s,
	}
}

func (q *trainingDeferQueue) start(ctx context.Context) {
	if q == nil || q.srv == nil {
		return
	}
	interval := envconfig.TrainingQueuePollInterval()
	ticker := time.NewTicker(interval)
	defer ticker.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			q.drainOnce(ctx)
			q.evictExpired()
		}
	}
}

func (q *trainingDeferQueue) enqueue(kind string, payload json.RawMessage) (string, error) {
	if q == nil {
		return "", errors.New("training defer queue unavailable")
	}
	max := envconfig.TrainingQueueMaxDepth()
	q.mu.Lock()
	defer q.mu.Unlock()
	waiting := 0
	for _, id := range q.order {
		if j, ok := q.items[id]; ok && j.state == deferredStateWaiting {
			waiting++
		}
	}
	if max > 0 && waiting >= max {
		return "", fmt.Errorf("training defer queue full (max %d)", max)
	}
	id, err := newDeferredJobID()
	if err != nil {
		return "", err
	}
	q.items[id] = &deferredTrainingJob{
		id:         id,
		kind:       kind,
		payload:    payload,
		enqueuedAt: time.Now(),
		state:      deferredStateWaiting,
		fifoSeq:    AllocCrossQueueSeq(),
	}
	q.order = append(q.order, id)
	return id, nil
}

func (q *trainingDeferQueue) lookupSnapshot(id string) (deferredTrainingJob, bool) {
	q.mu.Lock()
	defer q.mu.Unlock()
	j, ok := q.items[id]
	if !ok {
		return deferredTrainingJob{}, false
	}
	return *j, true
}

func (q *trainingDeferQueue) cancel(id string) (bool, error) {
	if q == nil {
		return false, errors.New("training defer queue unavailable")
	}
	q.mu.Lock()
	defer q.mu.Unlock()
	j, ok := q.items[id]
	if !ok {
		return false, nil
	}
	switch j.state {
	case deferredStateWaiting:
		j.state = deferredStateCancelled
		j.finishedAt = time.Now()
		q.removeFromOrderLocked(id)
		return true, nil
	case deferredStateCancelled:
		return true, nil
	default:
		return false, fmt.Errorf("defer job %q is %s, not waiting", id, j.state)
	}
}

// coordinationStats returns defer-queue counts for the Python runtime /health mirror.
func (q *trainingDeferQueue) coordinationStats() map[string]any {
	if q == nil {
		return map[string]any{"defer_waiting": 0, "defer_tracked": 0}
	}
	q.mu.Lock()
	defer q.mu.Unlock()
	waiting := 0
	for _, j := range q.items {
		if j.state == deferredStateWaiting {
			waiting++
		}
	}
	return map[string]any{
		"defer_waiting":      waiting,
		"defer_tracked":      len(q.items),
		"fifo_go_oldest_defer": q.oldestWaitingFifoSeqLocked(),
	}
}

func (q *trainingDeferQueue) oldestWaitingFifoSeq() uint64 {
	if q == nil {
		return 0
	}
	q.mu.Lock()
	defer q.mu.Unlock()
	return q.oldestWaitingFifoSeqLocked()
}

func (q *trainingDeferQueue) waitingFifoSeq(id string) uint64 {
	if q == nil {
		return 0
	}
	q.mu.Lock()
	defer q.mu.Unlock()
	j, ok := q.items[id]
	if !ok || j.state != deferredStateWaiting {
		return 0
	}
	return j.fifoSeq
}

func (q *trainingDeferQueue) oldestWaitingFifoSeqLocked() uint64 {
	var min uint64
	for _, id := range q.order {
		j, ok := q.items[id]
		if !ok || j.state != deferredStateWaiting || j.fifoSeq == 0 {
			continue
		}
		if min == 0 || j.fifoSeq < min {
			min = j.fifoSeq
		}
	}
	return min
}

func (q *trainingDeferQueue) listEntries() []deferredTrainingJob {
	if q == nil {
		return nil
	}
	q.mu.Lock()
	defer q.mu.Unlock()
	out := make([]deferredTrainingJob, 0, len(q.items))
	for _, j := range q.items {
		if !envconfig.TrainingQueueListAll() && j.state != deferredStateWaiting {
			continue
		}
		out = append(out, *j)
	}
	slices.SortFunc(out, func(a, b deferredTrainingJob) int {
		return a.enqueuedAt.Compare(b.enqueuedAt)
	})
	return out
}

func (q *trainingDeferQueue) removeFromOrderLocked(id string) {
	for i, x := range q.order {
		if x == id {
			q.order = append(q.order[:i], q.order[i+1:]...)
			return
		}
	}
}

func (q *trainingDeferQueue) markPromotedLocked(id, jobID string) {
	if j, ok := q.items[id]; ok {
		j.state = deferredStatePromoted
		j.promotedID = jobID
		j.finishedAt = time.Now()
		q.removeFromOrderLocked(id)
	}
}

func (q *trainingDeferQueue) markFailedOrRetryLocked(id, msg string) {
	j, ok := q.items[id]
	if !ok {
		return
	}
	max := envconfig.TrainingQueueRetryMax()
	if max > 0 && j.retryCount < max {
		j.retryCount++
		j.state = deferredStateWaiting
		j.errMsg = msg
		j.finishedAt = time.Time{}
		j.nextRetryAt = time.Now().Add(envconfig.TrainingQueueRetryInterval())
		if !slices.Contains(q.order, id) {
			q.order = append(q.order, id)
		}
		slog.Info("deferred training job scheduled for retry",
			"defer_id", id, "attempt", j.retryCount, "max", max, "retry_at", j.nextRetryAt)
		return
	}
	j.state = deferredStateFailed
	j.errMsg = msg
	j.finishedAt = time.Now()
	q.removeFromOrderLocked(id)
}

func (q *trainingDeferQueue) evictExpired() {
	if q == nil {
		return
	}
	ttl := envconfig.TrainingQueueTombstoneTTL()
	if ttl <= 0 {
		return
	}
	now := time.Now()
	q.mu.Lock()
	defer q.mu.Unlock()
	for id, j := range q.items {
		if j.state == deferredStateWaiting || j.finishedAt.IsZero() {
			continue
		}
		if now.Sub(j.finishedAt) > ttl {
			delete(q.items, id)
			q.removeFromOrderLocked(id)
		}
	}
}

func (q *trainingDeferQueue) drainOnce(ctx context.Context) {
	if q == nil || q.srv == nil || q.srv.training == nil {
		return
	}
	for {
		id := q.nextPendingID()
		if id == "" {
			return
		}
		if err := checkTrainingAllowedWindow(); err != nil {
			return
		}
		if err := q.srv.checkTrainingSubmitAllowed(ctx); err != nil {
			return
		}
		if q.srv.crossFifoBlocksDeferPromote(id) {
			return
		}
		q.mu.Lock()
		j, ok := q.items[id]
		if !ok || j.state != deferredStateWaiting {
			q.mu.Unlock()
			continue
		}
		kind, payload := j.kind, j.payload
		q.mu.Unlock()

		jobID, err := q.srv.training.SubmitTrainingJobDirect(ctx, kind, payload)
		if err != nil {
			if TrainingSubmitConflict(err) {
				return
			}
			slog.Warn("deferred training submit failed", "defer_id", id, "error", err)
			q.mu.Lock()
			q.markFailedOrRetryLocked(id, err.Error())
			q.mu.Unlock()
			continue
		}
		q.mu.Lock()
		q.markPromotedLocked(id, jobID)
		q.mu.Unlock()
		slog.Info("deferred training job promoted", "defer_id", id, "job_id", jobID)
		// Wan (run_script) needs exclusive GPU for the promoted worker job; acquire here
		// because /v1/videos skipped the lease while the job sat on defer-*.
		if kind == "run_script" {
			q.srv.acquireVideoExclusiveGPU(ctx, jobID)
		}
	}
}

func (q *trainingDeferQueue) nextPendingID() string {
	q.mu.Lock()
	defer q.mu.Unlock()
	now := time.Now()
	for _, id := range q.order {
		j, ok := q.items[id]
		if !ok || j.state != deferredStateWaiting {
			continue
		}
		if !j.nextRetryAt.IsZero() && now.Before(j.nextRetryAt) {
			continue
		}
		return id
	}
	return ""
}

func newDeferredJobID() (string, error) {
	var b [8]byte
	if _, err := rand.Read(b[:]); err != nil {
		return "", err
	}
	return deferredJobIDPrefix + hex.EncodeToString(b[:]), nil
}

func isDeferredTrainingJobID(id string) bool {
	return strings.HasPrefix(id, deferredJobIDPrefix)
}

func (q *trainingDeferQueue) promotedJobID(deferID string) (string, bool) {
	j, ok := q.lookupSnapshot(deferID)
	if !ok || j.promotedID == "" {
		return "", false
	}
	return j.promotedID, j.state == deferredStatePromoted
}
