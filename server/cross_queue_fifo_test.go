package server

import (
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/gin-gonic/gin"
)

func TestCrossQueueSeqHandler_loopbackOnly(t *testing.T) {
	gin.SetMode(gin.TestMode)
	r := gin.New()
	internal := r.Group("/internal", internalLoopbackOnly())
	internal.POST("/cross-queue-seq", (&Server{}).CrossQueueSeqHandler)

	remote := httptest.NewRequest(http.MethodPost, "/internal/cross-queue-seq", nil)
	remote.RemoteAddr = "203.0.113.1:12345"
	w := httptest.NewRecorder()
	r.ServeHTTP(w, remote)
	if w.Code != http.StatusForbidden {
		t.Fatalf("remote status %d", w.Code)
	}

	local := httptest.NewRequest(http.MethodPost, "/internal/cross-queue-seq", nil)
	local.RemoteAddr = "127.0.0.1:12345"
	w2 := httptest.NewRecorder()
	r.ServeHTTP(w2, local)
	if w2.Code != http.StatusOK {
		t.Fatalf("loopback status %d body %s", w2.Code, w2.Body.String())
	}
}

func TestAllocCrossQueueSeqMonotonic(t *testing.T) {
	a := AllocCrossQueueSeq()
	b := AllocCrossQueueSeq()
	if a == 0 || b == 0 || b <= a {
		t.Fatalf("expected increasing non-zero seq, got %d then %d", a, b)
	}
}

func TestMinNonZeroUint64(t *testing.T) {
	if minNonZeroUint64(0, 0) != 0 {
		t.Fatal("expected 0")
	}
	if minNonZeroUint64(5, 0) != 5 {
		t.Fatal("expected 5")
	}
	if minNonZeroUint64(9, 3) != 3 {
		t.Fatal("expected 3")
	}
}

func TestPendingQueueOldestFifoSeq(t *testing.T) {
	q := newPendingQueue(8)
	if q.OldestFifoSeq() != 0 {
		t.Fatal("expected empty oldest 0")
	}
	q.Push(&LlmRequest{fifoSeq: 12})
	q.Push(&LlmRequest{fifoSeq: 7})
	if got := q.OldestFifoSeq(); got != 7 {
		t.Fatalf("oldest=%d want 7", got)
	}
}

func TestCrossFifoBlocksDeferPromoteWhenGgmlAhead(t *testing.T) {
	s := &Server{}
	s.sched = &Scheduler{pending: newPendingQueue(8)}
	s.sched.pending.Push(&LlmRequest{fifoSeq: 3})
	s.trainingDefer = &trainingDeferQueue{
		items: map[string]*deferredTrainingJob{
			"defer-1": {id: "defer-1", state: deferredStateWaiting, fifoSeq: 20},
		},
		order: []string{"defer-1"},
	}
	if !s.crossFifoBlocksDeferPromote("defer-1") {
		t.Fatal("expected block when ggml ticket 3 < defer 20")
	}
	s.sched.pending = newPendingQueue(8)
	s.setRuntimeFifoOldest(5)
	if !s.crossFifoBlocksDeferPromote("defer-1") {
		t.Fatal("expected block when runtime ticket 5 < defer 20")
	}
	s.setRuntimeFifoOldest(0)
	s.sched.loadingFifoSeq.Store(4)
	if !s.crossFifoBlocksDeferPromote("defer-1") {
		t.Fatal("expected block when loading ggml ticket 4 < defer 20")
	}
}

func TestSchedYieldToRuntimeFifo(t *testing.T) {
	s := &Server{}
	s.sched = &Scheduler{pending: newPendingQueue(8)}
	s.sched.pending.Push(&LlmRequest{fifoSeq: 20})
	s.setRuntimeFifoOldest(5)
	if !s.schedYieldToRuntimeFifo() {
		t.Fatal("expected yield when runtime ticket is older")
	}
	s.setRuntimeFifoOldest(25)
	if s.schedYieldToRuntimeFifo() {
		t.Fatal("expected no yield when ggml is older")
	}
}
