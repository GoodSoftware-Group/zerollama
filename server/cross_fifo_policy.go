package server

import (
	"net/http"

	"github.com/gin-gonic/gin"
)

func (s *Server) setRuntimeFifoOldest(v uint64) {
	if s == nil {
		return
	}
	s.runtimeFifoMu.Lock()
	s.runtimeFifoOldest = v
	s.runtimeFifoMu.Unlock()
}

func (s *Server) cachedRuntimeFifoOldest() uint64 {
	if s == nil {
		return 0
	}
	s.runtimeFifoMu.RLock()
	defer s.runtimeFifoMu.RUnlock()
	return s.runtimeFifoOldest
}

// oldestInferenceFifoSeq is the smallest ticket among runtime (mirrored) and ggml work.
func (s *Server) oldestInferenceFifoSeq() uint64 {
	if s == nil {
		return 0
	}
	var ggml uint64
	if s.sched != nil {
		ggml = s.sched.oldestGgmlFifoSeq()
	}
	return minNonZeroUint64(s.cachedRuntimeFifoOldest(), ggml)
}

// schedYieldToRuntimeFifo blocks ggml pending scheduling while runtime has older FIFO work.
func (s *Server) schedYieldToRuntimeFifo() bool {
	if s == nil || s.sched == nil {
		return false
	}
	ggmlOldest := s.sched.oldestGgmlFifoSeq()
	if ggmlOldest == 0 {
		return false
	}
	rt := s.cachedRuntimeFifoOldest()
	return rt > 0 && rt < ggmlOldest
}

// crossFifoBlocksDeferPromote returns true when inference (runtime or ggml) is ahead of defer head.
func (s *Server) crossFifoBlocksDeferPromote(deferID string) bool {
	if s == nil || s.trainingDefer == nil {
		return false
	}
	deferSeq := s.trainingDefer.waitingFifoSeq(deferID)
	if deferSeq == 0 {
		return false
	}
	ahead := s.oldestInferenceFifoSeq()
	return ahead > 0 && ahead < deferSeq
}

func (s *Server) fifoCoordinationFields() map[string]any {
	if s == nil {
		return nil
	}
	var ggmlOldest, deferOldest uint64
	if s.sched != nil {
		ggmlOldest = s.sched.oldestGgmlFifoSeq()
	}
	if s.trainingDefer != nil {
		deferOldest = s.trainingDefer.oldestWaitingFifoSeq()
	}
	inferenceOldest := s.oldestInferenceFifoSeq()
	return map[string]any{
		"fifo_go_oldest_ggml":        ggmlOldest,
		"fifo_go_oldest_defer":       deferOldest,
		"fifo_go_oldest_inference":   inferenceOldest,
		"fifo_go_oldest":             minNonZeroUint64(ggmlOldest, deferOldest),
		"fifo_runtime_oldest":        s.cachedRuntimeFifoOldest(),
	}
}

// CrossQueueSeqHandler allocates a global FIFO ticket for the Python runtime.
func (s *Server) CrossQueueSeqHandler(c *gin.Context) {
	_ = s
	c.JSON(http.StatusOK, gin.H{"seq": AllocCrossQueueSeq()})
}
