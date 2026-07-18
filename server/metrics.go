package server

import (
	"fmt"
	"net/http"
	"strings"
	"sync/atomic"

	"github.com/gin-gonic/gin"
)

// Prometheus counters for GET /api/metrics (hand-rolled; no prometheus client dep).
// Why hand-rolled: scrapers need text exposition without pulling a client library into
// the serve binary. Why also instrument runtime proxy: ggml-only counters undercount on Linux.
var (
	metricsRequestsOK      atomic.Uint64
	metricsRequestsError   atomic.Uint64
	metricsRequestsEmpty   atomic.Uint64
	metricsRequestsUnstable atomic.Uint64
	metricsQueueRejects    atomic.Uint64
	metricsRunnerCrashes   atomic.Uint64
)

func metricsIncRequestResult(result string) {
	switch result {
	case "ok":
		metricsRequestsOK.Add(1)
	case "error":
		metricsRequestsError.Add(1)
	case "empty_generation":
		metricsRequestsEmpty.Add(1)
	case "host_unstable":
		metricsRequestsUnstable.Add(1)
	}
}

func metricsIncQueueReject()  { metricsQueueRejects.Add(1) }
func metricsIncRunnerCrash()  { metricsRunnerCrashes.Add(1) }

// MetricsHandler serves GET /api/metrics as Prometheus text exposition.
// Why separate from /api/status JSON: fleet scrapers expect Prometheus; operators keep jq on status.
func (s *Server) MetricsHandler(c *gin.Context) {
	ctx := c.Request.Context()
	var ggmlPending, ggmlActive, ggmlLoaded int
	var loadsPaused int
	if s != nil && s.sched != nil {
		snap := s.sched.InferenceFleetSnapshot()
		ggmlPending = snap.Pending
		ggmlActive = snap.Active
		ggmlLoaded = snap.Loaded
		if snap.LoadsPaused {
			loadsPaused = 1
		}
	}

	var runtimeWaiting, runtimeRunning, llamaLoaded int
	if runtimeHealthProbeRequired() {
		h := runtimeInferenceHealth(ctx)
		if h.ok {
			runtimeWaiting = h.waiting
			runtimeRunning = h.running
			if h.llamaLoaded {
				llamaLoaded = 1
			}
		}
	}

	var b strings.Builder
	writeGauge := func(name, help string, v float64) {
		fmt.Fprintf(&b, "# HELP %s %s\n# TYPE %s gauge\n%s %g\n", name, help, name, name, v)
	}
	writeCounter := func(name, help string, v uint64) {
		fmt.Fprintf(&b, "# HELP %s %s\n# TYPE %s counter\n%s %d\n", name, help, name, name, v)
	}

	writeGauge("zerollama_ggml_pending", "GGML scheduler pending queue depth", float64(ggmlPending))
	writeGauge("zerollama_ggml_active", "GGML active runner refs", float64(ggmlActive))
	writeGauge("zerollama_ggml_loaded", "GGML loaded ready runners", float64(ggmlLoaded))
	writeGauge("zerollama_ggml_loads_paused", "GGML loads paused (1=true)", float64(loadsPaused))
	writeGauge("zerollama_runtime_waiting", "Python runtime waiting queue depth", float64(runtimeWaiting))
	writeGauge("zerollama_runtime_running", "Python runtime running requests", float64(runtimeRunning))
	writeGauge("zerollama_runtime_llama_loaded", "Python runtime has a loaded GGUF (1=true)", float64(llamaLoaded))

	fmt.Fprintf(&b, "# HELP zerollama_inference_requests_total Completed inference requests by result\n")
	fmt.Fprintf(&b, "# TYPE zerollama_inference_requests_total counter\n")
	fmt.Fprintf(&b, "zerollama_inference_requests_total{result=%q} %d\n", "ok", metricsRequestsOK.Load())
	fmt.Fprintf(&b, "zerollama_inference_requests_total{result=%q} %d\n", "error", metricsRequestsError.Load())
	fmt.Fprintf(&b, "zerollama_inference_requests_total{result=%q} %d\n", "empty_generation", metricsRequestsEmpty.Load())
	fmt.Fprintf(&b, "zerollama_inference_requests_total{result=%q} %d\n", "host_unstable", metricsRequestsUnstable.Load())

	writeCounter("zerollama_inference_queue_rejects_total", "Requests rejected for max queue (ErrMaxQueue)", metricsQueueRejects.Load())
	writeCounter("zerollama_runner_crashes_total", "Runner/process crash or exit errors", metricsRunnerCrashes.Load())

	c.Data(http.StatusOK, "text/plain; version=0.0.4; charset=utf-8", []byte(b.String()))
}
