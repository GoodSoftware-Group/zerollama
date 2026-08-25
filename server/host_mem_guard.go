package server

import (
	"log/slog"

	"github.com/gin-gonic/gin"
	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/discover"
	"github.com/ollama/ollama/envconfig"
)

func currentHostMemPressure() discover.HostMemPressure {
	return discover.HostMemSnapshot(
		envconfig.HostMemPressureRatio(),
		envconfig.HostSwapPressureRatio(),
		envconfig.HostSwapPressureFloor(),
	)
}

func (s *Server) hostMemGuard() gin.HandlerFunc {
	return func(c *gin.Context) {
		if !envconfig.HostMemGuardEnabled() {
			c.Next()
			return
		}
		p := currentHostMemPressure()
		if !p.Pressure {
			c.Next()
			return
		}
		slog.Warn("refusing inference: host RAM/swap pressure",
			"anon", p.Cgroup.Anon,
			"limit", p.Cgroup.Limit,
			"swap", p.Cgroup.SwapCurrent,
		)
		writeBusyUnavailable(c, p.ClientMessage())
		c.Abort()
	}
}

func hostMemoryStatusAPI() *api.HostMemoryStatus {
	p := currentHostMemPressure()
	st := &api.HostMemoryStatus{
		Pressure:         p.Pressure,
		Guard:            envconfig.HostMemGuardEnabled(),
		LimitBytes:       p.Cgroup.Limit,
		CurrentBytes:     p.Cgroup.Current,
		AnonBytes:        p.Cgroup.Anon,
		SwapCurrentBytes: p.Cgroup.SwapCurrent,
		Reason:           p.Reason,
	}
	if p.Cgroup.HasSwapMax {
		st.SwapMaxBytes = p.Cgroup.SwapMax
	}
	return st
}
