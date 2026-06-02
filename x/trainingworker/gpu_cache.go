package trainingworker

import (
	"os"
	"strconv"
	"strings"
	"time"
)

func trainingGPUStatusTTL() time.Duration {
	ms := 250
	if v := strings.TrimSpace(os.Getenv("ZEROLLAMA_TRAINING_GPU_STATUS_TTL_MS")); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n > 0 {
			ms = n
		}
	}
	return time.Duration(ms) * time.Millisecond
}

// LastGPUHealthOK is false when the most recent refresh failed (used for fail-closed policy).
func (c *Client) LastGPUHealthOK() bool {
	if c == nil {
		return true
	}
	c.gpuMu.Lock()
	defer c.gpuMu.Unlock()
	return c.gpuHealthErr == nil
}
