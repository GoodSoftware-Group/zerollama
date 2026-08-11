package server

import (
	"context"
	"encoding/json"
	"io"
	"log/slog"
	"net"
	"net/http"
	"strings"
	"time"

	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/version"
)

// watchLoopbackServeIdentity periodically GETs the connectable host /api/version
// and verifies this process is what clients will talk to.
//
// WHY: on Darwin/BSD, a process can bind 127.0.0.1:PORT after we already hold
// *:PORT (wildcard). Listen succeeds at startup, then later loopback traffic is
// stolen by the more-specific bind — empty replies / wrong servers, with no
// listen() failure to trigger a restart. Primary defense is claimLoopbackGuards
// (cmd) holding 127.0.0.1/::1 for the life of serve; this watcher is the
// remaining alarm if a steal still happens on an unclaimed address. We cannot
// reclaim the port without killing the other process, so we scream in the log.
func watchLoopbackServeIdentity(ctx context.Context) {
	base := strings.TrimSuffix(envconfig.ConnectableHost().String(), "/")
	if base == "" {
		return
	}
	_, port, _ := net.SplitHostPort(envconfig.Host().Host)
	if port == "" {
		port = "11434"
	}
	expectVersion := version.Version
	client := &http.Client{Timeout: 2 * time.Second}

	check := func() {
		req, err := http.NewRequestWithContext(ctx, http.MethodGet, base+"/api/version", nil)
		if err != nil {
			return
		}
		resp, err := client.Do(req)
		if err != nil {
			slog.Error("loopback serve identity: connectable host not answering as this process",
				"url", base+"/api/version",
				"error", err,
				"hint", "another process likely bound 127.0.0.1 on this port after serve started; run lsof -nP -iTCP:"+port+" -sTCP:LISTEN and close the non-zerollama listener",
			)
			return
		}
		defer resp.Body.Close()
		body, _ := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
		if resp.StatusCode != http.StatusOK {
			slog.Error("loopback serve identity: unexpected status on connectable host",
				"url", base+"/api/version",
				"status", resp.StatusCode,
				"body", strings.TrimSpace(string(body)),
			)
			return
		}
		var payload struct {
			Distribution string `json:"distribution"`
			Version      string `json:"version"`
		}
		if err := json.Unmarshal(body, &payload); err != nil {
			slog.Error("loopback serve identity: non-JSON response on connectable host (port shadowed?)",
				"url", base+"/api/version",
				"error", err,
			)
			return
		}
		if payload.Distribution != "zerollama" {
			slog.Error("loopback serve identity: wrong distribution answering on connectable host",
				"url", base+"/api/version",
				"distribution", payload.Distribution,
				"version", payload.Version,
			)
			return
		}
		if expectVersion != "" && payload.Version != "" && payload.Version != expectVersion {
			slog.Warn("loopback serve identity: version mismatch on connectable host",
				"url", base+"/api/version",
				"expected", expectVersion,
				"got", payload.Version,
				"hint", "another zerollama/ollama binary may be answering loopback",
			)
		}
	}

	// First probe shortly after accept begins (Serve is about to start).
	select {
	case <-ctx.Done():
		return
	case <-time.After(2 * time.Second):
		check()
	}

	t := time.NewTicker(30 * time.Second)
	defer t.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-t.C:
			check()
		}
	}
}
