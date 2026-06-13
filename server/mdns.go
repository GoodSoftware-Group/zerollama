package server

import (
	"context"
	"log/slog"
	"net"

	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/fleet/mdns"
	"github.com/ollama/ollama/version"
)

// startNodeMDNS registers this zerollama node on the LAN when ZEROLLAMA_MDNS=1.
func startNodeMDNS(ctx context.Context, ln net.Listener) {
	if ln == nil {
		return
	}
	port, err := mdns.PortFromAddr(ln.Addr())
	if err != nil {
		slog.Warn("mdns not started", "error", err)
		return
	}
	startMDNSAdvertisement(ctx, port)
}

// startMDNSAdvertisement registers this zerollama node on the LAN when ZEROLLAMA_MDNS=1.
// Why opt-in: surprise multicast on shared networks is undesirable; homelab operators enable explicitly.
func startMDNSAdvertisement(ctx context.Context, port int) {
	if !envconfig.MDNSEnabled() {
		return
	}
	if port <= 0 {
		slog.Warn("mdns not started", "reason", "invalid port", "port", port)
		return
	}

	srv, err := mdns.Register(mdns.RegisterOpts{
		Service: mdns.ServiceNode,
		Port:    port,
		TXT: map[string]string{
			"role":    "node",
			"version": version.Version,
		},
	})
	if err != nil {
		slog.Warn("mdns registration failed", "error", err)
		return
	}

	go func() {
		<-ctx.Done()
		srv.Shutdown()
		slog.Info("mdns unregistered", "service", mdns.ServiceNode)
	}()
}
