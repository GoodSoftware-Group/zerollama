package cmd

import (
	"context"
	"fmt"
	"log/slog"
	"net"
	"net/http"
	"os"
	"os/signal"
	"strconv"
	"strings"
	"syscall"
	"time"

	"github.com/spf13/cobra"

	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/fleet"
	"github.com/ollama/ollama/fleet/mdns"
	"github.com/ollama/ollama/version"
)

// NewFleetCommand registers `zerollama fleet serve` — F3 management node.
func NewFleetCommand() *cobra.Command {
	var listen string
	var peers string
	var pollInterval time.Duration
	var mdnsBrowse bool
	var mdnsAdvertise bool

	cmd := &cobra.Command{
		Use:   "fleet",
		Short: "Fleet management for multi-node zerollama routing",
	}
	serveCmd := &cobra.Command{
		Use:   "serve",
		Short: "Start the fleet management node (warm-model routing)",
		Long:  "Poll zerollama peers via GET /api/status and assign agents to the best node for a model. Peers may be static, discovered via mDNS (F4), or both.",
		RunE: func(_ *cobra.Command, _ []string) error {
			return runFleetServe(listen, peers, pollInterval, mdnsBrowse, mdnsAdvertise)
		},
	}
	serveCmd.Flags().StringVar(&listen, "listen", envconfig.FleetListen(), "Fleet management listen address (host:port)")
	serveCmd.Flags().StringVar(&peers, "peers", envconfig.FleetPeers(), "Comma-separated zerollama base URLs (optional when --mdns)")
	serveCmd.Flags().DurationVar(&pollInterval, "poll-interval", envconfig.FleetPollInterval(), "Peer status poll interval")
	serveCmd.Flags().BoolVar(&mdnsBrowse, "mdns", envconfig.FleetMDNS(), "Browse LAN for zerollama nodes (_zerollama._tcp)")
	serveCmd.Flags().BoolVar(&mdnsAdvertise, "mdns-advertise", envconfig.FleetMDNSAdvertise(), "Advertise fleet endpoint on LAN (_zerollama-fleet._tcp)")
	cmd.AddCommand(serveCmd)
	return cmd
}

func runFleetServe(listen, peers string, pollInterval time.Duration, mdnsBrowse, mdnsAdvertise bool) error {
	if strings.TrimSpace(listen) == "" {
		listen = envconfig.FleetListen()
	}

	var peerURLs []string
	if strings.TrimSpace(peers) != "" {
		parsed, err := fleet.ParsePeers(peers)
		if err != nil {
			return err
		}
		peerURLs = parsed
	}

	if len(peerURLs) == 0 && !mdnsBrowse {
		return fmt.Errorf("fleet peers required: set --peers or ZEROLLAMA_FLEET_PEERS, or enable --mdns / ZEROLLAMA_FLEET_MDNS")
	}

	manager, err := fleet.NewManager(fleet.Config{
		Peers:        peerURLs,
		PollInterval: pollInterval,
		Discovery:    fleet.DiscoveryConfig{Enabled: mdnsBrowse},
	})
	if err != nil {
		return err
	}

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	if mdnsBrowse {
		go fleet.RunMDNSDiscovery(ctx, manager)
	}
	go manager.Run(ctx)

	ln, err := net.Listen("tcp", listen)
	if err != nil {
		return err
	}

	srv := &http.Server{
		Handler:           fleet.NewServer(manager).Handler(),
		ReadHeaderTimeout: 5 * time.Second,
	}
	slog.Info("fleet management listening", "addr", ln.Addr().String(), "static_peers", len(peerURLs), "mdns_browse", mdnsBrowse, "mdns_advertise", mdnsAdvertise)

	if mdnsAdvertise {
		if port, err := mdns.PortFromAddr(ln.Addr()); err == nil {
			startFleetMDNS(ctx, port)
		} else {
			slog.Warn("fleet mdns advertise skipped", "error", err)
		}
	}

	errCh := make(chan error, 1)
	go func() {
		errCh <- srv.Serve(ln)
	}()

	select {
	case <-ctx.Done():
		shutdownCtx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		_ = srv.Shutdown(shutdownCtx)
		return nil
	case err := <-errCh:
		if err == http.ErrServerClosed {
			return nil
		}
		return err
	}
}

func portFromListen(listen string) (int, error) {
	listen = strings.TrimSpace(listen)
	if listen == "" {
		return 0, fmt.Errorf("empty listen address")
	}
	_, portStr, err := net.SplitHostPort(listen)
	if err != nil {
		if strings.HasPrefix(listen, ":") {
			portStr = strings.TrimPrefix(listen, ":")
		} else if _, perr := strconv.Atoi(listen); perr == nil {
			portStr = listen
		} else {
			return 0, err
		}
	}
	port, err := strconv.Atoi(portStr)
	if err != nil || port <= 0 {
		return 0, fmt.Errorf("invalid listen port %q", listen)
	}
	return port, nil
}

func startFleetMDNS(ctx context.Context, port int) {
	srv, err := mdns.Register(mdns.RegisterOpts{
		Service: mdns.ServiceFleet,
		Port:    port,
		TXT: map[string]string{
			"role":    "fleet",
			"version": version.Version,
		},
	})
	if err != nil {
		slog.Warn("fleet mdns registration failed", "error", err)
		return
	}
	go func() {
		<-ctx.Done()
		srv.Shutdown()
		slog.Info("fleet mdns unregistered", "service", mdns.ServiceFleet)
	}()
}
