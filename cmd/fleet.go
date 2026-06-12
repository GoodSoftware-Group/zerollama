package cmd

import (
	"context"
	"fmt"
	"log/slog"
	"net/http"
	"os"
	"os/signal"
	"strings"
	"syscall"
	"time"

	"github.com/spf13/cobra"

	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/fleet"
)

// NewFleetCommand registers `zerollama fleet serve` — F3 management node.
// Why a subcommand not a separate binary: same release artifact, shared envconfig, operators already run zerollama.
func NewFleetCommand() *cobra.Command {
	var listen string
	var peers string
	var pollInterval time.Duration

	cmd := &cobra.Command{
		Use:   "fleet",
		Short: "Fleet management for multi-node zerollama routing",
	}
	serveCmd := &cobra.Command{
		Use:   "serve",
		Short: "Start the fleet management node (warm-model routing)",
		Long:  "Poll zerollama peers via GET /api/status and assign agents to the best node for a model.",
		RunE: func(_ *cobra.Command, _ []string) error {
			return runFleetServe(listen, peers, pollInterval)
		},
	}
	serveCmd.Flags().StringVar(&listen, "listen", envconfig.FleetListen(), "Fleet management listen address (host:port)")
	serveCmd.Flags().StringVar(&peers, "peers", envconfig.FleetPeers(), "Comma-separated zerollama base URLs")
	serveCmd.Flags().DurationVar(&pollInterval, "poll-interval", envconfig.FleetPollInterval(), "Peer status poll interval")
	cmd.AddCommand(serveCmd)
	return cmd
}

func runFleetServe(listen, peers string, pollInterval time.Duration) error {
	if strings.TrimSpace(peers) == "" {
		peers = envconfig.FleetPeers()
	}
	if strings.TrimSpace(peers) == "" {
		return fmt.Errorf("fleet peers required: set --peers or ZEROLLAMA_FLEET_PEERS")
	}
	if strings.TrimSpace(listen) == "" {
		listen = envconfig.FleetListen()
	}

	peerURLs, err := fleet.ParsePeers(peers)
	if err != nil {
		return err
	}

	manager, err := fleet.NewManager(fleet.Config{
		Peers:        peerURLs,
		PollInterval: pollInterval,
	})
	if err != nil {
		return err
	}

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	go manager.Run(ctx)

	srv := &http.Server{
		Addr:              listen,
		Handler:           fleet.NewServer(manager).Handler(),
		ReadHeaderTimeout: 5 * time.Second,
	}
	slog.Info("fleet management listening", "addr", listen, "peers", len(peerURLs))

	errCh := make(chan error, 1)
	go func() {
		errCh <- srv.ListenAndServe()
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
