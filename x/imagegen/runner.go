// Package imagegen provides a unified MLX runner for both LLM and image generation models.
package imagegen

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"log/slog"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/x/imagegen/mlx"
	"github.com/ollama/ollama/x/internal/mlxthread"
)

func Execute(args []string) error {
	slog.SetDefault(slog.New(slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{Level: envconfig.LogLevel()})))

	fs := flag.NewFlagSet("mlx-runner", flag.ExitOnError)
	modelName := fs.String("model", "", "path to model")
	port := fs.Int("port", 0, "port to listen on")

	if err := fs.Parse(args); err != nil {
		return err
	}
	if *modelName == "" {
		return fmt.Errorf("--model is required")
	}
	if *port == 0 {
		return fmt.Errorf("--port is required")
	}

	// CUDA graph capture in libmlxc corrupts batched eval on RTX 5080 class GPUs.
	// Must be set before any MLX GPU call (static init in use_cuda_graphs()).
	_ = os.Setenv("MLX_USE_CUDA_GRAPHS", "false")
	_ = os.Setenv("MLX_DISABLE_COMPILE", "1")

	mode := detectModelMode(*modelName)
	slog.Info("starting mlx runner", "model", *modelName, "port", *port, "mode", mode)
	if mode != ModeImageGen {
		return fmt.Errorf("imagegen runner only supports image generation models")
	}

	if err := mlx.InitMLX(); err != nil {
		slog.Error("unable to initialize MLX", "error", err)
		return err
	}
	slog.Info("MLX library initialized")

	server, err := newServer(*modelName, *port)
	if err != nil {
		return fmt.Errorf("failed to create server: %w", err)
	}

	worker, err := mlxthread.Start("imagegen", func() error {
		if mlx.GPUIsAvailable() {
			mlx.SetDefaultDeviceGPU()
			// CUDA graph capture (mlx.compile) corrupts eval on RTX 5080 class GPUs.
			mlx.DisableCompile()
			mlx.SetCacheLimit(0)
			// WHY Linux-only 12GiB: survival clamp for ~16GB CUDA cards. On Metal UMA
			// (often 64–128GB) the same cap aborts text-encoder materialize mid-batch
			// and surfaces as empty mlx_stream / bogus "GPU OOM". Leave MLX's default
			// Metal working-set limit; override with ZEROLLAMA_IMAGEGEN_MEMORY_LIMIT.
			mlx.ApplyImagegenMemoryLimit()
			// Set CUDA pool release threshold to 0: freed buffers are returned to the
			// OS immediately instead of being held in the async pool. This trades
			// allocation latency for reduced peak VRAM, critical on 16 GB GPUs.
			mlx.SetCudaPoolThreshold(0)
		}
		// Load the model on this OS thread so all weight arrays share the same
		// GPU stream as inference and export operations. The CUDA MLX backend uses
		// thread_local CommandEncoders — cross-thread stream access is not supported.
		return server.loadImageModel()
	})
	if err != nil {
		return err
	}
	server.mlxThread = worker

	mux := http.NewServeMux()
	mux.HandleFunc("/health", server.healthHandler)
	mux.HandleFunc("/completion", server.completionHandler)

	httpServer := &http.Server{Addr: fmt.Sprintf("127.0.0.1:%d", *port), Handler: mux}

	done := make(chan struct{})
	go func() {
		sigCh := make(chan os.Signal, 1)
		signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)
		<-sigCh
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		_ = httpServer.Shutdown(ctx)
		_ = worker.Stop(ctx, func() { mlx.ClearCache() })
		close(done)
	}()

	slog.Info("mlx runner listening", "addr", httpServer.Addr)
	if err := httpServer.ListenAndServe(); err != http.ErrServerClosed {
		return err
	}
	<-done
	return nil
}

func detectModelMode(modelName string) ModelMode {
	modelType := DetectModelType(modelName)
	if modelType != "" {
		switch modelType {
		case "ZImagePipeline", "FluxPipeline", "Flux2KleinPipeline":
			return ModeImageGen
		}
	}
	return ModeLLM
}

type server struct {
	modelName string
	port      int
	mlxThread *mlxthread.Thread
	imageModel ImageModel
}

func newServer(modelName string, port int) (*server, error) {
	s := &server{modelName: modelName, port: port}
	return s, nil
}

func (s *server) healthHandler(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(HealthResponse{Status: "ok"})
}

func (s *server) completionHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	var req Request
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	s.handleImageCompletion(w, r, req)
}
