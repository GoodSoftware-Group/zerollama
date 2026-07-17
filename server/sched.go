package server

import (
	"cmp"
	"context"
	"errors"
	"fmt"
	"log/slog"
	"reflect"
	"slices"
	"sort"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/discover"
	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/format"
	"github.com/ollama/ollama/fs/ggml"
	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/logutil"
	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/server/vram"
	"github.com/ollama/ollama/types/model"
	"github.com/ollama/ollama/x/imagegen"
	"github.com/ollama/ollama/x/mlxrunner"
)

type LlmRequest struct {
	ctx             context.Context //nolint:containedctx
	model           *Model
	opts            api.Options
	sessionDuration *api.Duration
	successCh       chan *runnerRef
	errCh           chan error
	schedAttempts   uint
	fifoSeq         uint64

	// contextShift is a llama-server launch attribute resolved from the
	// request-level shift option before scheduling.
	contextShift bool
	shift        *bool
}

// failIfCanceled reports ctx cancellation on errCh so scheduleRunner does not hang.
// Returns true when the request should stop scheduling.
func (pending *LlmRequest) failIfCanceled() bool {
	err := pending.ctx.Err()
	if err == nil {
		return false
	}
	schedLogWarn("request canceled, failing", pending, "err", err)
	select {
	case pending.errCh <- err:
		schedLogDebug("canceled err sent on errCh", pending)
	default:
		schedLogWarn("errCh full, could not deliver cancel", pending)
	}
	return true
}

type Scheduler struct {
	pending       *pendingQueue
	finishedReqCh chan *LlmRequest
	expiredCh     chan *runnerRef
	unloadedCh    chan any

	// fifoYield, when set, blocks pendingPopNext while runtime has an older ticket.
	fifoYield func() bool

	// loadedMu protects loaded and activeLoading
	loadedMu sync.Mutex

	// activeLoading is the model that we are currently working on loading,
	// including by evicting one or more other models. We can only load
	// one model at a time but new requests to models that already loaded can
	// happen in parallel
	activeLoading    llm.LlamaServer
	activeLoadingKey string // schedulerModelKey for activeLoading; avoids killing same-model load probes
	loaded           map[string]*runnerRef

	loadFn          func(req *LlmRequest, systemInfo ml.SystemInfo, gpus []ml.DeviceInfo, requireFull bool) bool
	newServerFn     func(systemInfo ml.SystemInfo, gpus []ml.DeviceInfo, model string, f *ggml.GGML, adapters []string, projectors []string, opts api.Options, numParallel int, config llm.LlamaServerConfig) (llm.LlamaServer, error)
	getGpuFn        func(ctx context.Context, runners []ml.FilteredRunnerDiscovery) []ml.DeviceInfo
	getSystemInfoFn func() ml.SystemInfo
	waitForRecovery time.Duration

	// When true, GetRunner blocks until ResumeLoads (used when training needs VRAM).
	loadsPaused atomic.Bool

	// loadingFifoSeq is the ticket of the pending request currently being scheduled/loaded.
	loadingFifoSeq atomic.Uint64

	// mlxGate defers unkeyed MLX generate while keyed agent sessions are active.
	mlxGate mlxAgentGate
}

// Default automatic value for number of models we allow per GPU
// Model will still need to fit in VRAM, but loading many small models
// on a large GPU can cause stalling
var defaultModelsPerGPU = 3

var ErrMaxQueue = errors.New("server busy, please try again.  maximum pending requests exceeded")

func InitScheduler(ctx context.Context) *Scheduler {
	maxQueue := envconfig.MaxQueue()
	sched := &Scheduler{
		pending:         newPendingQueue(int(maxQueue)),
		finishedReqCh:   make(chan *LlmRequest, maxQueue),
		expiredCh:       make(chan *runnerRef, maxQueue),
		unloadedCh:      make(chan any, maxQueue),
		loaded:          make(map[string]*runnerRef),
		newServerFn:     llm.NewLlamaServer,
		getGpuFn:        discover.GPUDevices,
		getSystemInfoFn: discover.GetSystemInfo,
		waitForRecovery: 5 * time.Second,
	}
	sched.loadFn = sched.load
	sched.mlxGate = *newMLXAgentGate()
	return sched
}

// schedulerModelKey returns the scheduler map key for a model.
// GGUF-backed models use ModelPath; safetensors/image models without a
// ModelPath use manifest digest so distinct models don't collide.
func schedulerModelKey(m *Model) string {
	if m == nil {
		return ""
	}
	if m.ModelPath != "" {
		return m.ModelPath
	}
	if m.Digest != "" {
		return "digest:" + m.Digest
	}
	if m.Name != "" {
		return "name:" + m.Name
	}
	if m.ShortName != "" {
		return "short:" + m.ShortName
	}
	return ""
}

// context must be canceled to decrement ref count and release the runner
func (s *Scheduler) GetRunner(c context.Context, m *Model, opts api.Options, sessionDuration *api.Duration, shift *bool) (chan *runnerRef, chan error, uint64) {
	if opts.NumCtx < 4 {
		opts.NumCtx = 4
	}

	if m.CheckCapabilities(model.CapabilityVision) == nil {
		// multimodal models require at least 2048 context
		opts.NumCtx = max(opts.NumCtx, 2048)
	}

	req := &LlmRequest{
		ctx:             c,
		model:           m,
		opts:            opts,
		sessionDuration: sessionDuration,
		successCh:       make(chan *runnerRef, 1),
		errCh:           make(chan error, 1),
		shift:           shift,
	}
	req.contextShift = runnerContextShift(req)

	for s.loadsPaused.Load() {
		select {
		case <-c.Done():
			req.errCh <- c.Err()
			return req.successCh, req.errCh, 0
		case <-time.After(50 * time.Millisecond):
		}
	}

	key := schedulerModelKey(req.model)
	s.loadedMu.Lock()
	runner := s.loaded[key]
	s.loadedMu.Unlock()
	if runner != nil && !runner.needsReload(c, req) {
		attrs := s.schedSnapshot()
		attrs = append(attrs, schedRunnerAttrs(runner)...)
		schedLogInfo("GetRunner fast path (already loaded)", req, attrs...)
		req.useLoadedRunner(runner, s.finishedReqCh)
		return req.successCh, req.errCh, 0
	}
	req.fifoSeq = AllocCrossQueueSeq()
	if !s.pending.Push(req) {
		schedLogWarn("pending queue full", req, "max_queue", envconfig.MaxQueue())
		req.errCh <- ErrMaxQueue
	} else {
		attrs := s.schedSnapshot()
		attrs = append(attrs, "needs_reload", runner != nil)
		schedLogInfo("queued for load", req, attrs...)
		if req.ctx != nil {
			go s.dropPendingOnCancel(req)
		}
	}
	return req.successCh, req.errCh, req.fifoSeq
}

func resolveContextShift(shift *bool, m *Model) bool {
	if shift != nil {
		return *shift
	}
	return supportsContextShift(m)
}

// runnerContextShift is the context-shift flag stored on runnerRef and compared in needsReload.
// MLX and other path-less runners do not use llama-server --context-shift; always false.
func runnerContextShift(req *LlmRequest) bool {
	if req == nil || req.model == nil || req.model.ModelPath == "" {
		return false
	}
	return resolveContextShift(req.shift, req.model)
}

func supportsContextShift(m *Model) bool {
	if m == nil {
		return true
	}
	if m.Config.ModelFamily == "deepseek2" || slices.Contains(m.Config.ModelFamilies, "deepseek2") {
		return false
	}
	return true
}

// Returns immediately, spawns go routines for the scheduler which will shutdown when ctx is done
func (s *Scheduler) Run(ctx context.Context) {
	slog.Debug("starting llm scheduler")
	go func() {
		s.processPending(ctx)
	}()

	go s.processFinishedRequests(ctx)
	go s.processExpiredRunners(ctx)
	go s.processSchedWatchdog(ctx)
}

// scheduleExpiredRunner queues an unload without blocking the completion loop.
func (s *Scheduler) scheduleExpiredRunner(runner *runnerRef) {
	go func() {
		s.expiredCh <- runner
	}()
}

func (s *Scheduler) processFinishedRequest(finished *LlmRequest) {
	finishedKey := schedulerModelKey(finished.model)
	s.loadedMu.Lock()
	runner := s.loaded[finishedKey]
	s.loadedMu.Unlock()
	if runner == nil {
		slog.Error("finished request signal received after model unloaded", "modelPath", finishedKey)
		return
	}
	runner.refMu.Lock()
	runner.refCount--
	if runner.refCount <= 0 {
		runner.busySince = time.Time{}
		runner.lastUsedAt = time.Now()
		if runner.sessionDuration <= 0 {
			slog.Debug("runner with zero duration has gone idle, expiring to unload", "runner", runner)
			if runner.expireTimer != nil {
				runner.expireTimer.Stop()
				runner.expireTimer = nil
			}
			s.scheduleExpiredRunner(runner)
		} else if runner.expireTimer == nil {
			slog.Debug("runner with non-zero duration has gone idle, adding timer", "runner", runner, "duration", runner.sessionDuration)
			runner.expireTimer = time.AfterFunc(runner.sessionDuration, func() {
				slog.Debug("timer expired, expiring to unload", "runner", runner)
				runner.refMu.Lock()
				defer runner.refMu.Unlock()
				if runner.expireTimer != nil {
					runner.expireTimer.Stop()
					runner.expireTimer = nil
				}
				s.scheduleExpiredRunner(runner)
			})
			runner.expiresAt = time.Now().Add(runner.sessionDuration)
		} else {
			slog.Debug("runner with non-zero duration has gone idle, resetting timer", "runner", runner, "duration", runner.sessionDuration)
			runner.expireTimer.Reset(runner.sessionDuration)
			runner.expiresAt = time.Now().Add(runner.sessionDuration)
		}
	}
	slog.Debug("after processing request finished event", "runner", runner, "refCount", runner.refCount)
	runner.refMu.Unlock()
}

func (s *Scheduler) processExpiredRunner(runner *runnerRef) {
	schedLogDebug("expiredCh: unload requested", nil, schedRunnerAttrs(runner)...)
	runner.refMu.Lock()
	refCount := runner.refCount
	if refCount > 0 {
		runner.refMu.Unlock()
		schedLogDebug("expiredCh: victim still referenced, will retry", nil, schedRunnerAttrs(runner)...)
		go func(r *runnerRef) {
			time.Sleep(10 * time.Millisecond)
			s.scheduleExpiredRunner(r)
		}(runner)
		return
	}

	s.loadedMu.Lock()
	slog.Debug("got lock to unload expired event", "runner", runner)
	runnerToUnload := s.loaded[runner.modelKey]
	if runnerToUnload == nil {
		s.loadedMu.Unlock()
		runner.refMu.Unlock()
		slog.Debug("duplicate expired event, ignoring", "runner", runner)
		return
	}
	if runner.pid != runnerToUnload.pid {
		slog.Debug("orphaned runner shutting down", "orphan", runner, "loaded", runnerToUnload)
		runner.unload()
		s.loadedMu.Unlock()
		runner.refMu.Unlock()
		return
	}

	slog.Debug("starting background wait for VRAM recovery", "runner", runner)
	runnersSnapshot := make([]ml.FilteredRunnerDiscovery, 0, len(s.loaded))
	for _, r := range s.loaded {
		runnersSnapshot = append(runnersSnapshot, r)
	}
	finished := s.waitForVRAMRecovery(runner, runnersSnapshot)
	runner.unload()
	delete(s.loaded, runner.modelKey)
	s.loadedMu.Unlock()
	runner.refMu.Unlock()
	go func() {
		<-finished
		schedLogDebug("runner unloaded, signaling scheduler", nil, schedRunnerAttrs(runner)...)
		s.unloadedCh <- struct{}{}
	}()
}

func (s *Scheduler) processFinishedRequests(ctx context.Context) {
	for {
		select {
		case <-ctx.Done():
			slog.Debug("shutting down scheduler finished loop")
			return
		case finished := <-s.finishedReqCh:
			s.processFinishedRequest(finished)
		}
	}
}

func (s *Scheduler) processExpiredRunners(ctx context.Context) {
	for {
		select {
		case <-ctx.Done():
			slog.Debug("shutting down scheduler expired loop")
			return
		case runner := <-s.expiredCh:
			s.processExpiredRunner(runner)
		}
	}
}

// processCompleted drives finished/expired handling synchronously (tests only).
func (s *Scheduler) processCompleted(ctx context.Context) {
	for {
		select {
		case <-ctx.Done():
			return
		case finished := <-s.finishedReqCh:
			s.processFinishedRequest(finished)
		case runner := <-s.expiredCh:
			s.processExpiredRunner(runner)
		}
	}
}

func (s *Scheduler) processPending(ctx context.Context) {
	for {
		select {
		case <-ctx.Done():
			slog.Debug("shutting down scheduler pending loop")
			return
		case <-s.pending.WakeCh():
			s.processOnePending(ctx)
		case <-s.unloadedCh:
			if s.pending.Len() > 0 {
				s.pending.notify()
			} else {
				slog.Debug("ignoring unload event with no pending requests")
			}
		}
	}
}

// dropPendingOnCancel removes a queued request when its client context ends.
func (s *Scheduler) dropPendingOnCancel(req *LlmRequest) {
	if req == nil || req.ctx == nil {
		return
	}
	done := req.ctx.Done()
	if done == nil {
		return
	}
	<-done
	if s.pending.Remove(req) {
		s.pending.notify()
	}
}

func (s *Scheduler) notifyPendingIfQueued() {
	if s.pending.Len() > 0 {
		s.pending.notify()
	}
}

func (s *Scheduler) processOnePending(ctx context.Context) {
	maxRunners := envconfig.MaxRunners()
	pending := s.pendingPopNext()
	if pending == nil {
		return
	}
	if pending.fifoSeq != 0 {
		s.loadingFifoSeq.Store(pending.fifoSeq)
		defer s.loadingFifoSeq.Store(0)
	}
	// Block other requests until we get this pending request running
	pending.schedAttempts++
	schedLogDebug("dequeued pending request", pending, s.schedSnapshot()...)

	if pending.failIfCanceled() {
		s.notifyPendingIfQueued()
		return
	}
	logutil.Trace("processing incoming request", "model", pending.model.ModelPath)
	schedLoopStart := time.Now()

	for {
		if pending.failIfCanceled() {
			schedLogDebug("exiting schedule loop (canceled)", pending, "loop_elapsed", time.Since(schedLoopStart))
			break
		}
		var runnerToExpire *runnerRef
		pendingKey := schedulerModelKey(pending.model)
		s.loadedMu.Lock()
		runner := s.loaded[pendingKey]
		loadedCount := len(s.loaded)
		runnersSnapshot := make([]ml.FilteredRunnerDiscovery, 0, len(s.loaded))
		for _, r := range s.loaded {
			runnersSnapshot = append(runnersSnapshot, r)
		}
		s.loadedMu.Unlock()

		if runner != nil {
			if runner.needsReload(ctx, pending) {
				schedLogInfo("loaded runner needs reload, will evict", pending, schedRunnerAttrs(runner)...)
				runnerToExpire = runner
			} else {
				// Runner is usable, return it
				schedLogInfo("using loaded runner", pending, schedRunnerAttrs(runner)...)
				pending.useLoadedRunner(runner, s.finishedReqCh)
				break
			}
		} else if conflict := s.findConcurrencyGroupConflict(pending.model); conflict != nil {
			schedLogInfo("concurrency group conflict, will evict", pending, schedRunnerAttrs(conflict)...)
			runnerToExpire = conflict
		} else if maxRunners > 0 && loadedCount >= int(maxRunners) {
			schedLogInfo("max loaded models, picking eviction victim", pending, "max_runners", maxRunners, "loaded_count", loadedCount)
			runnerToExpire = s.findRunnerToUnload()
		} else {
			// Either no models are loaded or below envconfig.MaxRunners
			// Get a refreshed GPU list
			var gpus []ml.DeviceInfo
			if pending.opts.NumGPU == 0 {
				gpus = []ml.DeviceInfo{}
			} else {
				logutil.Trace("refreshing GPU list", "model", pending.model.ModelPath)
				gpus = s.getGpuFn(ctx, runnersSnapshot)
			}
			logutil.Trace("refreshing system information", "model", pending.model.ModelPath)
			systemInfo := s.getSystemInfoFn()
			if maxRunners <= 0 {
				// No user specified MaxRunners, so figure out what automatic setting to use for the next load attempt
				if pending.opts.NumGPU == 0 {
					// Need to get actual GPU list to set the correct default max models
					logutil.Trace("refreshing GPU list", "model", pending.model.ModelPath)
					g := s.getGpuFn(ctx, runnersSnapshot)
					maxRunners = uint(defaultModelsPerGPU * max(len(g), 1))
				} else {
					maxRunners = uint(defaultModelsPerGPU * max(len(gpus), 1))
				}
				slog.Debug("updating default concurrency", "OLLAMA_MAX_LOADED_MODELS", maxRunners, "gpu_count", len(gpus))
			}

			// Update free memory from currently loaded models
			logutil.Trace("updating free space", "gpu_count", len(gpus), "model", pending.model.ModelPath)
			s.updateFreeSpace(gpus)

			if loadedCount == 0 {
				// No models loaded. Load the model but prefer the best fit.
				schedLogInfo("loading first model (VRAM empty)", pending)
				s.loadFn(pending, systemInfo, gpus, false)
				break
			}

			// More than one loaded model, so we have to see if the
			// new one fits
			schedLogDebug("probing load alongside existing models", pending, "loaded_count", loadedCount)
			needEvict := s.loadFn(pending, systemInfo, gpus, true)
			if !needEvict {
				schedLogInfo("new model fits without eviction", pending)
				break
			}

			schedLogInfo("new model requires eviction", pending)
			runnerToExpire = s.findRunnerToUnload()
		}

		if runnerToExpire == nil {
			// While we were performing load calculations, the loaded runner(s) unloaded in parallel
			// so findRunnerToUnload returned no runners.  We'll try again and the loadedCount should be zero
			schedLogDebug("eviction victim nil after race, retrying", pending)
			continue
		}
		s.drainSameModelPending(ctx, runnerToExpire)
		schedLogInfo("evicting runner for VRAM", pending, schedRunnerAttrs(runnerToExpire)...)
		// Trigger an expiration to unload once it's done
		evictionLocked := false
		for {
			if runnerToExpire.refMu.TryLock() {
				evictionLocked = true
				break
			}
			if pending.failIfCanceled() {
				break
			}
			select {
			case <-ctx.Done():
				return
			case <-time.After(10 * time.Millisecond):
			}
		}
		if !evictionLocked {
			continue
		}
		evictRefCount := runnerToExpire.refCount
		if runnerToExpire.expireTimer != nil {
			runnerToExpire.expireTimer.Stop()
			runnerToExpire.expireTimer = nil
		}
		runnerToExpire.sessionDuration = 0
		if runnerToExpire.refCount <= 0 {
			schedLogDebug("victim idle, sending expiredCh immediately", pending, "ref_count", evictRefCount)
		} else {
			schedLogInfo("victim still in use, forcing expiration after refs drain", pending, "ref_count", evictRefCount)
		}
		// Always queue expiration; processExpiredRunner retries until refCount reaches zero.
		s.expiredCh <- runnerToExpire
		runnerToExpire.refMu.Unlock()

		evictionPaused := false
		if !s.loadsPaused.Load() {
			s.PauseNewLoads()
			evictionPaused = true
			schedLogDebug("paused new loads during eviction wait", pending)
		}
		resumeAfterEvict := func() {
			if evictionPaused {
				s.ResumeLoads()
			}
		}

		// Wait for the unload to happen
		evictWaitStart := time.Now()
		select {
		case <-ctx.Done():
			resumeAfterEvict()
			slog.Debug("shutting down scheduler pending loop")
			return
		case <-pending.ctx.Done():
			evictAttrs := append([]any{"evict_wait", time.Since(evictWaitStart)}, schedRunnerAttrs(runnerToExpire)...)
			schedLogWarn("client canceled while waiting for eviction unload", pending, evictAttrs...)
			resumeAfterEvict()
			pending.failIfCanceled()
			break
		case <-s.unloadedCh:
			evictAttrs := append([]any{"evict_wait", time.Since(evictWaitStart)}, schedRunnerAttrs(runnerToExpire)...)
			schedLogInfo("eviction unload completed", pending, evictAttrs...)
			resumeAfterEvict()
			continue
		}
	}
	s.notifyPendingIfQueued()
}

// pendingPopNext returns the next pending request, preferring models already loaded.
// Canceled requests are discarded in batch so stale client disconnects do not block the queue.
func (s *Scheduler) pendingPopNext() *LlmRequest {
	for {
		if s.fifoYield != nil && s.fifoYield() {
			return nil
		}
		prefer := make(map[string]struct{})
		s.loadedMu.Lock()
		for key := range s.loaded {
			prefer[key] = struct{}{}
		}
		s.loadedMu.Unlock()
		req := s.pending.PopPreferringKeys(prefer)
		if req == nil {
			return nil
		}
		if req.ctx.Err() == nil {
			return req
		}
		req.failIfCanceled()
	}
}

// drainSameModelPending dispatches queued requests for runner's model before eviction.
func (s *Scheduler) drainSameModelPending(ctx context.Context, runner *runnerRef) {
	matched := s.pending.DrainMatching(runner.modelKey)
	if len(matched) == 0 {
		return
	}
	slog.Debug("draining same-model pending before eviction", "runner", runner, "count", len(matched))
	var requeue []*LlmRequest
	for _, req := range matched {
		if req.ctx.Err() != nil {
			continue
		}
		if runner.needsReload(ctx, req) {
			requeue = append(requeue, req)
			continue
		}
		req.useLoadedRunner(runner, s.finishedReqCh)
	}
	if len(requeue) > 0 {
		s.pending.RequeueFront(requeue)
	}
}

// Complete the pending request and send the runner back to the requester
// Wires up a finished event after the request context is completed
// Updates session duration, and resets expiration timer
func (pending *LlmRequest) useLoadedRunner(runner *runnerRef, finished chan *LlmRequest) {
	if err := runner.waitUntilReady(pending.ctx); err != nil {
		pending.errCh <- err
		return
	}
	runner.refMu.Lock()
	defer runner.refMu.Unlock()
	if runner.loading || runner.llama == nil {
		pending.errCh <- errors.New("runner unavailable")
		return
	}
	wasIdle := runner.refCount == 0
	runner.refCount++
	now := time.Now()
	runner.lastUsedAt = now
	if wasIdle {
		runner.busySince = now
	}
	if runner.expireTimer != nil {
		runner.expireTimer.Stop()
		runner.expireTimer = nil
	}
	if pending.sessionDuration != nil {
		runner.sessionDuration = pending.sessionDuration.Duration
	}
	pending.successCh <- runner
	go func() {
		<-pending.ctx.Done()
		slog.Debug("context for request finished", "runner", runner)
		finished <- pending
	}()
}

// clearActiveLoading closes the in-flight load subprocess. llama is the handle for the
// current load attempt; if another goroutine already cleared activeLoading (e.g. VRAM
// handoff), llama is closed directly so the subprocess is not leaked.
func (s *Scheduler) clearActiveLoading(llama llm.LlamaServer) {
	s.loadedMu.Lock()
	defer s.loadedMu.Unlock()
	if s.activeLoading != nil {
		s.activeLoading.Close()
		s.activeLoading = nil
		s.activeLoadingKey = ""
		return
	}
	if llama != nil {
		llama.Close()
	}
}

// load creates a new model based on req and loads it. If requireFull is true then the model must be loaded fully onto GPUs
// (if any). Returns whether the scheduler needs to evict a model to make this one fit.
func (s *Scheduler) load(req *LlmRequest, systemInfo ml.SystemInfo, gpus []ml.DeviceInfo, requireFull bool) bool {
	loadStart := time.Now()
	schedLogDebug("load() begin", req, "require_full", requireFull, "active_loading", s.schedActiveLoadingPath())
	numParallel := max(int(envconfig.NumParallel()), 1)

	// Embedding models should always be loaded with parallel=1
	if req.model.CheckCapabilities(model.CapabilityCompletion) != nil {
		numParallel = 1
	}

	// Some architectures are not safe with num_parallel > 1.
	// ref: https://github.com/ollama/ollama/issues/4165
	if slices.Contains([]string{"mllama", "qwen3vl", "qwen3vlmoe", "qwen35", "qwen35moe", "qwen3next", "lfm2", "lfm2moe", "nemotron_h", "nemotron_h_moe"}, req.model.Config.ModelFamily) && numParallel != 1 {
		numParallel = 1
		slog.Warn("model architecture does not currently support parallel requests", "architecture", req.model.Config.ModelFamily)
	}

	sessionDuration := envconfig.KeepAlive()
	if req.sessionDuration != nil {
		sessionDuration = req.sessionDuration.Duration
	}

	s.loadedMu.Lock()
	llama := s.activeLoading

	if llama == nil {
		schedLogDebug("creating new llama server subprocess", req, "num_parallel", numParallel)
		slog.Info(
			"loading model",
			"name", req.model.ShortName,
			"path", req.model.ModelPath,
		)
		ctx := req.ctx
		if ctx == nil {
			ctx = context.Background()
		}
		if deferInferenceToRuntime(req.model) || darwinRuntimeMetalBlocksGgml(ctx, req.model) {
			slog.Info(
				"skipping ggml runner load; model uses python runtime",
				"name", req.model.ShortName,
				"path", req.model.ModelPath,
			)
			req.errCh <- ErrRuntimeInferenceModel
			s.loadedMu.Unlock()
			return false
		}
		if skip, skipErr := schedSkipGgmlRunnerLoad(req.model); skip {
			slog.Info(
				"skipping ggml runner load; edge / llama-server policy",
				"name", req.model.ShortName,
				"path", req.model.ModelPath,
				"error", skipErr,
			)
			req.errCh <- skipErr
			s.loadedMu.Unlock()
			return false
		}
		if darwinGgmlContentionWithRuntime(ctx, req.model) {
			slog.Info(
				"skipping ggml runner load; runtime sidecar holds Metal on darwin",
				"name", req.model.ShortName,
			)
			req.errCh <- ErrDarwinMetalContention
			s.loadedMu.Unlock()
			return false
		}
		// Why after contention checks: PrepareForLegacyRunner evicts the runtime sidecar;
		// do not handoff VRAM for a ggml load we are about to skip.
		vram.PrepareForLegacyRunner(ctx)
		var err error
		if !req.model.IsMLX() {
			f, loadErr := llm.LoadModel(req.model.ModelPath, 1024)
			if loadErr != nil {
				slog.Info("failed to load model metadata", "model", req.model.ModelPath, "error", loadErr)
				req.errCh <- loadErr
				s.loadedMu.Unlock()
				return false
			}
			config := llamaServerConfigForModel(req.model, req.contextShift, req.opts)
			llama, err = s.newServerFn(systemInfo, gpus, req.model.ModelPath, f, req.model.AdapterPaths, req.model.ProjectorPaths, req.opts, numParallel, config)
			if err != nil {
				// some older models are not compatible with newer versions of llama.cpp
				// show a generalized compatibility error until there is a better way to
				// check for model compatibility
				if errors.Is(err, ggml.ErrUnsupportedFormat) || strings.Contains(err.Error(), "failed to load model") {
					err = fmt.Errorf("%v: this model may be incompatible with your version of Ollama. If you previously pulled this model, try updating it by running `zerollama pull %s`", err, req.model.ShortName)
				}
			}
		} else {
			modelName := req.model.ShortName
			if slices.Contains(req.model.Config.Capabilities, "image") {
				llama, err = imagegen.NewServer(modelName)
			} else {
				llama, err = mlxrunner.NewClient(modelName, req.opts.NumCtx)
			}
		}
		if err != nil {
			slog.Info("failed to create server", "model", req.model.ShortName, "error", err)
			req.errCh <- err
			s.loadedMu.Unlock()
			return false
		}

		s.activeLoading = llama
		s.activeLoadingKey = schedulerModelKey(req.model)
		schedLogDebug("activeLoading set", req, "path", llama.ModelPath(), "model_key", s.activeLoadingKey)
	} else {
		wantPath := req.model.ModelPath
		if wantPath == "" {
			wantPath = req.model.ShortName
		}
		if s.activeLoading.ModelPath() != wantPath {
			panic(fmt.Errorf("attempting to load different model after eviction (original %v new %v)", s.activeLoading.ModelPath(), wantPath))
		}
		schedLogDebug("reusing activeLoading llama server", req, "path", wantPath)
	}

	s.loadedMu.Unlock()

	systemTotalMemory := systemInfo.TotalMemory
	systemFreeMemory := systemInfo.FreeMemory
	systemSwapFreeMemory := systemInfo.FreeSwap
	slog.Info("system memory", "total", format.HumanBytes2(systemTotalMemory), "free", format.HumanBytes2(systemFreeMemory), "free_swap", format.HumanBytes2(systemSwapFreeMemory))

	for _, gpu := range gpus {
		available := gpu.FreeMemory - envconfig.GpuOverhead() - gpu.MinimumMemory()
		if gpu.FreeMemory < envconfig.GpuOverhead()+gpu.MinimumMemory() {
			available = 0
		}
		slog.Info("gpu memory", "id", gpu.ID, "library", gpu.Library,
			"available", format.HumanBytes2(available),
			"free", format.HumanBytes2(gpu.FreeMemory),
			"minimum", format.HumanBytes2(gpu.MinimumMemory()),
			"overhead", format.HumanBytes2(envconfig.GpuOverhead()))
	}

	schedLogDebug("calling llama.Load (fit/alloc/commit)", req, "gpu_count", len(gpus))
	llamaLoadStart := time.Now()
	gpuIDs, err := llama.Load(req.ctx, systemInfo, gpus, requireFull)
	schedLogLoadPhase(req, "llama.Load returned", llamaLoadStart, "err", err, "need_evict", errors.Is(err, llm.ErrLoadRequiredFull))
	if err != nil {
		if errors.Is(err, llm.ErrLoadRequiredFull) {
			if !requireFull {
				// No other models loaded, yet we still don't fit, so report an error
				schedLogWarn("model too large for system memory", req, "require_full", requireFull)
				s.clearActiveLoading(llama)
				req.errCh <- err
			} else {
				// Fit probe alongside an loaded model: partial GPU is fine after eviction.
				// Drop the probe subprocess so eviction sees accurate VRAM and the retry
				// can commit with requireFull=false (hybrid GPU+CPU offload).
				schedLogInfo("Load requires eviction (ErrLoadRequiredFull)", req)
				s.clearActiveLoading(llama)
				schedLogDebug("cleared load probe before eviction", req)
			}
			return true
		}

		schedLogWarn("Load failed", req, "error", err, "total_elapsed", time.Since(loadStart))
		s.clearActiveLoading(llama)
		req.errCh <- err
		return false
	}

	// Determine if we have discrete GPUs which we should monitor VRAM usage on during shutdown
	discreteGPUs := false
iGPUScan:
	for _, devid := range gpuIDs {
		for _, dev := range gpus {
			if dev.DeviceID == devid {
				if !dev.Integrated {
					discreteGPUs = true
					break iGPUScan
				}
			}
		}
	}

	totalSize, vramSize := llama.MemorySize()
	runner := &runnerRef{
		model:           req.model,
		modelPath:       req.model.ModelPath,
		modelKey:        schedulerModelKey(req.model),
		llama:           llama,
		Options:         &req.opts,
		sessionDuration: sessionDuration,
		gpus:            gpuIDs,
		discreteGPUs:    discreteGPUs,
		isImagegen:      slices.Contains(req.model.Config.Capabilities, "image"),
		totalSize:       totalSize,
		vramSize:        vramSize,
		loading:         true,
		loadDone:        make(chan struct{}),
		lastUsedAt:      time.Now(),
		pid:             llama.Pid(),
		contextShift:    runnerContextShift(req),
	}
	runner.numParallel = numParallel

	s.loadedMu.Lock()
	if oldRunner, ok := s.loaded[runner.modelKey]; ok {
		// Shouldn't happen, but safeguard against leaking a runner
		slog.Warn("model was still loaded", "old_runner", oldRunner, "new_runner", runner)
		oldRunner.refMu.Lock()
		oldRunner.unload()
		oldRunner.refMu.Unlock()
	}
	s.activeLoading = nil
	s.activeLoadingKey = ""
	s.loaded[runner.modelKey] = runner
	slog.Info("loaded runners", "count", len(s.loaded))
	s.loadedMu.Unlock()
	schedLogDebug("runner registered, waiting for subprocess ready", req,
		"pid", runner.pid, "vram_size", runner.vramSize, "llama_load_elapsed", time.Since(llamaLoadStart))

	go func() {
		defer close(runner.loadDone)
		waitStart := time.Now()
		if err = llama.WaitUntilRunning(req.ctx); err != nil {
			schedLogWarn("WaitUntilRunning failed", req, "error", err, "wait_elapsed", time.Since(waitStart), "total_elapsed", time.Since(loadStart))
			runner.refMu.Lock()
			runner.loading = false
			runner.refMu.Unlock()
			req.errCh <- err
			s.scheduleExpiredRunner(runner)
			return
		}
		schedLogInfo("model ready", req,
			"pid", runner.pid,
			"wait_elapsed", time.Since(waitStart),
			"total_elapsed", time.Since(loadStart),
		)
		slog.Info(
			"model ready",
			"name", runner.model.ShortName,
			"path", runner.modelPath,
			"pid", runner.pid,
		)
		runner.refMu.Lock()
		if runner.pid < 0 {
			runner.pid = llama.Pid()
		}
		syncRunnerLoadOptions(runner)
		runner.refCount++
		runner.loading = false
		runner.lastUsedAt = time.Now()
		runner.refMu.Unlock()
		go func() {
			<-req.ctx.Done()
			slog.Debug("context for request finished")
			s.finishedReqCh <- req
		}()
		req.successCh <- runner
	}()

	return false
}

// waitUntilReady blocks until the runner finishes its initial load or ctx is canceled.
func (runner *runnerRef) waitUntilReady(ctx context.Context) error {
	for {
		runner.refMu.Lock()
		loading := runner.loading
		done := runner.loadDone
		llama := runner.llama
		runner.refMu.Unlock()

		if !loading {
			if llama == nil {
				return errors.New("runner unavailable")
			}
			return nil
		}
		if done == nil {
			select {
			case <-ctx.Done():
				return ctx.Err()
			case <-time.After(50 * time.Millisecond):
			}
			continue
		}
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-done:
			runner.refMu.Lock()
			loading = runner.loading
			llama = runner.llama
			runner.refMu.Unlock()
			if llama == nil || loading {
				return errors.New("runner failed to load")
			}
			return nil
		}
	}
}

func (s *Scheduler) updateFreeSpace(allGpus []ml.DeviceInfo) {
	if len(allGpus) == 0 {
		return
	}
	predMap := map[ml.DeviceID]uint64{} // Sum up the total predicted usage per GPU for all runners
	s.loadedMu.Lock()
	runners := make([]*runnerRef, 0, len(s.loaded))
	for _, r := range s.loaded {
		runners = append(runners, r)
	}
	s.loadedMu.Unlock()
	for _, r := range runners {
		r.refMu.Lock()
		if r.llama != nil {
			for _, gpu := range allGpus {
				predMap[gpu.DeviceID] += r.llama.VRAMByGPU(gpu.DeviceID)
			}
		} else {
			slog.Warn("unexpected nil runner reference, memory prediction may be incorrect")
		}
		r.refMu.Unlock()
	}

	// Now that we've summed up all the GPU usage predictions across all the loaded runners, update the gpu list
	for i := range allGpus {
		if p, ok := predMap[allGpus[i].DeviceID]; ok {
			slog.Debug("gpu reported", "gpu", allGpus[i].ID, "library", allGpus[i].Library, "available", format.HumanBytes2(allGpus[i].FreeMemory))
			if p > allGpus[i].TotalMemory {
				// Shouldn't happen
				slog.Warn("predicted usage exceeds VRAM", "gpu", allGpus[i].ID, "totalMemory", allGpus[i].TotalMemory, "predicted", p)
				allGpus[i].FreeMemory = 0
			} else if (allGpus[i].TotalMemory - p) < allGpus[i].FreeMemory { // predicted free is smaller than reported free, use it
				// TODO maybe we should just always trust our numbers, since cuda's free memory reporting is laggy
				// and we might unload models we didn't actually need to.  The risk is if some other GPU intensive app is loaded
				// after we start our first runner, then we'll never account for that, so picking the smallest free value seems prudent.
				allGpus[i].FreeMemory = allGpus[i].TotalMemory - p
			}
			slog.Info("updated VRAM based on existing loaded models", "gpu", allGpus[i].ID, "library", allGpus[i].Library, "total", format.HumanBytes2(allGpus[i].TotalMemory), "available", format.HumanBytes2(allGpus[i].FreeMemory))
		}
	}
}

// TODO consolidate sched_types.go
type runnerRef struct {
	refMu    sync.Mutex
	refCount uint // prevent unloading if > 0

	llama        llm.LlamaServer
	pid          int
	loading      bool          // True only during initial load, then false forever
	loadDone     chan struct{} // closed when initial load finishes (success or failure)
	lastUsedAt   time.Time     // updated on acquire and when going idle (LRU reclaim)
	busySince    time.Time     // set when refCount goes 0→1; cleared when idle
	gpus         []ml.DeviceID // Recorded at time of provisioning
	discreteGPUs bool          // True if all devices are discrete GPUs - used to skip VRAM recovery check for iGPUs
	isImagegen   bool          // True if loaded via imagegen runner (vs mlxrunner)
	vramSize     uint64
	totalSize    uint64

	sessionDuration time.Duration
	expireTimer     *time.Timer
	expiresAt       time.Time

	model       *Model
	modelPath   string
	modelKey    string
	numParallel int
	contextShift bool
	loadedMeta  api.LoadedModelMetadata
	*api.Options
}

// The refMu must already be held when calling unload
func (runner *runnerRef) unload() {
	if runner.expireTimer != nil {
		runner.expireTimer.Stop()
		runner.expireTimer = nil
	}
	if runner.llama != nil {
		runner.llama.Close()
	}
	runner.model = nil
	runner.Options = nil
	runner.gpus = nil
	runner.contextShift = false
}

// syncRunnerLoadOptions updates runner.Options to the context size actually allocated
// at load. llm.NewLlamaServer receives opts by value and may clamp NumCtx to
// n_ctx_train without writing back to the scheduler's LlmRequest.opts copy.
func syncRunnerLoadOptions(runner *runnerRef) {
	if runner == nil || runner.Options == nil || runner.llama == nil {
		return
	}
	effective := runner.llama.ContextLength()
	if effective <= 0 {
		return
	}
	if runner.Options.NumCtx != effective {
		slog.Debug("sync runner num_ctx to effective load size",
			"model", runner.model.ShortName,
			"requested", runner.Options.NumCtx,
			"effective", effective,
		)
	}
	runner.Options.NumCtx = effective
	// Only clamp NumBatch down when it exceeds the effective context — preserve explicit user requests.
	if runner.Options.NumBatch > effective {
		runner.Options.NumBatch = effective
	}
	runner.loadedMeta = probeRunnerMetadata(runner)
}

func (runner *runnerRef) needsReload(ctx context.Context, req *LlmRequest) bool {
	slog.Debug("evaluating already loaded", "model", schedulerModelKey(req.model))
	runner.refMu.Lock()
	defer runner.refMu.Unlock()

	reloadReason := func(reason string, attrs ...any) bool {
		// Why INFO + reason: needs_reload=true alone did not explain cold reload loops
		// (common: num_ctx mismatch after MLX context cap fix).
		args := append([]any{"model", schedulerModelKey(req.model), "reload_reason", reason}, attrs...)
		slog.Info("runner needs reload", args...)
		fields := map[string]any{
			"model":         schedulerModelKey(req.model),
			"reload_reason": reason,
		}
		for i := 0; i+1 < len(attrs); i += 2 {
			if k, ok := attrs[i].(string); ok {
				fields[k] = attrs[i+1]
			}
		}
		RecordAgentStatsEvent("runner_reload", fields)
		return true
	}

	// Check if runner type (imagegen vs mlxrunner) matches what's requested.
	wantImagegen := slices.Contains(req.model.Config.Capabilities, "image")
	if runner.isImagegen != wantImagegen {
		return reloadReason("runner_type_mismatch",
			"loaded_imagegen", runner.isImagegen,
			"want_imagegen", wantImagegen,
		)
	}

	// Reuse the runner while the initial load is still in progress.
	if runner.loading {
		return false
	}

	timeout := 10 * time.Second

	if runner.Options == nil {
		return reloadReason("nil_options")
	}

	// Don't reload runner if num_gpu=-1 was provided
	optsExisting := runner.Options.Runner
	optsNew := req.opts.Runner
	if optsNew.NumGPU < 0 {
		optsExisting.NumGPU = -1
		optsNew.NumGPU = -1
	}

	if req.model.ModelPath != "" {
		contextShift := runnerContextShift(req)
		if runner.contextShift != contextShift {
			return reloadReason("context_shift_changed",
				"loaded", runner.contextShift,
				"want", contextShift,
			)
		}
	}

	ctx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()

	// Compare against the loaded runner's effective KV size, not only stored Options.
	// Stored Options can still reflect the pre-clamp request when n_ctx_train is smaller.
	// Only reload when the request needs MORE context than the loaded KV — requesting less
	// is fine since llama.cpp just uses fewer slots from the pre-allocated KV.
	// Also normalize optsNew.NumCtx for the DeepEqual below: a smaller-than-loaded ctx
	// is served by the existing runner without reload.
	if runner.llama != nil {
		if effective := runner.llama.ContextLength(); effective > 0 {
			if optsNew.NumCtx > effective {
				return reloadReason("num_ctx_exceeds_loaded_kv",
					"loaded_ctx", effective,
					"want_ctx", optsNew.NumCtx,
				)
			}
			if optsNew.NumCtx < effective {
				optsNew.NumCtx = optsExisting.NumCtx // treat "fits in loaded ctx" as same
			}
		}
	}

	// Runner options (num_ctx, num_gpu, num_batch, …) are fixed at llama.Load time.
	// Manifest or request changes to num_ctx require a reload — KV is pre-sized at load.
	if !reflect.DeepEqual(runner.model.AdapterPaths, req.model.AdapterPaths) {
		return reloadReason("adapter_paths_changed")
	}
	if !reflect.DeepEqual(runner.model.ProjectorPaths, req.model.ProjectorPaths) {
		return reloadReason("projector_paths_changed")
	}
	if !runner.model.IsMLX() && !reflect.DeepEqual(optsExisting, optsNew) {
		return reloadReason("runner_options_changed",
			"loaded_num_ctx", optsExisting.NumCtx,
			"want_num_ctx", optsNew.NumCtx,
			"loaded_num_gpu", optsExisting.NumGPU,
			"want_num_gpu", optsNew.NumGPU,
		)
	}
	if runner.llama != nil {
		if runner.model.IsMLX() {
			// MLX runs inference on a single worker thread; Ping can block behind prefill
			// or fail spuriously while the subprocess is healthy.
		if runner.llama.HasExited() {
			attrs := []any{"pid", runner.pid}
			if er, ok := runner.llama.(interface{ ExitError() error }); ok {
				if err := er.ExitError(); err != nil {
					attrs = append(attrs, "exit_err", err, "exit", llm.ExitStatusFromError(err))
				}
			}
			if sr, ok := runner.llama.(interface{ LastStatusError() string }); ok {
				if msg := sr.LastStatusError(); msg != "" {
					attrs = append(attrs, "status_err", msg)
				}
			}
			return reloadReason("mlx_runner_exited", attrs...)
		}
		} else if runner.llama.Ping(ctx) != nil {
			return reloadReason("ping_failed")
		}
		if sb, ok := runner.llama.(interface{ BinaryStale() bool }); ok && sb.BinaryStale() {
			return reloadReason("llama_server_binary_rebuilt", "pid", runner.pid)
		}
	}

	return false
}

// Free memory reporting on GPUs can lag for a while even after the runner
// exits, so we have to keep checking until we see the available memory recover,
// otherwise subsequent model loads will get far less layers loaded or worse
// case, may completely fall back to CPU mode.
// This routine must be called before the runner unloads so it can establish
// a before and after GPU memory allocation.  The returned channel
// will be notified when we're done waiting, or have timed out and should
// proceed anyway
// filterRunnerDiscovery returns runners except skip (used while evicting skip).
func filterRunnerDiscovery(runners []ml.FilteredRunnerDiscovery, skip *runnerRef) []ml.FilteredRunnerDiscovery {
	if skip == nil {
		return runners
	}
	out := make([]ml.FilteredRunnerDiscovery, 0, len(runners))
	for _, r := range runners {
		if rr, ok := r.(*runnerRef); ok && rr == skip {
			continue
		}
		out = append(out, r)
	}
	return out
}

func (s *Scheduler) waitForVRAMRecovery(runner *runnerRef, runners []ml.FilteredRunnerDiscovery) chan any {
	finished := make(chan any, 1)

	// CPU, Metal and iGPUs don't need checking, so no waiting required
	if len(runner.gpus) == 0 || !runner.discreteGPUs ||
		(len(runner.gpus) == 1 && runner.gpus[0].Library == "Metal") {
		finished <- struct{}{}
		slog.Debug("no need to wait for VRAM recovery", "runner", runner)
		return finished
	}
	start := time.Now()

	// Do not query the runner we are about to kill.
	refreshRunners := filterRunnerDiscovery(runners, runner)

	// Establish a baseline before we unload
	gpusBefore := s.getGpuFn(context.Background(), refreshRunners)
	var totalMemoryBefore, freeMemoryBefore uint64
	for _, gpu := range gpusBefore {
		totalMemoryBefore += gpu.TotalMemory
		freeMemoryBefore += gpu.FreeMemory
	}
	totalMemoryNow := totalMemoryBefore
	freeMemoryNow := freeMemoryBefore

	go func() {
		// typical convergence is 0.5-1.5s - If it takes too long to discover and converge, let the scheduler estimate VRAM usage
		ctx, cancel := context.WithTimeout(context.Background(), s.waitForRecovery)
		defer cancel()
		ticker := time.NewTicker(250 * time.Millisecond)
		defer ticker.Stop()
		for {
			select {
			case <-ticker.C:
				// Query GPUs, look for free to go back up
				gpusNow := s.getGpuFn(ctx, refreshRunners)
				totalMemoryNow = 0
				freeMemoryNow = 0
				for _, gpu := range gpusNow {
					totalMemoryNow += gpu.TotalMemory
					freeMemoryNow += gpu.FreeMemory
				}
				// NVML/bootstrap often lags after CUDA runner exit; credit unloaded VRAM.
				if freeMemoryNow <= freeMemoryBefore && runner.vramSize > 0 {
					freeMemoryNow = freeMemoryBefore + runner.vramSize
					if totalMemoryBefore > 0 && freeMemoryNow > totalMemoryBefore {
						freeMemoryNow = totalMemoryBefore
					}
				}
				if freeMemoryNow > freeMemoryBefore {
					logutil.Trace("gpu VRAM convergence", "percent", int(float32(freeMemoryNow-freeMemoryBefore)/float32(runner.vramSize)*100))
				} else {
					logutil.Trace("gpu VRAM convergence", "percent", 0)
				}
				// If we're within ~75% of the estimated memory usage recovered, bail out
				if float32(freeMemoryNow-freeMemoryBefore) > float32(runner.vramSize)*0.75 {
					slog.Debug(fmt.Sprintf("gpu VRAM free memory converged after %0.2f seconds", time.Since(start).Seconds()), "free_before", format.HumanBytes2(freeMemoryBefore), "free_now", format.HumanBytes2(freeMemoryNow), "runner", runner)
					finished <- struct{}{}
					return
				}
			case <-ctx.Done():
				slog.Debug("gpu VRAM usage didn't recover within timeout", "seconds", time.Since(start).Seconds(), "free_before", format.HumanBytes2(freeMemoryBefore), "free_now", format.HumanBytes2(freeMemoryNow), "runner", runner)
				finished <- struct{}{}
				return
			}
		}
	}()
	return finished
}

func (runner *runnerRef) LogValue() slog.Value {
	if runner == nil {
		return slog.StringValue("nil")
	}
	modelID := runner.modelPath
	if modelID == "" {
		modelID = runner.modelKey
	}
	attrs := []slog.Attr{}
	if runner.model != nil {
		attrs = append(attrs, slog.String("name", runner.model.Name))
	}
	if len(runner.gpus) > 0 {
		attrs = append(attrs,
			slog.Any("inference", runner.gpus),
		)
	}
	attrs = append(attrs,
		slog.String("size", format.HumanBytes2(runner.totalSize)),
		slog.String("vram", format.HumanBytes2(runner.vramSize)),
		slog.Int("parallel", runner.numParallel),
		slog.Int("pid", runner.pid),
		slog.String("model", modelID),
	)
	if runner.Options != nil {
		attrs = append(attrs, slog.Int("num_ctx", runner.Options.NumCtx))
	}
	return slog.GroupValue(attrs...)
}

// Implements discover.RunnerDiscovery
func (runner *runnerRef) GetPort() int {
	if runner.llama != nil {
		return runner.llama.GetPort()
	}
	return -1
}

func (runner *runnerRef) GetDeviceInfos(ctx context.Context) []ml.DeviceInfo {
	if runner.llama != nil {
		return runner.llama.GetDeviceInfos(ctx)
	}
	return nil
}

func (runner *runnerRef) GetActiveDeviceIDs() []ml.DeviceID {
	return runner.gpus
}

func (runner *runnerRef) HasExited() bool {
	if runner.llama != nil {
		return runner.llama.HasExited()
	}
	return true
}

type ByDurationAndName []*runnerRef

func (a ByDurationAndName) Len() int      { return len(a) }
func (a ByDurationAndName) Swap(i, j int) { a[i], a[j] = a[j], a[i] }
func (a ByDurationAndName) Less(i, j int) bool {
	// Primary sort by session duration (uint64 to handle negatives)
	d1 := uint64(a[i].sessionDuration)
	d2 := uint64(a[j].sessionDuration)
	if d1 != d2 {
		return d1 < d2
	}
	// Secondary sort by model key/path lex order
	n1 := a[i].modelPath
	if n1 == "" {
		n1 = a[i].modelKey
	}
	n2 := a[j].modelPath
	if n2 == "" {
		n2 = a[j].modelKey
	}
	return n1 < n2
}

// TODO - future consideration to pick runners based on size
// type BySize []*runnerRef
// func (a BySize) Len() int           { return len(a) }
// func (a BySize) Swap(i, j int)      { a[i], a[j] = a[j], a[i] }
// func (a BySize) Less(i, j int) bool { return a[i].vramSize < a[j].vramSize }

// findRunnerToUnload finds a runner to unload to make room for a new model
func (s *Scheduler) findRunnerToUnload() *runnerRef {
	s.loadedMu.Lock()
	runnerList := make([]*runnerRef, 0, len(s.loaded))
	for _, r := range s.loaded {
		runnerList = append(runnerList, r)
	}
	s.loadedMu.Unlock()
	if len(runnerList) == 0 {
		slog.Debug("no loaded runner to unload")
		return nil
	}

	// In the future we can enhance the algorithm to be smarter about picking the optimal runner to unload
	// e.g., if we have multiple options, will one make room for the request?
	sort.Sort(ByDurationAndName(runnerList))

	protected := s.mlxGate.protectedModelKeys()

	// First try to find a runner that's already idle (skip fulfillment-protected models).
	for _, runner := range runnerList {
		if _, skip := protected[runner.modelKey]; skip {
			continue
		}
		runner.refMu.Lock()
		rc := runner.refCount
		runner.refMu.Unlock()
		if rc == 0 {
			schedLogDebug("findRunnerToUnload: idle victim", nil, schedRunnerAttrs(runner)...)
			return runner
		}
	}
	// None appear idle, just wait for the one with the shortest duration (still skip protected).
	for _, runner := range runnerList {
		if _, skip := protected[runner.modelKey]; skip {
			continue
		}
		victimAttrs := append([]any{"runner_count", len(runnerList)}, schedRunnerAttrs(runner)...)
		schedLogDebug("findRunnerToUnload: no idle runners, picking shortest keep-alive", nil, victimAttrs...)
		return runner
	}
	if len(protected) > 0 {
		slog.Debug("findRunnerToUnload: only fulfillment-protected runners loaded", "protected", len(protected))
	}
	return nil
}

func (s *Scheduler) unloadRunnersExcept(keepModelKey string) {
	// keepModelKey: preserve one loaded model during targeted VRAM prep (e.g. imagegen
	// reload). Empty keepModelKey means UnloadAllRunners — defer runners with refCount>0
	// so active HTTP streams (image NDJSON) finish instead of losing the client while the
	// MLX subprocess keeps running orphaned in the background.
	s.loadedMu.Lock()
	if s.activeLoading != nil {
		if keepModelKey != "" && s.activeLoadingKey == keepModelKey {
			slog.Debug("keeping activeLoading probe for requested model", "model_key", keepModelKey)
		} else {
			slog.Debug("shutting down currently loading runner", "model_key", s.activeLoadingKey)
			s.activeLoading.Close()
			s.activeLoading = nil
			s.activeLoadingKey = ""
		}
	}

	runners := make([]*runnerRef, 0, len(s.loaded))
	deferred := make([]*runnerRef, 0)
	for key, runner := range s.loaded {
		if keepModelKey != "" && key == keepModelKey {
			continue
		}
		runner.refMu.Lock()
		refs := runner.refCount
		runner.refMu.Unlock()
		// UnloadAllRunners (empty keepModelKey): defer in-use runners so active
		// requests (e.g. image generation) finish instead of losing the HTTP stream
		// while the MLX subprocess keeps running.
		if keepModelKey == "" && refs > 0 {
			runner.refMu.Lock()
			if runner.expireTimer != nil {
				runner.expireTimer.Stop()
				runner.expireTimer = nil
			}
			runner.sessionDuration = 0
			runner.refMu.Unlock()
			deferred = append(deferred, runner)
			continue
		}
		runners = append(runners, runner)
		delete(s.loaded, key)
	}
	s.loadedMu.Unlock()

	for _, runner := range deferred {
		slog.Info("deferring unload of in-use runner", "model", runner.modelPath, "ref_count", runner.refCount)
		s.scheduleExpiredRunner(runner)
	}

	for _, runner := range runners {
		runner.refMu.Lock()
		if runner.expireTimer != nil {
			runner.expireTimer.Stop()
			runner.expireTimer = nil
		}
		runner.sessionDuration = 0
		runner.refCount = 0
		runner.refMu.Unlock()
		if runner.llama != nil {
			slog.Debug("shutting down runner", "model", runner.modelPath, "pid", runner.pid)
			runner.llama.Close()
		}
	}
}

// keepModelKeyForUnload returns the scheduler map key to preserve during VRAM prep.
// Uses the loaded runner's key when present so digest vs path aliases match.
func (s *Scheduler) keepModelKeyForUnload(m *Model) string {
	if m == nil {
		return ""
	}
	if runner := s.findLoadedRunner(m); runner != nil {
		return runner.modelKey
	}
	return schedulerModelKey(m)
}

func (s *Scheduler) unloadAllRunners() {
	s.unloadRunnersExcept("")
}

// PauseNewLoads blocks new inference runner scheduling until [Scheduler.ResumeLoads].
// Used when training hits CUDA OOM: we must not let a new chat load grab VRAM between eviction
// and Python's retry. Why atomic bool + spin in GetRunner: minimal change vs redesigning the scheduler queue.
func (s *Scheduler) PauseNewLoads() {
	// Swap returns the previous value; skip log when already paused (e.g. coordination ticker).
	if s.loadsPaused.Swap(true) {
		return
	}
	slog.Info("scheduler: paused new loads", "reason", "training_or_runtime_backlog")
}

// ResumeLoads allows inference scheduling after an eviction / training retry.
func (s *Scheduler) ResumeLoads() {
	if !s.loadsPaused.Swap(false) {
		return
	}
	slog.Info("scheduler: resumed new loads")
	s.pending.notify()
}

// UnloadAllRunners evicts all loaded inference models (exported for training worker).
func (s *Scheduler) UnloadAllRunners() {
	s.unloadAllRunners()
}

// UnloadOtherRunners evicts loaded models except keepModelKey (empty keepModelKey evicts all).
func (s *Scheduler) UnloadOtherRunners(keepModelKey string) {
	s.unloadRunnersExcept(keepModelKey)
}

// pendingOldestFifoSeq is the smallest FIFO ticket among ggml pending loads (0 if none).
func (s *Scheduler) pendingOldestFifoSeq() uint64 {
	if s == nil || s.pending == nil {
		return 0
	}
	return s.pending.OldestFifoSeq()
}

// oldestGgmlFifoSeq is the smallest ticket among ggml pending and in-flight load work.
func (s *Scheduler) oldestGgmlFifoSeq() uint64 {
	if s == nil {
		return 0
	}
	pending := s.pendingOldestFifoSeq()
	loading := s.loadingFifoSeq.Load()
	return minNonZeroUint64(pending, loading)
}

// InferenceFleetSnapshot is a consistent ggml scheduler view for GET /api/status.
type InferenceFleetSnapshot struct {
	Pending      int
	Active       int
	Loaded       int
	LoadsPaused  bool
	Loading      bool
	LoadedModels       []string
	LoadedModelDetails []api.GgmlLoadedModelStatus
}

// InferenceFleetSnapshot returns a point-in-time ggml scheduler view for fleet polling.
// loaded and loaded_models count only ready runners (not still loading); loading=true
// when a new model load probe is in flight via activeLoading.
func (s *Scheduler) InferenceFleetSnapshot() InferenceFleetSnapshot {
	if s == nil {
		return InferenceFleetSnapshot{}
	}
	snap := InferenceFleetSnapshot{
		Pending:     s.pending.Len(),
		LoadsPaused: s.loadsPaused.Load(),
	}
	type readyEntry struct {
		name string
		meta api.LoadedModelMetadata
	}
	var ready []readyEntry

	s.loadedMu.Lock()
	snap.Loading = s.activeLoading != nil
	for _, runner := range s.loaded {
		runner.refMu.Lock()
		if runner.refCount > 0 {
			snap.Active++
		}
		if !runner.loading && runner.model != nil && runner.llama != nil {
			name := runner.modelKey
			if runner.model.ShortName != "" {
				name = runner.model.ShortName
			}
			meta := runner.loadedMeta
			runner.refMu.Unlock()
			if meta.ProbedAt.IsZero() {
				meta = probeRunnerMetadata(runner)
			}
			ready = append(ready, readyEntry{name: name, meta: meta})
		} else {
			runner.refMu.Unlock()
		}
	}
	if snap.Loading {
		snap.Active++
	}
	s.loadedMu.Unlock()

	slices.SortFunc(ready, func(a, b readyEntry) int {
		return strings.Compare(a.name, b.name)
	})
	snap.Loaded = len(ready)
	snap.LoadedModels = make([]string, 0, len(ready))
	snap.LoadedModelDetails = make([]api.GgmlLoadedModelStatus, 0, len(ready))
	for _, e := range ready {
		snap.LoadedModels = append(snap.LoadedModels, e.name)
		snap.LoadedModelDetails = append(snap.LoadedModelDetails, api.GgmlLoadedModelStatus{
			Name:                e.name,
			LoadedModelMetadata: e.meta,
		})
	}
	return snap
}

// ProcessModelsSnapshot returns loaded runners for GET /api/ps (mutex-safe).
func (s *Scheduler) ProcessModelsSnapshot() []api.ProcessModelResponse {
	if s == nil {
		return nil
	}
	s.loadedMu.Lock()
	runners := make([]*runnerRef, 0, len(s.loaded))
	for _, r := range s.loaded {
		runners = append(runners, r)
	}
	s.loadedMu.Unlock()

	models := make([]api.ProcessModelResponse, 0, len(runners))
	now := time.Now()
	sessionSnap := s.mlxGate.activeSessionsSnapshot(now)
	for _, runner := range runners {
		runner.refMu.Lock()
		if runner.loading || runner.model == nil {
			runner.refMu.Unlock()
			continue
		}
		mr := buildProcessModelResponse(runner)
		runner.refMu.Unlock()

		meta := loadedMetadataForRunner(runner)
		mr.LoadedMetadata = &meta
		if mr.ContextLength == 0 && meta.NumCtx > 0 {
			mr.ContextLength = meta.NumCtx
		}
		modelKey := runner.modelKey
		if modelKey == "" && runner.model != nil {
			modelKey = schedulerModelKey(runner.model)
		}
		if sessions := sessionSnap[modelKey]; len(sessions) > 0 {
			mr.Zerollama = &api.ProcessZerollamaInfo{Sessions: sessions}
		}
		models = append(models, mr)
	}

	slices.SortStableFunc(models, func(i, j api.ProcessModelResponse) int {
		return cmp.Compare(j.ExpiresAt.Unix(), i.ExpiresAt.Unix())
	})
	return models
}

// LoadedRunnersForDiscovery returns loaded ggml runners for GPU free-memory refresh.
// Why: GPUDevices(nil, nil) on every /api/show was slow; load-path suggest passes
// these runners so discovery reuses in-process free bytes instead of bootstrap-only.
func (s *Scheduler) LoadedRunnersForDiscovery() []ml.FilteredRunnerDiscovery {
	if s == nil {
		return nil
	}
	s.loadedMu.Lock()
	defer s.loadedMu.Unlock()
	out := make([]ml.FilteredRunnerDiscovery, 0, len(s.loaded))
	for _, runner := range s.loaded {
		out = append(out, runner)
	}
	return out
}

// InferenceBacklog returns pending requests, active refs, and resident runner count.
// loaded counts every runner in the scheduler map (including still loading), not only
// ready runners — training idle-wait uses this to block while VRAM is occupied.
func (s *Scheduler) InferenceBacklog() (pending int, active int, loaded int) {
	if s == nil {
		return 0, 0, 0
	}
	pending = s.pending.Len()
	s.loadedMu.Lock()
	loaded = len(s.loaded)
	for _, runner := range s.loaded {
		runner.refMu.Lock()
		if runner.refCount > 0 {
			active++
		}
		runner.refMu.Unlock()
	}
	if s.activeLoading != nil {
		active++
	}
	s.loadedMu.Unlock()
	return pending, active, loaded
}

// WaitStatus returns a streaming progress snapshot for a scheduler ticket.
func (s *Scheduler) WaitStatus(ticket uint64) (status, detail string, position, queueDepth int) {
	if s == nil || ticket == 0 {
		return "", "", 0, 0
	}
	position, queueDepth = s.pending.FifoPosition(ticket)
	if position > 0 {
		return "queued", fmt.Sprintf("queued (#%d of %d)", position, queueDepth), position, queueDepth
	}
	s.loadedMu.Lock()
	loading := s.activeLoading != nil
	loadingTicket := s.loadingFifoSeq.Load() == ticket
	s.loadedMu.Unlock()
	if loading && loadingTicket {
		return "loading", "loading model into memory", 0, queueDepth
	}
	if loading {
		return "loading", "loading model into memory", 0, queueDepth
	}
	return "loading", "starting inference", 0, queueDepth
}

// InferenceBusy reports whether ggml inference has queued, in-flight, or resident work.
func (s *Scheduler) InferenceBusy() bool {
	pending, active, loaded := s.InferenceBacklog()
	if pending > 0 || active > 0 {
		return true
	}
	if loaded > 0 && envconfig.TrainingWaitGgmlLoaded() {
		return true
	}
	return false
}

// findLoadedRunner returns the in-memory runner for a model.
// Primary key is schedulerModelKey (GGUF path or digest); fall back to ShortName/Name
// because stop/unload requests resolve names via GetModel while the loaded map may
// have been keyed before alias normalization.
func (s *Scheduler) findLoadedRunner(model *Model) *runnerRef {
	if model == nil {
		return nil
	}
	key := schedulerModelKey(model)
	s.loadedMu.Lock()
	defer s.loadedMu.Unlock()
	if runner, ok := s.loaded[key]; ok {
		return runner
	}
	for _, runner := range s.loaded {
		if runner.model == nil {
			continue
		}
		if model.ShortName != "" && runner.model.ShortName == model.ShortName {
			return runner
		}
		if model.Name != "" && runner.model.Name == model.Name {
			return runner
		}
	}
	return nil
}

// expireRunner unloads a loaded model immediately (stop CLI, keep_alive:0, post-create
// eviction). Called from HTTP handlers that do not hold a runner ref themselves.
// Why always scheduleExpiredRunner: a prior version only unloaded when refCount<=0,
// so stop returned "unload" while the model stayed in /api/ps; processExpiredRunner
// already retries until refs drop.
func (s *Scheduler) expireRunner(model *Model) {
	if s == nil {
		return
	}
	runner := s.findLoadedRunner(model)
	if runner == nil {
		return
	}
	runner.refMu.Lock()
	runner.expiresAt = time.Now()
	if runner.expireTimer != nil {
		runner.expireTimer.Stop()
		runner.expireTimer = nil
	}
	runner.sessionDuration = 0
	s.scheduleExpiredRunner(runner)
	runner.refMu.Unlock()
}
