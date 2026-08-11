package cmd

import (
	"context"
	"fmt"
	"os"
	"slices"
	"strings"
	"time"

	"github.com/olekukonko/tablewriter"
	"github.com/spf13/cobra"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/cmd/benchcache"
	"github.com/ollama/ollama/internal/modelhealth"
	"github.com/ollama/ollama/types/model"
)

const (
	benchKindCompletion = "completion"
	benchKindImage      = "image"
	benchKindVideoGen   = "video_gen"
)

var benchPromptWords = []string{
	"the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog",
	"a", "bright", "sunny", "day", "in", "the", "meadow", "where",
	"flowers", "bloom", "and", "birds", "sing", "their", "morning",
	"songs", "while", "gentle", "breeze", "carries", "sweet", "scent",
	"of", "pine", "trees", "across", "rolling", "hills", "toward",
	"distant", "mountains", "covered", "with", "fresh", "snow",
	"beneath", "clear", "blue", "sky", "children", "play", "near",
	"old", "stone", "bridge", "that", "crosses", "winding", "river",
}

func benchPromptForEpoch(epoch int) string {
	// WHY offset per epoch: defeat KV prefix reuse so timed epochs measure decode, not cached prefill.
	offset := epoch * 7
	n := len(benchPromptWords)
	words := make([]string, 32)
	for i := range words {
		words[i] = benchPromptWords[((i+offset)%n+n)%n]
	}
	return strings.Join(words, " ")
}

// benchModelKind picks bench cache kind from manifest capabilities.
// WHY skip remote (non-LM Studio): cloud models are not local inference; bench measures this host.
func benchModelKind(m api.ListModelResponse) string {
	if m.RemoteModel != "" && !strings.EqualFold(m.RemoteHost, "lmstudio") {
		return ""
	}
	if m.Digest == "" {
		return ""
	}
	caps := m.Capabilities
	if slices.Contains(caps, model.CapabilityImage) {
		return benchKindImage
	}
	if slices.Contains(caps, model.CapabilityVideoGen) {
		return benchKindVideoGen
	}
	if len(caps) == 0 {
		return benchKindCompletion
	}
	if slices.Contains(caps, model.CapabilityCompletion) {
		return benchKindCompletion
	}
	if slices.Contains(caps, model.CapabilityEmbedding) ||
		slices.Contains(caps, model.CapabilitySpeech) {
		return ""
	}
	return ""
}

func isBenchableModel(m api.ListModelResponse) bool {
	return benchModelKind(m) != ""
}

func matchesBenchFilter(name string, filters []string) bool {
	if len(filters) == 0 {
		return true
	}
	lowerName := strings.ToLower(name)
	for _, filter := range filters {
		if strings.HasPrefix(lowerName, strings.ToLower(filter)) {
			return true
		}
	}
	return false
}

func selectBenchModels(all []api.ListModelResponse, filters []string) []api.ListModelResponse {
	var selected []api.ListModelResponse
	for _, m := range all {
		if !isBenchableModel(m) {
			continue
		}
		if !matchesBenchFilter(m.Name, filters) {
			continue
		}
		selected = append(selected, m)
	}
	return selected
}

const (
	benchMinEvalDuration   = time.Millisecond
	benchMaxSanityTokPerSec = 10_000 // reject partial-stream metrics (e.g. 333k tok/s glitches)
)

func benchTokPerSec(metrics *api.Metrics) float64 {
	// WHY EvalDuration not TotalDuration: load and prefill are paid in warmup; ls column is decode tok/s.
	if metrics == nil || metrics.EvalCount <= 0 || metrics.EvalDuration < benchMinEvalDuration {
		return 0
	}
	rate := float64(metrics.EvalCount) / metrics.EvalDuration.Seconds()
	if rate > benchMaxSanityTokPerSec {
		return 0
	}
	return rate
}

func benchGenerateOnce(ctx context.Context, client *api.Client, modelName string, epoch, maxTokens, numCtx int, loadTimeout, genTimeout time.Duration) (*api.Metrics, error) {
	options := map[string]any{
		"num_predict": maxTokens,
		"temperature": 0.0,
	}
	if numCtx > 0 {
		options["num_ctx"] = numCtx
	}

	stream := true
	req := &api.GenerateRequest{
		Model:  modelName,
		Prompt: benchPromptForEpoch(epoch),
		// Raw skips chat template so we measure decode throughput, not formatted replies.
		Raw:     true,
		Stream:  &stream,
		Options: options,
	}

	timeout := genTimeout
	if epoch < 0 {
		// Warmup pays model load; allow longer deadline than timed epochs.
		timeout = loadTimeout
	}

	ctx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()

	var metrics *api.Metrics
	var sawDone bool
	var responseBytes int
	err := client.Generate(ctx, req, func(resp api.GenerateResponse) error {
		responseBytes += len(resp.Response)
		// WHY Done-only: streaming chunks can carry full EvalCount with near-zero EvalDuration,
		// producing nonsense tok/s (e.g. 333k) if we accept intermediate updates.
		if resp.Done {
			sawDone = true
			m := resp.Metrics
			metrics = &m
		}
		return nil
	})
	if err != nil {
		return nil, err
	}
	if metrics == nil || metrics.EvalCount <= 0 {
		return nil, benchNoMetricsError(sawDone, responseBytes, numCtx)
	}
	return metrics, nil
}

func benchNoMetricsError(sawDone bool, responseBytes, numCtx int) error {
	switch {
	case !sawDone && responseBytes > 0:
		hint := "stream aborted after partial output"
		if numCtx >= 8192 {
			hint += "; try --num-ctx 2048 on tight VRAM hosts"
		}
		return fmt.Errorf("no metrics received (%s)", hint)
	case sawDone:
		return fmt.Errorf("no metrics received (done with eval_count=0)")
	default:
		return fmt.Errorf("no metrics received (empty stream)")
	}
}

func benchUnloadModel(client *api.Client, modelName string, timeout time.Duration) {
	// WHY KeepAlive=0 between models: fair VRAM for next tag; avoid warm-runner skew in multi-model bench.
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()

	zero := api.Duration{Duration: 0}
	req := &api.GenerateRequest{
		Model:     modelName,
		KeepAlive: &zero,
	}
	_ = client.Generate(ctx, req, func(api.GenerateResponse) error { return nil })
}

func benchUnloadAllLoaded(ctx context.Context, client *api.Client, timeout time.Duration) {
	// WHY unload every /api/ps entry: OOM or partial load can leave a different tag resident than lastBenched.
	listCtx, cancel := context.WithTimeout(ctx, 30*time.Second)
	resp, err := client.ListRunning(listCtx)
	cancel()
	if err != nil || len(resp.Models) == 0 {
		return
	}
	per := timeout / time.Duration(len(resp.Models))
	if per < 30*time.Second {
		per = 30 * time.Second
	}
	for _, m := range resp.Models {
		name := m.Name
		if name == "" {
			name = m.Model
		}
		if name == "" {
			continue
		}
		benchUnloadModel(client, name, per)
	}
	_ = benchWaitUnloaded(ctx, client, timeout)
}

func benchWaitUnloaded(ctx context.Context, client *api.Client, timeout time.Duration) error {
	deadline := time.Now().Add(timeout)
	var lastCount int
	for time.Now().Before(deadline) {
		listCtx, cancel := context.WithTimeout(ctx, 10*time.Second)
		resp, err := client.ListRunning(listCtx)
		cancel()
		if err == nil {
			if len(resp.Models) == 0 {
				return nil
			}
			lastCount = len(resp.Models)
		}
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-time.After(2 * time.Second):
		}
	}
	if lastCount > 0 {
		return fmt.Errorf("%d model(s) still loaded after %s", lastCount, timeout)
	}
	return fmt.Errorf("unload wait timed out after %s", timeout)
}

func benchWaitServer(ctx context.Context, client *api.Client, timeout time.Duration) error {
	deadline := time.Now().Add(timeout)
	var lastErr error
	for time.Now().Before(deadline) {
		if _, err := client.List(ctx); err == nil {
			return nil
		} else {
			lastErr = err
		}
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-time.After(2 * time.Second):
		}
	}
	if lastErr != nil {
		return fmt.Errorf("server not ready: %w", lastErr)
	}
	return fmt.Errorf("server not ready within %s", timeout)
}

type benchModelResult struct {
	kind        string
	rate        float64
	genSec      float64
	epochsOK    int
	epochsTotal int
	partial     bool
}

func benchModel(ctx context.Context, client *api.Client, m api.ListModelResponse, warmup, epochs, maxTokens, numCtx int, loadTimeout, genTimeout time.Duration, minEpochs int) (benchModelResult, error) {
	for i := range warmup {
		_, err := benchGenerateOnce(ctx, client, m.Name, -(i + 1), maxTokens, numCtx, loadTimeout, genTimeout)
		if err != nil {
			// WHY warn not fail: slow first load should not skip timed epochs that would succeed.
			fmt.Fprintf(os.Stderr, "warning: warmup %d/%d for %s failed: %v\n", i+1, warmup, m.Name, err)
		}
	}

	var rates []float64
	for epoch := range epochs {
		metrics, err := benchGenerateOnce(ctx, client, m.Name, epoch, maxTokens, numCtx, loadTimeout, genTimeout)
		if err != nil {
			fmt.Fprintf(os.Stderr, "warning: %s: epoch %d/%d: %v\n", m.Name, epoch+1, epochs, err)
			continue
		}
		if rate := benchTokPerSec(metrics); rate > 0 {
			rates = append(rates, rate)
		}
	}

	if len(rates) < minEpochs {
		return benchModelResult{kind: benchKindCompletion, epochsTotal: epochs}, fmt.Errorf("only %d/%d epochs produced metrics (need %d)", len(rates), epochs, minEpochs)
	}

	var sum float64
	for _, rate := range rates {
		sum += rate
	}
	return benchModelResult{
		kind:        benchKindCompletion,
		rate:        sum / float64(len(rates)),
		epochsOK:    len(rates),
		epochsTotal: epochs,
		partial:     len(rates) < epochs,
	}, nil
}

func benchPreflightSkip(name string, skipCheck bool) (string, bool) {
	if skipCheck {
		return "", false
	}
	report, err := modelhealth.CheckName(name)
	if err != nil {
		return "", false
	}
	if modelhealth.IsBenchable(report) {
		return "", false
	}
	return fmt.Sprintf("%s (%s)", report.Detail, report.FixHint), true
}

// benchClampNumCtx lowers context for huge / MoE models so force-bench
// with the default 8192 does not OOM-kill the runner (e.g. qwen3-coder-next).
func benchClampNumCtx(m api.ListModelResponse, numCtx int) int {
	if numCtx <= 0 {
		return numCtx
	}
	const (
		softBytes = 40 << 30 // ~40 GiB on-disk → clamp to 4096
		hardBytes = 70 << 30 // ~70 GiB on-disk → clamp to 2048
	)
	maxCtx := numCtx
	if m.Size >= hardBytes {
		maxCtx = 2048
	} else if m.Size >= softBytes {
		maxCtx = 4096
	}
	// Dense-parameter MoE / large active-count models are especially KV-heavy.
	if m.Details.ExpertCount > 1 || m.Details.ActiveParameterCount > 0 {
		if m.Size >= softBytes && maxCtx > 2048 {
			maxCtx = 2048
		}
	}
	if maxCtx < numCtx {
		return maxCtx
	}
	return numCtx
}

func runBench(cmd *cobra.Command, args []string) error {
	epochs, _ := cmd.Flags().GetInt("epochs")
	maxTokens, _ := cmd.Flags().GetInt("tokens")
	numCtx, _ := cmd.Flags().GetInt("num-ctx")
	warmup, _ := cmd.Flags().GetInt("warmup")
	force, _ := cmd.Flags().GetBool("force")
	timeoutSec, _ := cmd.Flags().GetInt("timeout")
	loadTimeoutSec, _ := cmd.Flags().GetInt("load-timeout")
	minEpochs, _ := cmd.Flags().GetInt("min-epochs")
	skipHealth, _ := cmd.Flags().GetBool("skip-health-check")

	videoTimeoutSec, _ := cmd.Flags().GetInt("video-timeout")

	genTimeout := time.Duration(timeoutSec) * time.Second
	loadTimeout := time.Duration(loadTimeoutSec) * time.Second
	videoTimeout := time.Duration(videoTimeoutSec) * time.Second
	if minEpochs < 1 {
		minEpochs = 1
	}
	if minEpochs > epochs {
		minEpochs = epochs
	}

	client, err := api.ClientFromEnvironment()
	if err != nil {
		return err
	}

	listResp, err := client.List(cmd.Context())
	if err != nil {
		return err
	}

	cache, err := benchcache.Load()
	if err != nil {
		return err
	}

	models := selectBenchModels(listResp.Models, args)
	if len(models) == 0 {
		return fmt.Errorf("no local models to benchmark")
	}

	type result struct {
		name    string
		kind    string
		perf    string
		skipped bool
		partial bool
		err     error
	}
	results := make([]result, 0, len(models))

	if numCtx > 0 {
		fmt.Fprintf(os.Stderr, "bench num_ctx=%d (use --num-ctx to override)\n", numCtx)
	}

	for _, m := range models {
		kind := benchModelKind(m)

		if kind != benchKindVideoGen {
			benchUnloadAllLoaded(cmd.Context(), client, genTimeout)
			_ = benchWaitServer(cmd.Context(), client, 30*time.Second)
		}

		if detail, skip := benchPreflightSkip(m.Name, skipHealth); skip {
			fmt.Fprintf(os.Stderr, "skip %s (unhealthy: %s)\n", m.Name, detail)
			results = append(results, result{name: m.Name, kind: kind, err: fmt.Errorf("unhealthy")})
			continue
		}

		if !force {
			if entry, ok := cache[m.Digest]; ok && entry.Cached() {
				fmt.Fprintf(os.Stderr, "skip %s (cached %s, use --force to re-bench)\n", m.Name, entry.PerfString())
				results = append(results, result{name: m.Name, kind: kind, perf: entry.PerfString(), skipped: true})
				continue
			}
		}

		fmt.Fprintf(os.Stderr, "benching %s (%s)...\n", m.Name, kind)
		var br benchModelResult
		var err error
		modelNumCtx := numCtx
		if kind == benchKindCompletion {
			modelNumCtx = benchClampNumCtx(m, numCtx)
			if modelNumCtx != numCtx {
				fmt.Fprintf(os.Stderr, "  num_ctx clamped %d → %d for large model\n", numCtx, modelNumCtx)
			}
		}
		switch kind {
		case benchKindImage:
			br, err = benchImageModel(cmd.Context(), client, m, warmup, epochs, loadTimeout, genTimeout, minEpochs)
		case benchKindVideoGen:
			br, err = benchVideoModel(cmd.Context(), m, videoTimeout)
		default:
			br, err = benchModel(cmd.Context(), client, m, warmup, epochs, maxTokens, modelNumCtx, loadTimeout, genTimeout, minEpochs)
		}
		if err != nil {
			fmt.Fprintf(os.Stderr, "warning: %s: %v\n", m.Name, err)
			results = append(results, result{name: m.Name, kind: kind, err: err})
			if kind != benchKindVideoGen {
				benchUnloadAllLoaded(cmd.Context(), client, 120*time.Second)
				_ = benchWaitServer(cmd.Context(), client, 120*time.Second)
			}
			continue
		}

		if br.rate <= 0 {
			fmt.Fprintf(os.Stderr, "warning: %s: no valid tok/s metrics\n", m.Name)
			results = append(results, result{name: m.Name, kind: kind, err: fmt.Errorf("invalid metrics")})
			benchUnloadAllLoaded(cmd.Context(), client, genTimeout)
			continue
		}

		entry := benchcache.Entry{
			Model:     m.Name,
			Kind:      br.kind,
			TokPerSec: br.rate,
			GenSec:    br.genSec,
			BenchedAt: time.Now().UTC(),
		}
		cache[m.Digest] = entry
		if err := cache.Save(); err != nil {
			return fmt.Errorf("save bench cache: %w", err)
		}

		perf := entry.PerfString()
		if br.partial {
			fmt.Fprintf(os.Stderr, "%s  %s (%d/%d epochs)\n", m.Name, perf, br.epochsOK, br.epochsTotal)
		} else {
			fmt.Fprintf(os.Stderr, "%s  %s\n", m.Name, perf)
		}
		results = append(results, result{name: m.Name, kind: kind, perf: perf, partial: br.partial})
	}

	benchUnloadAllLoaded(cmd.Context(), client, genTimeout)

	var tableData [][]string
	for _, r := range results {
		if r.err != nil {
			tableData = append(tableData, []string{r.name, "error"})
			continue
		}
		perfStr := r.perf
		if r.partial {
			perfStr += "*"
		}
		tableData = append(tableData, []string{r.name, perfStr})
	}

	if len(tableData) > 0 {
		table := tablewriter.NewWriter(os.Stdout)
		table.SetHeader([]string{"NAME", "PERF"})
		table.SetHeaderAlignment(tablewriter.ALIGN_LEFT)
		table.SetAlignment(tablewriter.ALIGN_LEFT)
		table.SetHeaderLine(false)
		table.SetBorder(false)
		table.SetNoWhiteSpace(true)
		table.SetTablePadding("    ")
		table.AppendBulk(tableData)
		table.Render()
	}

	return nil
}

// NewBenchCommand registers `zerollama bench`.
// WHY client-only: measures the same HTTP path agents use; no server schema or manifest layer required.
func NewBenchCommand() *cobra.Command {
	cmd := &cobra.Command{
		Use:     "bench [MODEL...]",
		Short:   "Benchmark local models and cache perf for ls",
		Long: `Run a short generation benchmark for local models and save results
to ~/.ollama/bench.json. Text models cache decode tok/s; image and video_gen
models cache average wall seconds per generation. Results appear in the PERF
column of zerollama ls.

With no model names, benchmarks all local models (completion, image, video_gen).
Cached results are skipped unless --force is set.`,
		Args:    cobra.ArbitraryArgs,
		PreRunE: checkServerHeartbeat,
		RunE:    runBench,
	}

	cmd.Flags().Int("epochs", 3, "Number of timed epochs to average")
	cmd.Flags().Int("tokens", 128, "Maximum output tokens per epoch")
	cmd.Flags().Int("num-ctx", 8192, "Context window for bench loads (lower fits large models on dual-GPU tensor parallel)")
	cmd.Flags().Int("warmup", 1, "Warmup runs before timing")
	cmd.Flags().Bool("force", false, "Re-bench models that already have cached results")
	cmd.Flags().Int("timeout", 600, "Per-request generation timeout in seconds (image models)")
	cmd.Flags().Int("load-timeout", 900, "Warmup timeout in seconds (includes model load)")
	cmd.Flags().Int("video-timeout", 7200, "Video generation poll timeout in seconds")
	cmd.Flags().Int("min-epochs", 1, "Minimum successful epochs required (allows partial results)")
	cmd.Flags().Bool("skip-health-check", false, "Benchmark even when local blob health check fails")

	return cmd
}
