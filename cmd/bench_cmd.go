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

func isBenchableModel(m api.ListModelResponse) bool {
	// WHY same remote filter as ls: cloud catalog stubs are not local generate targets.
	if m.RemoteModel != "" && !strings.EqualFold(m.RemoteHost, "lmstudio") {
		return false
	}
	if m.Digest == "" {
		return false // WHY: cannot key cache without stable digest
	}
	caps := m.Capabilities
	if len(caps) == 0 {
		return true // legacy tags default to completion
	}
	if slices.Contains(caps, model.CapabilityCompletion) {
		return true
	}
	// WHY skip non-chat models: generate bench is meaningless on embed/image/video_gen/speech-only tags.
	if slices.Contains(caps, model.CapabilityEmbedding) ||
		slices.Contains(caps, model.CapabilityImage) ||
		slices.Contains(caps, model.CapabilityVideoGen) ||
		slices.Contains(caps, model.CapabilitySpeech) {
		return false
	}
	return true
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

func benchTokPerSec(metrics *api.Metrics) float64 {
	// WHY EvalDuration not TotalDuration: load and prefill are paid in warmup; ls column is decode tok/s.
	if metrics == nil || metrics.EvalCount <= 0 || metrics.EvalDuration <= 0 {
		return 0
	}
	return float64(metrics.EvalCount) / metrics.EvalDuration.Seconds()
}

func benchGenerateOnce(ctx context.Context, client *api.Client, modelName string, epoch, maxTokens int, loadTimeout, genTimeout time.Duration) (*api.Metrics, error) {
	options := map[string]any{
		"num_predict": maxTokens,
		"temperature": 0.0,
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
	err := client.Generate(ctx, req, func(resp api.GenerateResponse) error {
		if resp.Metrics.EvalCount > 0 || resp.Metrics.EvalDuration > 0 {
			m := resp.Metrics
			metrics = &m
		}
		if resp.Done {
			m := resp.Metrics
			metrics = &m
		}
		return nil
	})
	if err != nil {
		return nil, err
	}
	if metrics == nil || metrics.EvalCount <= 0 {
		return nil, fmt.Errorf("no metrics received")
	}
	return metrics, nil
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

type benchModelResult struct {
	rate       float64
	epochsOK   int
	epochsTotal int
	partial    bool
}

func benchModel(ctx context.Context, client *api.Client, m api.ListModelResponse, warmup, epochs, maxTokens int, loadTimeout, genTimeout time.Duration, minEpochs int) (benchModelResult, error) {
	for i := range warmup {
		_, err := benchGenerateOnce(ctx, client, m.Name, -(i + 1), maxTokens, loadTimeout, genTimeout)
		if err != nil {
			// WHY warn not fail: slow first load should not skip timed epochs that would succeed.
			fmt.Fprintf(os.Stderr, "warning: warmup %d/%d for %s failed: %v\n", i+1, warmup, m.Name, err)
		}
	}

	var rates []float64
	for epoch := range epochs {
		metrics, err := benchGenerateOnce(ctx, client, m.Name, epoch, maxTokens, loadTimeout, genTimeout)
		if err != nil {
			fmt.Fprintf(os.Stderr, "warning: %s: epoch %d/%d: %v\n", m.Name, epoch+1, epochs, err)
			continue
		}
		if rate := benchTokPerSec(metrics); rate > 0 {
			rates = append(rates, rate)
		}
	}

	if len(rates) < minEpochs {
		return benchModelResult{epochsTotal: epochs}, fmt.Errorf("only %d/%d epochs produced metrics (need %d)", len(rates), epochs, minEpochs)
	}

	var sum float64
	for _, rate := range rates {
		sum += rate
	}
	return benchModelResult{
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

func runBench(cmd *cobra.Command, args []string) error {
	epochs, _ := cmd.Flags().GetInt("epochs")
	maxTokens, _ := cmd.Flags().GetInt("tokens")
	warmup, _ := cmd.Flags().GetInt("warmup")
	force, _ := cmd.Flags().GetBool("force")
	timeoutSec, _ := cmd.Flags().GetInt("timeout")
	loadTimeoutSec, _ := cmd.Flags().GetInt("load-timeout")
	minEpochs, _ := cmd.Flags().GetInt("min-epochs")
	skipHealth, _ := cmd.Flags().GetBool("skip-health-check")

	genTimeout := time.Duration(timeoutSec) * time.Second
	loadTimeout := time.Duration(loadTimeoutSec) * time.Second
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
		name      string
		tokPerSec float64
		skipped   bool
		partial   bool
		err       error
	}
	results := make([]result, 0, len(models))

	for _, m := range models {
		if detail, skip := benchPreflightSkip(m.Name, skipHealth); skip {
			fmt.Fprintf(os.Stderr, "skip %s (unhealthy: %s)\n", m.Name, detail)
			results = append(results, result{name: m.Name, err: fmt.Errorf("unhealthy")})
			continue
		}

		if !force {
			if entry, ok := cache[m.Digest]; ok && entry.TokPerSec > 0 {
				// WHY digest lookup: name may change via cp; weights unchanged → skip re-bench unless --force.
				fmt.Fprintf(os.Stderr, "skip %s (cached %.1f tok/s, use --force to re-bench)\n", m.Name, entry.TokPerSec)
				results = append(results, result{name: m.Name, tokPerSec: entry.TokPerSec, skipped: true})
				continue
			}
		}

		fmt.Fprintf(os.Stderr, "benching %s...\n", m.Name)
		br, err := benchModel(cmd.Context(), client, m, warmup, epochs, maxTokens, loadTimeout, genTimeout, minEpochs)
		if err != nil {
			fmt.Fprintf(os.Stderr, "warning: %s: %v\n", m.Name, err)
			results = append(results, result{name: m.Name, err: err})
			benchUnloadModel(client, m.Name, genTimeout)
			continue
		}

		cache[m.Digest] = benchcache.Entry{
			Model:     m.Name,
			TokPerSec: br.rate,
			BenchedAt: time.Now().UTC(),
		}
		// WHY save per model: multi-model bench can take tens of minutes; preserve progress on interrupt.
		if err := cache.Save(); err != nil {
			return fmt.Errorf("save bench cache: %w", err)
		}

		if br.partial {
			fmt.Fprintf(os.Stderr, "%s  %.1f tok/s (%d/%d epochs)\n", m.Name, br.rate, br.epochsOK, br.epochsTotal)
		} else {
			fmt.Fprintf(os.Stderr, "%s  %.1f tok/s\n", m.Name, br.rate)
		}
		results = append(results, result{name: m.Name, tokPerSec: br.rate, partial: br.partial})
		benchUnloadModel(client, m.Name, genTimeout)
	}

	var tableData [][]string
	for _, r := range results {
		if r.err != nil {
			tableData = append(tableData, []string{r.name, "error"})
			continue
		}
		rateStr := fmt.Sprintf("%.1f", r.tokPerSec)
		if r.partial {
			rateStr += "*"
		}
		tableData = append(tableData, []string{r.name, rateStr})
	}

	if len(tableData) > 0 {
		table := tablewriter.NewWriter(os.Stdout)
		table.SetHeader([]string{"NAME", "TOK/S"})
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
		Short:   "Benchmark local models and cache tok/s for ls",
		Long: `Run a short generation benchmark for local models and save estimated tok/s
to ~/.ollama/bench.json. Results appear in the TOK/S column of zerollama ls.

With no model names, benchmarks all local text models. Cached results are skipped
unless --force is set. Models with missing blobs are skipped unless --skip-health-check.`,
		Args:    cobra.ArbitraryArgs,
		PreRunE: checkServerHeartbeat,
		RunE:    runBench,
	}

	cmd.Flags().Int("epochs", 3, "Number of timed epochs to average")
	cmd.Flags().Int("tokens", 128, "Maximum output tokens per epoch")
	cmd.Flags().Int("warmup", 1, "Warmup runs before timing")
	cmd.Flags().Bool("force", false, "Re-bench models that already have cached results")
	cmd.Flags().Int("timeout", 600, "Per-request generation timeout in seconds")
	cmd.Flags().Int("load-timeout", 900, "Warmup timeout in seconds (includes model load)")
	cmd.Flags().Int("min-epochs", 1, "Minimum successful epochs required (allows partial results)")
	cmd.Flags().Bool("skip-health-check", false, "Benchmark even when local blob health check fails")

	return cmd
}
