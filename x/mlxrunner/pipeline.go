package mlxrunner

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"log/slog"
	"sort"
	"time"

	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/logutil"
	"github.com/ollama/ollama/x/mlxrunner/batch"
	"github.com/ollama/ollama/x/mlxrunner/mlx"
	sampler "github.com/ollama/ollama/x/mlxrunner/sample"
	"github.com/ollama/ollama/x/tokenizer"
)

func prefillChunkSize() int {
	return loadPrefillConfig().chunkSize
}

// Prepare tokenizes the prompt and validates it against the model's
// context length. It is safe to call from any goroutine. On success it
// populates request.Tokens and adjusts request.Options.NumPredict.
func (r *Runner) Prepare(request *Request) error {
	if r.Model == nil {
		return errors.New("model not loaded")
	}

	var tokens []int32
	if len(request.Tokens) > 0 {
		tokens = append([]int32(nil), request.Tokens...)
	} else {
		tokens = r.Tokenizer.Encode(request.Prompt, r.Tokenizer.AddBOS())
	}
	if len(tokens) == 0 {
		return errors.New("empty prompt")
	}

	reserve := 256
	if request.Options.NumPredict > 0 {
		reserve = request.Options.NumPredict
	}
	maxReserve := r.contextLength / 2
	if maxReserve < 32 {
		maxReserve = 32
	}
	if reserve > maxReserve {
		reserve = maxReserve
	}
	maxPrompt := r.contextLength - reserve
	if maxPrompt < 1 {
		maxPrompt = r.contextLength - 1
	}
	if len(tokens) > maxPrompt {
		dropped := len(tokens) - maxPrompt
		tokens = tokens[dropped:]
		slog.Warn("mlx prepare tail-truncated prompt to fit context",
			"dropped_tokens", dropped,
			"kept_tokens", len(tokens),
			"context_length", r.contextLength,
			"reserve", reserve,
		)
	}

	if len(tokens) >= r.contextLength {
		return fmt.Errorf("input length (%d tokens) exceeds the model's maximum context length (%d tokens)", len(tokens), r.contextLength)
	}

	// Cap generation to stay within the model's context length
	maxGenerate := r.contextLength - len(tokens)
	if request.Options.NumPredict <= 0 {
		request.Options.NumPredict = maxGenerate
	} else {
		request.Options.NumPredict = min(request.Options.NumPredict, maxGenerate)
	}

	request.Tokens = tokens
	return nil
}

// The runner serializes requests today so we just use a fixed slot ID.
const pipelineSlot = 0

func (r *Runner) TextGenerationPipeline(ctx context.Context, request Request) error {
	mlx.ResetPeakMemory()
	var sample, nextSample sampler.Result

	defer func() {
		r.Sampler.Remove(pipelineSlot)
		mlx.Unpin(sample.Arrays()...)
		mlx.Unpin(nextSample.Arrays()...)
		mlx.Sweep()
		mlx.ClearCache()

		if slog.Default().Enabled(context.TODO(), logutil.LevelTrace) {
			mlx.LogArrays()
			r.cache.dumpTree()
		}
		slog.Info("peak memory", "size", mlx.PrettyBytes(mlx.PeakMemory()))
	}()

	inputs := request.Tokens

	session := r.cache.begin(r.Model, inputs)
	defer session.close()

	caches := session.caches
	tokens := session.remaining
	pcfg := effectivePrefillConfig(len(inputs), loadPrefillConfig())
	prefillChunk := pcfg.chunkSize
	if len(inputs) > defaultMTPMaxPromptTokens {
		slog.Info("long mlx prefill memory policy",
			"prompt_tokens", len(inputs),
			"chunk_size", prefillChunk,
			"clear_cache_every", pcfg.clearCacheEvery,
			"materialize_every", pcfg.materializeEvery,
		)
	}
	snapshotInterval := trieSnapshotInterval(pcfg, len(inputs))
	snapshotOffsets := prefillSnapshotOffsets(len(inputs), snapshotInterval)
	if snapshotInterval == 0 && pcfg.snapshotInterval > 0 && len(inputs) > defaultMTPMaxPromptTokens {
		slog.Info("prefill snapshots disabled for long prompt",
			"prompt_tokens", len(inputs),
			"threshold", defaultMTPMaxPromptTokens,
		)
	} else if len(inputs) > defaultMTPMaxPromptTokens && snapshotInterval > 0 {
		slog.Info("trie prefix snapshots enabled for long prompt",
			"prompt_tokens", len(inputs),
			"interval", snapshotInterval,
			"offsets", len(snapshotOffsets),
		)
	}

	cachedPrefix := session.cachedPrefix

	materializeCaches := func() {
		state := make([]*mlx.Array, 0, 2*len(caches))
		for _, c := range caches {
			state = append(state, c.State()...)
		}
		if len(state) == 0 {
			return
		}
		mlx.Eval(state...)
	}

	session.schedulePrefillSnapshots(snapshotOffsets)

	prefillStart := time.Now()
	lastHeartbeat := prefillStart
	now := prefillStart
	total, processed := len(tokens), 0
	position := len(inputs) - len(tokens)
	chunkNum := 0
	logProgressEvery := max(1, total/prefillChunk/10) // ~10 progress lines for long prompts
	emitPrefillHeartbeat := func() {
		if time.Since(lastHeartbeat) < 30*time.Second {
			return
		}
		select {
		case <-ctx.Done():
		case request.Responses <- CompletionResponse{
			PrefillProcessed: processed,
			PrefillTotal:     total,
		}:
			lastHeartbeat = time.Now()
		default:
		}
	}
	for total-processed > 1 {
		if err := ctx.Err(); err != nil {
			slog.Warn("mlx prefill canceled", "processed", processed, "total", total, "error", err)
			return err
		}

		n := min(prefillChunk, total-processed-1)

		r.Model.Forward(&batch.Batch{
			InputIDs:     mlx.FromValues(tokens[processed:processed+n], 1, n),
			SeqOffsets:   []int32{int32(position)},
			SeqQueryLens: []int32{int32(n)},
		}, caches)
		mlx.Sweep()
		if pcfg.materializeEvery <= 1 || chunkNum%pcfg.materializeEvery == 0 || processed+n >= total-1 {
			materializeCaches()
		}
		processed += n
		position += n
		chunkNum++
		if chunkNum%logProgressEvery == 0 || processed >= total-1 {
			slog.Info("Prompt processing progress",
				"processed", processed,
				"total", total,
				"chunk", prefillChunk,
				"active_memory", mlx.PrettyBytes(mlx.ActiveMemory()),
				"peak_memory", mlx.PrettyBytes(mlx.PeakMemory()),
			)
		}
		emitPrefillHeartbeat()
		logutil.TraceContext(ctx, "mlx prompt forward", "processed", processed, "total", total, "tokens", n, "memory", mlx.Memory{})

		if pcfg.clearCacheEvery > 0 && chunkNum%pcfg.clearCacheEvery == 0 {
			mlx.Sweep()
			mlx.ClearCache()
		}
	}

	// Attach the snapshots captured during prefill to the trie.
	session.attachPrefillSnapshots()

	slog.Info("prefill complete",
		"prompt_tokens", len(inputs),
		"cached_tokens", len(inputs)-len(tokens),
		"prefill_tokens", total,
		"elapsed", time.Since(prefillStart).Round(time.Millisecond),
		"tok_per_sec", float64(total)/max(time.Since(prefillStart).Seconds(), 0.001),
	)

	// Register the sampler after prefill completes.
	r.Sampler.Add(pipelineSlot, request.SamplerOpts, inputs)
	useMTP := len(inputs) <= mtpMaxPromptTokens()
	if !useMTP && r.Draft != nil {
		slog.Info("MTP disabled for long prompt; using standard decode",
			"prompt_tokens", len(inputs),
			"max_mtp_prompt_tokens", mtpMaxPromptTokens(),
		)
	}
	if useMTP && r.useGreedyMTP(request.SamplerOpts) {
		return r.runGreedyMTPDecode(ctx, request, session, caches, tokens[processed:], &position, now)
	}
	if useMTP && r.useSampleMTP(request.SamplerOpts) {
		return r.runSampleMTPDecode(ctx, request, session, caches, tokens[processed:], &position, now)
	}

	step := func(token *mlx.Array) sampler.Result {
		fwd := r.Model.Forward(&batch.Batch{
			InputIDs:     token,
			SeqOffsets:   []int32{int32(position)},
			SeqQueryLens: []int32{int32(token.Dim(1))},
		}, caches)
		position += token.Dim(1)
		logits := r.Model.Unembed(fwd)
		logits = logits.Slice(mlx.Slice(), mlx.Slice(logits.Dim(1)-1), mlx.Slice()).Squeeze(1)

		sample := r.Sampler.Sample([]int{pipelineSlot}, logits)
		mlx.Pin(sample.Arrays()...)
		mlx.Sweep()
		mlx.AsyncEval(sample.Arrays()...)
		return sample
	}

	sample = step(mlx.FromValues(tokens[processed:], 1, total-processed))
	logutil.TraceContext(ctx, "mlx decode seed", "tokens", total-processed, "memory", mlx.Memory{})

	dec := decoder{
		tokenizer:       r.Tokenizer,
		wantLogprobs:    request.SamplerOpts.Logprobs,
		wantTopLogprobs: request.SamplerOpts.TopLogprobs,
	}

	final := CompletionResponse{
		Done:                  true,
		PromptEvalCount:       len(inputs),
		PromptEvalCachedCount: cachedPrefix,
		EvalCount:             request.Options.NumPredict,
		DoneReason:            1,
	}
	for i := range request.Options.NumPredict {
		if err := ctx.Err(); err != nil {
			return err
		}

		nextSample = step(sample.Token.ExpandDims(-1))

		if i == 0 {
			mlx.Eval(sample.Arrays()...)
			final.PromptEvalDuration = time.Since(now)
			now = time.Now()
		}

		output := int32(sample.Token.Int())
		session.outputs = append(session.outputs, output)
		if i == 0 {
			slog.Info("mlx decode first token",
				"token_id", output,
				"prompt_eval_ms", final.PromptEvalDuration.Milliseconds(),
			)
			logutil.TraceContext(ctx, "mlx decode first token", "memory", mlx.Memory{})
		}

		if r.Tokenizer.IsEOS(output) {
			final.DoneReason = 0
			final.EvalCount = i
			break
		}

		if resp, ok := dec.decode(sample); ok {
			select {
			case <-ctx.Done():
				return ctx.Err()
			case request.Responses <- resp:
			}
		}

		mlx.Unpin(sample.Arrays()...)
		sample, nextSample = nextSample, sampler.Result{}

		if i%256 == 0 {
			mlx.ClearCache()
		}
	}

	final.EvalDuration = time.Since(now)
	if tail, ok := dec.flush(); ok {
		select {
		case <-ctx.Done():
			return ctx.Err()
		case request.Responses <- tail:
		}
	}
	select {
	case <-ctx.Done():
		return ctx.Err()
	case request.Responses <- final:
		return nil
	}
}

// decoder serializes sampled tokens into response chunks, holding bytes
// whose UTF-8 sequence hasn't completed yet and the logprobs that belong
// with those bytes so Content and Logprobs stay aligned when a chunk does
// flush.
type decoder struct {
	tokenizer       *tokenizer.Tokenizer
	buf             bytes.Buffer
	logprobs        []llm.Logprob
	wantLogprobs    bool
	wantTopLogprobs int
}

func (d *decoder) decode(res sampler.Result) (CompletionResponse, bool) {
	output := int32(res.Token.Int())
	d.buf.WriteString(d.tokenizer.Decode([]int32{output}))
	d.logprobs = append(d.logprobs, buildLogprob(res, d.wantLogprobs, d.wantTopLogprobs, d.tokenizer.Decode)...)

	content := flushValidUTF8Prefix(&d.buf)
	if content == "" {
		return CompletionResponse{}, false
	}
	resp := CompletionResponse{Content: content, Logprobs: d.logprobs}
	d.logprobs = nil
	return resp, true
}

func (d *decoder) flush() (CompletionResponse, bool) {
	if d.buf.Len() == 0 {
		return CompletionResponse{}, false
	}
	content := d.buf.String()
	d.buf.Reset()
	resp := CompletionResponse{Content: content, Logprobs: d.logprobs}
	d.logprobs = nil
	return resp, true
}

// buildLogprob converts the sampler's logprob tensors into the wire-format
// llm.Logprob entries the caller wants. The sampler populates its logprob
// tensors whenever any registered slot requested them, so the caller must
// gate emission on its own request config (wantLogprobs / wantTopLogprobs)
// rather than on whether the tensors happen to be non-nil.
func buildLogprob(sample sampler.Result, wantLogprobs bool, wantTopLogprobs int, decode func([]int32) string) []llm.Logprob {
	if !wantLogprobs || sample.Logprob == nil {
		return nil
	}
	tok := func(id int32) string { return decode([]int32{id}) }

	out := llm.Logprob{
		TokenLogprob: llm.TokenLogprob{
			Token:   tok(int32(sample.Token.Int())),
			Logprob: float64(sample.Logprob.Floats()[0]),
		},
	}

	if wantTopLogprobs > 0 && sample.TopTokens != nil {
		ids := sample.TopTokens.Ints()
		vals := sample.TopLogprobs.Floats()
		pairs := make([]llm.TokenLogprob, len(ids))
		for i, id := range ids {
			pairs[i] = llm.TokenLogprob{
				Token:   tok(int32(id)),
				Logprob: float64(vals[i]),
			}
		}
		// The sampler emits the top maxK across registered slots via
		// Argpartition, which leaves entries unsorted.
		sort.Slice(pairs, func(i, j int) bool {
			return pairs[i].Logprob > pairs[j].Logprob
		})
		if wantTopLogprobs < len(pairs) {
			pairs = pairs[:wantTopLogprobs]
		}
		out.TopLogprobs = pairs
	}
	return []llm.Logprob{out}
}
