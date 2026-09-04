package mlxrunner

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"log/slog"
	"sort"
	"strings"
	"time"

	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/logutil"
	"github.com/ollama/ollama/x/mlxrunner/batch"
	"github.com/ollama/ollama/x/mlxrunner/cache"
	"github.com/ollama/ollama/x/mlxrunner/mlx"
	"github.com/ollama/ollama/x/mlxrunner/model/base"
	sampler "github.com/ollama/ollama/x/mlxrunner/sample"
	"github.com/ollama/ollama/x/tokenizer"
	"github.com/ollama/ollama/x/uma"
)

func modelSlidingWindow(m base.Model) int {
	type sliding interface {
		SlidingWindowSize() int
	}
	s, ok := m.(sliding)
	if !ok {
		return 0
	}
	return s.SlidingWindowSize()
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
		return fmt.Errorf("%s", llm.ContextOverflowMessage(len(tokens), r.contextLength))
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

	defer func() {
		r.Sampler.Remove(pipelineSlot)
		mlx.Sweep()
		mlx.ClearCache()

		if slog.Default().Enabled(context.TODO(), logutil.LevelTrace) {
			mlx.LogArrays()
			r.cache.dumpTree()
		}
		slog.Info("peak memory", "size", mlx.PrettyBytes(mlx.PeakMemory()))
	}()

	inputs := request.Tokens

	session := r.cache.begin(r.Model, inputs, request.PromptCacheKey, request.CacheReset)
	defer session.close()
	caches := session.caches

	// Built before prefill so a drafter with draft caches follows the prompt
	// through prefill alongside the target.
	spec := r.spec.open(request, caches)
	defer spec.close()

	seed, position, promptEval, err := r.prefill(ctx, request, session, spec)
	if err != nil {
		return err
	}

	// Sampler registration + first dispatch/prime can Eval before decode's
	// per-step leases; hold one admission window for that setup.
	var d decoder
	if err := func() error {
		if err := uma.LeaseBegin("decode-prime"); err != nil {
			return err
		}
		defer uma.LeaseEnd()
		defer mlx.Synchronize() // first AsyncEval must finish under HOLD
		r.Sampler.Add(pipelineSlot, request.SamplerOpts, inputs)
		r.Sampler.SetSlotLogitBias(pipelineSlot, request.Options.LogitBias)
		if spec != nil {
			d = spec.decoder(seed, position)
		} else {
			// Prefill seed is 1-D token ids (same layout as sampler.Result.Token);
			// ExpandDims(0) yields InputIDs [1, L] for Forward.
			d = r.pipelinedDecoder(nil, caches, seed.ExpandDims(0), position)
		}
		return nil
	}(); err != nil {
		return err
	}
	defer d.close()
	return r.decode(ctx, request, session, d, promptEval)
}

// completionPromptMetrics reports prompt tokens evaluated this request vs restored
// from the MLX prefix trie (mirrors llama-server cache_n / prompt_eval_count).
func completionPromptMetrics(session *cacheSession) (evalCount, cachedCount int) {
	if session == nil {
		return 0, 0
	}
	total := len(session.inputs)
	cached := session.cachedPrefix
	if cached < 0 {
		cached = 0
	}
	if cached > total {
		cached = total
	}
	return total - cached, cached
}

// prefill evaluates the prompt in chunks, leaving one token for decode to
// seed from, and schedules the prompt's periodic snapshots. It returns the
// seed tokens, the resume position, and the prompt-evaluation duration.
//
// prefillSeedCount is the T=0 exactness contract (mtplx 2.6.0): speculative
// and plain AR partition the prompt the same way (body + last token). A
// whole-prompt spec prefill vs a split AR prefill rounds the last bit
// differently and can flip greedy argmax at a near-tie.
const prefillSeedCount = 1

func prefillBodyLen(remaining int) int {
	if remaining <= prefillSeedCount {
		return 0
	}
	return remaining - prefillSeedCount
}
func (r *Runner) prefill(ctx context.Context, request Request, session *cacheSession, spec *speculationSession) (*mlx.Array, int, time.Duration, error) {
	start := time.Now()
	inputs := session.inputs
	tokens := session.remaining
	caches := session.caches

	cfg := defaultPrefillConfig(len(inputs), modelSlidingWindow(r.Model))
	cfg = tunePrefillConfig(cfg, prefillTuneInput{
		total:          len(inputs),
		cachedPrefix:   session.cachedPrefix,
		remaining:      len(session.remaining),
		promptCacheKey: request.PromptCacheKey,
		fastPath:       session.fastPath,
		sameBranch:     session.sameBranch,
	})
	before := cfg.chunkSize
	if rec, err := mlx.MaxRecommendedWorkingSetSize(); err == nil {
		cfg = capPrefillChunkForWorkingSet(cfg, mlx.ActiveMemory(), rec)
		if cfg.chunkSize < before {
			slog.Info("mlx tune",
				"hint", "prefill chunk shrunk because unified memory is ≥75% of the recommended working set",
				"chunk", cfg.chunkSize,
				"was", before,
				"tune", "unset OLLAMA_MLX_PREFILL_CHUNK; or raise RAM headroom / shorter prompt",
			)
		}
	}
	prefillChunk := cfg.chunkSize
	if prefillChunk < 1 {
		prefillChunk = defaultPrefillChunk
	}

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

	session.schedulePrefillSnapshots(buildTriePrefillSnapshotOffsets(cfg, len(inputs), request.PromptCacheKey, modelSlidingWindow(r.Model)))

	total, processed := len(tokens), 0
	position := len(inputs) - len(tokens)
	chunkIdx := 0
	for body := prefillBodyLen(total - processed); body > 0; body = prefillBodyLen(total - processed) {
		if err := ctx.Err(); err != nil {
			return nil, 0, 0, err
		}

		n := min(prefillChunk, body)
		remainingAfter := total - processed - n
		isLastChunk := remainingAfter <= prefillSeedCount
		chunkIdx++

		tForwardStart := time.Now()
		var (
			tForward, tSweep, tMaterialize time.Duration
		)
		if err := func() error {
			if err := uma.LeaseBegin("prefill"); err != nil {
				return err
			}
			defer uma.LeaseEnd()
			defer mlx.Synchronize()
			chunkIDs := mlx.FromValues(tokens[processed:processed+n], 1, n)
			hidden := r.Model.Forward(&batch.Batch{
				InputIDs:     chunkIDs,
				SeqOffsets:   []int32{int32(position)},
				SeqQueryLens: []int32{int32(n)},
			}, caches)
			spec.committed(chunkIDs, hidden, position)
			tForward = time.Since(tForwardStart)

			tSweepStart := time.Now()
			mlx.Sweep()
			tSweep = time.Since(tSweepStart)

			shouldMaterialize := cfg.materializeEvery <= 1 || chunkIdx%cfg.materializeEvery == 0 || isLastChunk
			if shouldMaterialize {
				tMatStart := time.Now()
				materializeCaches()
				tMaterialize = time.Since(tMatStart)
			}
			return nil
		}(); err != nil {
			return nil, 0, 0, err
		}

		processed += n
		position += n

		var tClear time.Duration
		if cfg.clearCacheEvery > 0 && (chunkIdx%cfg.clearCacheEvery == 0 || isLastChunk) {
			tClearStart := time.Now()
			mlx.ClearCache()
			tClear = time.Since(tClearStart)
		}

		emitPrefillProgress(ctx, request.Responses, processed, total)
		slog.Info("Prompt processing progress", "processed", processed, "total", total)
		logutil.TraceContext(ctx, "mlx prompt forward", "processed", processed, "total", total, "tokens", n, "memory", mlx.Memory{})
		slog.Debug("mlx prefill chunk",
			"chunk", chunkIdx,
			"tokens", n,
			"forward_ms", tForward.Milliseconds(),
			"sweep_ms", tSweep.Milliseconds(),
			"materialize_ms", tMaterialize.Milliseconds(),
			"clear_ms", tClear.Milliseconds(),
			"memory", mlx.Memory{},
		)
	}

	// Settle before attaching: snapshots attach only at offsets every cache
	// has crossed, and the draft caches stay a pair short of the target
	// until the seed completes the frontier pair.
	// Seed is 1-D token ids (matches sampler.Result.Token / settle / draft propose).
	var seed *mlx.Array
	var attachMs int64
	if err := func() error {
		if err := uma.LeaseBegin("prefill-seed"); err != nil {
			return err
		}
		defer uma.LeaseEnd()
		defer mlx.Synchronize()
		seed = mlx.FromValues(tokens[processed:], len(tokens)-processed)
		if spec != nil {
			spec.settle(seed)
		}
		tAttachStart := time.Now()
		session.attachPrefillSnapshots()
		attachMs = time.Since(tAttachStart).Milliseconds()
		return nil
	}(); err != nil {
		return nil, 0, 0, err
	}
	slog.Debug("mlx prefill snapshots attached", "elapsed_ms", attachMs)

	promptEval := time.Since(start)
	prefillTokens := processed
	attrs := []any{
		"prompt_tokens", len(inputs),
		"cached_tokens", session.cachedPrefix,
		"prefill_tokens", prefillTokens,
		"elapsed", promptEval,
		"chunk_size", prefillChunk,
	}
	if prefillTokens > 0 && promptEval > 0 {
		attrs = append(attrs, "tok_per_sec", float64(prefillTokens)/promptEval.Seconds())
	}
	if len(inputs) > defaultMTPMaxPromptTokens && prefillSnapshotInterval(cfg, len(inputs)) == 0 {
		slog.Info("prefill MTP snapshots disabled for long prompt", "prompt_tokens", len(inputs),
			"trie_restore", strings.TrimSpace(request.PromptCacheKey) != "")
	}
	if session.cachedPrefix == 0 && len(inputs) > defaultMTPMaxPromptTokens && strings.TrimSpace(request.PromptCacheKey) != "" {
		slog.Debug("mlx prefix cache miss despite prompt_cache_key; full prefill required",
			"prompt_cache_key", request.PromptCacheKey,
			"prompt_tokens", len(inputs),
			"hint", "prompt tokens changed since last turn, runner reloaded, or trie restore failed",
		)
	}
	slog.Info("prefill complete", attrs...)

	return seed, position, promptEval, nil
}

func emitPrefillProgress(ctx context.Context, responses chan<- CompletionResponse, processed, total int) {
	if responses == nil || processed <= 0 {
		return
	}
	resp := CompletionResponse{
		PrefillProcessed: processed,
		PrefillTotal:     total,
	}
	select {
	case <-ctx.Done():
	case responses <- resp:
	}
}

// A decoder produces each run of tokens to emit, owning its own dispatch and
// synchronization; the decode loop owns the budget, emission, and
// cancellation. next may return none while its first tokens are in flight.
type decoder interface {
	next(remaining int) ([]sampler.Result, error)

	// drain ends production, returning any results sampled but never
	// delivered through next and the position the next forward would have
	// taken; the decoder remains closeable.
	drain() ([]sampler.Result, int)

	close()
}

// decode drives either decoder and owns where generation stops — at an EOS
// or the NumPredict budget. Every produced token is recorded so the caches
// never rest ahead of session.outputs; tokens past the stop are recorded but
// not streamed or counted.
func (r *Runner) decode(ctx context.Context, request Request, session *cacheSession, d decoder, promptEval time.Duration) error {
	// A sampled-but-undelivered result is still a produced token; record it.
	defer func() {
		results, _ := d.drain()
		for _, res := range results {
			session.outputs = append(session.outputs, int32(res.Token.Int()))
		}
	}()

	detok := detokenizer{
		tokenizer:       r.Tokenizer,
		wantLogprobs:    request.SamplerOpts.Logprobs,
		wantTopLogprobs: request.SamplerOpts.TopLogprobs,
	}
	stops := nonemptyStops(request.Options.Stop)
	var stopHold string

	final := CompletionResponse{Done: true, DoneReason: 1}
	final.PromptEvalCount, final.PromptEvalCachedCount = completionPromptMetrics(session)
	final.PromptEvalDuration = promptEval
	now := time.Now()

	// Release MLX's cached free buffers every clearCacheInterval tokens so the
	// allocator's pool does not grow unbounded over a long generation.
	const clearCacheInterval = 256

	generated := 0
	for generated < request.Options.NumPredict {
		if err := ctx.Err(); err != nil {
			return err
		}

		var results []sampler.Result
		if err := func() error {
			if err := uma.LeaseBegin("decode"); err != nil {
				return err
			}
			defer uma.LeaseEnd()
			defer mlx.Synchronize() // AsyncEval drain before RELEASE; Int() after is host-side
			var nextErr error
			results, nextErr = d.next(request.Options.NumPredict - generated)
			return nextErr
		}(); err != nil {
			return err
		}

		// Record the whole run before streaming any of it: a cancelled
		// stream returns early and must not leave the caches ahead of
		// session.outputs.
		done := false
		stream := len(results)
		for i, res := range results {
			// Int evaluates the array before reading it; a raw data read
			// on a lazy array races its evaluation and returns garbage.
			id := int32(res.Token.Int())
			session.outputs = append(session.outputs, id)
			if done {
				continue
			}
			if r.Tokenizer.IsEOS(id) {
				final.DoneReason = 0
				done = true
				stream = i
				continue
			}
			generated++
			if generationShouldStop(session.outputs) {
				logLoopStop(session.outputs)
				final.FinishDetails = finishDetailsRepetitionLoop
				done = true
				stream = i + 1
				continue
			}
			if generated >= request.Options.NumPredict {
				done = true
				stream = i + 1
			}
		}

		for _, res := range results[:stream] {
			resp, ok := detok.detokenize(res)
			if !ok {
				continue
			}
			stopHold += resp.Content
			emit, rest, matched := flushStopHold(stopHold, stops, false)
			stopHold = rest
			if emit != "" {
				resp.Content = emit
				select {
				case <-ctx.Done():
					return ctx.Err()
				case request.Responses <- resp:
				}
			}
			if matched != "" {
				final.DoneReason = 0
				final.StopSequence = matched
				done = true
				break
			}
		}

		if done {
			break
		}

		if generated%clearCacheInterval == 0 {
			mlx.ClearCache()
		}
	}

	if emit, _, _ := flushStopHold(stopHold, stops, true); emit != "" {
		select {
		case <-ctx.Done():
			return ctx.Err()
		case request.Responses <- CompletionResponse{Content: emit}:
		}
	}

	final.EvalCount = generated
	final.EvalDuration = time.Since(now)
	if r.spec != nil && r.spec.depth != nil {
		r.spec.depth.saveRoundCost(r.modelName)
	}
	select {
	case <-ctx.Done():
		return ctx.Err()
	case request.Responses <- final:
		return nil
	}
}

// pipelinedDecoder decodes one token per call, one call ahead of emission:
// the next token's chain is dispatched before the returned one is
// synchronized, so the device runs ahead of host emission.
type pipelinedDecoder struct {
	r *Runner
	// spec, when non-nil, receives every forwarded token and settles its
	// drafter at close, keeping a non-drafting session's draft KV level.
	spec     *speculationSession
	caches   []cache.Cache
	position int
	sample   sampler.Result // in flight: sampled, not yet forwarded
}

func (r *Runner) pipelinedDecoder(spec *speculationSession, caches []cache.Cache, seed *mlx.Array, position int) *pipelinedDecoder {
	t := &pipelinedDecoder{r: r, spec: spec, caches: caches, position: position}
	t.sample = t.dispatch(seed)
	return t
}

// dispatch builds one forward-and-sample chain without reading the token's
// value, so it is in flight before the previous token is synchronized.
func (t *pipelinedDecoder) dispatch(token *mlx.Array) sampler.Result {
	r := t.r
	hidden := r.Model.Forward(&batch.Batch{
		InputIDs:     token,
		SeqOffsets:   []int32{int32(t.position)},
		SeqQueryLens: []int32{int32(token.Dim(1))},
	}, t.caches)
	t.spec.committed(token, hidden, t.position)
	t.position += token.Dim(1)
	logits := r.Model.Unembed(hidden)
	next := r.Sampler.Sample([]int{pipelineSlot}, logits.Slice(mlx.Slice(), mlx.Slice(logits.Dim(1)-1), mlx.Slice()).Squeeze(1))
	mlx.Pin(next.Arrays()...)
	mlx.Sweep()
	mlx.AsyncEval(next.Arrays()...)
	return next
}

func (t *pipelinedDecoder) next(int) ([]sampler.Result, error) {
	out := t.sample
	t.sample = t.dispatch(out.Token.ExpandDims(-1))
	mlx.Unpin(out.Arrays()...)
	return []sampler.Result{out}, nil
}

// drain ends production: it returns the in-flight sample (sampled but never
// forwarded) and the position its forward would have taken. The decoder
// keeps the sample for close.
func (t *pipelinedDecoder) drain() ([]sampler.Result, int) {
	return []sampler.Result{t.sample}, t.position
}

func (t *pipelinedDecoder) close() {
	// The in-flight sample's forward was never dispatched; its report settles
	// the drafter level with the caches' resting offset.
	if t.spec != nil {
		t.spec.settle(t.sample.Token)
	}
	mlx.Unpin(t.sample.Arrays()...)
}

// detokenizer serializes sampled tokens into response chunks, holding bytes
// whose UTF-8 sequence hasn't completed yet and the logprobs that belong
// with those bytes so Content and Logprobs stay aligned when a chunk does
// flush.
type detokenizer struct {
	tokenizer       *tokenizer.Tokenizer
	buf             bytes.Buffer
	logprobs        []llm.Logprob
	wantLogprobs    bool
	wantTopLogprobs int
}

func (d *detokenizer) detokenize(res sampler.Result) (CompletionResponse, bool) {
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

	id := int(sample.Token.Int())
	out := llm.Logprob{
		TokenLogprob: llm.TokenLogprob{
			Token:   tok(int32(id)),
			Logprob: float64(sample.Logprob.Floats()[0]),
			ID:      llm.IntPtr(id),
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
				ID:      llm.IntPtr(int(id)),
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
