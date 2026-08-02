// Package zimage implements the Z-Image diffusion transformer model.
package zimage

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strconv"
	"time"

	"github.com/ollama/ollama/x/imagegen/cache"
	latentfile "github.com/ollama/ollama/x/imagegen/latents"
	"github.com/ollama/ollama/x/imagegen/manifest"
	"github.com/ollama/ollama/x/imagegen/mlx"
	"github.com/ollama/ollama/x/imagegen/size"
	"github.com/ollama/ollama/x/imagegen/tokenizer"
	"github.com/ollama/ollama/x/imagegen/vae"
)

// GenerateConfig holds all options for image generation.
type GenerateConfig struct {
	Prompt         string
	NegativePrompt string                     // Empty = no CFG
	CFGScale       float32                    // Only used if NegativePrompt is set (default: 4.0)
	Width          int32                      // Image width (default: max side for VRAM)
	Height         int32                      // Image height (default: max side for VRAM)
	AspectRatio    string                     // Optional preset: 16:9, 9:16, 3:2, 2:3, 1:1
	Steps          int                        // Denoising steps (default: 9 for turbo)
	Seed           int64                      // Random seed
	Progress       func(step, totalSteps int) // Optional progress callback
	CapturePath    string                     // GPU capture path (debug)

	// TeaCache options (timestep embedding aware caching)
	TeaCache          bool    // TeaCache is always enabled for faster inference
	TeaCacheThreshold float32 // Threshold for cache reuse (default: 0.1, lower = more aggressive)

	// Fused QKV (fuse Q/K/V projections into single matmul)
	FusedQKV bool // Enable fused QKV projection (default: false)
}

// Model represents a Z-Image diffusion model.
type Model struct {
	ModelName   string
	Tokenizer   *tokenizer.Tokenizer
	TextEncoder *Qwen3TextEncoder
	Transformer *Transformer
	VAEDecoder  *VAEDecoder
	manifest    *manifest.ModelManifest
	qkvFused    bool // Track if QKV has been fused (do only once)
	needsReload struct {
		textEncoder bool
		transformer bool
		vae         bool
	}
}

// Load loads the Z-Image model from ollama blob storage.
func (m *Model) Load(modelName string) error {
	fmt.Printf("Loading Z-Image model from manifest: %s...\n", modelName)
	start := time.Now()

	if mlx.GPUIsAvailable() {
		mlx.SetDefaultDeviceGPU()
		mlx.DisableCompile()
		mlx.SetCacheLimit(0)
		// See runner.go — 12GiB is CUDA 16g only; Metal keeps MLX default / env override.
		mlx.ApplyImagegenMemoryLimit()
	}

	m.ModelName = modelName

	// Load manifest
	mf, err := manifest.LoadManifest(modelName)
	if err != nil {
		return fmt.Errorf("load manifest: %w", err)
	}
	m.manifest = mf

	// Load tokenizer from manifest with config
	fmt.Print("  Loading tokenizer... ")
	tokData, err := mf.ReadConfig("tokenizer/tokenizer.json")
	if err != nil {
		return fmt.Errorf("tokenizer: %w", err)
	}

	// Try to read tokenizer config files from manifest
	tokConfig := &tokenizer.TokenizerConfig{}
	if data, err := mf.ReadConfig("tokenizer/tokenizer_config.json"); err == nil {
		tokConfig.TokenizerConfigJSON = data
	}
	if data, err := mf.ReadConfig("tokenizer/generation_config.json"); err == nil {
		tokConfig.GenerationConfigJSON = data
	}
	if data, err := mf.ReadConfig("tokenizer/special_tokens_map.json"); err == nil {
		tokConfig.SpecialTokensMapJSON = data
	}

	tok, err := tokenizer.LoadFromBytesWithConfig(tokData, tokConfig)
	if err != nil {
		return fmt.Errorf("tokenizer: %w", err)
	}
	m.Tokenizer = tok
	fmt.Println("✓")

	if mlx.GPUIsAvailable() {
		// Defer text encoder to first generate so serve startup does not hold ~4.5GB
		// while other models may also be loading on tight VRAM hosts. CPU/Metal paths
		// load immediately because they are not contending with ggml on the same card.
		m.TextEncoder = nil
		m.needsReload.textEncoder = true
		fmt.Println("  Text encoder... deferred until first generate")
	} else {
		m.TextEncoder = &Qwen3TextEncoder{}
		if err := m.TextEncoder.Load(mf, "text_encoder/config.json"); err != nil {
			return fmt.Errorf("text encoder: %w", err)
		}
		mlx.UntrackWeights(m.TextEncoder)
		fmt.Printf("  (%.1f GB, peak %.1f GB)\n",
			float64(mlx.MetalGetActiveMemory())/(1024*1024*1024),
			float64(mlx.MetalGetPeakMemory())/(1024*1024*1024))
	}

	fmt.Println("  Transformer... deferred until after text encoding")
	fmt.Println("  VAE decoder... deferred until decode")
	m.needsReload.transformer = true
	m.needsReload.vae = true

	mem := mlx.MetalGetActiveMemory()
	fmt.Printf("  Loaded in %.2fs (%.1f GB VRAM)\n", time.Since(start).Seconds(), float64(mem)/(1024*1024*1024))

	return nil
}

func (m *Model) ensureTextEncoder() error {
	if !m.needsReload.textEncoder && m.TextEncoder != nil {
		return nil
	}
	m.TextEncoder = &Qwen3TextEncoder{}
	if err := m.TextEncoder.Load(m.manifest, "text_encoder/config.json"); err != nil {
		return fmt.Errorf("reload text encoder: %w", err)
	}
	mlx.UntrackWeights(m.TextEncoder)
	m.needsReload.textEncoder = false
	return nil
}

func (m *Model) ensureTransformer() error {
	if !m.needsReload.transformer && m.Transformer != nil {
		return nil
	}
	m.Transformer = &Transformer{}
	if err := m.Transformer.Load(m.manifest); err != nil {
		return fmt.Errorf("reload transformer: %w", err)
	}
	mlx.UntrackWeights(m.Transformer)
	m.needsReload.transformer = false
	m.qkvFused = false
	return nil
}

func (m *Model) freeTextEncoderWeights() {
	if m.TextEncoder == nil {
		return
	}
	fmt.Printf("  [freeTextEncoder] releasing %d arrays\n", len(mlx.Collect(m.TextEncoder)))
	before := mlx.MetalGetActiveMemory()
	mlx.ReleaseStruct(m.TextEncoder)
	m.TextEncoder = nil
	// ResumeCleanup drops MLX graph nodes that still reference freed encoder weights;
	// without this, transformer load sees inflated active memory and OOMs on 16GB.
	mlx.ResumeCleanup()
	mlx.Sync()
	mlx.TrimVRAM()
	fmt.Printf("  [freeTextEncoder] active=%.2fGB→%.2fGB\n",
		float64(before)/(1<<30), float64(mlx.MetalGetActiveMemory())/(1<<30))
	runtime.GC()
	m.needsReload.textEncoder = true
}

func (m *Model) reloadVAEDecoder() error {
	m.VAEDecoder = nil
	m.needsReload.vae = true
	return m.ensureVAEDecoder()
}

func (m *Model) ensureVAEDecoderCPU() error {
	if !m.needsReload.vae && m.VAEDecoder != nil {
		return nil
	}
	fmt.Print("  VAE decoder (CPU)... ")
	m.VAEDecoder = &VAEDecoder{}
	if err := m.VAEDecoder.LoadOnCPU(m.manifest); err != nil {
		return fmt.Errorf("load VAE on CPU: %w", err)
	}
	mlx.UntrackWeights(m.VAEDecoder)
	m.needsReload.vae = false
	fmt.Println("✓")
	return nil
}

func (m *Model) ensureVAEDecoder() error {
	if !m.needsReload.vae && m.VAEDecoder != nil {
		return nil
	}
	m.VAEDecoder = &VAEDecoder{}
	if err := m.VAEDecoder.Load(m.manifest); err != nil {
		return fmt.Errorf("reload VAE decoder: %w", err)
	}
	mlx.UntrackWeights(m.VAEDecoder)
	m.VAEDecoder.pinWeights()
	m.needsReload.vae = false
	return nil
}

func (m *Model) freeTransformerWeights() {
	if m.Transformer == nil {
		return
	}
	if mlx.GPUIsAvailable() {
		// Keep transformer weights resident between CUDA requests. Reloading after
		// denoise previously leaked handles and the second load OOMs on 16GB cards.
		// Idle VRAM stays higher until keep_alive expires — acceptable vs reload cost.
		mlx.ClearCache()
		mlx.TrimVRAM()
		fmt.Printf("  [freeTransformer] keeping transformer resident (%.2f GB active)\n",
			float64(mlx.MetalGetActiveMemory())/(1<<30))
		return
	}
	fmt.Printf("  [freeTransformer] releasing %d arrays\n", len(mlx.Collect(m.Transformer)))
	before := mlx.MetalGetActiveMemory()
	if m.VAEDecoder != nil {
		m.VAEDecoder.pinWeights()
	}
	mlx.ReleaseStruct(m.Transformer)
	m.Transformer = nil
	m.needsReload.transformer = true
	m.qkvFused = false
	if m.VAEDecoder != nil {
		m.VAEDecoder.pinWeights()
	}
	mlx.Sync()
	mlx.TrimVRAM()
	fmt.Printf("  [freeTransformer] active=%.2fGB→%.2fGB\n",
		float64(before)/(1<<30), float64(mlx.MetalGetActiveMemory())/(1<<30))
	runtime.GC()
}

// Generate creates an image from a prompt.
func (m *Model) Generate(prompt string, width, height int32, steps int, seed int64) (*mlx.Array, error) {
	return m.GenerateFromConfig(context.Background(), &GenerateConfig{
		Prompt: prompt,
		Width:  width,
		Height: height,
		Steps:  steps,
		Seed:   seed,
	})
}

// GenerateWithProgress creates an image with progress callback.
func (m *Model) GenerateWithProgress(prompt string, width, height int32, steps int, seed int64, progress func(step, totalSteps int)) (*mlx.Array, error) {
	return m.GenerateFromConfig(context.Background(), &GenerateConfig{
		Prompt:   prompt,
		Width:    width,
		Height:   height,
		Steps:    steps,
		Seed:     seed,
		Progress: progress,
	})
}

// GenerateWithCFG creates an image with classifier-free guidance.
func (m *Model) GenerateWithCFG(prompt, negativePrompt string, width, height int32, steps int, seed int64, cfgScale float32, progress func(step, totalSteps int)) (*mlx.Array, error) {
	return m.GenerateFromConfig(context.Background(), &GenerateConfig{
		Prompt:         prompt,
		NegativePrompt: negativePrompt,
		CFGScale:       cfgScale,
		Width:          width,
		Height:         height,
		Steps:          steps,
		Seed:           seed,
		Progress:       progress,
	})
}

// GenerateFromConfig generates an image using the unified config struct.
func (m *Model) GenerateFromConfig(ctx context.Context, cfg *GenerateConfig) (*mlx.Array, error) {
	start := time.Now()
	result, err := m.generate(ctx, cfg)
	if err != nil {
		return nil, err
	}
	if cfg.NegativePrompt != "" {
		fmt.Printf("Generated with CFG (scale=%.1f) in %.2fs (%d steps)\n", cfg.CFGScale, time.Since(start).Seconds(), cfg.Steps)
	} else {
		fmt.Printf("Generated in %.2fs (%d steps)\n", time.Since(start).Seconds(), cfg.Steps)
	}
	return result, nil
}

// GenerateImage implements runner.ImageModel interface.
func (m *Model) GenerateImage(ctx context.Context, prompt string, width, height int32, steps int, seed int64, progress func(step, total int)) (*mlx.Array, error) {
	return m.GenerateFromConfig(ctx, &GenerateConfig{
		Prompt:   prompt,
		Width:    width,
		Height:   height,
		Steps:    steps,
		Seed:     seed,
		Progress: progress,
	})
}

// generate is the internal denoising pipeline.
func (m *Model) generate(ctx context.Context, cfg *GenerateConfig) (*mlx.Array, error) {
	if err := m.ensureTextEncoder(); err != nil {
		return nil, err
	}

	// Apply defaults and aspect presets
	maxSide := maxImageSideForVRAM()
	var err error
	cfg.Width, cfg.Height, err = size.Resolve(cfg.Width, cfg.Height, cfg.AspectRatio, maxSide)
	if err != nil {
		return nil, err
	}
	origW, origH := cfg.Width, cfg.Height
	cfg.Width, cfg.Height = size.Clamp(cfg.Width, cfg.Height, maxSide)
	if cfg.Width != origW || cfg.Height != origH {
		fmt.Printf("  Output: %dx%d (VRAM clamp from %dx%d)\n", cfg.Width, cfg.Height, origW, origH)
	}
	if cfg.Steps <= 0 {
		cfg.Steps = 9 // Z-Image turbo default
	}
	if cfg.CFGScale <= 0 {
		cfg.CFGScale = 4.0
	}
	// TeaCache enabled by default (disabled on GPU — unreliable with CUDA graphs off)
	cfg.TeaCache = true
	if cfg.TeaCacheThreshold <= 0 {
		cfg.TeaCacheThreshold = 0.15
	}
	if mlx.GPUIsAvailable() {
		cfg.TeaCache = false
	}

	useCFG := cfg.NegativePrompt != ""

	// Text encoding with padding to multiple of 32
	var posEmb, negEmb *mlx.Array
	{
		posEmb, _ = m.TextEncoder.EncodePrompt(m.Tokenizer, cfg.Prompt, 512, false)
		if useCFG {
			negEmb, _ = m.TextEncoder.EncodePrompt(m.Tokenizer, cfg.NegativePrompt, 512, false)
		}

		// Pad both to same length (multiple of 32)
		maxLen := posEmb.Shape()[1]
		if useCFG && negEmb.Shape()[1] > maxLen {
			maxLen = negEmb.Shape()[1]
		}
		if pad := (32 - (maxLen % 32)) % 32; pad > 0 {
			maxLen += pad
		}

		posEmb = padToLength(posEmb, maxLen)
		if useCFG {
			negEmb = padToLength(negEmb, maxLen)
			mlx.Keep(posEmb, negEmb)
			mlx.Eval(posEmb, negEmb)
		} else {
			mlx.Keep(posEmb)
			mlx.Eval(posEmb)
		}
	}

	// Text encoder (~4.5GB) is not needed during denoise; free after embeddings are materialized.
	m.freeTextEncoderWeights()

	if err := m.ensureTransformer(); err != nil {
		return nil, err
	}
	mlx.TrimVRAM()

	// Enable fused QKV if requested (only fuse once)
	if cfg.FusedQKV && !m.qkvFused {
		m.Transformer.FuseAllQKV()
		m.qkvFused = true
		fmt.Println("  Fused QKV enabled")
	}

	tcfg := m.Transformer.TransformerConfig
	latentH := cfg.Height / 8
	latentW := cfg.Width / 8
	hTok := latentH / tcfg.PatchSize
	wTok := latentW / tcfg.PatchSize

	// Scheduler
	scheduler := NewFlowMatchEulerScheduler(DefaultFlowMatchSchedulerConfig())
	scheduler.SetTimestepsWithMu(cfg.Steps, CalculateShift(hTok*wTok))

	// Init latents [B, C, H, W]
	var latents *mlx.Array
	{
		latents = scheduler.InitNoise([]int32{1, tcfg.InChannels, latentH, latentW}, cfg.Seed)
		mlx.Eval(latents)
	}

	// RoPE cache
	var ropeCache *RoPECache
	{
		ropeCache = m.Transformer.PrepareRoPECache(hTok, wTok, posEmb.Shape()[1])
		mlx.Keep(ropeCache.ImgCos, ropeCache.ImgSin, ropeCache.CapCos, ropeCache.CapSin,
			ropeCache.UnifiedCos, ropeCache.UnifiedSin)
		mlx.Eval(ropeCache.UnifiedCos)
	}

	// Pre-compute batched embeddings for CFG (outside the loop for efficiency)
	var batchedEmb *mlx.Array
	if useCFG {
		// Concatenate embeddings once: [1, L, D] + [1, L, D] -> [2, L, D]
		batchedEmb = mlx.Concatenate([]*mlx.Array{posEmb, negEmb}, 0)
		mlx.Keep(batchedEmb)
		mlx.Eval(batchedEmb)
	}

	// TeaCache for timestep-aware caching
	// For CFG mode, we cache pos/neg separately, skip early steps, and always compute CFG fresh
	var teaCache *cache.TeaCache
	if cfg.TeaCache {
		skipEarly := 0
		if useCFG {
			skipEarly = 3 // Skip first 3 steps for CFG to preserve structure
		}
		teaCache = cache.NewTeaCache(&cache.TeaCacheConfig{
			Threshold:      cfg.TeaCacheThreshold,
			RescaleFactor:  1.0,
			SkipEarlySteps: skipEarly,
		})
		if useCFG {
			fmt.Printf("  TeaCache enabled (CFG mode): threshold=%.2f, skip first %d steps\n", cfg.TeaCacheThreshold, skipEarly)
		} else {
			fmt.Printf("  TeaCache enabled: threshold=%.2f\n", cfg.TeaCacheThreshold)
		}
	}

	// cleanup frees all kept arrays when we need to abort early
	cleanup := func() {
		posEmb.Free()
		if negEmb != nil {
			negEmb.Free()
		}
		ropeCache.ImgCos.Free()
		ropeCache.ImgSin.Free()
		ropeCache.CapCos.Free()
		ropeCache.CapSin.Free()
		ropeCache.UnifiedCos.Free()
		ropeCache.UnifiedSin.Free()
		if batchedEmb != nil {
			batchedEmb.Free()
		}
		if teaCache != nil {
			teaCache.Free()
		}
		latents.Free()
	}

	// Denoising loop
	if cfg.Progress != nil {
		cfg.Progress(0, cfg.Steps) // Start at 0%
	}
	for i := 0; i < cfg.Steps; i++ {
		// Check for cancellation
		if ctx != nil {
			select {
			case <-ctx.Done():
				cleanup()
				return nil, ctx.Err()
			default:
			}
		}
		stepStart := time.Now()

		// GPU capture on step 2 if requested
		if cfg.CapturePath != "" && i == 1 {
			mlx.MetalStartCapture(cfg.CapturePath)
		}

		tCurr := scheduler.Timesteps[i]
		var noisePred *mlx.Array

		// TeaCache: check if we should compute or reuse cached output
		shouldCompute := teaCache == nil || teaCache.ShouldCompute(i, tCurr)

		if shouldCompute {
			// Flush CUDA pool-reserved-but-freed memory before each forward pass.
			mlx.TrimVRAM()
			timestep := mlx.ToBFloat16(mlx.NewArray([]float32{1.0 - tCurr}, []int32{1}))
			patches := PatchifyLatents(latents, tcfg.PatchSize)

			var output *mlx.Array
			if useCFG {
				// CFG Batching: single forward pass with batch=2
				// Tile patches: [1, L, D] -> [2, L, D]
				batchedPatches := mlx.Tile(patches, []int32{2, 1, 1})
				// Tile timestep: [1] -> [2]
				batchedTimestep := mlx.Tile(timestep, []int32{2})

				// Single batched forward pass (RoPE broadcasts from [1,L,H,D] to [2,L,H,D])
				batchedOutput := m.Transformer.Forward(batchedPatches, batchedTimestep, batchedEmb, ropeCache)

				// Split output: [2, L, D] -> pos [1, L, D], neg [1, L, D]
				outputShape := batchedOutput.Shape()
				L := outputShape[1]
				D := outputShape[2]
				posOutput := mlx.Slice(batchedOutput, []int32{0, 0, 0}, []int32{1, L, D})
				negOutput := mlx.Slice(batchedOutput, []int32{1, 0, 0}, []int32{2, L, D})

				// Convert to noise predictions (unpatchify and negate)
				posPred := UnpatchifyLatents(posOutput, tcfg.PatchSize, latentH, latentW, tcfg.InChannels)
				posPred = mlx.Neg(posPred)
				negPred := UnpatchifyLatents(negOutput, tcfg.PatchSize, latentH, latentW, tcfg.InChannels)
				negPred = mlx.Neg(negPred)

				// Cache pos/neg separately for TeaCache
				if teaCache != nil {
					teaCache.UpdateCFGCache(posPred, negPred, tCurr)
					mlx.Keep(teaCache.Arrays()...)
				}

				// Apply CFG: noisePred = neg + scale * (pos - neg)
				diff := mlx.Sub(posPred, negPred)
				scaledDiff := mlx.MulScalar(diff, cfg.CFGScale)
				noisePred = mlx.Add(negPred, scaledDiff)
			} else {
				// Non-CFG forward pass
				output = m.Transformer.Forward(patches, timestep, posEmb, ropeCache)
				noisePred = UnpatchifyLatents(output, tcfg.PatchSize, latentH, latentW, tcfg.InChannels)
				noisePred = mlx.Neg(noisePred)

				// Update TeaCache
				if teaCache != nil {
					teaCache.UpdateCache(noisePred, tCurr)
					mlx.Keep(teaCache.Arrays()...)
				}
			}
		} else if useCFG && teaCache != nil && teaCache.HasCFGCache() {
			// CFG mode: get cached pos/neg and compute CFG fresh
			posPred, negPred := teaCache.GetCFGCached()
			diff := mlx.Sub(posPred, negPred)
			scaledDiff := mlx.MulScalar(diff, cfg.CFGScale)
			noisePred = mlx.Add(negPred, scaledDiff)
			fmt.Printf("    [TeaCache: reusing cached pos/neg outputs]\n")
		} else {
			// Non-CFG mode: reuse cached noise prediction
			noisePred = teaCache.GetCached()
			fmt.Printf("    [TeaCache: reusing cached output]\n")
		}

		oldLatents := latents

		latents = scheduler.Step(noisePred, latents, i)

		if err := mlx.EvalErr(latents); err != nil {
			oldLatents.Free()
			cleanup()
			return nil, fmt.Errorf(
				"denoise step %d failed (%.2fs, %dx%d): %w",
				i+1, time.Since(stepStart).Seconds(), cfg.Width, cfg.Height, err,
			)
		}
		oldLatents.Free()

		stepDur := time.Since(stepStart)

		if cfg.CapturePath != "" && i == 1 {
			mlx.MetalStopCapture()
		}

		activeMem := float64(mlx.MetalGetActiveMemory()) / (1024 * 1024 * 1024)
		peakMem := float64(mlx.MetalGetPeakMemory()) / (1024 * 1024 * 1024)
		fmt.Printf("  Step %d/%d: t=%.4f (%.2fs) [%.1f GB active, %.1f GB peak]\n",
			i+1, cfg.Steps, tCurr, stepDur.Seconds(), activeMem, peakMem)

		if cfg.Progress != nil {
			cfg.Progress(i+1, cfg.Steps) // Report completed step
		}
	}

	// CUDA path: write latents to disk and decode in a fresh CPU subprocess.
	// WHY not in-process GPU VAE: MLX CUDA allocator state after denoise caused
	// heap corruption / OOM when loading VAE weights in the same process (5080 16GB).
	if mlx.GPUIsAvailable() {
		mlx.Keep(latents)
		if err := mlx.EvalErr(latents); err != nil {
			cleanup()
			return nil, fmt.Errorf("finalize latents: %w", err)
		}
		latentsPath, err := exportLatentTensor(latents, "zimage-latents-")
		if err != nil {
			cleanup()
			return nil, fmt.Errorf("export latents: %w", err)
		}
		fmt.Printf("  Exported latents: %s\n", latentsPath)
		latents.Free()
		cleanup()
		m.freeTransformerWeights()
		mlx.Sync()
		mlx.TrimVRAM()
		return m.decodeLatentsSubprocess(latentsPath, cfg.Width, cfg.Height)
	}

	// Metal path: capture latents on CPU then decode.
	var latentShape []int32
	var latentData []float32
	if !mlx.GPUIsAvailable() {
		mlx.Keep(latents)
		if err := mlx.EvalErr(latents); err != nil {
			cleanup()
			return nil, fmt.Errorf("finalize latents: %w", err)
		}
		latentShape = latents.Shape()
		latentData = mlx.RawFloat32Slice(latents)
		if len(latentData) == 0 {
			latentData = mlx.HostFloat32Slice(latents)
		}
		if len(latentData) == 0 {
			cleanup()
			return nil, fmt.Errorf("latents raw read failed")
		}
		latents.Free()
	} else {
		cleanup()
		return nil, fmt.Errorf("CUDA decode path did not run")
	}

	// Free denoising temporaries before VAE decode
	posEmb.Free()
	if negEmb != nil {
		negEmb.Free()
	}
	ropeCache.ImgCos.Free()
	ropeCache.ImgSin.Free()
	ropeCache.CapCos.Free()
	ropeCache.CapSin.Free()
	ropeCache.UnifiedCos.Free()
	ropeCache.UnifiedSin.Free()
	if batchedEmb != nil {
		batchedEmb.Free()
	}
	if teaCache != nil {
		hits, misses := teaCache.Stats()
		fmt.Printf("  TeaCache stats: %d hits, %d misses (%.1f%% cache rate)\n",
			hits, misses, float64(hits)/float64(hits+misses)*100)
		teaCache.Free()
	}

	var latentsDecode *mlx.Array
	m.freeTransformerWeights()
	mlx.Sync()
	mlx.TrimVRAM()
	latentsDecode = mlx.NewArray(latentData, latentShape)
	mlx.Untrack(latentsDecode)
	mlx.Eval(latentsDecode)

	var decoded *mlx.Array
	if m.VAEDecoder == nil {
		latentsDecode.Release()
		return nil, fmt.Errorf("VAE decoder not loaded")
	}
	if latentH > 64 || latentW > 64 {
		m.VAEDecoder.Tiling = vae.DefaultTilingConfig()
	}
	decoded = m.VAEDecoder.Decode(latentsDecode)
	latentsDecode.Release()
	mlx.Sync()

	cpuImage, err := exportDecodedToCPU(decoded)
	decoded.Free()
	if err != nil {
		return nil, err
	}
	return cpuImage, nil
}

func maxImageSideForVRAM() int32 {
	return size.MaxSide(mlx.GPUIsAvailable())
}

func clampResolutionForVRAM(w, h int32) (int32, int32) {
	return size.Clamp(w, h, maxImageSideForVRAM())
}

func exportLatentTensor(arr *mlx.Array, prefix string) (string, error) {
	if arr == nil || !arr.Valid() {
		return "", fmt.Errorf("invalid tensor")
	}
	f, err := os.CreateTemp("", prefix+"*.bin")
	if err != nil {
		return "", err
	}
	path := f.Name()
	f.Close()
	if err := mlx.ExportLatentsBin(path, arr); err != nil {
		os.Remove(path)
		return "", err
	}
	return path, nil
}

func (m *Model) decodeLatentsSubprocess(latentsPath string, width, height int32) (*mlx.Array, error) {
	exe, err := os.Executable()
	if err != nil {
		return nil, err
	}
	if eval, err := filepath.EvalSymlinks(exe); err == nil {
		exe = eval
	}

	outFile, err := os.CreateTemp("", "zimage-decoded-*.bin")
	if err != nil {
		return nil, err
	}
	outPath := outFile.Name()
	outFile.Close()
	defer os.Remove(outPath)
	defer os.Remove(latentsPath)

	fmt.Println("  VAE decode via subprocess (CPU)...")
	cmd := exec.Command(
		exe, "runner", "--imagegen-decode-latents",
		"--model", m.ModelName,
		"--latents-file", latentsPath,
		"--output", outPath,
		"--width", strconv.Itoa(int(width)),
		"--height", strconv.Itoa(int(height)),
	)
	cmd.Env = os.Environ()
	cmd.Stderr = os.Stderr
	out, err := cmd.Output()
	if err != nil {
		return nil, fmt.Errorf("vae subprocess: %w", err)
	}
	var status struct {
		OK bool `json:"ok"`
	}
	if json.Unmarshal(out, &status) != nil || !status.OK {
		return nil, fmt.Errorf("vae subprocess bad status: %s", string(out))
	}

	img, err := latentfile.LoadBin(outPath)
	if err != nil {
		return nil, fmt.Errorf("load decoded image: %w", err)
	}
	return img, nil
}

func (m *Model) decodeHandoffSubprocess(metaPath string) (*mlx.Array, error) {
	exe, err := os.Executable()
	if err != nil {
		return nil, err
	}
	if eval, err := filepath.EvalSymlinks(exe); err == nil {
		exe = eval
	}

	outFile, err := os.CreateTemp("", "zimage-decoded-*.bin")
	if err != nil {
		return nil, err
	}
	outPath := outFile.Name()
	outFile.Close()
	defer os.Remove(outPath)
	defer os.Remove(metaPath)

	fmt.Println("  VAE decode via subprocess (CPU handoff)...")
	cmd := exec.Command(
		exe, "runner", "--imagegen-decode-latents",
		"--model", m.ModelName,
		"--denoise-meta", metaPath,
		"--output", outPath,
	)
	cmd.Env = os.Environ()
	cmd.Stderr = os.Stderr
	out, err := cmd.Output()
	if err != nil {
		return nil, fmt.Errorf("vae subprocess: %w", err)
	}
	var status struct {
		OK bool `json:"ok"`
	}
	if json.Unmarshal(out, &status) != nil || !status.OK {
		return nil, fmt.Errorf("vae subprocess bad status: %s", string(out))
	}

	img, err := latentfile.LoadBin(outPath)
	if err != nil {
		return nil, fmt.Errorf("load decoded image: %w", err)
	}
	return img, nil
}

// DecodeFromHandoff applies the final Euler step on CPU and runs VAE decode.
func (m *Model) DecodeFromHandoff(modelName string, meta *latentfile.FinalStepMeta) (*mlx.Array, error) {
	m.ModelName = modelName
	mf, err := manifest.LoadManifest(modelName)
	if err != nil {
		return nil, fmt.Errorf("load manifest: %w", err)
	}
	m.manifest = mf

	sample, err := latentfile.LoadBin(meta.SamplePath)
	if err != nil {
		return nil, fmt.Errorf("load sample: %w", err)
	}
	defer sample.Release()
	noise, err := latentfile.LoadBin(meta.NoisePath)
	if err != nil {
		return nil, fmt.Errorf("load noise: %w", err)
	}
	defer noise.Release()

	latentH := meta.Height / 8
	latentW := meta.Width / 8
	patchSize := int32(2)
	scheduler := NewFlowMatchEulerScheduler(DefaultFlowMatchSchedulerConfig())
	scheduler.SetTimestepsWithMu(meta.Steps, CalculateShift((latentH/patchSize)*(latentW/patchSize)))
	dt := scheduler.Sigmas[meta.StepIdx+1] - scheduler.Sigmas[meta.StepIdx]
	latentsArr := mlx.Add(sample, mlx.MulScalar(noise, dt))
	mlx.Keep(latentsArr)
	mlx.Eval(latentsArr)

	mlx.SetDefaultDeviceCPU()
	if err := m.ensureVAEDecoderCPU(); err != nil {
		return nil, err
	}
	if latentH > 64 || latentW > 64 {
		m.VAEDecoder.Tiling = vae.DefaultTilingConfig()
	}
	decoded := m.VAEDecoder.Decode(latentsArr)
	if decoded == nil || !decoded.Valid() {
		return nil, fmt.Errorf("VAE decode failed")
	}
	mlx.EvalMaterialize(decoded)
	mlx.Sync()
	return exportDecodedToCPU(decoded)
}

// DecodeLatentsFromFile loads latents from disk and runs VAE decode in a fresh MLX process.
func (m *Model) DecodeLatentsFromFile(modelName, latentsPath string, width, height int32) (*mlx.Array, error) {
	m.ModelName = modelName
	mf, err := manifest.LoadManifest(modelName)
	if err != nil {
		return nil, fmt.Errorf("load manifest: %w", err)
	}
	m.manifest = mf

	latentsArr, err := latentfile.LoadBin(latentsPath)
	if err != nil {
		return nil, err
	}
	defer latentsArr.Release()

	// Fresh subprocess: decode on CPU to avoid CUDA heap issues after denoise.
	mlx.SetDefaultDeviceCPU()
	if err := m.ensureVAEDecoderCPU(); err != nil {
		return nil, err
	}
	latentH := height / 8
	latentW := width / 8
	if latentH > 64 || latentW > 64 {
		m.VAEDecoder.Tiling = vae.DefaultTilingConfig()
	}
	decoded := m.VAEDecoder.Decode(latentsArr)
	if decoded == nil || !decoded.Valid() {
		return nil, fmt.Errorf("VAE decode failed")
	}
	mlx.EvalMaterialize(decoded)
	mlx.Sync()
	return exportDecodedToCPU(decoded)
}

func exportDecodedToCPU(gpu *mlx.Array) (*mlx.Array, error) {
	if gpu == nil || !gpu.Valid() {
		return nil, fmt.Errorf("invalid decode output")
	}
	shape := gpu.Shape()
	if len(shape) != 4 || shape[1] != 3 {
		return nil, fmt.Errorf("unexpected decode shape %v", shape)
	}

	cpu := mlx.MaterializeOnCPU(gpu)
	if cpu == nil {
		return nil, fmt.Errorf("failed to materialize decode output")
	}
	read := cpu
	if cpu.Dtype() != mlx.DtypeFloat32 {
		read = mlx.AsType(cpu, mlx.DtypeFloat32)
		mlx.EvalMaterialize(read)
	}
	data := mlx.HostFloat32Slice(read)
	if read != cpu {
		read.Release()
	}
	cpu.Release()

	expected := int(shape[0] * shape[1] * shape[2] * shape[3])
	if len(data) < expected {
		return nil, fmt.Errorf("decode export: got %d floats, want %d", len(data), expected)
	}
	if len(data) > expected {
		data = data[:expected]
	}
	out := mlx.NewArrayFloat32(data, shape)
	mlx.Untrack(out)
	return out, nil
}

// padToLength pads a sequence tensor to the target length by repeating the last token.
func padToLength(x *mlx.Array, targetLen int32) *mlx.Array {
	shape := x.Shape()
	currentLen := shape[1]
	if currentLen >= targetLen {
		return x
	}
	padLen := targetLen - currentLen
	lastToken := mlx.Slice(x, []int32{0, currentLen - 1, 0}, []int32{shape[0], currentLen, shape[2]})
	padding := mlx.Tile(lastToken, []int32{1, padLen, 1})
	return mlx.Concatenate([]*mlx.Array{x, padding}, 1)
}

// CalculateShift computes the mu shift value for dynamic scheduling
func CalculateShift(imgSeqLen int32) float32 {
	baseSeqLen := float32(256)
	maxSeqLen := float32(4096)
	baseShift := float32(0.5)
	maxShift := float32(1.15)

	m := (maxShift - baseShift) / (maxSeqLen - baseSeqLen)
	b := baseShift - m*baseSeqLen
	return float32(imgSeqLen)*m + b
}
