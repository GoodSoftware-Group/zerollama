package llm

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/ml"
)

func TestSelectGpuProfile_RTX5080(t *testing.T) {
	repo := findRepoRoot(t)
	t.Setenv("ZEROLLAMA_REPO", repo)
	t.Setenv("ZEROLLAMA_GPU_PROFILE", "1")
	t.Setenv("ZEROLLAMA_LLAMA_FORK", "0")
	t.Setenv("ZEROLLAMA_GPU_PROFILE_ID", "")

	gpus := []ml.DeviceInfo{{
		Description: "NVIDIA GeForce RTX 5080",
		TotalMemory: 16 << 30,
	}}
	p := SelectGpuProfile(gpus)
	if p == nil {
		t.Fatal("expected rtx-5080 profile")
	}
	if p.ID != "rtx-5080" {
		t.Fatalf("id = %q, want rtx-5080", p.ID)
	}
	if p.Source != "match" {
		t.Fatalf("source = %q, want match", p.Source)
	}
	if p.NParallel != 2 {
		t.Fatalf("n_parallel = %d, want 2", p.NParallel)
	}
	if p.BatchSize != 1024 || p.UBatchSize != 256 {
		t.Fatalf("batch/ubatch = %d/%d, want 1024/256", p.BatchSize, p.UBatchSize)
	}
	if p.CacheTypeK != "q8_0" || p.CacheTypeV != "q8_0" {
		t.Fatalf("kv = %s/%s, want q8_0/q8_0", p.CacheTypeK, p.CacheTypeV)
	}
	if !p.FlashAttn {
		t.Fatal("expected flash_attn")
	}
	if p.DraftMax != 16 || p.DraftMin != 4 {
		t.Fatalf("draft max/min = %d/%d, want 16/4", p.DraftMax, p.DraftMin)
	}
}

func TestSelectGpuProfile_NvidiaSMIFallback(t *testing.T) {
	repo := findRepoRoot(t)
	t.Setenv("ZEROLLAMA_REPO", repo)
	t.Setenv("ZEROLLAMA_GPU_PROFILE", "1")
	t.Setenv("ZEROLLAMA_LLAMA_FORK", "0")
	t.Setenv("ZEROLLAMA_GPU_PROFILE_ID", "")
	// Empty device list — must still match via nvidia-smi on this host.
	p := SelectGpuProfile(nil)
	if p == nil {
		t.Skip("nvidia-smi unavailable or no matching profile on this host")
	}
	if p.ID != "rtx-5080" && p.Source != "bucket" && p.Source != "match" {
		t.Fatalf("unexpected profile %+v", p)
	}
	if p.BatchSize != 1024 || p.CacheTypeK != "q8_0" {
		t.Fatalf("expected L1 throughput knobs, got %+v", p)
	}
}

func TestSelectGpuProfile_Disabled(t *testing.T) {
	repo := findRepoRoot(t)
	t.Setenv("ZEROLLAMA_REPO", repo)
	t.Setenv("ZEROLLAMA_GPU_PROFILE", "0")
	gpus := []ml.DeviceInfo{{Description: "NVIDIA GeForce RTX 5080", TotalMemory: 16 << 30}}
	if p := SelectGpuProfile(gpus); p != nil {
		t.Fatalf("expected nil when disabled, got %+v", p)
	}
}

func TestSelectGpuProfile_ForcedID(t *testing.T) {
	repo := findRepoRoot(t)
	t.Setenv("ZEROLLAMA_REPO", repo)
	t.Setenv("ZEROLLAMA_GPU_PROFILE", "1")
	t.Setenv("ZEROLLAMA_LLAMA_FORK", "0")
	t.Setenv("ZEROLLAMA_GPU_PROFILE_ID", "rtx-5080")
	p := SelectGpuProfile(nil)
	if p == nil || p.ID != "rtx-5080" || p.Source != "env" {
		t.Fatalf("forced id failed: %+v", p)
	}
}

func TestSelectGpuProfile_VRAMBucket(t *testing.T) {
	repo := findRepoRoot(t)
	t.Setenv("ZEROLLAMA_REPO", repo)
	t.Setenv("ZEROLLAMA_GPU_PROFILE", "1")
	t.Setenv("ZEROLLAMA_LLAMA_FORK", "0")
	t.Setenv("ZEROLLAMA_GPU_PROFILE_ID", "")
	// Unknown name, 16 GiB → small bucket → rtx-5080
	gpus := []ml.DeviceInfo{{
		Description: "Some Unknown GPU",
		TotalMemory: 16 << 30,
	}}
	p := SelectGpuProfile(gpus)
	if p == nil || p.ID != "rtx-5080" || p.Source != "bucket" {
		t.Fatalf("bucket select failed: %+v", p)
	}
}

func TestApplyGpuProfileToLaunch(t *testing.T) {
	profile := &GpuProfile{
		ID:         "rtx-5080",
		Source:     "match",
		NParallel:  2,
		BatchSize:  1024,
		UBatchSize: 256,
		CacheTypeK: "q8_0",
		CacheTypeV: "q8_0",
		FlashAttn:  true,
		DraftMax:   16,
		DraftPMin:  0.5,
	}
	t.Setenv("OLLAMA_KV_CACHE_TYPE", "")
	t.Setenv("OLLAMA_FLASH_ATTENTION", "")

	opts := api.DefaultOptions()
	launch := llamaServerLaunchConfig{
		opts:        opts,
		numParallel: 4, // fitted high → cap to profile 2
		kvCacheType: "f16",
		config:      LlamaServerConfig{EnableMTP: true},
	}
	ApplyGpuProfileToLaunch(&launch, profile)
	if launch.numParallel != 2 {
		t.Fatalf("np = %d, want 2", launch.numParallel)
	}
	if launch.opts.NumBatch != 1024 || launch.ubatchSize != 256 {
		t.Fatalf("batch/ub = %d/%d", launch.opts.NumBatch, launch.ubatchSize)
	}
	if launch.kvCacheType != "q8_0" {
		t.Fatalf("kv = %q, want q8_0", launch.kvCacheType)
	}
	if !launch.forceFlashAttn {
		t.Fatal("expected forceFlashAttn")
	}
	if launch.opts.DraftNumPredict != 16 || launch.draftPMin != 0.5 {
		t.Fatalf("MTP draft max/pmin = %d/%v", launch.opts.DraftNumPredict, launch.draftPMin)
	}
}

func TestApplyGpuProfileToLaunch_RespectsExplicitKV(t *testing.T) {
	profile := &GpuProfile{ID: "rtx-5080", CacheTypeK: "q8_0", BatchSize: 1024, NParallel: 2}
	t.Setenv("OLLAMA_KV_CACHE_TYPE", "q4_0")
	launch := llamaServerLaunchConfig{
		opts:        api.Options{Runner: api.Runner{KvCacheType: ""}},
		numParallel: 1,
		kvCacheType: "q4_0",
	}
	ApplyGpuProfileToLaunch(&launch, profile)
	if launch.kvCacheType != "q4_0" {
		t.Fatalf("explicit KV overwritten: %q", launch.kvCacheType)
	}
}

func TestApplyGpuProfileToLaunch_EmbeddingSkipped(t *testing.T) {
	profile := &GpuProfile{ID: "rtx-5080", NParallel: 2, BatchSize: 1024, CacheTypeK: "q8_0"}
	launch := llamaServerLaunchConfig{
		embedding:   true,
		numParallel: 1,
		kvCacheType: "f16",
		opts:        api.DefaultOptions(),
	}
	ApplyGpuProfileToLaunch(&launch, profile)
	if launch.kvCacheType != "f16" || launch.numParallel != 1 {
		t.Fatalf("embedding should skip profile: %+v", launch)
	}
}

func TestApplyGpuProfileToLaunch_DraftOnlyWhenSpec(t *testing.T) {
	profile := &GpuProfile{ID: "rtx-5080", DraftMax: 16, DraftPMin: 0.5, BatchSize: 1024, CacheTypeK: "q8_0"}
	t.Setenv("OLLAMA_KV_CACHE_TYPE", "")
	t.Setenv("OLLAMA_FLASH_ATTENTION", "")

	plain := llamaServerLaunchConfig{opts: api.DefaultOptions(), numParallel: 1, kvCacheType: "f16"}
	before := plain.opts.DraftNumPredict
	ApplyGpuProfileToLaunch(&plain, profile)
	if plain.opts.DraftNumPredict != before || plain.draftPMin != 0 {
		t.Fatalf("plain load should not take draft knobs: predict=%d→%d pmin=%v", before, plain.opts.DraftNumPredict, plain.draftPMin)
	}

	mtp := llamaServerLaunchConfig{
		opts:        api.DefaultOptions(),
		numParallel: 1,
		kvCacheType: "f16",
		config:      LlamaServerConfig{EnableMTP: true},
	}
	ApplyGpuProfileToLaunch(&mtp, profile)
	if mtp.opts.DraftNumPredict != 16 || mtp.draftPMin != 0.5 {
		t.Fatalf("MTP load should take draft knobs: predict=%d pmin=%v", mtp.opts.DraftNumPredict, mtp.draftPMin)
	}
}

func TestApplyGpuProfileToLaunch_DoesNotRaiseNP(t *testing.T) {
	profile := &GpuProfile{ID: "rtx-5080", NParallel: 2, BatchSize: 1024, CacheTypeK: "q8_0", FlashAttn: true}
	t.Setenv("OLLAMA_KV_CACHE_TYPE", "")
	t.Setenv("OLLAMA_FLASH_ATTENTION", "")
	launch := llamaServerLaunchConfig{
		opts:        api.DefaultOptions(),
		numParallel: 1, // VRAM-fitted — must not raise to 2
		kvCacheType: "f16",
	}
	ApplyGpuProfileToLaunch(&launch, profile)
	if launch.numParallel != 1 {
		t.Fatalf("must not raise np above fitted: got %d", launch.numParallel)
	}
	if launch.kvCacheType != "q8_0" {
		t.Fatalf("kv should still apply: %q", launch.kvCacheType)
	}
}

func TestAppendBatchArgsWithUBatch(t *testing.T) {
	opts := api.Options{Runner: api.Runner{NumBatch: 1024}}
	got := appendBatchArgsWithUBatch(nil, opts, false, 1, 256)
	want := []string{"-b", "1024", "-ub", "256"}
	if len(got) != len(want) {
		t.Fatalf("got %v want %v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("got %v want %v", got, want)
		}
	}
}

func TestLaunchFlashAttentionMode_ProfileForce(t *testing.T) {
	t.Setenv("OLLAMA_FLASH_ATTENTION", "")
	launch := llamaServerLaunchConfig{forceFlashAttn: true}
	if got := launchFlashAttentionMode(launch); got != ml.FlashAttentionEnabled {
		t.Fatalf("got %v, want enabled", got)
	}
}

func findRepoRoot(t *testing.T) string {
	t.Helper()
	wd, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	dir := wd
	for i := 0; i < 6; i++ {
		if _, err := os.Stat(filepath.Join(dir, "runtime", "configs", "gpu", "rtx-5080.json")); err == nil {
			return dir
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			break
		}
		dir = parent
	}
	t.Fatal("could not find repo root with runtime/configs/gpu/rtx-5080.json")
	return ""
}
