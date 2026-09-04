// Host-RAM / CPU admission for Wan video jobs.
//
// WHY: exclusive GPU stops chat from competing for VRAM, but Wan still runs as a
// host-RAM-heavy PyTorch child (mmgp shuttles + optional CPU VAE). Without a gate,
// a 16 GiB CT accepts TI2V, thrash-swaps through decode, pegs every core, and
// OOM-kills serve — the opposite of zerollama's job as coordinator.
package server

import (
	"fmt"
	"log/slog"
	"runtime"
	"strconv"
	"strings"

	"github.com/ollama/ollama/discover"
	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/format"
	"github.com/ollama/ollama/types/model"
)

type wanHostMem struct {
	TotalMemory uint64
	FreeMemory  uint64
}

// Overridable for tests.
var wanReadHostMem = func() (wanHostMem, error) {
	m, err := discover.GetCPUMem()
	if err != nil {
		return wanHostMem{}, err
	}
	return wanHostMem{TotalMemory: m.TotalMemory, FreeMemory: m.FreeMemory}, nil
}

const (
	// Reserve host RAM for Go serve + embed + OS so Wan cannot RLIMIT itself into killing us.
	wanDefaultHostReserveGiB = 2
	// mmgp profile 5 + CPU VAE peaks well above 16 GiB (T5 shuttle + DiT pages + decode).
	wanMMGPCPUVAEMinHostGiB = 24
	// mmgp with GPU VAE still needs host for async shuttles / T5; far less than CPU decode.
	wanMMGPGPUVAEMinHostGiB = 12
	// CPU VAE without mmgp (e.g. 1.3b T2V on 16g).
	wanCPUVAEMinHostGiB = 14
	wanDefaultOMPThreads = 2
)

// wanHostPlan is the coordinated host-side policy for one Wan job.
type wanHostPlan struct {
	VAECPU         string // "0" | "1"
	AllowGPUVAE    bool
	MinHostRAMGiB  int
	OMPThreads     int
	RlimitASGiB    int // 0 = unset
	HostReserveGiB int
}

func planWanHost(cfg model.VideoGenerationConfig) wanHostPlan {
	mmgp := wanMMGP(cfg) == "1"
	vaeCPU := wanVAECPUResolved(cfg, mmgp)
	plan := wanHostPlan{
		VAECPU:         vaeCPU,
		AllowGPUVAE:    vaeCPU == "0" && mmgp,
		MinHostRAMGiB:  wanMinHostRAMGiB(cfg, mmgp, vaeCPU == "1"),
		OMPThreads:     wanOMPThreads(),
		HostReserveGiB: wanHostReserveGiB(),
	}
	plan.RlimitASGiB = wanRlimitASGiB(plan.HostReserveGiB)
	return plan
}

// wanVAECPUResolved picks VAE placement. Explicit ZEROLLAMA_WAN_VAE_CPU wins.
// Under mmgp, default GPU VAE — CPU VAE + mmgp both thrash host RAM on ≤24 GiB boxes.
func wanVAECPUResolved(cfg model.VideoGenerationConfig, mmgp bool) string {
	if v := strings.TrimSpace(envconfig.Var("ZEROLLAMA_WAN_VAE_CPU")); v != "" {
		return v
	}
	if mmgp {
		return "0"
	}
	if cfg.VRAMTier == "16g" {
		return "1"
	}
	return "0"
}

func wanMinHostRAMGiB(cfg model.VideoGenerationConfig, mmgp, vaeCPU bool) int {
	computed := 8
	switch {
	case mmgp && vaeCPU:
		computed = wanMMGPCPUVAEMinHostGiB
	case mmgp:
		computed = wanMMGPGPUVAEMinHostGiB
	case vaeCPU:
		computed = wanCPUVAEMinHostGiB
	}
	if v := strings.TrimSpace(envconfig.Var("ZEROLLAMA_WAN_MIN_HOST_RAM_GIB")); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n > 0 {
			force := strings.TrimSpace(envconfig.Var("ZEROLLAMA_WAN_MIN_HOST_RAM_FORCE"))
			if force == "1" || strings.EqualFold(force, "true") {
				return n
			}
			// WHY raise-only: serve_gpu_example exports :-14; that must not undercut
			// mmgp+CPU VAE's 24 GiB floor and re-enable thrash-OOM on 16 GiB CTs.
			if n > computed {
				return n
			}
		}
	}
	return computed
}

func wanOMPThreads() int {
	if v := strings.TrimSpace(envconfig.Var("ZEROLLAMA_WAN_OMP_NUM_THREADS")); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n > 0 {
			return n
		}
	}
	n := runtime.NumCPU() / 4
	if n < wanDefaultOMPThreads {
		n = wanDefaultOMPThreads
	}
	if n > 4 {
		n = 4
	}
	return n
}

// admitLtxHostRAM rejects LTX jobs that cannot start without thrashing the CT.
// Floor defaults to Wan mmgp+GPU-VAE class (12 GiB); raise via ZEROLLAMA_LTX_MIN_HOST_RAM_GIB.
func admitLtxHostRAM(cfg model.VideoGenerationConfig) error {
	minGiB := ltxDefaultMinHostGiB
	if isLtx2BProfile(cfg.Profile) {
		minGiB = ltx2bMinHostGiB
	}
	if v := strings.TrimSpace(envconfig.Var("ZEROLLAMA_LTX_MIN_HOST_RAM_GIB")); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n > 0 {
			force := strings.TrimSpace(envconfig.Var("ZEROLLAMA_LTX_MIN_HOST_RAM_FORCE"))
			if force == "1" || strings.EqualFold(force, "true") || n > minGiB {
				minGiB = n
			}
		}
	}
	plan := wanHostPlan{
		VAECPU:         "0",
		AllowGPUVAE:    true,
		MinHostRAMGiB:  minGiB,
		OMPThreads:     wanOMPThreads(),
		HostReserveGiB: wanHostReserveGiB(),
	}
	plan.RlimitASGiB = wanRlimitASGiB(plan.HostReserveGiB)
	return admitWanHostRAM(cfg, plan)
}

func wanHostReserveGiB() int {
	if v := strings.TrimSpace(envconfig.Var("ZEROLLAMA_WAN_HOST_RESERVE_GIB")); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n >= 0 {
			return n
		}
	}
	return wanDefaultHostReserveGiB
}

func wanRlimitASGiB(reserveGiB int) int {
	// WHY default off: RLIMIT_AS caps *virtual* address space. CUDA maps GPU VRAM into
	// the process VA — a 14 GiB AS on a 16 GiB host makes cudaGetDeviceCount fail with
	// "out of memory" before any real alloc. Containment is host admission + OMP caps;
	// operators who want a hard AS cap set ZEROLLAMA_WAN_RLIMIT_AS_GIB explicitly
	// (use a large value, e.g. 96+, not host_RAM−reserve).
	if v := strings.TrimSpace(envconfig.Var("ZEROLLAMA_WAN_RLIMIT_AS_GIB")); v != "" {
		if v == "0" || strings.EqualFold(v, "off") {
			return 0
		}
		if n, err := strconv.Atoi(v); err == nil && n > 0 {
			return n
		}
	}
	_ = reserveGiB
	return 0
}

// admitWanHostRAM rejects configs that cannot run without thrashing the CT.
// Checks TotalMemory (box class) and FreeMemory (can start now).
func admitWanHostRAM(cfg model.VideoGenerationConfig, plan wanHostPlan) error {
	need := uint64(plan.MinHostRAMGiB) * format.GibiByte
	mem, err := wanReadHostMem()
	if err != nil {
		slog.Warn("wan host admission: meminfo unavailable; allowing submit", "error", err)
		return nil
	}
	total := mem.TotalMemory
	free := mem.FreeMemory
	if total > 0 && total < need {
		return fmt.Errorf(
			"host RAM too small for this Wan plan: need ≥%d GiB total (have %s); mmgp_cpu_vae needs ~%d GiB, mmgp+gpu_vae ~%d GiB — raise CT RAM, set ZEROLLAMA_WAN_VAE_CPU=0 with WAN_MMGP=1, or ZEROLLAMA_WAN_MIN_HOST_RAM_GIB to override",
			plan.MinHostRAMGiB, format.HumanBytes2(total), wanMMGPCPUVAEMinHostGiB, wanMMGPGPUVAEMinHostGiB,
		)
	}
	// Free must cover the working set; leave reserve for serve (already outside need when
	// need is based on profile — still require free ≥ need - small reclaim slack is wrong;
	// require free ≥ need so we don't start into swap thrash).
	if free > 0 && free < need {
		return fmt.Errorf(
			"insufficient free host RAM for Wan: need ≥%d GiB MemAvailable (have %s); unload other workloads or retry — zerollama refuses to start a job that would thrash the box",
			plan.MinHostRAMGiB, format.HumanBytes2(free),
		)
	}
	slog.Info("wan host admission ok",
		"min_gib", plan.MinHostRAMGiB,
		"total", format.HumanBytes2(total),
		"available", format.HumanBytes2(free),
		"vae_cpu", plan.VAECPU,
		"mmgp", wanMMGP(cfg),
		"omp_threads", plan.OMPThreads,
		"rlimit_as_gib", plan.RlimitASGiB,
	)
	return nil
}

func applyWanHostPlanEnv(env map[string]string, plan wanHostPlan) {
	env["WAN_VAE_CPU"] = plan.VAECPU
	if plan.AllowGPUVAE {
		env["WAN_ALLOW_GPU_VAE"] = "1"
	}
	omp := strconv.Itoa(plan.OMPThreads)
	env["WAN_OMP_NUM_THREADS"] = omp
	env["OMP_NUM_THREADS"] = omp
	env["MKL_NUM_THREADS"] = omp
	env["OPENBLAS_NUM_THREADS"] = omp
	env["TORCH_NUM_THREADS"] = omp
	env["NUMEXPR_NUM_THREADS"] = omp
	if plan.RlimitASGiB > 0 {
		env["WAN_RLIMIT_AS_GIB"] = strconv.Itoa(plan.RlimitASGiB)
	}
	env["WAN_HOST_RESERVE_GIB"] = strconv.Itoa(plan.HostReserveGiB)
}
