package server

import (
	"strings"
	"testing"

	"github.com/ollama/ollama/format"
	"github.com/ollama/ollama/types/model"
)

func TestPlanWanHostMMGPUsesGPUVAE(t *testing.T) {
	t.Setenv("ZEROLLAMA_WAN_VAE_CPU", "")
	t.Setenv("ZEROLLAMA_WAN_MMGP", "")
	t.Setenv("ZEROLLAMA_WAN_MIN_HOST_RAM_GIB", "")
	t.Setenv("ZEROLLAMA_WAN_MIN_HOST_RAM_FORCE", "")
	t.Setenv("ZEROLLAMA_WAN_OMP_NUM_THREADS", "2")
	t.Setenv("ZEROLLAMA_WAN_RLIMIT_AS_GIB", "0")

	cfg := model.VideoGenerationConfig{VRAMTier: "16g", Profile: wanProfile22TI2V5B}
	plan := planWanHost(cfg)
	if plan.VAECPU != "0" {
		t.Fatalf("VAECPU=%q want 0 (GPU under mmgp)", plan.VAECPU)
	}
	if !plan.AllowGPUVAE {
		t.Fatal("AllowGPUVAE want true")
	}
	if plan.MinHostRAMGiB != wanMMGPGPUVAEMinHostGiB {
		t.Fatalf("MinHostRAMGiB=%d want %d", plan.MinHostRAMGiB, wanMMGPGPUVAEMinHostGiB)
	}
	if plan.OMPThreads != 2 {
		t.Fatalf("OMPThreads=%d want 2", plan.OMPThreads)
	}
}

func TestWanMinHostRAMRaiseOnly(t *testing.T) {
	t.Setenv("ZEROLLAMA_WAN_MIN_HOST_RAM_FORCE", "")
	t.Setenv("ZEROLLAMA_WAN_MIN_HOST_RAM_GIB", "14") // serve default — must not undercut 24
	got := wanMinHostRAMGiB(model.VideoGenerationConfig{}, true, true)
	if got != wanMMGPCPUVAEMinHostGiB {
		t.Fatalf("raise-only: got %d want %d", got, wanMMGPCPUVAEMinHostGiB)
	}
	t.Setenv("ZEROLLAMA_WAN_MIN_HOST_RAM_GIB", "30")
	got = wanMinHostRAMGiB(model.VideoGenerationConfig{}, true, true)
	if got != 30 {
		t.Fatalf("raise: got %d want 30", got)
	}
	t.Setenv("ZEROLLAMA_WAN_MIN_HOST_RAM_FORCE", "1")
	t.Setenv("ZEROLLAMA_WAN_MIN_HOST_RAM_GIB", "10")
	got = wanMinHostRAMGiB(model.VideoGenerationConfig{}, true, true)
	if got != 10 {
		t.Fatalf("force: got %d want 10", got)
	}
}

func TestAdmitWanHostRAMRejectsSmallTotal(t *testing.T) {
	prev := wanReadHostMem
	t.Cleanup(func() { wanReadHostMem = prev })
	wanReadHostMem = func() (wanHostMem, error) {
		return wanHostMem{
			TotalMemory: 16 * format.GibiByte,
			FreeMemory:  14 * format.GibiByte,
		}, nil
	}
	plan := wanHostPlan{MinHostRAMGiB: wanMMGPCPUVAEMinHostGiB, VAECPU: "1"}
	err := admitWanHostRAM(model.VideoGenerationConfig{VRAMTier: "16g", Profile: wanProfile22TI2V5B}, plan)
	if err == nil || !strings.Contains(err.Error(), "too small") {
		t.Fatalf("want total too small error, got %v", err)
	}
}

func TestAdmitWanHostRAMOKForMMGPGPUVAE(t *testing.T) {
	prev := wanReadHostMem
	t.Cleanup(func() { wanReadHostMem = prev })
	wanReadHostMem = func() (wanHostMem, error) {
		return wanHostMem{
			TotalMemory: 16 * format.GibiByte,
			FreeMemory:  14 * format.GibiByte,
		}, nil
	}
	cfg := model.VideoGenerationConfig{VRAMTier: "16g", Profile: wanProfile22TI2V5B}
	plan := planWanHost(cfg)
	if err := admitWanHostRAM(cfg, plan); err != nil {
		t.Fatal(err)
	}
}

func TestAdmitLtxHostRAMOK(t *testing.T) {
	prev := wanReadHostMem
	t.Cleanup(func() { wanReadHostMem = prev })
	t.Setenv("ZEROLLAMA_LTX_MIN_HOST_RAM_GIB", "")
	t.Setenv("ZEROLLAMA_LTX_MIN_HOST_RAM_FORCE", "")
	t.Setenv("ZEROLLAMA_WAN_OMP_NUM_THREADS", "2")
	t.Setenv("ZEROLLAMA_WAN_RLIMIT_AS_GIB", "0")
	wanReadHostMem = func() (wanHostMem, error) {
		return wanHostMem{
			TotalMemory: 24 * format.GibiByte,
			FreeMemory:  20 * format.GibiByte,
		}, nil
	}
	cfg := model.VideoGenerationConfig{VRAMTier: "16g", Profile: ltxProfile13BDistill}
	if err := admitLtxHostRAM(cfg); err != nil {
		t.Fatal(err)
	}
}

func TestAdmitLtxHostRAMRejectsTinyBox(t *testing.T) {
	prev := wanReadHostMem
	t.Cleanup(func() { wanReadHostMem = prev })
	t.Setenv("ZEROLLAMA_LTX_MIN_HOST_RAM_GIB", "")
	wanReadHostMem = func() (wanHostMem, error) {
		return wanHostMem{
			TotalMemory: 8 * format.GibiByte,
			FreeMemory:  6 * format.GibiByte,
		}, nil
	}
	err := admitLtxHostRAM(model.VideoGenerationConfig{VRAMTier: "16g", Profile: ltxProfile13BDistill})
	if err == nil || !strings.Contains(err.Error(), "too small") {
		t.Fatalf("want too small, got %v", err)
	}
}
