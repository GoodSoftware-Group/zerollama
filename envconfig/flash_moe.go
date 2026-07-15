package envconfig

import (
	"strconv"
	"strings"
)

// Flash-MoE env helpers (anemll-flash-llama.cpp).
// Why env + manifest: global enable for serve, per-model sidecar via Modelfile options.
// See docs/flash-moe.md.

// FlashMoEEnabled is true when ZEROLLAMA_FLASH_MOE=1 (use anemll-flash-llama.cpp llama-server + sidecar flags).
func FlashMoEEnabled() bool {
	return Bool("ZEROLLAMA_FLASH_MOE")()
}

// FlashMoESidecar returns the routed-expert sidecar directory from ZEROLLAMA_FLASH_MOE_SIDECAR.
func FlashMoESidecar() string {
	return strings.TrimSpace(Var("ZEROLLAMA_FLASH_MOE_SIDECAR"))
}

// FlashMoEMode returns Flash-MoE runtime mode (default slot-bank when enabled).
func FlashMoEMode() string {
	if v := strings.TrimSpace(Var("ZEROLLAMA_FLASH_MOE_MODE")); v != "" {
		return v
	}
	return "slot-bank"
}

// FlashMoESlotBank returns slot-bank capacity per layer (0 = omit flag, llama-server default).
func FlashMoESlotBank() int {
	return intEnv("ZEROLLAMA_FLASH_MOE_SLOT_BANK", 0)
}

// FlashMoETopK returns routed expert top-k override (0 = model default).
func FlashMoETopK() int {
	return intEnv("ZEROLLAMA_FLASH_MOE_TOPK", 0)
}

// FlashMoEPrefetchTemporal is true when one-step temporal prefetch is enabled.
func FlashMoEPrefetchTemporal() bool {
	return Bool("ZEROLLAMA_FLASH_MOE_PREFETCH")()
}

// FlashMoEAutoExtract is true when `pull` should auto-extract a Flash-MoE
// sidecar for newly pulled MoE GGUF tags that don't have one yet
// (ZEROLLAMA_FLASH_MOE_AUTO_EXTRACT=1). Opt-in — why: extraction reads the
// full GGUF and can take minutes on 100GB+ MoE models; pull must not
// silently balloon in time for operators who never asked for slot-bank
// streaming. See docs/flash-moe.md.
func FlashMoEAutoExtract() bool {
	return Bool("ZEROLLAMA_FLASH_MOE_AUTO_EXTRACT")()
}

// FlashMoELlamaServerBin returns an override path for the Flash-MoE llama-server binary.
func FlashMoELlamaServerBin() string {
	return strings.TrimSpace(Var("ZEROLLAMA_FLASH_MOE_LLAMA_SERVER_BIN"))
}

// FlashMoERepo returns the anemll-flash-llama.cpp checkout used by build scripts.
func FlashMoERepo() string {
	return localRepoPath("FLASH_MOE_REPO", "anemll-flash-llama.cpp")
}

func intEnv(key string, def int) int {
	v := strings.TrimSpace(Var(key))
	if v == "" {
		return def
	}
	n, err := strconv.Atoi(v)
	if err != nil {
		return def
	}
	return n
}
