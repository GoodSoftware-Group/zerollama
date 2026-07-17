package cmd

import (
	"fmt"
	"os"
	"path/filepath"
	"runtime"

	"github.com/ollama/ollama/discover"
	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/llm"
)

func doctorCheckFlashMoE(_ string) doctorCheck {
	// Why report binary when disabled: operators should see "binary ready" before setting env.
	name := "flash-moe llama-server (experimental)"
	if runtime.GOOS != "darwin" {
		return doctorCheck{Name: name, Status: "warn", Detail: "Flash-MoE build is darwin-only today"}
	}

	repo := envconfig.FlashMoERepo()
	repoOK := pathExists(filepath.Join(repo, "CMakeLists.txt"))

	bin, binErr := llm.FindFlashMoELlamaServer()
	binOK := binErr == nil

	enabled := envconfig.FlashMoEEnabled()
	sidecar := envconfig.FlashMoESidecar()

	if !enabled && sidecar == "" {
		detail := "not enabled — set ZEROLLAMA_FLASH_MOE=1 + ZEROLLAMA_FLASH_MOE_SIDECAR for RAM-busting MoE"
		if inv, err := discover.ListFlashMoEInventory(); err == nil && len(inv) > 0 {
			needSidecar := 0
			for _, e := range inv {
				if !e.SidecarReady {
					needSidecar++
				}
			}
			detail += fmt.Sprintf("; found %d local MoE tag(s) — run ./zerollama flash-moe-resolve --list", len(inv))
			if needSidecar > 0 && !envconfig.FlashMoEAutoExtract() {
				detail += fmt.Sprintf(" (%d without a sidecar — set ZEROLLAMA_FLASH_MOE_AUTO_EXTRACT=1 to extract on next pull)", needSidecar)
			}
		}
		switch {
		case binOK:
			detail = fmt.Sprintf("not enabled (binary ready @ %s)", bin)
		case repoOK:
			detail += "; run ./scripts/build/build_flash_moe_llama_server.sh"
		default:
			detail += fmt.Sprintf("; repo missing at %s", repo)
		}
		fixHint := "export ZEROLLAMA_FLASH_MOE=1 ZEROLLAMA_FLASH_MOE_SIDECAR=/path/to/sidecar"
		if !repoOK {
			fixHint = "git clone --branch Server-Flash-Moe --depth 1 https://github.com/Anemll/anemll-flash-llama.cpp.git " + repo
		} else if !binOK {
			fixHint = "./scripts/build/build_flash_moe_llama_server.sh"
		}
		return doctorCheck{Name: name, Status: "warn", Detail: detail, FixHint: fixHint}
	}

	if !repoOK {
		return doctorCheck{
			Name:    name,
			Status:  "warn",
			Detail:  fmt.Sprintf("anemll-flash-llama.cpp missing at %s", repo),
			FixHint: "git clone --branch Server-Flash-Moe --depth 1 https://github.com/Anemll/anemll-flash-llama.cpp.git " + repo,
		}
	}

	if !binOK {
		return doctorCheck{
			Name:    name,
			Status:  "warn",
			Detail:  "flash-moe llama-server not built",
			FixHint: "./scripts/build/build_flash_moe_llama_server.sh",
		}
	}

	if sidecar == "" {
		return doctorCheck{
			Name:    name,
			Status:  "warn",
			Detail:  fmt.Sprintf("ZEROLLAMA_FLASH_MOE=1 but sidecar unset; binary ok @ %s", bin),
			FixHint: "export ZEROLLAMA_FLASH_MOE_SIDECAR=/path/to/extracted/sidecar",
		}
	}
	if st, err := os.Stat(sidecar); err != nil || !st.IsDir() {
		return doctorCheck{
			Name:    name,
			Status:  "warn",
			Detail:  fmt.Sprintf("sidecar missing: %s", sidecar),
			FixHint: "see docs/flash-moe.md — flashmoe_sidecar.py extract",
		}
	}

	return doctorCheck{
		Name:   name,
		Status: "ok",
		Detail: fmt.Sprintf("binary=%s sidecar=%s mode=%s", bin, sidecar, envconfig.FlashMoEMode()),
	}
}

func pathExists(path string) bool {
	st, err := os.Stat(path)
	return err == nil && !st.IsDir()
}
