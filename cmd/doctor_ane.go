package cmd

import (
	"fmt"
	"os"
	"path/filepath"
	"runtime"

	"github.com/spf13/cobra"

	"github.com/ollama/ollama/discover"
)

func doctorCheckANE(repo string) doctorCheck {
	// Why warn-only: private ANE APIs are experimental; missing bridge must not block serve.
	if runtime.GOOS != "darwin" {
		return doctorCheck{
			Name:   "apple neural engine (experimental)",
			Status: "warn",
			Detail: "ANE probe runs on darwin only",
		}
	}

	aneRepo := discover.ANERepoPath()
	bridgeLib := filepath.Join(aneRepo, "bridge", "libane_bridge.dylib")
	if st, err := os.Stat(bridgeLib); err != nil || st.IsDir() {
		return doctorCheck{
			Name:    "apple neural engine (experimental)",
			Status:  "warn",
			Detail:  fmt.Sprintf("maderix/ane bridge missing at %s", bridgeLib),
			FixHint: "git clone https://github.com/maderix/ane ~/Sites/inference/ane && ./scripts/ane_probe_build.sh",
		}
	}

	bin := discover.FindANEProbeBin()
	if bin == "" {
		return doctorCheck{
			Name:    "apple neural engine (experimental)",
			Status:  "warn",
			Detail:  fmt.Sprintf("ane-probe not built (bridge ok @ %s)", aneRepo),
			FixHint: "./scripts/ane_probe_build.sh",
		}
	}

	res, err := discover.ProbeANE(nil)
	if err != nil {
		return doctorCheck{
			Name:    "apple neural engine (experimental)",
			Status:  "warn",
			Detail:  fmt.Sprintf("probe failed: %v", err),
			FixHint: "rebuild: ./scripts/ane_probe_build.sh; uses private ANE APIs — may break on macOS updates",
		}
	}

	benchDetail := doctorANEBenchDetail()
	handoffDetail := doctorANEHandoffDetail()
	hybridDetail := doctorANEHybridDetail()
	prefillDetail := discover.DoctorPrefillDetail(nil)
	modelDetail := doctorANEModelInventoryDetail()

	_ = repo
	return doctorCheck{
		Name:   "apple neural engine (experimental)",
		Status: "ok",
		Detail: fmt.Sprintf("smoke eval %.2f ms/op compile=%d; %s; %s; %s; %s; %s; bin=%s",
			res.EvalMS, res.CompileCount, benchDetail, handoffDetail, hybridDetail, prefillDetail, modelDetail, bin),
	}
}

func doctorANEHybridDetail() string {
	if !discover.ANEDraftLabEnabled() {
		return "hybrid lab: ZEROLLAMA_ANE_DRAFT off"
	}
	router, err := discover.ProbeANEDraftRouterSmoke(nil, "", 2, true)
	if err != nil {
		return fmt.Sprintf("draft router (ANE_DRAFT=1): %v", err)
	}
	return fmt.Sprintf("draft router %s avg eval %.3f ms map %.3f ms (%d steps)",
		router.Tag, router.AvgEvalMS, router.AvgMapFillMS, len(router.Steps))
}

func NewANEProbeCommand() *cobra.Command {
	return &cobra.Command{
		Use:    "ane-probe",
		Hidden: true,
		Short:  "Run Apple Neural Engine smoke probe (maderix/ane bridge)",
		RunE: func(cmd *cobra.Command, _ []string) error {
			return discover.RunANEProbe(cmd.Context(), os.Stdout)
		},
	}
}
