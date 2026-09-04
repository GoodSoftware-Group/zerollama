package cmd

import (
	"runtime"
	"strings"

	"github.com/ollama/ollama/x/mlxrunner"
)

func doctorCheckMLXKnobs() doctorCheck {
	const name = "mlx knobs (tune sheet)"
	knobs := mlxrunner.KnobSnapshot()
	detail := mlxrunner.FormatKnobs(knobs)
	status := "ok"
	fix := ""
	for _, k := range knobs {
		if k.Env == "ZEROLLAMA_MLX_PLD" && k.Value == "off" {
			status = "warn"
			fix = "unset ZEROLLAMA_MLX_PLD unless this shell is an AR bench vs mlx-serve"
			break
		}
	}
	return doctorCheck{Name: name, Status: status, Detail: detail, FixHint: fix}
}

func doctorCheckMLXLastRun() doctorCheck {
	const name = "mlx last decode"
	detail, warn := mlxrunner.LastRunTuneReport()
	status := "ok"
	fix := ""
	if warn {
		status = "warn"
		fix = "follow the hint; one parked novel chat is normal — warn only if many recent runs park"
	}
	if runtime.GOOS != "darwin" && strings.Contains(detail, "no last MLX decode") {
		return doctorCheck{Name: name, Status: "ok", Detail: "skipped (no MLX last-run on this host)"}
	}
	return doctorCheck{Name: name, Status: status, Detail: detail, FixHint: fix}
}

func doctorCheckMLXRoundCost() doctorCheck {
	const name = "mlx round-cost tables"
	detail, warn := mlxrunner.RoundCostTuneReport()
	status := "ok"
	fix := ""
	if warn {
		status = "warn"
		fix = "run an echo or code prompt so depth can leave 0, or unset ZEROLLAMA_MLX_PLD=off"
	}
	if runtime.GOOS != "darwin" && strings.Contains(detail, "no mlx-round-cost") {
		return doctorCheck{Name: name, Status: "ok", Detail: "skipped (no MLX tables on this host)"}
	}
	return doctorCheck{Name: name, Status: status, Detail: detail, FixHint: fix}
}
