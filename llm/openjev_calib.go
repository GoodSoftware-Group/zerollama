package llm

// OpenJevCalibGate is the DG7/DG9 checklist for considering calibrated:true.
// WHY a struct (not auto-flip): holdout ranking alone is not a calibration study;
// operator_signoff must stay manual (see docs/diffusion-gemma-readout-design.md).
// Serve still requires ZEROLLAMA_OPENJEV_CALIBRATED (+ NOUL_CALIBRATED for noul).
type OpenJevCalibGate struct {
	ChoiceHoldoutN        int     `json:"choice_holdout_n"`
	ChoiceHoldoutAccuracy float64 `json:"choice_holdout_accuracy"`
	NoulHoldoutN          int     `json:"noul_holdout_n,omitempty"`
	NoulHoldoutAccuracy   float64 `json:"noul_holdout_accuracy,omitempty"`
	NoulBias              float64 `json:"noul_bias,omitempty"`
	FixtureN              int     `json:"fixture_n"`
	FixtureAccuracy       float64 `json:"fixture_accuracy"`
	MarkerRate            float64 `json:"marker_rate"`
	Temperature           float64 `json:"temperature"`
	OperatorSignoff       bool    `json:"operator_signoff"`
}

// OpenJevCalibGateResult is the evaluated gate (always Calibrated=false unless all pass + signoff).
type OpenJevCalibGateResult struct {
	GateReady    bool     `json:"gate_ready"`
	Calibrated   bool     `json:"calibrated"`
	FailedChecks []string `json:"failed_checks,omitempty"`
	Note         string   `json:"note"`
}

// DG7/DG9 thresholds — raise only with a larger labeled study; do not loosen to flip calibrated.
const (
	OpenJevCalibMinChoiceHoldoutN = 5
	OpenJevCalibMinHoldoutAcc     = 0.9
	OpenJevCalibMinNoulHoldoutN   = 2
	OpenJevCalibMinFixtureAcc     = 0.9
	OpenJevCalibMinMarkerRate     = 0.9
)

// EvalOpenJevCalibGate returns whether the lab gate is ready. Calibrated is true only
// when gate_ready and OperatorSignoff — callers must still refuse auto-flip in serve.
// Noul floors apply when NoulHoldoutN > 0 (DG9); omitted noul fields keep DG8 choice/score gate.
func EvalOpenJevCalibGate(g OpenJevCalibGate) OpenJevCalibGateResult {
	var fail []string
	if g.ChoiceHoldoutN < OpenJevCalibMinChoiceHoldoutN {
		fail = append(fail, "choice_holdout_n")
	}
	if g.ChoiceHoldoutAccuracy < OpenJevCalibMinHoldoutAcc {
		fail = append(fail, "choice_holdout_accuracy")
	}
	if g.NoulHoldoutN > 0 {
		if g.NoulHoldoutN < OpenJevCalibMinNoulHoldoutN {
			fail = append(fail, "noul_holdout_n")
		}
		if g.NoulHoldoutAccuracy < OpenJevCalibMinHoldoutAcc {
			fail = append(fail, "noul_holdout_accuracy")
		}
	}
	if g.FixtureN == 0 || g.FixtureAccuracy < OpenJevCalibMinFixtureAcc {
		fail = append(fail, "fixture_accuracy")
	}
	if g.MarkerRate < OpenJevCalibMinMarkerRate {
		fail = append(fail, "marker_rate")
	}
	ready := len(fail) == 0
	cal := ready && g.OperatorSignoff
	note := "DG9: gate_ready means choice+marker floors (+ noul when present); serve opt-in is ZEROLLAMA_OPENJEV_CALIBRATED (+ NOUL_CALIBRATED for noul)."
	if !ready {
		note = "DG7/DG9: gate not ready — " + note
	} else if !g.OperatorSignoff {
		note = "DG9: gate_ready; set CALIBRATED=1 (choice/score) and NOUL_CALIBRATED=1 (noul) for calibrated:true"
	}
	return OpenJevCalibGateResult{
		GateReady:    ready,
		Calibrated:   cal,
		FailedChecks: fail,
		Note:         note,
	}
}
