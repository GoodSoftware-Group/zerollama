package llm

import (
	"testing"

	"github.com/stretchr/testify/require"
)

func TestEvalOpenJevCalibGateNotReady(t *testing.T) {
	r := EvalOpenJevCalibGate(OpenJevCalibGate{
		ChoiceHoldoutN:        3,
		ChoiceHoldoutAccuracy: 1.0,
		FixtureN:              10,
		FixtureAccuracy:       1.0,
		MarkerRate:            1.0,
		Temperature:           0.5,
		OperatorSignoff:       false,
	})
	require.False(t, r.GateReady)
	require.False(t, r.Calibrated)
	require.Contains(t, r.FailedChecks, "choice_holdout_n")
}

func TestEvalOpenJevCalibGateReadyNeedsSignoff(t *testing.T) {
	r := EvalOpenJevCalibGate(OpenJevCalibGate{
		ChoiceHoldoutN:        5,
		ChoiceHoldoutAccuracy: 1.0,
		FixtureN:              18,
		FixtureAccuracy:       1.0,
		MarkerRate:            1.0,
		Temperature:           0.5,
		OperatorSignoff:       false,
	})
	require.True(t, r.GateReady)
	require.False(t, r.Calibrated)
}

func TestEvalOpenJevCalibGateSignoff(t *testing.T) {
	r := EvalOpenJevCalibGate(OpenJevCalibGate{
		ChoiceHoldoutN:        5,
		ChoiceHoldoutAccuracy: 1.0,
		FixtureN:              18,
		FixtureAccuracy:       1.0,
		MarkerRate:            1.0,
		Temperature:           0.5,
		OperatorSignoff:       true,
	})
	require.True(t, r.GateReady)
	require.True(t, r.Calibrated)
}

func TestEvalOpenJevCalibGateNoulHoldout(t *testing.T) {
	bad := EvalOpenJevCalibGate(OpenJevCalibGate{
		ChoiceHoldoutN:        6,
		ChoiceHoldoutAccuracy: 1.0,
		NoulHoldoutN:          2,
		NoulHoldoutAccuracy:   0.5,
		FixtureN:              20,
		FixtureAccuracy:       1.0,
		MarkerRate:            1.0,
		Temperature:           0.5,
	})
	require.False(t, bad.GateReady)
	require.Contains(t, bad.FailedChecks, "noul_holdout_accuracy")

	ok := EvalOpenJevCalibGate(OpenJevCalibGate{
		ChoiceHoldoutN:        6,
		ChoiceHoldoutAccuracy: 1.0,
		NoulHoldoutN:          2,
		NoulHoldoutAccuracy:   1.0,
		NoulBias:              0,
		FixtureN:              20,
		FixtureAccuracy:       1.0,
		MarkerRate:            1.0,
		Temperature:           0.5,
	})
	require.True(t, ok.GateReady)
	require.False(t, ok.Calibrated)
}
