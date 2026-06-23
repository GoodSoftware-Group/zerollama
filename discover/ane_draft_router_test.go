package discover

import "testing"

func TestANEDraftRouterNotStarted(t *testing.T) {
	r := NewANEDraftRouter()
	if r.Ready() {
		t.Fatal("expected not ready")
	}
	_, err := r.DraftStep(t.Context(), 0.01)
	if err == nil {
		t.Fatal("expected error when not started")
	}
}

func TestRouterSmokeStepsDefault(t *testing.T) {
	if defaultRouterSteps(true) != 3 {
		t.Fatal("quick steps")
	}
	if defaultRouterSteps(false) != 5 {
		t.Fatal("full steps")
	}
}
