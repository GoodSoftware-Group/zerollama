package server

import (
	"errors"
	"fmt"
	"time"

	"github.com/ollama/ollama/envconfig"
)

// ErrTrainingOutsideWindow is returned when submit is outside ZEROLLAMA_TRAINING_ALLOWED_WINDOW.
var ErrTrainingOutsideWindow = errors.New("training submit outside allowed window")

// ErrTrainingWindowMisconfigured is returned when ZEROLLAMA_TRAINING_ALLOWED_WINDOW is set but invalid.
var ErrTrainingWindowMisconfigured = errors.New("training allowed window misconfigured")

func checkTrainingAllowedWindow() error {
	if envconfig.TrainingAllowedWindowMisconfigured() {
		return fmt.Errorf(
			"%w: fix ZEROLLAMA_TRAINING_ALLOWED_WINDOW (expected HH:MM-HH:MM, e.g. 22:00-06:00); got %q",
			ErrTrainingWindowMisconfigured,
			envconfig.Var("ZEROLLAMA_TRAINING_ALLOWED_WINDOW"),
		)
	}
	if !envconfig.TrainingAllowedWindowEnabled() {
		return nil
	}
	if envconfig.WithinTrainingAllowedWindow(time.Now()) {
		return nil
	}
	label := envconfig.TrainingAllowedWindowLabel()
	return fmt.Errorf(
		"%w: allowed %s; use priority high to bypass or wait for the window",
		ErrTrainingOutsideWindow,
		label,
	)
}
