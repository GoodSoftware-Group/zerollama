package server

import (
	"errors"
	"log/slog"
	"net/http"
)

// showInfoError tags which GetModelInfo step failed so /api/show logs are actionable
// when clients (Hermes model picker, context probes) hit 500s.
type showInfoError struct {
	stage string
	err   error
}

func (e *showInfoError) Error() string {
	if e == nil || e.err == nil {
		return ""
	}
	if e.stage == "" {
		return e.err.Error()
	}
	return e.stage + ": " + e.err.Error()
}

func (e *showInfoError) Unwrap() error {
	if e == nil {
		return nil
	}
	return e.err
}

func showErr(stage string, err error) error {
	if err == nil {
		return nil
	}
	if stage == "" {
		return err
	}
	var tagged *showInfoError
	if errors.As(err, &tagged) {
		return err
	}
	return &showInfoError{stage: stage, err: err}
}

func showErrorStage(err error) string {
	var tagged *showInfoError
	if errors.As(err, &tagged) && tagged.stage != "" {
		return tagged.stage
	}
	return ""
}

func logShowHandlerOutcome(model, userAgent string, status int, err error) {
	if status == http.StatusOK {
		return
	}
	stage := showErrorStage(err)
	attrs := []any{
		"route", "/api/show",
		"model", model,
		"status", status,
	}
	if stage != "" {
		attrs = append(attrs, "show_stage", stage)
	}
	if userAgent != "" {
		attrs = append(attrs, "user_agent", userAgent)
	}
	if err != nil {
		attrs = append(attrs, "error", err.Error())
	}

	switch status {
	case http.StatusInternalServerError:
		slog.Error("api/show failed", attrs...)
	case http.StatusNotFound:
		// Hermes and model pickers probe many names; keep 404 at debug.
		slog.Debug("api/show model not found", attrs...)
	case http.StatusBadRequest:
		slog.Debug("api/show bad request", attrs...)
	default:
		slog.Warn("api/show rejected", attrs...)
	}
}
