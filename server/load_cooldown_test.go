package server

import (
	"context"
	"errors"
	"testing"
	"time"

)

func TestLoadCooldownBackoff(t *testing.T) {
	t.Setenv("ZEROLLAMA_LOAD_COOLDOWN", "100ms")
	t.Setenv("ZEROLLAMA_LOAD_COOLDOWN_MAX", "1s")
	s := InitScheduler(t.Context())
	key := "m1"
	boom := errors.New("weights corrupt")
	s.noteLoadFailure(key, boom)
	err := s.cooldownErr(key)
	var ce *LoadCooldownError
	if !errors.As(err, &ce) {
		t.Fatalf("want LoadCooldownError, got %v", err)
	}
	if !errors.Is(err, ErrLoadCooldown) {
		t.Fatal("errors.Is ErrLoadCooldown")
	}
	if ce.Failures != 1 {
		t.Fatalf("failures=%d", ce.Failures)
	}
	s.noteLoadFailure(key, boom)
	err = s.cooldownErr(key)
	if !errors.As(err, &ce) {
		t.Fatal("second failure")
	}
	if ce.Failures != 2 {
		t.Fatalf("failures=%d", ce.Failures)
	}
	s.clearLoadCooldown(key)
	if s.cooldownErr(key) != nil {
		t.Fatal("cleared cooldown should allow load")
	}
}

func TestLoadCooldownSkipsCancel(t *testing.T) {
	t.Setenv("ZEROLLAMA_LOAD_COOLDOWN", "10s")
	s := InitScheduler(t.Context())
	s.noteLoadFailure("k", context.Canceled)
	if s.cooldownErr("k") != nil {
		t.Fatal("canceled load must not cooldown")
	}
}

func TestLoadCooldownDisabled(t *testing.T) {
	t.Setenv("ZEROLLAMA_LOAD_COOLDOWN", "0")
	s := InitScheduler(t.Context())
	s.noteLoadFailure("k", errors.New("fail"))
	if s.cooldownErr("k") != nil {
		t.Fatal("disabled cooldown")
	}
}

func TestLoadCooldownExpires(t *testing.T) {
	t.Setenv("ZEROLLAMA_LOAD_COOLDOWN", "20ms")
	t.Setenv("ZEROLLAMA_LOAD_COOLDOWN_MAX", "20ms")
	s := InitScheduler(t.Context())
	s.noteLoadFailure("k", errors.New("fail"))
	if s.cooldownErr("k") == nil {
		t.Fatal("expected cooldown")
	}
	time.Sleep(40 * time.Millisecond)
	if s.cooldownErr("k") != nil {
		t.Fatal("cooldown should expire")
	}
}
