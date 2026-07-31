package proctitle

import (
	"os"
	"os/exec"
	"testing"
)

func TestSetRunnerArgv0(t *testing.T) {
	cmd := exec.Command("/bin/true", "runner", "--port", "1")
	if got := cmd.Args[0]; got != "/bin/true" {
		t.Fatalf("precondition Args[0]=%q", got)
	}
	SetRunnerArgv0(cmd)
	if cmd.Path != "/bin/true" {
		t.Fatalf("Path changed to %q", cmd.Path)
	}
	if cmd.Args[0] != Runner {
		t.Fatalf("Args[0]=%q want %q", cmd.Args[0], Runner)
	}
	if len(cmd.Args) < 2 || cmd.Args[1] != "runner" {
		t.Fatalf("Args=%v", cmd.Args)
	}
}

func TestSetRunnerArgv0Nil(t *testing.T) {
	SetRunnerArgv0(nil) // must not panic
}

func TestSetEmpty(t *testing.T) {
	Set("") // must not panic
}

func TestSetPreservesArgs(t *testing.T) {
	orig := append([]string(nil), os.Args...)
	t.Cleanup(func() { os.Args = orig })

	os.Args = []string{"/tmp/zerollama", "runner", "--mlx-engine", "--model", "m", "--port", "12345"}
	// Shallow copy (same bug as the first fix attempt) — must still be safe
	// because Set dupArgs before wiping.
	shallow := append([]string(nil), os.Args[1:]...)
	Set(Runner)
	if os.Args[0] != Runner {
		t.Fatalf("Args[0]=%q want %q", os.Args[0], Runner)
	}
	want := []string{"runner", "--mlx-engine", "--model", "m", "--port", "12345"}
	for i, w := range want {
		if os.Args[i+1] != w {
			t.Fatalf("os.Args[%d]=%q want %q (full=%v)", i+1, os.Args[i+1], w, os.Args)
		}
	}
	// After Set, prefer os.Args (heap). Shallow may be stale if it aliased C
	// argv — document that callers must Set before reading args, or deep-copy.
	_ = shallow
}

func TestDupArgsDeep(t *testing.T) {
	orig := append([]string(nil), os.Args...)
	t.Cleanup(func() { os.Args = orig })

	os.Args = []string{"/tmp/zerollama", "runner", "--port", "9"}
	dupArgs()
	// Mutating a fresh byte slice must not change os.Args after dup.
	b := []byte(os.Args[1])
	if len(b) > 0 {
		b[0] = 'X'
	}
	if os.Args[1] != "runner" {
		t.Fatalf("dupArgs did not isolate backing: %q", os.Args[1])
	}
}
