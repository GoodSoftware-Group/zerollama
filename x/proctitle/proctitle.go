// Package proctitle sets the process title shown in ps / Activity Monitor.
//
// Serve and runner share one binary; spawn sites rewrite argv0 and entry hooks
// call Set so operators can tell zerollama-serve from zerollama-runner.
package proctitle

import (
	"os"
	"os/exec"
)

// Canonical titles for the shared zerollama binary.
const (
	Serve  = "zerollama-serve"
	Runner = "zerollama-runner"
)

// Set updates os.Args[0] and the OS process title when supported.
//
// WHY copy os.Args first: on Darwin (and some Unix), Go's os.Args strings alias
// the C argv block. setproctitle memsets that block for ps — without a copy,
// runner flags (--mlx-engine, --port, …) become empty and the child falls back
// to llamarunner on :8080 while the parent waits on the real ephemeral port.
func Set(name string) {
	if name == "" {
		return
	}
	dupArgs()
	os.Args[0] = name
	setOS(name)
}

// dupArgs forces each os.Args element onto the Go heap so later C argv
// mutation cannot clobber flag strings.
func dupArgs() {
	for i, a := range os.Args {
		os.Args[i] = string(append([]byte(nil), a...))
	}
}

// SetRunnerArgv0 sets cmd.Args[0] to Runner while leaving cmd.Path as the real
// executable. Call after exec.Command(exe, …) so children show as zerollama-runner.
func SetRunnerArgv0(cmd *exec.Cmd) {
	if cmd == nil || len(cmd.Args) == 0 {
		return
	}
	cmd.Args[0] = Runner
}
