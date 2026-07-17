//go:build !windows

package server

import "golang.org/x/sys/unix"

// exitProcess terminates the whole process immediately without Go runtime.exit
// or libc atexit handlers.
//
// WHY not syscall.SYS_EXIT: on Linux that kills only the calling thread. The
// second Ctrl+C handler runs in a goroutine — SYS_EXIT left the process alive
// (log said "forced exit" then graceful teardown continued) and further SIGINTs
// had no reader. unix.Exit uses exit_group(2).
// Embedded CPython/torch atexit can SIGSEGV under os.Exit after training shutdown.
func exitProcess(code int) {
	unix.Exit(code)
}
