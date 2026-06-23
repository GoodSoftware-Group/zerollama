//go:build !windows

package server

import "syscall"

// exitProcess terminates immediately without Go runtime.exit or libc atexit handlers.
// Embedded CPython/torch registers atexit hooks; os.Exit can SIGSEGV after training shutdown.
func exitProcess(code int) {
	syscall.RawSyscall(syscall.SYS_EXIT, uintptr(code&0xff), 0, 0)
}
