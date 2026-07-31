//go:build linux

package proctitle

import (
	"unsafe"

	"golang.org/x/sys/unix"
)

func setOS(name string) {
	// PR_SET_NAME is capped at 15 bytes (16 with NUL). Prefer full argv0 via
	// spawn Args[0]; this is a short kernel comm for top/htop.
	var buf [16]byte
	n := copy(buf[:15], name)
	buf[n] = 0
	_ = unix.Prctl(unix.PR_SET_NAME, uintptr(unsafe.Pointer(&buf[0])), 0, 0, 0)
}
