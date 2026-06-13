//go:build windows

// Platform-specific free-space lookup for LM Studio import disk checks.
// Why a separate file: Unix uses syscall.Statfs; Windows uses GetDiskFreeSpaceExW.
package lmstudio

import (
	"fmt"
	"syscall"
	"unsafe"
)

func freeBytesOS(path string) (int64, error) {
	kernel32 := syscall.NewLazyDLL("kernel32.dll")
	getDiskFreeEx := kernel32.NewProc("GetDiskFreeSpaceExW")

	pathPtr, err := syscall.UTF16PtrFromString(path)
	if err != nil {
		return 0, fmt.Errorf("utf16: %w", err)
	}

	var avail, total, free uint64
	r1, _, e := getDiskFreeEx.Call(
		uintptr(unsafe.Pointer(pathPtr)),
		uintptr(unsafe.Pointer(&avail)),
		uintptr(unsafe.Pointer(&total)),
		uintptr(unsafe.Pointer(&free)),
	)
	if r1 == 0 {
		return 0, fmt.Errorf("GetDiskFreeSpaceEx %s: %w", path, e)
	}
	return int64(avail), nil
}
