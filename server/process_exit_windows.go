//go:build windows

package server

import "os"

func exitProcess(code int) {
	os.Exit(code)
}
