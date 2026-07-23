package cmd

import (
	"net"
	"runtime"
	"strings"
	"time"
)

func doctorCheckUMABroker() doctorCheck {
	const name = "uma broker"
	if runtime.GOOS != "darwin" {
		return doctorCheck{Name: name, Status: "ok", Detail: "darwin-only"}
	}
	sock := "/tmp/uma_daemon.sock"
	c, err := net.DialTimeout("unix", sock, 500*time.Millisecond)
	if err != nil {
		return doctorCheck{
			Name:    name,
			Status:  "warn",
			Detail:  "not running (" + sock + ")",
			FixHint: "make -C ../bmtl/hardware_lab/lanes/m4/uma_toolkit uma-daemon-install (or open UMAStatus.app); default ZEROLLAMA_UMA_SCHED=auto gates mlxrunner when broker is up",
		}
	}
	defer c.Close()
	_ = c.SetDeadline(time.Now().Add(time.Second))
	if _, err := c.Write([]byte("HELP\n")); err != nil {
		return doctorCheck{Name: name, Status: "warn", Detail: "write failed: " + err.Error()}
	}
	buf := make([]byte, 1024)
	n, err := c.Read(buf)
	if err != nil {
		return doctorCheck{Name: name, Status: "warn", Detail: "read failed: " + err.Error()}
	}
	reply := string(buf[:n])
	if !strings.Contains(reply, "HOLD_GPU") {
		return doctorCheck{
			Name:    name,
			Status:  "warn",
			Detail:  "broker up but lacks HOLD_GPU — upgrade uma_daemon",
			FixHint: "rebuild UMAStatus.app from bmtl uma_toolkit",
		}
	}
	return doctorCheck{
		Name:   name,
		Status: "ok",
		Detail: sock + " HOLD_GPU ready",
	}
}
