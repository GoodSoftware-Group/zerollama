package cmd

import (
	"net"
	"runtime"
	"strings"
	"time"
)

func umaBrokerTransact(sock, req string) (string, error) {
	c, err := net.DialTimeout("unix", sock, 500*time.Millisecond)
	if err != nil {
		return "", err
	}
	defer c.Close()
	_ = c.SetDeadline(time.Now().Add(time.Second))
	if _, err := c.Write([]byte(req)); err != nil {
		return "", err
	}
	buf := make([]byte, 2048)
	n, err := c.Read(buf)
	if err != nil {
		return "", err
	}
	return string(buf[:n]), nil
}

func doctorCheckUMABroker() doctorCheck {
	const name = "uma broker"
	if runtime.GOOS != "darwin" {
		return doctorCheck{Name: name, Status: "ok", Detail: "darwin-only"}
	}
	sock := "/tmp/uma_daemon.sock"
	info, err := umaBrokerTransact(sock, "INFO\n")
	if err != nil {
		return doctorCheck{
			Name:    name,
			Status:  "warn",
			Detail:  "not running (" + sock + ")",
			FixHint: "make -C ../bmtl/hardware_lab/lanes/m4/uma_toolkit uma-daemon-install (or open UMAStatus.app); default ZEROLLAMA_UMA_SCHED=auto gates when broker is up; set off to disable",
		}
	}
	reply := info
	if !strings.Contains(reply, "HOLD_GPU") {
		if help, herr := umaBrokerTransact(sock, "HELP\n"); herr == nil {
			reply = help
		}
	}
	if !strings.Contains(reply, "HOLD_GPU") {
		return doctorCheck{
			Name:    name,
			Status:  "warn",
			Detail:  "broker up but lacks HOLD_GPU — upgrade uma_daemon",
			FixHint: "rebuild UMAStatus.app from bmtl uma_toolkit",
		}
	}
	if !strings.Contains(reply, "HOLD_ANE") {
		return doctorCheck{
			Name:    name,
			Status:  "warn",
			Detail:  sock + " HOLD_GPU only — upgrade for HOLD_ANE/HOLD_AMX (M23)",
			FixHint: "rebuild UMAStatus.app from bmtl uma_toolkit (F0390)",
		}
	}
	return doctorCheck{
		Name:   name,
		Status: "ok",
		Detail: sock + " HOLD_GPU+ANE+AMX ready (M23)",
	}
}
