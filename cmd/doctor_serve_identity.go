package cmd

import (
	"encoding/json"
	"fmt"
	"net"
	"net/http"
	"net/url"
	"os/exec"
	"runtime"
	"strconv"
	"strings"
	"time"
)

// doctorCheckServeIdentity covers minefield trap 53: after a config edit / restart,
// prove which process is answering — not that the restart command exited 0.
// Always reports identity when reachable so operators can compare start times.
func doctorCheckServeIdentity() doctorCheck {
	const name = "serve identity (trap 53)"
	base, tcpOnly := doctorProbeGoAPI()
	if base == "" {
		detail := "no Go API reachable"
		if len(tcpOnly) > 0 {
			detail = "TCP open but /api/tags not answering on " + strings.Join(tcpOnly, ", ")
		}
		return doctorCheck{
			Name:    name,
			Status:  "warn",
			Detail:  detail,
			FixHint: "confirm the intended zerollama process holds the port (lsof -nP -iTCP:<port> -sTCP:LISTEN); crash-loops leave a stale healthy listener",
		}
	}

	ver := doctorFetchVersion(base)
	hostPort := doctorHostPortFromBase(base)
	pidInfo := doctorListenerIdentity(hostPort)

	parts := []string{fmt.Sprintf("base=%s", base)}
	if ver != "" {
		parts = append(parts, "version="+ver)
	}
	if pidInfo != "" {
		parts = append(parts, pidInfo)
	} else if hostPort != "" {
		parts = append(parts, "listener=unresolved (install lsof or check "+hostPort+")")
	}

	return doctorCheck{
		Name:    name,
		Status:  "ok",
		Detail:  strings.Join(parts, " · ") + " — after config edits, start time must be newer than the edit",
		FixHint: "kill by port, assert free, then start; never trust restart exit codes alone (minefield trap 53)",
	}
}

func doctorFetchVersion(base string) string {
	client := &http.Client{Timeout: 3 * time.Second}
	resp, err := client.Get(strings.TrimSuffix(base, "/") + "/api/version")
	if err != nil {
		return ""
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return ""
	}
	var body struct {
		Version string `json:"version"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&body); err != nil {
		return ""
	}
	return strings.TrimSpace(body.Version)
}

func doctorHostPortFromBase(base string) string {
	u, err := url.Parse(base)
	if err != nil || u.Host == "" {
		return ""
	}
	host := u.Hostname()
	port := u.Port()
	if port == "" {
		if u.Scheme == "https" {
			port = "443"
		} else {
			port = "80"
		}
	}
	if host == "" || port == "" {
		return ""
	}
	// Prefer loopback forms lsof understands.
	switch host {
	case "localhost", "::1":
		host = "127.0.0.1"
	}
	return net.JoinHostPort(host, port)
}

func doctorListenerIdentity(hostPort string) string {
	if hostPort == "" {
		return ""
	}
	_, port, err := net.SplitHostPort(hostPort)
	if err != nil || port == "" {
		return ""
	}
	if _, err := strconv.Atoi(port); err != nil {
		return ""
	}
	if runtime.GOOS != "darwin" && runtime.GOOS != "linux" {
		return ""
	}
	out, err := exec.Command("lsof", "-nP", "-iTCP:"+port, "-sTCP:LISTEN", "-Fpc").CombinedOutput()
	if err != nil {
		return ""
	}
	pid, cmd := "", ""
	for _, line := range strings.Split(string(out), "\n") {
		if len(line) < 2 {
			continue
		}
		switch line[0] {
		case 'p':
			if pid == "" {
				pid = line[1:]
			}
		case 'c':
			if cmd == "" {
				cmd = line[1:]
			}
		}
	}
	if pid == "" {
		return ""
	}
	psOut, err := exec.Command("ps", "-o", "lstart=,etime=,command=", "-p", pid).CombinedOutput()
	if err != nil {
		return fmt.Sprintf("pid=%s cmd=%s", pid, cmd)
	}
	fields := strings.TrimSpace(string(psOut))
	if fields == "" {
		return fmt.Sprintf("pid=%s cmd=%s", pid, cmd)
	}
	// ps line: "Tue Jul 28 13:00:00 2026 01:02:03 /path/zerollama serve"
	return fmt.Sprintf("pid=%s started/etime/cmd=%s", pid, fields)
}

// doctorCheckContextCeilings covers trap 55/61 arithmetic on a warm runner:
// advertised/served (num_ctx) vs trained context from loaded_metadata.
// Behavioural silent-fail ladders remain a hand-run (see docs).
func doctorCheckContextCeilings(m doctorLoadedModel) doctorCheck {
	name := fmt.Sprintf("context ceilings %s (trap 55/61)", m.Name)
	if m.NumCtx <= 0 && m.TrainCtx <= 0 {
		return doctorCheck{
			Name:   name,
			Status: "warn",
			Detail: "loaded_metadata missing num_ctx / train_context_length",
		}
	}
	parts := []string{}
	if m.NumCtx > 0 {
		parts = append(parts, fmt.Sprintf("served(num_ctx)=%d", m.NumCtx))
	}
	if m.TrainCtx > 0 {
		parts = append(parts, fmt.Sprintf("trained=%d", m.TrainCtx))
	}
	detail := strings.Join(parts, " ")
	if m.NumCtx > 0 && m.TrainCtx > 0 && m.NumCtx > m.TrainCtx {
		return doctorCheck{
			Name:    name,
			Status:  "warn",
			Detail:  detail + " — served num_ctx exceeds trained context_length (minefield trap 61)",
			FixHint: "lower num_ctx to the GGUF trained window; HTTP 200 at long prompts is not proof the head was read",
		}
	}
	if m.NumCtx > 0 && m.TrainCtx > 0 && m.TrainCtx > m.NumCtx*2 {
		return doctorCheck{
			Name:   name,
			Status: "ok",
			Detail: detail + " — served is a VRAM clamp below trained max; long-context still needs a cold ladder (trap 61 hand-run)",
		}
	}
	return doctorCheck{
		Name:   name,
		Status: "ok",
		Detail: detail + " — arithmetic aligned; behavioural long-context still needs a cold ladder (trap 61 hand-run)",
	}
}
