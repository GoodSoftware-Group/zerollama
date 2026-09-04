package cmd

import (
	"fmt"
	"io"
	"net"
	"net/http"
	"net/url"
	"os/exec"
	"runtime"
	"strconv"
	"strings"
	"time"

	"github.com/ollama/ollama/envconfig"
)

// portListener is one lsof LISTEN row (one process can have several n= addresses).
type portListener struct {
	PID     string
	Command string
	Name    string // e.g. *:11434, 127.0.0.1:11434, [::1]:11434
}

// parseLsofListenF parses `lsof -nP -iTCP:<port> -sTCP:LISTEN -Fpcn`.
func parseLsofListenF(out string) []portListener {
	var rows []portListener
	pid, cmd := "", ""
	for _, line := range strings.Split(out, "\n") {
		if len(line) < 2 {
			continue
		}
		switch line[0] {
		case 'p':
			pid = line[1:]
			cmd = ""
		case 'c':
			cmd = line[1:]
		case 'n':
			if pid == "" {
				continue
			}
			rows = append(rows, portListener{PID: pid, Command: cmd, Name: line[1:]})
		}
	}
	return rows
}

func uniqueListenPIDs(rows []portListener) []string {
	seen := map[string]bool{}
	var pids []string
	for _, r := range rows {
		if r.PID == "" || seen[r.PID] {
			continue
		}
		seen[r.PID] = true
		pids = append(pids, r.PID)
	}
	return pids
}

func looksLikeZerollamaListenCmd(cmd string) bool {
	c := strings.ToLower(cmd)
	return strings.Contains(c, "zerollama") || c == "ollama"
}

func emptyHTTPReply(err error) bool {
	if err == nil {
		return false
	}
	s := err.Error()
	if strings.Contains(s, "empty reply") {
		return true
	}
	// net/http: Head "http://127.0.0.1:11434/": EOF
	if strings.Contains(s, ": EOF") || strings.HasSuffix(s, "EOF") {
		return true
	}
	return strings.Contains(s, io.EOF.Error()) && (strings.Contains(s, "Head ") || strings.Contains(s, "Get "))
}

func listTCPListenersLsof(port string) []portListener {
	if runtime.GOOS != "darwin" && runtime.GOOS != "linux" {
		return nil
	}
	if _, err := strconv.Atoi(port); err != nil {
		return nil
	}
	out, err := exec.Command("lsof", "-nP", "-iTCP:"+port, "-sTCP:LISTEN", "-Fpcn").CombinedOutput()
	if err != nil && len(out) == 0 {
		return nil
	}
	return parseLsofListenF(string(out))
}

var listTCPListeners = listTCPListenersLsof

func summarizeListeners(rows []portListener) string {
	if len(rows) == 0 {
		return ""
	}
	var b strings.Builder
	for i, r := range rows {
		if i > 0 {
			b.WriteString("; ")
		}
		fmt.Fprintf(&b, "pid=%s cmd=%s %s", r.PID, r.Command, r.Name)
	}
	return b.String()
}

// formatPortHijack explains a split bind: wildcard zerollama + loopback Cursor (or similar).
// Darwin prefers the more-specific 127.0.0.1 bind, so localhost never reaches *:port.
// alt is an optional host:port where /api/version still works (LAN), or empty.
func formatPortHijack(target *url.URL, heartbeatErr error, rows []portListener, alt string) string {
	if target == nil {
		return ""
	}
	port := target.Port()
	if port == "" {
		return ""
	}
	pids := uniqueListenPIDs(rows)
	host := target.Hostname()
	loopbackClient := host == "localhost" || host == "127.0.0.1" || host == "::1"
	if ip := net.ParseIP(host); ip != nil && ip.IsLoopback() {
		loopbackClient = true
	}

	if len(pids) > 1 {
		var other []string
		for _, r := range rows {
			if looksLikeZerollamaListenCmd(r.Command) {
				continue
			}
			other = append(other, fmt.Sprintf("%s (pid %s) on %s", r.Command, r.PID, r.Name))
		}
		msg := fmt.Sprintf("port %s has %d LISTEN processes (split bind). macOS/Linux send 127.0.0.1 to the loopback socket, not zerollama's *:port. Listeners: %s",
			port, len(pids), summarizeListeners(rows))
		if len(other) > 0 {
			msg += ". Non-zerollama: " + strings.Join(other, ", ")
		}
		if alt != "" {
			msg += ". Serve still answers on " + alt + " — retry with OLLAMA_HOST=" + alt
		} else if loopbackClient {
			msg += ". Bypass: OLLAMA_HOST=<lan-ip>:" + port
		}
		if emptyHTTPReply(heartbeatErr) {
			msg += ". Empty/EOF reply is typical when Cursor/IDE grabbed 127.0.0.1:" + port
		}
		return msg
	}

	if len(rows) == 0 {
		return ""
	}
	if loopbackClient && emptyHTTPReply(heartbeatErr) {
		allForeign := true
		for _, r := range rows {
			if looksLikeZerollamaListenCmd(r.Command) {
				allForeign = false
				break
			}
		}
		if allForeign {
			return fmt.Sprintf("127.0.0.1:%s accepted then closed (EOF). Listener is %s — not zerollama. Close the IDE Ollama proxy or set OLLAMA_HOST to the real serve address.",
				port, summarizeListeners(rows))
		}
		if alt != "" {
			return fmt.Sprintf("localhost:%s returned EOF but serve is up at %s (loopback hijack). Retry: OLLAMA_HOST=%s", port, alt, alt)
		}
	}
	return ""
}

func probeServeOnNonLoopback(scheme, port string) string {
	if scheme == "" {
		scheme = "http"
	}
	addrs, err := net.InterfaceAddrs()
	if err != nil {
		return ""
	}
	client := &http.Client{Timeout: 800 * time.Millisecond}
	for _, a := range addrs {
		ipnet, ok := a.(*net.IPNet)
		if !ok || ipnet.IP == nil || ipnet.IP.IsLoopback() || ipnet.IP.IsUnspecified() {
			continue
		}
		ip := ipnet.IP
		if ip.To4() == nil {
			continue
		}
		base := (&url.URL{Scheme: scheme, Host: net.JoinHostPort(ip.String(), port)}).String()
		resp, err := client.Get(base + "/api/version")
		if err != nil {
			continue
		}
		_ = resp.Body.Close()
		if resp.StatusCode == http.StatusOK {
			return net.JoinHostPort(ip.String(), port)
		}
	}
	return ""
}

func wrapHeartbeatPortHijack(err error) error {
	if err == nil {
		return nil
	}
	u := envconfig.ConnectableHost()
	rows := listTCPListeners(u.Port())
	alt := ""
	if emptyHTTPReply(err) || len(uniqueListenPIDs(rows)) > 1 {
		alt = probeServeOnNonLoopback(u.Scheme, u.Port())
	}
	hint := formatPortHijack(u, err, rows, alt)
	if hint == "" {
		return err
	}
	return fmt.Errorf("%w\n%s", err, hint)
}

func doctorCheckPortHijack() doctorCheck {
	const name = "loopback port hijack"
	u := envconfig.ConnectableHost()
	port := u.Port()
	if port == "" {
		return doctorCheck{Name: name, Status: "ok", Detail: "no port on OLLAMA_HOST"}
	}
	rows := listTCPListeners(port)
	pids := uniqueListenPIDs(rows)
	if len(pids) > 1 {
		alt := probeServeOnNonLoopback(u.Scheme, port)
		hint := formatPortHijack(u, nil, rows, alt)
		return doctorCheck{
			Name:    name,
			Status:  "fail",
			Detail:  hint,
			FixHint: "stop the IDE/extension bound to 127.0.0.1:" + port + " (Cursor helper is a common thief), or use OLLAMA_HOST=<lan-ip>:" + port,
		}
	}
	if len(rows) == 0 {
		return doctorCheck{Name: name, Status: "ok", Detail: "no split bind on :" + port + " (nothing listening)"}
	}
	foreign := true
	for _, r := range rows {
		if looksLikeZerollamaListenCmd(r.Command) {
			foreign = false
			break
		}
	}
	if foreign {
		return doctorCheck{
			Name:    name,
			Status:  "fail",
			Detail:  ":" + port + " is held by " + summarizeListeners(rows) + " — not zerollama",
			FixHint: "that process (often Cursor extension-host) is answering localhost; disable its Ollama bind or point OLLAMA_HOST at a free port",
		}
	}
	return doctorCheck{
		Name:   name,
		Status: "ok",
		Detail: "one listener pid on :" + port + " — " + summarizeListeners(rows),
	}
}
