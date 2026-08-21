package cmd

import (
	"fmt"
	"net/http"
	"strings"
	"time"
)

// doctorCheckModelReadiness covers minefield trap 112: a live listener / PID
// is not the same as an authenticated, answering API (and a green /api/tags
// is not a loaded runner). Complements trap 53 (which process) with "is it
// actually serving".
func doctorCheckModelReadiness() doctorCheck {
	const name = "serving trap-112 (liveness ≠ readiness)"
	base, tcpOnly := doctorProbeGoAPI()
	if base == "" {
		detail := "no Go API reachable"
		if len(tcpOnly) > 0 {
			detail = "TCP open but /api/tags not answering on " + strings.Join(tcpOnly, ", ") + " (trap 112 — process/port ≠ API ready)"
			return doctorCheck{
				Name:    name,
				Status:  "warn",
				Detail:  detail,
				FixHint: "crash-loops leave a stale listener; require /api/version + /api/tags 200, not just lsof",
			}
		}
		return doctorCheck{
			Name:   name,
			Status: "ok",
			Detail: detail + " — skipped",
		}
	}

	client := &http.Client{Timeout: 5 * time.Second}
	verStatus, _ := doctorGETStatus(client, strings.TrimSuffix(base, "/")+"/api/version")
	tagsStatus, _ := doctorGETStatus(client, strings.TrimSuffix(base, "/")+"/api/tags")
	if verStatus != http.StatusOK || tagsStatus != http.StatusOK {
		return doctorCheck{
			Name:    name,
			Status:  "warn",
			Detail:  fmt.Sprintf("base=%s /api/version HTTP %d /api/tags HTTP %d (trap 112)", base, verStatus, tagsStatus),
			FixHint: "do not treat PID or TCP as model readiness; wait for version+tags 200",
		}
	}

	loaded, err := doctorFetchLoadedModels(base)
	n := 0
	if err == nil {
		n = len(loaded)
	}
	if n == 0 {
		return doctorCheck{
			Name:   name,
			Status: "ok",
			Detail: fmt.Sprintf("API ready on %s (version+tags 200); no warm runner — PID≠loaded model (trap 112)", base),
		}
	}
	return doctorCheck{
		Name:   name,
		Status: "ok",
		Detail: fmt.Sprintf("API ready on %s with %d loaded runner(s) (trap 112 liveness/readiness split clear)", base, n),
	}
}

func doctorGETStatus(client *http.Client, url string) (int, error) {
	resp, err := client.Get(url)
	if err != nil {
		return 0, err
	}
	defer resp.Body.Close()
	return resp.StatusCode, nil
}
