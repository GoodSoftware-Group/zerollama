package server

import (
	"fmt"
	"sort"
	"strings"
	"sync"

	"github.com/ollama/ollama/server/autoroute"
)

// Router metrics for GET /api/metrics (docs/auto-router.md §13). Only in-band rewrites
// count — dry runs (/v1/route, /api/router/decide) are probes, not traffic.
//
// WHY a mutex map, not atomics: labels are (router, surface, reason). Cardinality stays
// small — router names come from router.yaml plus "auto", surfaces and reasons are fixed.
type routeMetricKey struct {
	router, surface, reason string
}

var (
	routeMetricsMu        sync.Mutex
	routeDecisions        = map[routeMetricKey]uint64{}
	routeErrors           = map[routeMetricKey]uint64{}
	routeLatencyMsSum     uint64
	routeLatencyCount     uint64
	routeFallbackDecision uint64
)

func metricsRouteDecision(dec *RouterDecision, surface autoroute.Surface) {
	if dec == nil {
		return
	}
	reason := dec.Reason
	if reason == "" {
		reason = "classifier"
		if dec.Fallback {
			reason = "fallback"
		}
	}
	routeMetricsMu.Lock()
	defer routeMetricsMu.Unlock()
	routeDecisions[routeMetricKey{dec.Router, string(surface), reason}]++
	if dec.Fallback {
		routeFallbackDecision++
	}
	if dec.LatencyMs >= 0 {
		routeLatencyMsSum += uint64(dec.LatencyMs)
		routeLatencyCount++
	}
}

func metricsRouteError(router string, surface autoroute.Surface) {
	routeMetricsMu.Lock()
	defer routeMetricsMu.Unlock()
	routeErrors[routeMetricKey{router: router, surface: string(surface)}]++
}

// writeRouterMetrics appends router series in Prometheus text exposition.
func writeRouterMetrics(b *strings.Builder) {
	routeMetricsMu.Lock()
	defer routeMetricsMu.Unlock()

	fmt.Fprintf(b, "# HELP zerollama_router_decisions_total In-band router rewrites by router, surface and reason\n")
	fmt.Fprintf(b, "# TYPE zerollama_router_decisions_total counter\n")
	for _, k := range sortedRouteKeys(routeDecisions) {
		fmt.Fprintf(b, "zerollama_router_decisions_total{router=%q,surface=%q,reason=%q} %d\n", k.router, k.surface, k.reason, routeDecisions[k])
	}
	fmt.Fprintf(b, "# HELP zerollama_router_errors_total In-band router rewrites that failed (no candidate, classifier down)\n")
	fmt.Fprintf(b, "# TYPE zerollama_router_errors_total counter\n")
	for _, k := range sortedRouteKeys(routeErrors) {
		fmt.Fprintf(b, "zerollama_router_errors_total{router=%q,surface=%q} %d\n", k.router, k.surface, routeErrors[k])
	}
	fmt.Fprintf(b, "# HELP zerollama_router_fallback_total In-band router rewrites that used the router fallback\n")
	fmt.Fprintf(b, "# TYPE zerollama_router_fallback_total counter\n")
	fmt.Fprintf(b, "zerollama_router_fallback_total %d\n", routeFallbackDecision)
	fmt.Fprintf(b, "# HELP zerollama_router_suggest_total Cross-modal suggestions on chat turns by modality and source\n")
	fmt.Fprintf(b, "# TYPE zerollama_router_suggest_total counter\n")
	suggestMetricsMu.Lock()
	skeys := make([][2]string, 0, len(suggestCounts))
	for k := range suggestCounts {
		skeys = append(skeys, k)
	}
	sort.Slice(skeys, func(i, j int) bool { return skeys[i][0]+skeys[i][1] < skeys[j][0]+skeys[j][1] })
	for _, k := range skeys {
		fmt.Fprintf(b, "zerollama_router_suggest_total{modality=%q,source=%q} %d\n", k[0], k[1], suggestCounts[k])
	}
	suggestMetricsMu.Unlock()
	fmt.Fprintf(b, "# HELP zerollama_router_decision_latency_ms Router decision latency (sum/count)\n")
	fmt.Fprintf(b, "# TYPE zerollama_router_decision_latency_ms summary\n")
	fmt.Fprintf(b, "zerollama_router_decision_latency_ms_sum %d\n", routeLatencyMsSum)
	fmt.Fprintf(b, "zerollama_router_decision_latency_ms_count %d\n", routeLatencyCount)
}

func sortedRouteKeys(m map[routeMetricKey]uint64) []routeMetricKey {
	keys := make([]routeMetricKey, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	sort.Slice(keys, func(i, j int) bool {
		a, b := keys[i], keys[j]
		if a.router != b.router {
			return a.router < b.router
		}
		if a.surface != b.surface {
			return a.surface < b.surface
		}
		return a.reason < b.reason
	})
	return keys
}
