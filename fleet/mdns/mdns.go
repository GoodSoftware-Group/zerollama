// Package mdns implements DNS-SD (mDNS/Bonjour) for zerollama fleet discovery (F4).
//
// Why a separate package: GPU discovery lives in discover/; fleet LAN advertisement/browse is
// orthogonal and shared between zerollama serve (node) and zerollama fleet serve (manager).
package mdns

import (
	"context"
	"fmt"
	"log/slog"
	"net"
	"os"
	"strconv"
	"strings"

	"github.com/grandcat/zeroconf"
)

const (
	// ServiceNode is the DNS-SD type for zerollama inference nodes.
	ServiceNode = "_zerollama._tcp"
	// ServiceFleet is the DNS-SD type for the fleet management HTTP endpoint.
	ServiceFleet = "_zerollama-fleet._tcp"
	serviceDomain = "local."
)

// RegisterOpts configures mDNS registration.
type RegisterOpts struct {
	Service    string // ServiceNode or ServiceFleet
	Port       int
	Instance   string            // e.g. zerollama@host; default from hostname
	TXT        map[string]string // version, role, optional models hash
	Interfaces []net.Interface   // nil = all interfaces
}

// Register advertises a DNS-SD service until Shutdown is called.
func Register(opts RegisterOpts) (*zeroconf.Server, error) {
	if opts.Port <= 0 || opts.Port > 65535 {
		return nil, fmt.Errorf("mdns: invalid port %d", opts.Port)
	}
	if opts.Service == "" {
		return nil, fmt.Errorf("mdns: service type required")
	}
	instance := strings.TrimSpace(opts.Instance)
	if instance == "" {
		instance = defaultInstanceName()
	}
	txt := encodeTXT(opts.TXT)
	srv, err := zeroconf.Register(instance, opts.Service, serviceDomain, opts.Port, txt, opts.Interfaces)
	if err != nil {
		return nil, fmt.Errorf("mdns register %s: %w", opts.Service, err)
	}
	slog.Info("mdns registered", "service", opts.Service, "instance", instance, "port", opts.Port)
	return srv, nil
}

// BrowseOpts configures mDNS browse.
type BrowseOpts struct {
	Service string
	OnPeer  func(PeerEvent)
}

// PeerEvent is one discovered DNS-SD service instance.
type PeerEvent struct {
	URL      string
	Instance string
	TXT      map[string]string
}

// Browse watches for DNS-SD services until ctx is cancelled.
func Browse(ctx context.Context, opts BrowseOpts) error {
	if opts.Service == "" {
		return fmt.Errorf("mdns: service type required")
	}
	if opts.OnPeer == nil {
		return fmt.Errorf("mdns: OnPeer callback required")
	}

	resolver, err := zeroconf.NewResolver(nil)
	if err != nil {
		return fmt.Errorf("mdns resolver: %w", err)
	}

	entries := make(chan *zeroconf.ServiceEntry, 16)
	go func() {
		if err := resolver.Browse(ctx, opts.Service, serviceDomain, entries); err != nil && ctx.Err() == nil {
			slog.Warn("mdns browse stopped", "service", opts.Service, "error", err)
		}
	}()

	for {
		select {
		case <-ctx.Done():
			return ctx.Err()
		case entry, ok := <-entries:
			if !ok {
				return nil
			}
			if entry == nil {
				continue
			}
			url, err := PeerURL(entry)
			if err != nil {
				slog.Debug("mdns skip entry", "service", opts.Service, "error", err)
				continue
			}
			opts.OnPeer(PeerEvent{
				URL:      url,
				Instance: entry.Instance,
				TXT:      decodeTXT(entry.Text),
			})
		}
	}
}

// PeerURL builds an http base URL from a zeroconf service entry.
// Why prefer IPv4: LAN agents and curl examples use dotted-quad; IPv6 link-local needs brackets.
func PeerURL(entry *zeroconf.ServiceEntry) (string, error) {
	if entry == nil {
		return "", fmt.Errorf("nil service entry")
	}
	if entry.Port <= 0 {
		return "", fmt.Errorf("missing port")
	}
	host := pickHost(entry)
	if host == "" {
		return "", fmt.Errorf("no address for %q", entry.Instance)
	}
	return fmt.Sprintf("http://%s:%d", host, entry.Port), nil
}

func pickHost(entry *zeroconf.ServiceEntry) string {
	for _, ip := range entry.AddrIPv4 {
		if ip != nil && !ip.IsLoopback() {
			return ip.String()
		}
	}
	for _, ip := range entry.AddrIPv6 {
		if ip != nil && !ip.IsLoopback() {
			return "[" + ip.String() + "]"
		}
	}
	// Why skip loopback: a LAN peer URL must reach other hosts; 127.0.0.1 is useless off-box.
	if entry.HostName != "" {
		return strings.TrimSuffix(entry.HostName, ".")
	}
	return ""
}

func defaultInstanceName() string {
	host, err := os.Hostname()
	if err != nil || strings.TrimSpace(host) == "" {
		host = "zerollama"
	}
	return "zerollama@" + host
}

func encodeTXT(kv map[string]string) []string {
	if len(kv) == 0 {
		return nil
	}
	out := make([]string, 0, len(kv))
	for k, v := range kv {
		k = strings.TrimSpace(k)
		if k == "" {
			continue
		}
		out = append(out, k+"="+v)
	}
	return out
}

func decodeTXT(lines []string) map[string]string {
	if len(lines) == 0 {
		return nil
	}
	out := make(map[string]string, len(lines))
	for _, line := range lines {
		k, v, ok := strings.Cut(line, "=")
		if !ok || k == "" {
			continue
		}
		out[k] = v
	}
	return out
}

// PortFromAddr parses the TCP port from a net.Addr (listener or host:port string).
func PortFromAddr(addr net.Addr) (int, error) {
	if addr == nil {
		return 0, fmt.Errorf("nil addr")
	}
	_, portStr, err := net.SplitHostPort(addr.String())
	if err != nil {
		return 0, err
	}
	port, err := strconv.Atoi(portStr)
	if err != nil {
		return 0, err
	}
	return port, nil
}
