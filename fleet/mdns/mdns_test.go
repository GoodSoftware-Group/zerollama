package mdns

import (
	"net"
	"testing"

	"github.com/grandcat/zeroconf"
)

func TestPeerURLPrefersIPv4(t *testing.T) {
	entry := &zeroconf.ServiceEntry{
		Port:     11434,
		AddrIPv4: []net.IP{net.ParseIP("192.168.1.10")},
	}
	url, err := PeerURL(entry)
	if err != nil {
		t.Fatal(err)
	}
	if url != "http://192.168.1.10:11434" {
		t.Fatalf("url=%q", url)
	}
}

func TestPeerURLIPv6(t *testing.T) {
	entry := &zeroconf.ServiceEntry{
		Port:     11434,
		AddrIPv6: []net.IP{net.ParseIP("fe80::1")},
	}
	url, err := PeerURL(entry)
	if err != nil {
		t.Fatal(err)
	}
	if url != "http://[fe80::1]:11434" {
		t.Fatalf("url=%q", url)
	}
}

func TestPeerURLMissingAddr(t *testing.T) {
	_, err := PeerURL(&zeroconf.ServiceEntry{Port: 11434})
	if err == nil {
		t.Fatal("expected error")
	}
}

func TestPickHostSkipsLoopbackIPv4(t *testing.T) {
	entry := &zeroconf.ServiceEntry{
		Port:     11434,
		AddrIPv4: []net.IP{net.ParseIP("127.0.0.1")},
		HostName: "macbook.local.",
	}
	host := pickHost(entry)
	if host != "macbook.local" {
		t.Fatalf("host=%q want macbook.local", host)
	}
}

func TestPickHostLoopbackOnlyFails(t *testing.T) {
	entry := &zeroconf.ServiceEntry{
		Port:     11434,
		AddrIPv4: []net.IP{net.ParseIP("127.0.0.1")},
	}
	if pickHost(entry) != "" {
		t.Fatal("expected empty host when only loopback is advertised")
	}
}

func TestPortFromAddr(t *testing.T) {
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	defer ln.Close()
	port, err := PortFromAddr(ln.Addr())
	if err != nil {
		t.Fatal(err)
	}
	if port <= 0 {
		t.Fatalf("port=%d", port)
	}
}

func TestEncodeDecodeTXT(t *testing.T) {
	lines := encodeTXT(map[string]string{"version": "1.2.3", "role": "node"})
	got := decodeTXT(lines)
	if got["version"] != "1.2.3" || got["role"] != "node" {
		t.Fatalf("txt=%v", got)
	}
}
