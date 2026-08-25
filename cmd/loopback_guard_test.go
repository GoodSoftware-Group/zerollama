package cmd

import (
	"net"
	"net/url"
	"strconv"
	"testing"
)

func TestClaimLoopbackGuardsHoldsConnectable(t *testing.T) {
	probe, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	port := strconv.Itoa(probe.Addr().(*net.TCPAddr).Port)
	_ = probe.Close()

	bind := &url.URL{Host: net.JoinHostPort("0.0.0.0", port)}
	connect := &url.URL{Host: net.JoinHostPort("127.0.0.1", port)}

	guards, err := claimLoopbackGuards(bind, connect)
	if err != nil {
		t.Fatal(err)
	}
	if len(guards) == 0 {
		t.Fatal("expected at least the 127.0.0.1 guard")
	}
	defer func() {
		for _, g := range guards {
			_ = g.Close()
		}
	}()

	// Second claim of the required address must fail (we still hold it).
	_, err = claimLoopbackGuards(bind, connect)
	if err == nil {
		t.Fatal("expected occupied loopback to fail second claim")
	}

	// Specific (non-wildcard) bind needs no guards.
	specific := &url.URL{Host: net.JoinHostPort("127.0.0.1", port)}
	none, err := claimLoopbackGuards(specific, connect)
	if err != nil {
		t.Fatal(err)
	}
	if len(none) != 0 {
		t.Fatalf("expected no guards for specific bind, got %d", len(none))
	}
}
