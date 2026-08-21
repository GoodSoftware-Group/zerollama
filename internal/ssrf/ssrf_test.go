package ssrf

import (
	"net"
	"testing"
)

func TestValidateExternalURL_publicHTTPS(t *testing.T) {
	if err := ValidateExternalURL("https://example.com/image.png"); err != nil {
		t.Fatalf("example.com: %v", err)
	}
}

func TestValidateExternalURL_blocks(t *testing.T) {
	cases := []string{
		"http://localhost/secret",
		"http://127.0.0.1/secret",
		"http://10.0.0.1/secret",
		"http://172.16.0.1/secret",
		"http://192.168.1.1/secret",
		"http://169.254.169.254/latest/meta-data/",
		"ftp://example.com/file",
		"file:///etc/passwd",
		"http:///path",
		"http://myservice.local/api",
		"http://metadata.google.internal/computeMetadata/v1/",
		"http://0.0.0.0/",
		"http://[::1]/secret",
	}
	for _, raw := range cases {
		if err := ValidateExternalURL(raw); err == nil {
			t.Errorf("expected reject %q", raw)
		}
	}
}

func TestIsPublicIP(t *testing.T) {
	if IsPublicIP(net.ParseIP("8.8.8.8")) != true {
		t.Fatal("8.8.8.8 should be public")
	}
	if IsPublicIP(net.ParseIP("127.0.0.1")) {
		t.Fatal("loopback")
	}
	if IsPublicIP(net.ParseIP("0.0.0.0")) {
		t.Fatal("unspecified")
	}
	mapped := net.ParseIP("::ffff:192.168.1.1")
	if IsPublicIP(mapped) {
		t.Fatal("mapped private")
	}
}

func TestHostAllowed(t *testing.T) {
	if !HostAllowed("cdn-lfs.huggingface.co", "huggingface.co", "hf.co") {
		t.Fatal("cdn subdomain")
	}
	if HostAllowed("evil.example", "huggingface.co") {
		t.Fatal("other host")
	}
}
