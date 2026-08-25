// Package ssrf guards user-supplied outbound HTTP(S) URLs (LocalAI LA15 /
// ValidateExternalURL). Loopback, RFC1918, link-local, unspecified, and
// well-known metadata hostnames are rejected after DNS resolution.
package ssrf

import (
	"errors"
	"fmt"
	"net"
	"net/http"
	"net/url"
	"strings"
)

const maxRedirects = 10

var (
	errUnsupportedScheme = errors.New("unsupported URL scheme")
	errNoHostname        = errors.New("URL has no hostname")
	errInternalHost      = errors.New("requests to internal hosts are not allowed")
	errMetadataHost      = errors.New("requests to cloud metadata services are not allowed")
	errInternalNetwork   = errors.New("requests to internal network addresses are not allowed")
	errTooManyRedirects  = errors.New("too many redirects")
)

// IsPublicIP reports whether ip is on the public internet: not loopback,
// link-local, private (RFC 1918 / RFC 4193), or unspecified. Covers 0.0.0.0/8
// and IPv4-mapped IPv6 wrapping a private address.
func IsPublicIP(ip net.IP) bool {
	if ip == nil {
		return false
	}
	if ip.IsLoopback() ||
		ip.IsLinkLocalUnicast() ||
		ip.IsLinkLocalMulticast() ||
		ip.IsPrivate() ||
		ip.IsUnspecified() {
		return false
	}
	if ip4 := ip.To4(); ip4 != nil {
		return !ip4.IsLoopback() &&
			!ip4.IsLinkLocalUnicast() &&
			!ip4.IsPrivate() &&
			!ip4.IsUnspecified()
	}
	return true
}

// ValidateExternalURL checks scheme, hostname, metadata names, and resolved IPs.
func ValidateExternalURL(rawURL string) error {
	parsed, err := url.Parse(rawURL)
	if err != nil {
		return fmt.Errorf("invalid URL: %w", err)
	}
	return ValidateURL(parsed)
}

// ValidateURL is ValidateExternalURL for an already-parsed URL.
func ValidateURL(parsed *url.URL) error {
	if parsed == nil {
		return errors.New("invalid URL")
	}
	scheme := strings.ToLower(parsed.Scheme)
	if scheme != "http" && scheme != "https" {
		return fmt.Errorf("%w: %s", errUnsupportedScheme, parsed.Scheme)
	}

	hostname := parsed.Hostname()
	if hostname == "" {
		return errNoHostname
	}

	lower := strings.ToLower(strings.Trim(hostname, "[]"))
	if lower == "localhost" || strings.HasSuffix(lower, ".local") || strings.HasSuffix(lower, ".localhost") {
		return errInternalHost
	}
	if lower == "metadata.google.internal" || lower == "instance-data" {
		return errMetadataHost
	}

	if ip := net.ParseIP(lower); ip != nil {
		if !IsPublicIP(ip) {
			return errInternalNetwork
		}
		return nil
	}

	ips, err := net.LookupIP(hostname)
	if err != nil {
		return fmt.Errorf("failed to resolve hostname: %w", err)
	}
	if len(ips) == 0 {
		return errInternalNetwork
	}
	for _, ip := range ips {
		if !IsPublicIP(ip) {
			return errInternalNetwork
		}
	}
	return nil
}

// CheckRedirect re-validates each hop (gallery-style SSRF on redirects).
func CheckRedirect(req *http.Request, via []*http.Request) error {
	if len(via) >= maxRedirects {
		return errTooManyRedirects
	}
	if req == nil || req.URL == nil {
		return errors.New("redirect missing URL")
	}
	return ValidateURL(req.URL)
}

// HostAllowed reports whether host is huggingface.co / hf.co or a subdomain
// (used for LA8 downloads so a 302 cannot land on an arbitrary public IP).
func HostAllowed(host string, suffixes ...string) bool {
	h := strings.ToLower(strings.Trim(host, "[]"))
	for _, s := range suffixes {
		s = strings.ToLower(strings.TrimSpace(s))
		if s == "" {
			continue
		}
		if h == s || strings.HasSuffix(h, "."+s) {
			return true
		}
	}
	return false
}
