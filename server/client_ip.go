package server

import (
	"context"
	"net"
	"strings"

	"github.com/gin-gonic/gin"
)

type clientIPContextKey struct{}

// requestClientIP returns the best-effort client address for logs and /api/ps.
// Uses gin ClientIP (honors X-Forwarded-For when TrustedProxies are configured).
func requestClientIP(c *gin.Context) string {
	if c == nil || c.Request == nil {
		return ""
	}
	ip := strings.TrimSpace(c.ClientIP())
	if ip == "" {
		host, _, err := net.SplitHostPort(strings.TrimSpace(c.Request.RemoteAddr))
		if err == nil {
			ip = host
		} else {
			ip = strings.TrimSpace(c.Request.RemoteAddr)
		}
	}
	return normalizeClientIP(ip)
}

func normalizeClientIP(ip string) string {
	ip = strings.TrimSpace(ip)
	if ip == "" {
		return ""
	}
	// Strip zone id (fe80::1%eth0) and brackets.
	ip = strings.Trim(ip, "[]")
	if i := strings.IndexByte(ip, '%'); i >= 0 {
		ip = ip[:i]
	}
	return ip
}

func contextWithClientIP(ctx context.Context, ip string) context.Context {
	ip = normalizeClientIP(ip)
	if ctx == nil || ip == "" {
		return ctx
	}
	return context.WithValue(ctx, clientIPContextKey{}, ip)
}

func clientIPFromContext(ctx context.Context) string {
	if ctx == nil {
		return ""
	}
	ip, _ := ctx.Value(clientIPContextKey{}).(string)
	return normalizeClientIP(ip)
}
