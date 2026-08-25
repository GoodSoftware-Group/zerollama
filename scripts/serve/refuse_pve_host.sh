# Sourced by serve wrappers. Refuse to bind inference on the Proxmox hypervisor.
# Production is CT 1564 (hostname cudallama, 192.168.255.164:8080).
# Override (lab only): ZEROLLAMA_ALLOW_PVE_HOST=1
if [[ "${ZEROLLAMA_ALLOW_PVE_HOST:-}" == "1" ]]; then
  return 0 2>/dev/null || true
fi
_virt="$(systemd-detect-virt 2>/dev/null || true)"
if [[ -d /etc/pve && -d /var/lib/vz && "${_virt}" != "lxc" ]]; then
  echo "zerollama serve must not run on the PVE hypervisor ($(hostname -s))." >&2
  echo "Start inside CT 1564: pct exec 1564 -- ~/bin/serve.sh" >&2
  echo "Clients: http://192.168.255.164:8080" >&2
  exit 1
fi
unset _virt
