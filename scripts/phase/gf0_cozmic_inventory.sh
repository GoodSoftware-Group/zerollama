#!/usr/bin/env bash
# GF0 — cozmic inventory for glm-flash-lite ladder (read-only).
# Run ON cozmic (GPU host). Paste output into docs/glm-flash-lite-ladder.md Findings.
# Never starts serve / docker / downloads.
set -uo pipefail

echo "=== GF0 host ==="
hostname
date -u +%Y-%m-%dT%H:%MZ
whoami

echo "=== GPU ==="
if command -v nvidia-smi >/dev/null; then
  # current gen often parks at 1 via ASPM when idle — always record max too
  nvidia-smi --query-gpu=index,name,memory.total,memory.free,driver_version,pcie.link.gen.current,pcie.link.gen.max,pcie.link.width.current,pcie.link.width.max --format=csv \
    || echo "nvidia-smi query failed (driver down?)"
  echo "--- sysfs link (current vs max; Gen1 current alone is not proof of a Gen1 slot) ---"
  for d in /sys/bus/pci/devices/*/class; do
    [[ "$(cat "$d" 2>/dev/null)" == 0x030000 ]] || continue
    dev=$(dirname "$d")
    echo "$(basename "$dev"): cur=$(cat "$dev/current_link_speed" 2>/dev/null) x$(cat "$dev/current_link_width" 2>/dev/null) max=$(cat "$dev/max_link_speed" 2>/dev/null) x$(cat "$dev/max_link_width" 2>/dev/null)"
  done
  echo "--- compute apps ---"
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv || true
else
  echo "nvidia-smi missing"
fi

echo "=== RAM ==="
free -h
swapon --show || true
cat /sys/fs/cgroup/memory.max 2>/dev/null || true

echo "=== CPU ==="
nproc
lscpu | grep -E 'Model name|^CPU\(s\)|Thread|Core|Socket|NUMA' || true
grep -o -w -E 'avx2|fma|f16c' /proc/cpuinfo | sort -u || true

echo "=== DISK (NVMe / local fs) ==="
lsblk -o NAME,MODEL,SIZE,TYPE,FSTYPE,MOUNTPOINTS,ROTA,TRAN 2>/dev/null | head -40 || true
findmnt -t xfs,ext4 -o TARGET,SOURCE,FSTYPE,OPTIONS 2>/dev/null | head -25 || true
df -hT / /mnt/ssd2 /mnt/ollama_img /nvx /data 2>/dev/null || true

echo "=== PORTS (lab vs prod) ==="
ss -tlnp 2>/dev/null | grep -E ':30000|:2083|:11434|:8081|:8090|:8188' || echo "(none of watched ports listening)"

echo "=== TOOLS ==="
command -v docker || echo "docker: absent"
command -v hf || command -v huggingface-cli || echo "hf: absent"
command -v python3
python3 --version 2>/dev/null || true

echo "=== MODE pick (advisory) ==="
python3 - <<'PY'
import re
import shutil
import subprocess

def free_gib(path):
    try:
        return shutil.disk_usage(path).free / (1024**3)
    except OSError:
        return None

ram = None
try:
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                ram = int(line.split()[1]) / (1024**2)
                break
except OSError:
    pass

nvme = free_gib("/mnt/ssd2")
if nvme is None:
    nvme = free_gib("/")

print(f"MemAvailable≈{ram:.0f} GiB" if ram else "MemAvailable=?")
print(f"/mnt/ssd2 free≈{nvme:.0f} GiB" if nvme is not None else "disk free=?")

gpu_ok = False
try:
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
        text=True,
        timeout=10,
    )
    nums = [int(re.findall(r"\d+", x)[0]) for x in out.splitlines() if re.findall(r"\d+", x)]
    if nums:
        mib = max(nums)
        gpu_ok = mib >= 23000
        print(f"max GPU VRAM≈{mib} MiB  gpu_ok_24g={gpu_ok}")
except Exception as e:
    print(f"GPU probe failed: {e}")

avx = open("/proc/cpuinfo").read()
has_avx = all(f in avx for f in ("avx2", "fma", "f16c"))
print(f"avx2+fma+f16c={has_avx}")

if not gpu_ok:
    print("MODE: unsupported (need ≥~23 GiB VRAM)")
elif ram is not None and ram >= 240 and has_avx:
    print("MODE: fast (all-RAM) candidate — if you can spare ~238 GiB")
elif ram is not None and ram >= 60 and nvme is not None and nvme >= 250:
    print("MODE: nvme @ 55g candidate")
elif ram is not None and ram >= 17 and nvme is not None and nvme >= 250:
    print("MODE: nvme @ smaller cap candidate (expect <15–17 tok/s)")
else:
    print("MODE: not enough RAM/NVMe free — free space or pick another host")
PY

echo "=== GF0 done (read-only) ==="
