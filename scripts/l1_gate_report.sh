#!/usr/bin/env bash
# L1 gate verdict from l1_cuda_full_gate.json (or inline calibrate + concurrent dirs).
#
# Usage:
#   ./scripts/l1_gate_report.sh /tmp/l1-production-gate/gate.json
#   ./scripts/l1_gate_report.sh --calibrate /tmp/l1-cuda-calibrate --concurrent /tmp/l1-cuda-concurrent
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <l1-production-gate.json>" >&2
  echo "   or: $0 --calibrate <dir> --concurrent <dir>" >&2
  exit 1
fi

python3 - "$@" <<'PY'
import json
import sys
from pathlib import Path


def load_calibrate(dir_path: Path) -> dict:
    rows = {}
    for p in sorted(dir_path.glob("*.json")):
        d = json.loads(p.read_text())
        leg = d["legs"]["stock"]
        b = leg["bench"]
        gp = leg.get("gpu_profile") or {}
        rows[p.stem] = {
            "tok_s": b.get("decode_tok_per_s"),
            "n_parallel": gp.get("n_parallel"),
            "gpu_profile_id": gp.get("id"),
        }
    return rows


def load_concurrent(dir_path: Path) -> dict:
    rows = {}
    for p in sorted(dir_path.glob("*.json")):
        d = json.loads(p.read_text())
        gp = d.get("gpu_profile") or {}
        rows[d.get("label", p.stem)] = {
            "agg_tok_s": d.get("agg_tok_s_mean"),
            "n_concurrent": d.get("n_concurrent"),
            "n_parallel": gp.get("n_parallel"),
            "gpu_profile_id": gp.get("id"),
        }
    return rows


def pct_delta(on, off):
    if on is None or off is None or off == 0:
        return None
    return (on - off) / off * 100.0


def leg_ran(data, name):
    legs = data.get("legs") or {}
    if name in legs:
        return bool(legs[name])
    # Legacy gate.json without legs metadata: non-empty section means it ran.
    section = data.get(name) or {}
    return bool(section)


def main() -> int:
    argv = sys.argv[1:]
    if argv and argv[0] == "--calibrate":
        cal_dir = Path(argv[1])
        con_dir = Path(argv[3])
        data = {
            "calibrate": load_calibrate(cal_dir),
            "concurrent": load_concurrent(con_dir),
            "legs": {
                "calibrate": cal_dir.is_dir() and any(cal_dir.glob("*.json")),
                "concurrent": con_dir.is_dir() and any(con_dir.glob("*.json")),
            },
        }
    else:
        data = json.loads(Path(argv[0]).read_text())

    cal = data.get("calibrate") or {}
    con = data.get("concurrent") or {}
    model = data.get("model", "(unknown)")
    ran_cal = leg_ran(data, "calibrate")
    ran_con = leg_ran(data, "concurrent")

    print("== L1 gate report ==")
    print(f"model: {model}")

    off_ss = on_ss = ss_delta = None
    if ran_cal:
        off_ss = (cal.get("profile-off") or {}).get("tok_s")
        on_ss = (cal.get("profile-on-default") or {}).get("tok_s")
        ss_delta = pct_delta(on_ss, off_ss)
        print(f"single-stream OFF: {off_ss} tok/s")
        print(f"single-stream ON:  {on_ss} tok/s  (gpu={((cal.get('profile-on-default') or {}).get('gpu_profile_id'))})")
        if ss_delta is not None:
            print(f"single-stream delta: {ss_delta:+.1f}%")
    else:
        print("single-stream: (skipped)")

    off_c = on_c = c_delta = n_con = None
    if ran_con:
        off_c = (con.get("profile-off") or {}).get("agg_tok_s")
        on_c = (con.get("profile-on-default") or {}).get("agg_tok_s")
        c_delta = pct_delta(on_c, off_c)
        n_con = (con.get("profile-on-default") or con.get("profile-off") or {}).get("n_concurrent")
        print(f"concurrent OFF: {off_c} agg tok/s (n={n_con})")
        print(f"concurrent ON:  {on_c} agg tok/s")
        if c_delta is not None:
            print(f"concurrent delta: {c_delta:+.1f}%")
    else:
        print("concurrent: (skipped)")

    ss_min = float(data.get("single_stream_min_delta_pct", 0.0))
    pass_ss = (not ran_cal) or (ss_delta is not None and ss_delta >= ss_min)
    pass_con = (not ran_con) or (c_delta is not None and c_delta > 0)

    print()
    if pass_ss and pass_con:
        if ran_cal and ran_con:
            print("VERDICT: L1 CUDA gate PASS (single-stream non-regression + concurrent win)")
        elif ran_cal:
            print("VERDICT: L1 calibrate PASS (concurrent leg skipped)")
        elif ran_con:
            print("VERDICT: L1 concurrent PASS (calibrate leg skipped)")
        else:
            print("VERDICT: L1 FAIL — no gate legs ran")
            return 1
        return 0
    if ran_cal and not pass_ss:
        print(f"VERDICT: L1 FAIL — single-stream ON below OFF by >{-ss_min:.1f}% threshold")
    elif ran_con and not pass_con:
        print("VERDICT: L1 FAIL — concurrent profile ON did not beat OFF")
        print("  Tune runtime/configs/gpu/rtx-5080.json (n_parallel, batch_size, ubatch_size)")
    else:
        print("VERDICT: L1 FAIL")
    return 1


sys.exit(main())
PY
