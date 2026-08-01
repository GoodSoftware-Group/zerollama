#!/usr/bin/env python3
"""Insert a per-skill "Compatibility check" section into every SKILL.md.

These skills are written against zerollama tip/dev. Not every consumer runs
a build that new, so each skill should tell an agent how to verify the
endpoints/flags it relies on actually exist before assuming they work.

Idempotent: re-running replaces an existing "## Compatibility check" section
with a freshly-derived one (endpoints are re-scraped from the current body).

Usage: python3 .tools/add_compat_checks.py [--root .]
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

ENDPOINT_RE = re.compile(r"(GET|POST|PUT|DELETE)?\s*(/(?:api|v1)/[a-zA-Z0-9_/{}:.-]*)")
CLI_ONLY_SKILLS = {
    "diagnose-server-health",
    "install-zerollama",
    "configure-zerollama-env",
    "lmstudio-cache-import",
    "doctor-model",
}
MAX_ENDPOINTS = 4


def clean_path(path: str) -> str:
    return path.rstrip(".,);:'\"").rstrip("/")


def extract_endpoints(text: str) -> list[str]:
    seen: dict[str, str] = {}
    for method, path in ENDPOINT_RE.findall(text):
        path = clean_path(path)
        if not path or path in ("/api", "/v1"):
            continue
        if path not in seen or (not seen[path] and method):
            seen[path] = method
    # Prefer named-method entries, keep first-seen order, cap the list.
    ordered = list(seen.items())
    return [f"{m + ' ' if m else ''}{p}" for p, m in ordered[:MAX_ENDPOINTS]]


def probe_line(entry: str) -> str:
    method, _, path = entry.rpartition(" ") if " " in entry else ("GET", "", entry)
    path = path or entry
    if method in ("POST", "PUT", "DELETE"):
        return (
            f"curl -s -o /dev/null -w '%{{http_code}}\\n' -X {method} "
            f"http://localhost:11434{path} -d '{{}}'   # 400/422 = route exists; 404 = missing on this build"
        )
    return (
        f"curl -s -o /dev/null -w '%{{http_code}}\\n' "
        f"http://localhost:11434{path}   # 200/400 = route exists; 404 = missing on this build"
    )


def build_section(skill_name: str, endpoints: list[str]) -> str:
    lines = [
        "## Compatibility check",
        "",
        "This skill targets zerollama **tip/dev**, not a specific pinned",
        "release — not every server will have every endpoint/flag below yet.",
        "Verify before relying on this in an unattended flow, especially",
        "against a host you don't control:",
        "",
        "```bash",
        "zerollama --version                      # binary build",
        "curl -s http://localhost:11434/api/version | jq   # server build (if reachable)",
    ]
    is_cli_only = skill_name in CLI_ONLY_SKILLS
    if is_cli_only:
        lines.append("zerollama <subcommand> --help            # confirm the flag/subcommand exists before scripting it")
    for entry in endpoints:
        lines.append(probe_line(entry))
    lines.append("```")
    lines.append("")
    if is_cli_only and not endpoints:
        trigger = "An unrecognized flag/subcommand, or `--help` not mentioning an option this skill relies on,"
    else:
        trigger = "A **404** on an endpoint above (or an unrecognized flag/subcommand)"
    lines += [
        f"{trigger} means this build predates the feature this skill",
        "describes — check [`CHANGELOG.md`](../CHANGELOG.md) for when it",
        "landed, or upgrade (`git pull && ./scripts/build/build_zerollama_mac.sh`)",
        "rather than assuming the request shape is wrong.",
        "",
    ]
    return "\n".join(lines)


def insert_or_replace(text: str, section: str) -> str:
    pattern = re.compile(r"## Compatibility check\n.*?(?=\n## |\Z)", re.DOTALL)
    if pattern.search(text):
        # Use a function replacement — re.sub interprets backslash escapes
        # (e.g. our literal `\n` in curl format strings) in string replacements.
        return pattern.sub(lambda _m: section.rstrip("\n") + "\n", text, count=1)

    # Insert right before the first "## " heading after the title (i.e. before
    # "When to Use" / "Prerequisites" / whatever the skill leads with).
    lines = text.split("\n")
    insert_at = None
    seen_title = False
    for i, line in enumerate(lines):
        if line.startswith("# ") and not seen_title:
            seen_title = True
            continue
        if seen_title and line.startswith("## "):
            insert_at = i
            break
    if insert_at is None:
        return text.rstrip("\n") + "\n\n" + section + "\n"

    new_lines = lines[:insert_at] + [section, ""] + lines[insert_at:]
    return "\n".join(new_lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    changed = 0
    for skill_dir in sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith(".")):
        skill_md = skill_dir / "SKILL.md"
        if not skill_md.exists():
            continue
        text = skill_md.read_text()
        # Scrape endpoints from the body *excluding* any prior compatibility
        # section, so re-running doesn't feed its own generated probes back
        # in as "real" endpoints found in the skill's docs.
        existing_section_re = re.compile(r"## Compatibility check\n.*?(?=\n## |\Z)", re.DOTALL)
        body_for_scraping = existing_section_re.sub("", text)
        endpoints = extract_endpoints(body_for_scraping)
        section = build_section(skill_dir.name, endpoints)
        new_text = insert_or_replace(text, section)
        if new_text != text:
            skill_md.write_text(new_text)
            changed += 1
            print(f"updated {skill_dir.name} ({len(endpoints)} endpoint(s) probed)")

    print(f"\n{changed} skill(s) updated.")


if __name__ == "__main__":
    main()
