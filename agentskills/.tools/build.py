#!/usr/bin/env python3
"""Validate, index, and package agentskills/*/SKILL.md for distribution.

Usage:
    python3 .tools/build.py            # validate + write skills.json + README.md
    python3 .tools/build.py --package  # also zip each skill into dist/<name>.zip
    python3 .tools/build.py --strict   # exit non-zero on any warning (CI mode)

Run from the agentskills/ directory (or pass --root).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import zipfile
from pathlib import Path

import yaml

NAME_RE = re.compile(r"^[a-z0-9-]+$")
MAX_NAME_LEN = 64
MAX_DESC_LEN = 1024
REQUIRED_TOP_FIELDS = ["name", "description", "version", "author", "license", "platforms"]
REQUIRED_HERMES_FIELDS = ["tags", "category", "related_skills"]


def parse_frontmatter(text: str) -> dict:
    if not text.startswith("---"):
        raise ValueError("missing frontmatter delimiter")
    end = text.index("\n---", 3)
    fm = text[3:end]
    return yaml.safe_load(fm) or {}


def load_skills(root: Path):
    skills = {}
    errors: dict[str, list[str]] = {}
    for skill_dir in sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith(".")):
        skill_md = skill_dir / "SKILL.md"
        if not skill_md.exists():
            continue
        errs = []
        text = skill_md.read_text()
        try:
            fm = parse_frontmatter(text)
            end = text.index("\n---", 3)
        except Exception as e:  # noqa: BLE001
            errors[skill_dir.name] = [f"unparseable frontmatter: {e}"]
            continue

        for field in REQUIRED_TOP_FIELDS:
            if not fm.get(field):
                errs.append(f"missing required field: {field}")

        name = fm.get("name", "")
        if name and name != skill_dir.name:
            errs.append(f"name '{name}' does not match directory '{skill_dir.name}'")
        if name and not NAME_RE.match(name):
            errs.append(f"name '{name}' must be lowercase letters/numbers/hyphens only")
        if name and len(name) > MAX_NAME_LEN:
            errs.append(f"name '{name}' exceeds {MAX_NAME_LEN} chars")

        desc = fm.get("description", "")
        if len(desc) > MAX_DESC_LEN:
            errs.append(f"description exceeds {MAX_DESC_LEN} chars ({len(desc)})")

        hermes = (fm.get("metadata") or {}).get("hermes") or {}
        for field in REQUIRED_HERMES_FIELDS:
            if field not in hermes:
                errs.append(f"missing metadata.hermes.{field}")

        body_lines = text[end + 4 :].count("\n")
        if body_lines > 500:
            errs.append(f"body is {body_lines} lines, exceeds 500-line guideline")

        if errs:
            errors[skill_dir.name] = errs

        skills[skill_dir.name] = {
            "name": name or skill_dir.name,
            "description": desc,
            "version": fm.get("version"),
            "author": fm.get("author"),
            "license": fm.get("license"),
            "platforms": fm.get("platforms", []),
            "category": hermes.get("category"),
            "tags": hermes.get("tags", []),
            "related_skills": hermes.get("related_skills", []),
            "path": f"{skill_dir.name}/SKILL.md",
        }
    return skills, errors


def check_related_links(skills: dict, errors: dict):
    names = set(skills.keys())
    for name, info in skills.items():
        for rel in info.get("related_skills") or []:
            if rel not in names:
                errors.setdefault(name, []).append(
                    f"related_skills references unknown skill '{rel}'"
                )


def write_manifest(skills: dict, root: Path):
    manifest = {
        "schema_version": "1.0",
        "skill_count": len(skills),
        "skills": sorted(skills.values(), key=lambda s: s["name"]),
    }
    out = root / "skills.json"
    out.write_text(json.dumps(manifest, indent=2, sort_keys=False) + "\n")
    return out


def write_readme(skills: dict, root: Path):
    by_category: dict[str, list[dict]] = {}
    for info in skills.values():
        by_category.setdefault(info["category"] or "uncategorized", []).append(info)

    lines = [
        "# zerollama Agent Skills",
        "",
        f"{len(skills)} `SKILL.md` packages describing how to use "
        "[zerollama](https://github.com/GoodSoftware-Group/zerollama) — "
        "generated from the server's own OpenAPI spec, CLI, and source, "
        "intended for distribution to any agent/tool that can consume the "
        "`SKILL.md` (frontmatter + markdown) format.",
        "",
        "See [skills.json](skills.json) for the machine-readable manifest "
        "(name, description, version, category, tags, related_skills) used "
        "to generate this table.",
        "",
        "## Building",
        "",
        "This README and `skills.json` are generated — do not hand-edit them.",
        "",
        "```bash",
        "python3 .tools/build.py             # validate + regenerate README.md / skills.json",
        "python3 .tools/build.py --strict     # same, but exit non-zero on any issue (CI)",
        "python3 .tools/build.py --package    # also zip each skill into dist/<name>.zip",
        "```",
        "",
        "The validator checks: required frontmatter fields present, `name` "
        "matches its directory and is lowercase-hyphen-only (\u226464 chars), "
        "`description` \u22641024 chars, body \u2264500 lines, and every "
        "`related_skills` entry resolves to a real skill in this directory "
        "(catches stale/renamed cross-links before distribution).",
        "",
        "Every skill also carries a generated **Compatibility check** section "
        "(these docs target zerollama tip/dev, not a pinned release) \u2014 "
        "regenerate it after editing a skill's endpoints/CLI usage with:",
        "",
        "```bash",
        "python3 .tools/add_compat_checks.py   # idempotent; re-scrapes endpoints per skill",
        "```",
        "",
    ]
    for category in sorted(by_category):
        lines.append(f"## {category}")
        lines.append("")
        lines.append("| Skill | Description |")
        lines.append("|---|---|")
        for info in sorted(by_category[category], key=lambda s: s["name"]):
            desc = info["description"].replace("|", "\\|").replace("\n", " ")
            lines.append(f"| [`{info['name']}`]({info['name']}/SKILL.md) | {desc} |")
        lines.append("")

    out = root / "README.md"
    out.write_text("\n".join(lines).rstrip() + "\n")
    return out


def package_skills(skills: dict, root: Path):
    dist = root / "dist"
    dist.mkdir(exist_ok=True)
    for name in skills:
        skill_dir = root / name
        zip_path = dist / f"{name}.zip"
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for file in sorted(skill_dir.rglob("*")):
                if file.is_file():
                    zf.write(file, arcname=f"{name}/{file.relative_to(skill_dir)}")
    return dist


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".", help="agentskills/ directory")
    parser.add_argument("--package", action="store_true", help="also zip each skill into dist/")
    parser.add_argument("--strict", action="store_true", help="exit non-zero on any warning")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    skills, errors = load_skills(root)
    check_related_links(skills, errors)

    if errors:
        print(f"Found issues in {len(errors)} skill(s):\n")
        for name, errs in sorted(errors.items()):
            print(f"  {name}:")
            for e in errs:
                print(f"    - {e}")
        print()

    manifest_path = write_manifest(skills, root)
    readme_path = write_readme(skills, root)
    print(f"Wrote {manifest_path.relative_to(root.parent)}")
    print(f"Wrote {readme_path.relative_to(root.parent)}")

    if args.package:
        dist = package_skills(skills, root)
        print(f"Packaged {len(skills)} skill(s) into {dist.relative_to(root.parent)}/")

    print(f"\n{len(skills)} skills validated, {len(errors)} with issues.")
    if errors and args.strict:
        sys.exit(1)


if __name__ == "__main__":
    main()
