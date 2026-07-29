#!/usr/bin/env python3
"""
Repair leaked JSON objects in saved interactive generated dialogs.

Problem:
- Some saved dialogs contain lines like:
  System: {"type": "message", content: "..."}
  System: {"type": "api_call", method: "...", parameters: {...}}
  These are model outputs that were intended to be parsed and normalized.

This script reparses each System line using `parse_system_output` and replaces it
with the normalized conversation-log form:
- System: <message text>
- System: APICall(method='X', parameters={...})

Usage examples:
  python tools/repair_generated_dialogs.py --input_dir storage/12345678 --inplace
  python tools/repair_generated_dialogs.py --input_dir storage --output_dir storage_repaired
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# Ensure project root is on sys.path when running as a script from tools/
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from llm_interaction.system_output_parser import parse_system_output, format_for_conversation_log


def repair_generated_dialog_lines(lines: list[str]) -> tuple[list[str], int, int]:
    repaired: list[str] = []
    changed = 0
    sys_lines = 0
    for line in lines:
        if isinstance(line, str) and line.startswith("System:"):
            sys_lines += 1
            parsed = parse_system_output(line)
            normalized = format_for_conversation_log(parsed)
            if normalized != line:
                changed += 1
            repaired.append(normalized)
        else:
            repaired.append(line)
    return repaired, changed, sys_lines


def iter_dialog_json_files(input_dir: Path) -> list[Path]:
    return sorted(input_dir.rglob("dialog_*.json"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing dialog_*.json (searched recursively).")
    parser.add_argument("--inplace", action="store_true", help="Modify files in place.")
    parser.add_argument("--output_dir", type=str, default=None, help="Write repaired copies here (mirrors structure).")
    parser.add_argument("--dry_run", action="store_true", help="Do not write; only print summary.")
    args = parser.parse_args()

    input_dir = Path(args.input_dir).expanduser().resolve()
    if not input_dir.exists():
        raise SystemExit(f"input_dir not found: {input_dir}")

    if args.inplace and args.output_dir:
        raise SystemExit("Choose only one: --inplace or --output_dir")
    if not args.inplace and not args.output_dir and not args.dry_run:
        raise SystemExit("Specify --inplace or --output_dir (or use --dry_run).")

    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else None
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

    files = iter_dialog_json_files(input_dir)
    if not files:
        raise SystemExit(f"No dialog_*.json files found under: {input_dir}")

    total_files = 0
    total_changed_lines = 0
    total_system_lines = 0
    total_files_changed = 0

    for fp in files:
        total_files += 1
        try:
            data: dict[str, Any] = json.loads(fp.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"[skip] failed to read json: {fp} ({e})")
            continue

        lines = data.get("generated_dialog")
        if not isinstance(lines, list):
            continue

        repaired, changed, sys_lines = repair_generated_dialog_lines(lines)
        total_changed_lines += changed
        total_system_lines += sys_lines

        if changed == 0:
            continue

        total_files_changed += 1
        data["generated_dialog"] = repaired

        if args.dry_run:
            continue

        out_path = fp
        if output_dir:
            rel = fp.relative_to(input_dir)
            out_path = output_dir / rel
            out_path.parent.mkdir(parents=True, exist_ok=True)

        out_path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    print("Done.")
    print(f"files scanned: {total_files}")
    print(f"files changed: {total_files_changed}")
    print(f"system lines scanned: {total_system_lines}")
    print(f"system lines changed: {total_changed_lines}")


if __name__ == "__main__":
    main()

