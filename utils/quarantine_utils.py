#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quarantine analysis utilities.

Parse JSONL index and generate reports on patient rejections.
"""
import json
import logging
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime


def read_quarantine_index(quarantine_dir: Path) -> list[dict]:
    """
    Read JSONL index from quarantine_index.jsonl.

    Returns:
        List of dicts, one per quarantined patient
    """
    index_path = Path(quarantine_dir) / "quarantine_index.jsonl"
    if not index_path.exists():
        return []

    entries = []
    with open(index_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError as e:
                logging.warning(f"Malformed JSONL line: {line[:100]}, error: {e}")

    return entries


def analyze_quarantine(quarantine_dir: Path | str) -> dict:
    """
    Analyze quarantine directory and return statistics.

    Returns dict with:
      - total_quarantined: count
      - by_category: {category: count}
      - by_reason: {reason: count}
      - by_ptv_status: {status: count}
      - recent: list of recent 10 entries (sorted by timestamp)
    """
    quarantine_dir = Path(quarantine_dir)
    entries = read_quarantine_index(quarantine_dir)

    if not entries:
        return {
            "total_quarantined": 0,
            "by_category": {},
            "by_reason": {},
            "by_ptv_status": {},
            "recent": []
        }

    # Counters
    by_cat = Counter(e.get("reason_category", "unknown") for e in entries)
    by_reason = Counter(e.get("reason_slug", "unknown") for e in entries)
    by_ptv = Counter(e.get("ptv_final_status", "unknown") for e in entries if e.get("ptv_final_status"))

    # Recent entries (by timestamp)
    sorted_entries = sorted(
        entries,
        key=lambda e: e.get("timestamp", ""),
        reverse=True
    )
    recent = sorted_entries[:10]

    return {
        "total_quarantined": len(entries),
        "by_category": dict(by_cat.most_common()),
        "by_reason": dict(by_reason.most_common()),
        "by_ptv_status": dict(by_ptv.most_common()) if by_ptv else {},
        "recent": recent,
    }


def print_quarantine_summary(quarantine_dir: Path | str):
    """
    Print human-readable summary of quarantine state.
    """
    stats = analyze_quarantine(quarantine_dir)

    print("\n" + "=" * 80)
    print("QUARANTINE SUMMARY")
    print("=" * 80)
    print(f"Total Quarantined: {stats['total_quarantined']}\n")

    if stats['by_category']:
        print("By Category:")
        for cat, count in stats['by_category'].items():
            pct = 100.0 * count / stats['total_quarantined'] if stats['total_quarantined'] > 0 else 0
            print(f"  {cat:30s}: {count:4d} ({pct:5.1f}%)")
        print()

    if stats['by_reason']:
        print("Top Reasons (by slug):")
        for reason, count in list(stats['by_reason'].items())[:15]:
            pct = 100.0 * count / stats['total_quarantined'] if stats['total_quarantined'] > 0 else 0
            print(f"  {reason:40s}: {count:4d} ({pct:5.1f}%)")
        print()

    if stats['by_ptv_status']:
        print("PTV Final Status Distribution:")
        for status, count in stats['by_ptv_status'].items():
            pct = 100.0 * count / len([e for e in read_quarantine_index(quarantine_dir) if e.get('ptv_final_status')])
            print(f"  {status:30s}: {count:4d} ({pct:5.1f}%)")
        print()

    if stats['recent']:
        print("Recent Quarantines (last 10):")
        for entry in stats['recent']:
            ts = entry.get("timestamp", "N/A")
            pid = entry.get("pid", "?")
            cat = entry.get("reason_category", "?")
            reason = entry.get("reason", "?")[:50]
            print(f"  {ts} | {pid:15s} | {cat:20s} | {reason}")
        print()

    print("=" * 80 + "\n")


def export_quarantine_csv(quarantine_dir: Path | str, output_path: Path | str | None = None) -> str:
    """
    Export quarantine index to CSV format.

    Returns:
        CSV content as string (also writes to file if output_path provided)
    """
    entries = read_quarantine_index(Path(quarantine_dir))

    if not entries:
        return ""

    import csv
    import io

    # Extract all unique keys
    all_keys = set()
    for entry in entries:
        all_keys.update(entry.keys())

    fieldnames = sorted(all_keys)

    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(entries)

    csv_content = output.getvalue()

    if output_path:
        with open(output_path, 'w', encoding='utf-8', newline='') as f:
            f.write(csv_content)
        logging.info(f"Exported quarantine CSV to {output_path}")

    return csv_content


def get_quarantine_summary_table(quarantine_dir: Path | str) -> dict:
    """
    Get detailed breakdown of quarantine state.

    Returns:
        {
            "total": int,
            "categories": {category: {"count": int, "reasons": {reason: count}}},
            "timeline": {"date": count}
        }
    """
    entries = read_quarantine_index(Path(quarantine_dir))

    if not entries:
        return {"total": 0, "categories": {}, "timeline": {}}

    # By category → by reason
    by_cat_reason = defaultdict(lambda: defaultdict(int))
    timeline = defaultdict(int)

    for entry in entries:
        cat = entry.get("reason_category", "unknown")
        reason = entry.get("reason_slug", "unknown")
        ts = entry.get("timestamp", "")

        by_cat_reason[cat][reason] += 1

        # Extract date from ISO timestamp
        if ts:
            try:
                date = ts.split("T")[0]  # YYYY-MM-DD
                timeline[date] += 1
            except Exception:
                pass

    result = {"total": len(entries), "categories": {}, "timeline": dict(timeline)}

    for cat, reasons in by_cat_reason.items():
        result["categories"][cat] = {
            "count": sum(reasons.values()),
            "reasons": dict(reasons)
        }

    return result


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        quarantine_path = Path(sys.argv[1])
    else:
        # Try default location
        quarantine_path = Path(__file__).parent.parent / "quarantine"

    if quarantine_path.exists():
        print_quarantine_summary(quarantine_path)

        # Export CSV if requested
        if len(sys.argv) > 2:
            csv_out = sys.argv[2]
            export_quarantine_csv(quarantine_path, csv_out)
    else:
        print(f"Quarantine directory not found: {quarantine_path}")

