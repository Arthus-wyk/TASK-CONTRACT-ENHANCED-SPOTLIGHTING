from __future__ import annotations

import argparse
import csv
import json
from xml.sax.saxutils import escape
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any
from zipfile import ZIP_DEFLATED, ZipFile


DEFAULT_INPUT = Path("logs_my_agentdojo/workspace/token_usage.jsonl")
GROUP_FIELDS = ("provider", "model", "defense", "run_id", "operation")
DEFAULT_GROUP_BY = ("provider", "model", "defense")
MODEL_DEFENSE_GROUP_BY = ("model", "defense")


def flatten_numeric_values(value: Any, prefix: str = "") -> dict[str, float]:
    flattened: dict[str, float] = {}

    if isinstance(value, bool):
        return flattened

    if isinstance(value, (int, float)):
        key = prefix or "value"
        flattened[key] = float(value)
        return flattened

    if isinstance(value, dict):
        for key, item in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            flattened.update(flatten_numeric_values(item, child_prefix))

    return flattened


def parse_timestamp(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None

    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def format_number(value: float) -> str:
    if value.is_integer():
        return str(int(value))
    return f"{value:.6f}".rstrip("0").rstrip(".")


@dataclass
class Stats:
    record_count: int = 0
    error_count: int = 0
    malformed_count: int = 0
    duration_total: float = 0.0
    duration_min: float | None = None
    duration_max: float | None = None
    wait_duration_total: float = 0.0
    wait_duration_count: int = 0
    usage_totals: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    first_timestamp: datetime | None = None
    last_timestamp: datetime | None = None

    def add_record(self, record: dict[str, Any]) -> None:
        self.record_count += 1

        if record.get("error") is not None:
            self.error_count += 1

        timestamp = parse_timestamp(record.get("timestamp"))
        if timestamp is not None:
            if self.first_timestamp is None or timestamp < self.first_timestamp:
                self.first_timestamp = timestamp
            if self.last_timestamp is None or timestamp > self.last_timestamp:
                self.last_timestamp = timestamp

        duration = record.get("duration_seconds")
        if isinstance(duration, (int, float)) and not isinstance(duration, bool):
            duration = float(duration)
            self.duration_total += duration
            self.duration_min = duration if self.duration_min is None else min(self.duration_min, duration)
            self.duration_max = duration if self.duration_max is None else max(self.duration_max, duration)

        wait_duration = record.get("wait_duration_excluded_seconds")
        if isinstance(wait_duration, (int, float)) and not isinstance(wait_duration, bool):
            self.wait_duration_total += float(wait_duration)
            self.wait_duration_count += 1

        usage = record.get("usage")
        for key, value in flatten_numeric_values(usage).items():
            self.usage_totals[key] += value

    @property
    def avg_duration(self) -> float | None:
        if self.record_count == 0:
            return None
        return self.duration_total / self.record_count

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_count": self.record_count,
            "error_count": self.error_count,
            "malformed_count": self.malformed_count,
            "duration_seconds": {
                "total": self.duration_total,
                "average": self.avg_duration,
                "min": self.duration_min,
                "max": self.duration_max,
            },
            "wait_duration_excluded_seconds": {
                "total": self.wait_duration_total,
                "count": self.wait_duration_count,
            },
            "usage_totals": dict(sorted(self.usage_totals.items())),
            "time_range": {
                "first": self.first_timestamp.isoformat() if self.first_timestamp else None,
                "last": self.last_timestamp.isoformat() if self.last_timestamp else None,
            },
        }


def group_value_for(record: dict[str, Any], field: str) -> str:
    value = record.get(field, "unknown")
    if value is None:
        return "none"
    return str(value)


def group_values_for(record: dict[str, Any], group_by: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(group_value_for(record, field) for field in group_by)


def format_group_key(group_values: tuple[str, ...], group_by: tuple[str, ...]) -> str:
    if not group_by:
        return "all"

    parts = []
    for field, value in zip(group_by, group_values):
        parts.append(f"{field}={value}")
    return ", ".join(parts)


def read_stats(path: Path, group_by: tuple[str, ...]) -> tuple[Stats, dict[tuple[str, ...], Stats]]:
    overall = Stats()
    grouped: dict[tuple[str, ...], Stats] = {}

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            raw = line.strip()
            if not raw:
                continue

            try:
                record = json.loads(raw)
            except json.JSONDecodeError:
                overall.malformed_count += 1
                continue

            overall.add_record(record)
            key = group_values_for(record, group_by)
            grouped.setdefault(key, Stats()).add_record(record)

    return overall, grouped


def stats_row(
    label: str,
    stats: Stats,
    group_by: tuple[str, ...],
    group_values: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "scope": label,
        "group_key": "overall" if group_values is None else format_group_key(group_values, group_by),
        "record_count": stats.record_count,
        "error_count": stats.error_count,
        "malformed_count": stats.malformed_count,
        "first_timestamp": stats.first_timestamp.isoformat() if stats.first_timestamp else None,
        "last_timestamp": stats.last_timestamp.isoformat() if stats.last_timestamp else None,
        "duration_seconds_total": stats.duration_total,
        "duration_seconds_avg": stats.avg_duration,
        "duration_seconds_min": stats.duration_min,
        "duration_seconds_max": stats.duration_max,
        "wait_duration_excluded_seconds_total": stats.wait_duration_total,
        "wait_duration_excluded_seconds_count": stats.wait_duration_count,
    }

    for index, field in enumerate(group_by):
        row[field] = None if group_values is None else group_values[index]

    for key, value in sorted(stats.usage_totals.items()):
        row[f"usage.{key}"] = value

    return row


def build_export_rows(
    overall: Stats,
    grouped: dict[tuple[str, ...], Stats],
    group_by: tuple[str, ...],
) -> list[dict[str, Any]]:
    rows = [stats_row("overall", overall, group_by)]
    for group_values, stats in sorted(grouped.items()):
        rows.append(stats_row("group", stats, group_by, group_values))
    return rows


def build_model_defense_dict(
    grouped: dict[tuple[str, ...], Stats],
) -> dict[str, dict[str, Any]]:
    by_model: dict[str, dict[str, Any]] = {}
    for (model, defense), stats in sorted(grouped.items()):
        by_model.setdefault(model, {})[defense] = stats.to_dict()
    return by_model


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    columns: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                columns.append(key)

    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def write_excel(path: Path, rows: list[dict[str, Any]]) -> None:
    columns: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                columns.append(key)

    table = [columns]
    for row in rows:
        table.append([row.get(column) for column in columns])

    sheet_xml = build_sheet_xml(table)

    with ZipFile(path, "w", compression=ZIP_DEFLATED) as workbook:
        workbook.writestr(
            "[Content_Types].xml",
            """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>
  <Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>
  <Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>
</Types>""",
        )
        workbook.writestr(
            "_rels/.rels",
            """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>
</Relationships>""",
        )
        workbook.writestr(
            "xl/workbook.xml",
            """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"
 xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
  <sheets>
    <sheet name="summary" sheetId="1" r:id="rId1"/>
  </sheets>
</workbook>""",
        )
        workbook.writestr(
            "xl/_rels/workbook.xml.rels",
            """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>
</Relationships>""",
        )
        workbook.writestr(
            "xl/styles.xml",
            """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
  <fonts count="1">
    <font>
      <sz val="11"/>
      <name val="Calibri"/>
    </font>
  </fonts>
  <fills count="1">
    <fill><patternFill patternType="none"/></fill>
  </fills>
  <borders count="1">
    <border><left/><right/><top/><bottom/><diagonal/></border>
  </borders>
  <cellStyleXfs count="1">
    <xf numFmtId="0" fontId="0" fillId="0" borderId="0"/>
  </cellStyleXfs>
  <cellXfs count="1">
    <xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/>
  </cellXfs>
  <cellStyles count="1">
    <cellStyle name="Normal" xfId="0" builtinId="0"/>
  </cellStyles>
</styleSheet>""",
        )
        workbook.writestr("xl/worksheets/sheet1.xml", sheet_xml)


def build_sheet_xml(table: list[list[Any]]) -> str:
    rows_xml: list[str] = []
    for row_index, row in enumerate(table, start=1):
        cells_xml: list[str] = []
        for column_index, value in enumerate(row, start=1):
            cell_ref = f"{excel_column_name(column_index)}{row_index}"
            if value is None:
                continue

            if isinstance(value, bool):
                cell_xml = f'<c r="{cell_ref}" t="b"><v>{1 if value else 0}</v></c>'
            elif isinstance(value, (int, float)):
                cell_xml = f'<c r="{cell_ref}"><v>{value}</v></c>'
            else:
                cell_xml = (
                    f'<c r="{cell_ref}" t="inlineStr"><is><t>{escape(str(value))}</t></is></c>'
                )
            cells_xml.append(cell_xml)

        rows_xml.append(f'<row r="{row_index}">{"".join(cells_xml)}</row>')

    dimension_ref = f"A1:{excel_column_name(len(table[0]))}{len(table)}" if table and table[0] else "A1"
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
  <dimension ref="{dimension_ref}"/>
  <sheetData>
    {"".join(rows_xml)}
  </sheetData>
</worksheet>"""


def excel_column_name(index: int) -> str:
    letters: list[str] = []
    while index > 0:
        index, remainder = divmod(index - 1, 26)
        letters.append(chr(ord("A") + remainder))
    return "".join(reversed(letters))


def default_output_path(input_path: Path, suffix: str) -> Path:
    stem = input_path.stem
    if stem.endswith(".jsonl"):
        stem = stem[: -len(".jsonl")]
    return input_path.with_name(f"{stem}_summary{suffix}")


def print_stats(title: str, stats: Stats) -> None:
    print(title)
    print(f"  records: {stats.record_count}")
    print(f"  errors: {stats.error_count}")
    print(f"  malformed_lines: {stats.malformed_count}")

    if stats.first_timestamp or stats.last_timestamp:
        print(
            "  time_range: "
            f"{stats.first_timestamp.isoformat() if stats.first_timestamp else 'n/a'}"
            " -> "
            f"{stats.last_timestamp.isoformat() if stats.last_timestamp else 'n/a'}"
        )

    if stats.record_count > 0:
        avg_duration = stats.avg_duration
        print(
            "  duration_seconds: "
            f"total={format_number(stats.duration_total)}, "
            f"avg={format_number(avg_duration) if avg_duration is not None else 'n/a'}, "
            f"min={format_number(stats.duration_min) if stats.duration_min is not None else 'n/a'}, "
            f"max={format_number(stats.duration_max) if stats.duration_max is not None else 'n/a'}"
        )

    if stats.wait_duration_count > 0:
        print(
            "  wait_duration_excluded_seconds: "
            f"total={format_number(stats.wait_duration_total)}, "
            f"count={stats.wait_duration_count}"
        )

    if stats.usage_totals:
        print("  usage_totals:")
        for key, value in sorted(stats.usage_totals.items()):
            print(f"    {key}: {format_number(value)}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize duration_seconds and usage metrics from token_usage.jsonl."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Path to token_usage.jsonl (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--group-by",
        choices=GROUP_FIELDS,
        nargs="*",
        default=DEFAULT_GROUP_BY,
        help=(
            "Optional grouping fields. Defaults to provider model defense. "
            "Use '--group-by model defense' to compare only model/defense pairs."
        ),
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print JSON instead of human-readable text.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        help="Export the summary table to CSV.",
    )
    parser.add_argument(
        "--output-excel",
        type=Path,
        help="Export the summary table to Excel (.xlsx).",
    )
    args = parser.parse_args()

    if not args.input.exists():
        raise SystemExit(f"Input file not found: {args.input}")

    group_by = tuple(args.group_by)
    overall, grouped = read_stats(args.input, group_by)
    _model_defense_overall, model_defense_grouped = read_stats(args.input, MODEL_DEFENSE_GROUP_BY)
    export_rows = build_export_rows(overall, grouped, group_by)

    output_csv = args.output_csv
    output_excel = args.output_excel

    if not args.json and output_csv is None and output_excel is None:
        output_csv = default_output_path(args.input, ".csv")
        output_excel = default_output_path(args.input, ".xlsx")

    if output_csv == Path("."):
        output_csv = default_output_path(args.input, ".csv")
    if output_excel == Path("."):
        output_excel = default_output_path(args.input, ".xlsx")

    if output_csv is not None:
        write_csv(output_csv, export_rows)

    if output_excel is not None:
        write_excel(output_excel, export_rows)

    if args.json:
        payload = {
            "input": str(args.input),
            "group_by": list(group_by),
            "overall": overall.to_dict(),
            "groups": {
                format_group_key(key, group_by): stats.to_dict()
                for key, stats in sorted(grouped.items())
            },
            "by_model_and_defense": build_model_defense_dict(model_defense_grouped),
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    print(f"Input: {args.input}")
    print(f"Group by: {', '.join(group_by) if group_by else 'none'}")
    if output_csv is not None:
        print(f"CSV exported: {output_csv}")
    if output_excel is not None:
        print(f"Excel exported: {output_excel}")
    print()
    print_stats("Overall", overall)

    if grouped:
        print()
        print("Groups")
        print("=" * 80)
        for key, stats in sorted(grouped.items()):
            print_stats(format_group_key(key, group_by), stats)
            print()


if __name__ == "__main__":
    main()
