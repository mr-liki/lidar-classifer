#!/usr/bin/env python3
"""NAKSHA LAS/LAZ 1.4 -> compatible LAZ 1.2 batch utility.

Requires only Python standard library and external LASzip (laszip64.exe/laszip.exe).
Never modifies or overwrites inputs or existing outputs.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import math
import os
import shutil
import struct
import subprocess
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path

DEFAULT_INPUT = r"\\192.168.10.6\d\NT895\A13\LAS"
SUPPORTED = {6: 1, 7: 3, 8: 3}
MAX_LEGACY_POINTS = 0xFFFFFFFF


@dataclass(frozen=True)
class LasHeader:
    version: str
    point_format: int
    points: int
    scales: tuple[float, float, float]
    offsets: tuple[float, float, float]
    vlrs: frozenset[tuple[str, int]]
    compressed: bool


def read_header(path: Path) -> LasHeader:
    with path.open("rb") as file:
        first = file.read(375)
        if len(first) < 227 or first[:4] != b"LASF":
            raise ValueError("Not a valid LAS/LAZ header (LASF signature missing/truncated)")
        major, minor = first[24], first[25]
        header_size = struct.unpack_from("<H", first, 94)[0]
        point_data_start = struct.unpack_from("<I", first, 96)[0]
        vlr_count = struct.unpack_from("<I", first, 100)[0]
        fmt_raw = first[104]
        point_fmt = fmt_raw & 0x3F
        legacy_points = struct.unpack_from("<I", first, 107)[0]
        if header_size < 227 or header_size > 4096:
            raise ValueError(f"Unexpected header size: {header_size}")
        if point_data_start < header_size:
            raise ValueError("Point-data offset precedes LAS header")
        extended_points = 0
        if major == 1 and minor >= 4:
            if len(first) < 255:
                raise ValueError("Truncated LAS 1.4 header")
            extended_points = struct.unpack_from("<Q", first, 247)[0]
        count = extended_points or legacy_points
        scales = struct.unpack_from("<3d", first, 131)
        offsets = struct.unpack_from("<3d", first, 155)
        if any(not math.isfinite(x) for x in scales + offsets):
            raise ValueError("Invalid coordinate scale or offset")
        if any(x <= 0 for x in scales):
            raise ValueError("Invalid coordinate scale")
        file.seek(header_size)
        vlrs = set()
        for idx in range(vlr_count):
            current = file.tell()
            if current + 54 > point_data_start:
                raise ValueError("VLR headers overrun point-data offset")
            raw = file.read(54)
            if len(raw) != 54:
                raise ValueError("Truncated VLR")
            user = raw[2:18].split(b"\x00", 1)[0].decode("ascii", "replace").strip()
            rid, length = struct.unpack_from("<HH", raw, 18)
            vlrs.add((user.lower(), rid))
            if file.tell() + length > point_data_start:
                raise ValueError("VLR data overruns point-data offset")
            file.seek(length, os.SEEK_CUR)
        return LasHeader(
            version=f"{major}.{minor}",
            point_format=point_fmt,
            points=count,
            scales=tuple(scales),
            offsets=tuple(offsets),
            vlrs=frozenset(vlrs),
            compressed=bool(fmt_raw & 0x80) or ("laszip encoded", 22204) in vlrs,
        )


def verify_output(source: LasHeader, output_path: Path) -> LasHeader:
    output = read_header(output_path)
    if output.version != "1.2":
        raise ValueError(f"Expected LAZ 1.2, got LAS/LAZ {output.version}")
    expected_fmt = SUPPORTED.get(source.point_format)
    if expected_fmt is None:
        raise ValueError("Input point format cannot be verified in LAS 1.2 compatibility mode")
    if output.point_format != expected_fmt:
        raise ValueError(f"Expected point format {expected_fmt}, got {output.point_format}")
    if output.points != source.points:
        raise ValueError(f"Point count changed: {source.points} -> {output.points}")
    if ("laszip encoded", 22204) not in output.vlrs:
        raise ValueError("Missing LASzip compression VLR; file is not proven to be LAZ")
    if ("lascompatible", 22204) not in output.vlrs:
        raise ValueError("Missing LAS 1.4 compatibility VLR (extended attributes may be lost)")
    for field, a, b in zip(
        ("scale X", "scale Y", "scale Z", "offset X", "offset Y", "offset Z"),
        source.scales + source.offsets,
        output.scales + output.offsets,
    ):
        if not math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError(f"Coordinate {field} changed: {a} -> {b}")
    if output_path.stat().st_size <= 227:
        raise ValueError("Compressed output is suspiciously small")
    return output


def find_laszip(given: str | None) -> Path | None:
    if given:
        target = Path(given.strip('"'))
        return target if target.is_file() else None
    script_dir = Path(__file__).resolve().parent
    candidates = (
        script_dir / "laszip64.exe",
        script_dir / "laszip.exe",
        Path(r"C:\LAStools\bin\laszip64.exe"),
        Path(r"C:\LAStools\bin\laszip.exe"),
        Path(r"E:\lidar-classifer\tools\LAStools\bin\laszip64.exe"),
        Path(r"E:\lidar-classifer\tools\LAStools\bin\laszip.exe"),
    )
    for p in candidates:
        if p.is_file():
            return p
    for name in ("laszip64.exe", "laszip.exe", "laszip64", "laszip"):
        found = shutil.which(name)
        if found:
            return Path(found)
    return None


def write_report(dest: Path, rows: list[dict[str, object]]) -> Path:
    now = dt.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = dest / f"conversion_report_{now}.csv"
    fields = [
        "input", "output", "status", "source_version", "source_format",
        "source_points", "output_version", "output_format", "output_points", "details",
    ]
    with filename.open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return filename


def process(args: argparse.Namespace) -> int:
    source_dir = Path(args.input).expanduser()
    target_dir = Path(args.output).expanduser() if args.output else source_dir / "LAZ_1_2"
    if not source_dir.is_dir():
        print(f"ERROR: Input folder not accessible: {source_dir}", file=sys.stderr)
        print("Check VPN/network share permissions and UNC path.", file=sys.stderr)
        return 2
    files = sorted((p for p in source_dir.iterdir() if p.is_file() and p.suffix.lower() in (".las", ".laz")), key=lambda p: p.name.lower())
    if args.limit:
        files = files[:args.limit]
    if not files:
        print(f"No LAS/LAZ files directly inside {source_dir}")
        return 2
    tool = find_laszip(args.laszip)
    if not args.scan_only and tool is None:
        print("ERROR: LASzip executable not found. Place laszip64.exe or laszip.exe beside this script,", file=sys.stderr)
        print("or run: python convert_las14_to_laz12.py --laszip C:\\LAStools\\bin\\laszip64.exe", file=sys.stderr)
        print("Official download: https://rapidlasso.de/downloads/", file=sys.stderr)
        return 2
    if not args.scan_only:
        target_dir.mkdir(parents=True, exist_ok=True)
        log_dir = target_dir / "_conversion_logs"
        log_dir.mkdir(exist_ok=True)
    else:
        log_dir = None
    rows = []
    print(f"Input  : {source_dir}")
    print(f"Output : {target_dir}")
    print(f"Mode   : {'HEADER AUDIT' if args.scan_only else 'CONVERT (compatibility mode)'}")
    print(f"Files  : {len(files)}")
    if tool:
        print(f"LASzip : {tool}")
    print("NOTE: Extended LAS 1.4 attributes remain in compatibility Extra Bytes; legacy readers may not display them.\n")

    for index, source_path in enumerate(files, 1):
        destination = target_dir / (source_path.stem + ".laz")
        row: dict[str, object] = {
            "input": str(source_path), "output": str(destination), "status": "",
            "source_version": "", "source_format": "", "source_points": "",
            "output_version": "", "output_format": "", "output_points": "", "details": "",
        }
        temporary = None
        lock_path = None
        lock_handle = None
        try:
            src_header = read_header(source_path)
            row.update(source_version=src_header.version, source_format=src_header.point_format, source_points=src_header.points)
            if src_header.version != "1.4":
                row["status"] = "SKIP_SOURCE_VERSION"
                row["details"] = "Only LAS 1.4 inputs are converted"
            elif src_header.point_format not in SUPPORTED:
                row["status"] = "UNSUPPORTED_POINT_FORMAT"
                row["details"] = "Only 6, 7, 8 are supported in LAS 1.2 compatibility mode; waveform 9/10 require LAS 1.3"
            elif src_header.points > MAX_LEGACY_POINTS:
                row["status"] = "TOO_MANY_POINTS"
                row["details"] = "LAS 1.2 maximum is 4,294,967,295 points per file"
            elif args.scan_only:
                row["status"] = "ELIGIBLE"
                row["details"] = f"Would convert point format {src_header.point_format} -> {SUPPORTED[src_header.point_format]}"
            elif destination.exists():
                row["status"] = "SKIP_EXISTING"
                row["details"] = "Output already exists; untouched (not overwritten)"
                try:
                    old = verify_output(src_header, destination)
                    row.update(output_version=old.version, output_format=old.point_format, output_points=old.points)
                    row["details"] += "; header verified"
                except Exception as exc:
                    row["status"] = "EXISTING_UNVERIFIED"
                    row["details"] += f"; WARNING: {exc}"
            else:
                lock_path = target_dir / (destination.name + ".conversion.lock")
                try:
                    lock_handle = lock_path.open("x", encoding="utf-8")
                except FileExistsError:
                    row["status"] = "SKIP_LOCKED"
                    row["details"] = "Another conversion is active or a stale .conversion.lock file exists"
                else:
                    lock_handle.write(f"source={source_path}\n")
                    lock_handle.flush()
                    if destination.exists():
                        row["status"] = "SKIP_EXISTING"
                        row["details"] = "Output appeared after lock; untouched"
                    else:
                        temporary = target_dir / f".{destination.stem}.{uuid.uuid4().hex}.partial.laz"
                        logfile = log_dir / f"{source_path.stem}_{uuid.uuid4().hex[:8]}.log"
                        command = [str(tool), "-i", str(source_path), "-o", str(temporary), "-compatible"]
                        with logfile.open("wb") as stdout:
                            result = subprocess.run(command, stdout=stdout, stderr=subprocess.STDOUT, check=False)
                        if result.returncode != 0:
                            row["status"] = "FAILED"
                            row["details"] = f"LASzip exit code {result.returncode}; log: {logfile}"
                        elif not temporary.is_file():
                            row["status"] = "FAILED"
                            row["details"] = f"LASzip returned success without output; log: {logfile}"
                        else:
                            output = verify_output(src_header, temporary)
                            if args.deep_verify:
                                with logfile.open("ab") as stdout:
                                    stdout.write(b"\n=== LASzip full decompression integrity scan ===\n")
                                    stdout.flush()
                                    scan = subprocess.run([str(tool), "-i", str(temporary), "-check"], stdout=stdout, stderr=subprocess.STDOUT, check=False)
                                tail = logfile.read_bytes()[-16384:].decode("utf-8", "replace").lower()
                                if scan.returncode or "failed integrity check" in tail or "error:" in tail:
                                    raise ValueError(f"LASzip integrity scan failed; inspect {logfile}")
                            # On Windows os.rename refuses to overwrite an existing destination.
                            if destination.exists():
                                row["status"] = "SKIP_EXISTING"
                                row["details"] = "Output created by another process; untouched"
                            else:
                                os.rename(temporary, destination)
                                temporary = None
                                row["status"] = "CONVERTED"
                                row["details"] = f"Verified compatibility VLR and point count; log: {logfile}"
                                row.update(output_version=output.version, output_format=output.point_format, output_points=output.points)
        except Exception as exc:
            row["status"] = "FAILED"
            row["details"] = f"{type(exc).__name__}: {exc}"
        finally:
            if temporary is not None and temporary.exists():
                try:
                    temporary.unlink()
                except OSError:
                    pass
            if lock_handle is not None:
                lock_handle.close()
                try:
                    lock_path.unlink()
                except OSError:
                    pass
        rows.append(row)
        print(f"[{index:>3}/{len(files)}] {row['status']:<24} {source_path.name}  {row['source_points']} pts")
        if row["status"] not in ("CONVERTED", "SKIP_EXISTING", "ELIGIBLE"):
            print(f"          {row['details']}")

    counts = {status: sum(row["status"] == status for row in rows) for status in sorted(set(row["status"] for row in rows))}
    print("\nSUMMARY:", ", ".join(f"{name}={count}" for name, count in counts.items()))
    if not args.scan_only:
        report = write_report(target_dir, rows)
        print(f"Report : {report}")
    else:
        print("Audit-only mode: no files were written.")
    return 1 if any(r["status"] in ("FAILED", "EXISTING_UNVERIFIED", "UNSUPPORTED_POINT_FORMAT", "TOO_MANY_POINTS") for r in rows) else 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Batch LAS/LAZ 1.4 PF 6/7/8 -> LASzip compatibility LAZ 1.2")
    ap.add_argument("--input", default=DEFAULT_INPUT, help="Source LAS folder (UNC network paths supported)")
    ap.add_argument("--output", help="Output folder, default: <input>\\LAZ_1_2")
    ap.add_argument("--laszip", help="Path to laszip64.exe or laszip.exe")
    ap.add_argument("--scan-only", action="store_true", help="Inspect source LAS headers without conversion")
    ap.add_argument("--deep-verify", action="store_true", help="Also decompress each output in LASzip's read-only integrity-check mode")
    ap.add_argument("--limit", type=int, default=0, help="Limit to first N files for a pilot conversion")
    return process(ap.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
