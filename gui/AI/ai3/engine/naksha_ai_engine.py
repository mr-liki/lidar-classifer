#!/usr/bin/env python
"""naksha_ai_engine.py - isolated "AI model 3.0" LiDAR classification engine.

Runs the vendored pipeline steps in the standalone order:
  1. step1_preprocess   PDAL outlier + SMRF ground filter + HeightAboveGround
  2. step2_inference    Myria3D / RandLA-Net inference (FRACTAL-LidarHD checkpoint)
  3. step3_rule_engine  contract rule engine, spatial HAG cleanup, class mapping

Pipeline logic lives in the vendored steps, not here: this wrapper only decides what
contract to run them with and how to report progress.  The host starts this file with its
own runtime interpreter, so nothing here imports the host.

  python naksha_ai_engine.py --input IN.las --output OUT.laz --mapping MAP.json
          [--work-dir DIR] [--no-power-lines] [--device auto|cuda|cpu] [--selftest]

stdout protocol (flushed per line):
  PROGRESS <0-100> <message>   step 1: 0-25, step 2: 25-70, step 3: 70-98, write: 98-100
  RESULT {json}                final line: point count, timings, class histogram
  SELFTEST OK                  from --selftest when the environment is complete

stderr: ERROR <message>, exit 0 ok / 1 bad args / 2 step failed / 3 environment / 130 cancel.

The mapping JSON is written by the host: {"classes": {...}, "power_lines_enabled": true}.
The rule engine speaks its own internal codes (1 unclassified, 2 ground, 3 low, 4 medium,
5 high, 6 building, 7/18 noise, 9 water, 14 wire, 15 pole); the host's numbers become the
delivered codes through ptc_class_mapping, and the resolved contract is written beside the
output for provenance.  Bridge (17) is deliberately left unmapped: step 3 Rule N re-files it.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import math
import os
import re
import shutil
import signal
import sys
import tempfile
import threading
import time
import traceback

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

ENGINE_VERSION = "1.0.0"
ENGINE_NAME = "naksha_ai_engine"

CONFIG_PATH = os.path.join(_HERE, "config", "contract_spec.json")
MODELS_DIR = os.path.join(_HERE, "models")
CHECKPOINT_PATH = os.path.join(MODELS_DIR, "fractal_lidarhd_myria3d.ckpt")
INFERENCE_CONFIG_PATH = os.path.join(MODELS_DIR, "inference_config.yaml")
MYRIA3D_DIR = os.path.join(_HERE, "myria3d")

# Exit codes, part of the host contract.
EXIT_OK = 0
EXIT_BAD_ARGS = 1
EXIT_STEP_FAILED = 2
EXIT_ENVIRONMENT = 3
EXIT_CANCELLED = 130

# Progress bands. The host shows one bar, so the bands are part of the contract.
BAND_STEP1 = (0, 25)
BAND_STEP2 = (25, 70)
BAND_STEP3 = (70, 98)
BAND_WRITE = (98, 100)

# Internal rule-engine codes. This is the rule engine's own vocabulary and never changes;
# the host's numbering is applied by ptc_class_mapping at the end of step 3.
INTERNAL_UNCLASSIFIED = 1
INTERNAL_GROUND = 2
INTERNAL_LOW_VEG = 3
INTERNAL_MEDIUM_VEG = 4
INTERNAL_HIGH_VEG = 5
INTERNAL_BUILDING = 6
INTERNAL_LOW_NOISE = 7
INTERNAL_WATER = 9
INTERNAL_WIRE = 14
INTERNAL_POLE = 15
INTERNAL_BRIDGE = 17
INTERNAL_HIGH_NOISE = 18

CLASS_KEY_TO_INTERNAL = {
    "unclassified": INTERNAL_UNCLASSIFIED,
    "ground": INTERNAL_GROUND,
    "low_veg": INTERNAL_LOW_VEG,
    "medium_veg": INTERNAL_MEDIUM_VEG,
    "high_veg": INTERNAL_HIGH_VEG,
    "building": INTERNAL_BUILDING,
    "wire": INTERNAL_WIRE,
    "pole": INTERNAL_POLE,
    "low_noise": INTERNAL_LOW_NOISE,
    "high_noise": INTERNAL_HIGH_NOISE,
}

REQUIRED_MAPPING_KEYS = ("ground", "low_veg", "medium_veg", "high_veg", "building")

# Codes used when the host does not supply one. They are the host's own no-PTC convention
# (the same one the Premium engine keeps): LAS 0 unclassified, 7 low point, 18 high noise.
DEFAULT_UNCLASSIFIED = 0
DEFAULT_LOW_NOISE = 7
DEFAULT_HIGH_NOISE = 18

# Wall-clock cost per point of each step, measured on a 14.9 M point tile (RTX GPU, 20 cores).
# Only used to pace the progress bar while a step is running, never for any decision.
SECONDS_PER_POINT = {"step1": 12.5e-6, "step2": 22.0e-6, "step3": 8.5e-6}

# --------------------------------------------------------------------------------------- I/O
def _progress(percent, message: str) -> None:
    """One progress line on stdout, flushed so the host sees it immediately."""
    pct = int(max(0, min(100, round(float(percent)))))
    print(f"PROGRESS {pct} {message}", flush=True)


def _fail(message: str, code: int) -> int:
    print(f"ERROR {message}", file=sys.stderr, flush=True)
    return code


def _md5(path: str) -> str:
    digest = hashlib.md5()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _point_count(path: str) -> int:
    import laspy
    with laspy.open(path) as reader:
        return int(reader.header.point_count)


def _class_histogram(path: str) -> dict:
    """Classification counts without loading the whole cloud into memory."""
    import laspy
    import numpy as np
    counts = {}
    with laspy.open(path) as reader:
        for chunk in reader.chunk_iterator(2_000_000):
            codes, n = np.unique(np.asarray(chunk.classification), return_counts=True)
            for code, cnt in zip(codes, n):
                counts[int(code)] = counts.get(int(code), 0) + int(cnt)
    return dict(sorted(counts.items()))

# ------------------------------------------------------------------------ mapping -> contract
def load_mapping(path: str) -> dict:
    """Read and validate the host mapping JSON."""
    if not path:
        raise ValueError("--mapping is required")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"mapping file not found: {path}")
    with open(path, encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, dict):
        raise ValueError("mapping JSON must be an object")
    classes = raw.get("classes")
    if not isinstance(classes, dict) or not classes:
        raise ValueError("mapping JSON needs a non-empty 'classes' object")

    resolved = {}
    for key in CLASS_KEY_TO_INTERNAL:
        if key not in classes:
            continue
        value = classes[key]
        try:
            code = int(value)
        except (TypeError, ValueError):
            raise ValueError(f"mapping classes['{key}'] is not an integer: {value!r}")
        if not 0 <= code <= 255:
            raise ValueError(f"mapping classes['{key}']={code} is outside 0..255")
        resolved[key] = code

    missing = [k for k in REQUIRED_MAPPING_KEYS if k not in resolved]
    if missing:
        raise ValueError(f"mapping is missing required class key(s): {missing}")

    # Optional classes. "noise" is accepted as the single code for both outlier classes.
    if "noise" in classes:
        try:
            noise = int(classes["noise"])
        except (TypeError, ValueError):
            raise ValueError(f"mapping classes['noise'] is not an integer: {classes['noise']!r}")
        if not 0 <= noise <= 255:
            raise ValueError(f"mapping classes['noise']={noise} is outside 0..255")
        resolved.setdefault("low_noise", noise)
        resolved.setdefault("high_noise", noise)
    resolved.setdefault("unclassified", DEFAULT_UNCLASSIFIED)
    resolved.setdefault("low_noise", DEFAULT_LOW_NOISE)
    resolved.setdefault("high_noise", DEFAULT_HIGH_NOISE)
    resolved.setdefault("wire", INTERNAL_WIRE)
    resolved.setdefault("pole", INTERNAL_POLE)

    return {
        "classes": resolved,
        "power_lines_enabled": bool(raw.get("power_lines_enabled", True)),
    }

# ------------------------------------------------------------------------------- .ptc schema
# The client's class table (.ptc) is the engine's input schema: it says which code each internal
# class is delivered as. Resolution follows the original pipeline CLI: contracted folds first,
# then the exact class name, then auditable aliases, then the schema's default class.
INTERNAL_NAMES = {1: "Unclassified", 2: "Ground", 3: "Low Vegetation", 4: "Medium Vegetation",
                  5: "High Vegetation", 6: "Building", 7: "Low Point Noise", 9: "Water",
                  14: "Wire / Cable", 15: "Pole / Utility", 17: "Bridge / Deck",
                  18: "High Point Noise"}
PTC_ALIASES = {
    1: (None, "no 'Unclassified' class in the schema; uses its default class"),
    7: ("low point", "TerraScan's low-outlier class"),
    14: ("bare conductor", "generic conductor/wire"),
    15: ("pylon or pole", "utility pole or tower"),
    17: ("bridge", "bridge deck"),
}
PTC_FORCED_TARGETS = {
    7: ("low point", "noise consolidated into the schema's single noise container"),
    9: (None, "no Water class is delivered; water stays in the default class"),
    18: ("low point", "noise consolidated into the schema's single noise container"),
}
EMITTABLE_CODES = (1, 2, 3, 4, 5, 6, 7, 9, 14, 15, 17, 18)
# Classes that must exist in the schema by name: delivering them as "unclassified" would be wrong.
PTC_REQUIRED = {2: "Ground", 3: "Low vegetation", 4: "Medium vegetation",
                5: "High vegetation", 6: "Building"}


def _norm(name: str) -> str:
    text = re.sub(r"[^a-z0-9]+", " ", str(name).lower()).strip()
    tokens = [t[:-1] if len(t) > 3 and t.endswith("s") else t for t in text.split()]
    return " ".join(tokens)


def _rgb(token: str):
    match = re.fullmatch(r"(\d{1,3})\s*,\s*(\d{1,3})\s*,\s*(\d{1,3})", token or "")
    if not match:
        return None
    values = tuple(int(v) for v in match.groups())
    return values if all(0 <= v <= 255 for v in values) else None


_RGB_TRIPLE = re.compile(r"\d{1,3}\s*,\s*\d{1,3}\s*,\s*\d{1,3}")


def _tokenize(line: str) -> list:
    if "\t" in line:
        return [t.strip() for t in line.split("\t") if t.strip()]
    triples = []

    def stash(match):
        triples.append(match.group(0).replace(" ", ""))
        return f" __rgb{len(triples) - 1}__ "

    tokens = [t for t in re.split(r"[,\s;]+", _RGB_TRIPLE.sub(stash, line)) if t]
    restored = []
    for token in tokens:
        holder = re.fullmatch(r"__rgb(\d+)__", token)
        restored.append(triples[int(holder.group(1))] if holder else token)
    return restored


def parse_ptc_file(ptc_path: str) -> dict:
    """Parse a TerraScan-style class table (.ptc) or a plain ``id name r g b`` table.

    Binary .ptc files (Leica Cyclone point clouds use the extension too) hold no class table
    and are reported as such.
    """
    result = {"path": ptc_path, "md5": _md5(ptc_path), "format": "unknown", "classes": {},
              "ids": [], "default_class": None, "notes": []}
    with open(ptc_path, "rb") as handle:
        head = handle.read(4096)
    if head[:3] == b"PTC" or b"\x00" in head:
        result["format"] = "binary"
        result["notes"].append("binary payload detected; it holds no class table")
        return result
    with open(ptc_path, "r", encoding="utf-8", errors="ignore") as handle:
        raw_lines = handle.read().splitlines()

    colors, header_ids, view_lines = {}, {}, 0
    for line in raw_lines:
        line = line.rstrip("\r\n").strip()
        if not line or line.startswith("#") or line.startswith("//"):
            continue
        tokens = _tokenize(line)
        if not tokens:
            continue
        if tokens[0] == "*":
            numbers = [t for t in tokens[1:] if t.isdigit()]
            for token in tokens[1:]:
                rgb = _rgb(token)
                if rgb and numbers:
                    colors[int(numbers[0])] = rgb
                    break
            view_lines += 1
            continue
        if not tokens[0].isdigit():
            continue
        class_id = int(tokens[0])
        words = [t for t in tokens[1:] if _rgb(t) is None]
        if len(words) >= 2 and words[-1].isdigit():
            words = words[:-1]            # trailing numeric flag column some exports carry
        name = " ".join(words).strip()
        rgb = next((_rgb(t) for t in tokens[1:] if _rgb(t)), None)
        if rgb:
            colors[class_id] = rgb
        if name:
            header_ids[class_id] = name

    for class_id in sorted(set(header_ids) | set(colors)):
        result["classes"][class_id] = {"name": header_ids.get(class_id, f"Class {class_id}"),
                                       "color": colors.get(class_id)}
    result["ids"] = sorted(result["classes"])
    if not result["ids"]:
        result["format"] = "unrecognized"
        result["notes"].append("no class definitions could be read")
        return result
    result["format"] = "terrascan" if view_lines else "text-table"
    for class_id in result["ids"]:       # the default bucket: lowest id whose name has no word
        words = result["classes"][class_id]["name"].lower().split()
        if all(w in ("class", "default", "def") or not re.search(r"[a-z]", w) for w in words):
            result["default_class"] = class_id
            break
    return result


def resolve_ptc_mapping(ptc: dict) -> tuple:
    """Internal code -> delivered class id, plus a per-class provenance report."""
    by_norm = {}
    for class_id, info in ptc["classes"].items():
        by_norm.setdefault(_norm(info["name"]), class_id)
        parts = info["name"].split()
        if len(parts) > 1 and parts[0].isupper() and len(parts[0]) <= 5:
            by_norm.setdefault(_norm(" ".join(parts[1:])), class_id)   # "GRD Ground"
    default = ptc["default_class"]
    resolved, report = {}, []
    for code in EMITTABLE_CODES:
        internal = INTERNAL_NAMES[code]
        how, assumed = "name match", False
        forced = PTC_FORCED_TARGETS.get(code)
        if forced is not None:
            alias, reason = forced
            target = default if alias is None else by_norm.get(_norm(alias), default)
            how, assumed = f"forced target ({reason})", True
        else:
            target = by_norm.get(_norm(internal))
            alias, reason = PTC_ALIASES.get(code, (None, None))
            if target is None and alias:
                target = by_norm.get(_norm(alias))
                if target is not None:
                    how, assumed = f"alias '{alias}'", True
            if target is None and code in PTC_ALIASES and alias is None:
                target, how, assumed = default, f"alias default ({reason})", True
            if target is None:
                target, how, assumed = default, "unresolved", True
        resolved[code] = int(target)
        report.append({"code": code, "internal": internal, "target": int(target), "how": how,
                       "assumed": assumed,
                       "client_name": ptc["classes"].get(target, {}).get("name")})
    return resolved, report


def mapping_from_ptc(ptc_path: str) -> dict:
    """Build the engine mapping from a .ptc file. Raises ValueError with a user-facing message."""
    if not ptc_path:
        raise ValueError("--ptc is required")
    if not os.path.isfile(ptc_path):
        raise FileNotFoundError(f"PTC file not found: {ptc_path}")
    ptc = parse_ptc_file(ptc_path)
    if not ptc["ids"]:
        raise ValueError(f"{os.path.basename(ptc_path)} holds no readable class table "
                         f"({'; '.join(ptc['notes']) or ptc['format']}). Load the class "
                         "definition .ptc (a text file) in Display Mode.")
    if ptc["default_class"] is None:
        raise ValueError("the PTC has no unnamed default class (e.g. '0'); points the engine "
                         "cannot place would have nowhere to go")
    resolved, report = resolve_ptc_mapping(ptc)

    def how(code):
        return report[EMITTABLE_CODES.index(code)]["how"]

    missing = [name for code, name in PTC_REQUIRED.items() if how(code) == "unresolved"]
    if missing:
        raise ValueError("the PTC has no class for: " + ", ".join(missing) +
                         ". Class names must match (Ground, Low/Medium/High vegetation, Building).")
    warnings = []
    power_ok = how(14) != "unresolved" and how(15) != "unresolved"
    if not power_ok:
        warnings.append("The PTC has no 'Bare Conductors' / 'Pylons or Poles' class; wire and "
                        "pole extraction is turned off.")
    classes = {"unclassified": resolved[1], "ground": resolved[2], "low_veg": resolved[3],
               "medium_veg": resolved[4], "high_veg": resolved[5], "building": resolved[6],
               "low_noise": resolved[7], "high_noise": resolved[18],
               "wire": resolved[14], "pole": resolved[15]}
    return {"classes": classes, "power_lines_enabled": power_ok,
            "ptc": {"path": os.path.abspath(ptc_path), "md5": ptc["md5"], "format": ptc["format"],
                    "class_count": len(ptc["ids"]), "report": report},
            "warnings": warnings}


def _build_ptc_mapping(classes: dict) -> dict:
    """Host class key -> delivered code, in the rule engine's internal vocabulary.

    Bridge (internal 17) is deliberately absent: Rule N re-files those points before anything
    can deliver them, so an upstream bug cannot leak a class the host never asked for.
    """
    return {
        str(INTERNAL_UNCLASSIFIED): int(classes["unclassified"]),
        str(INTERNAL_GROUND): int(classes["ground"]),
        str(INTERNAL_LOW_VEG): int(classes["low_veg"]),
        str(INTERNAL_MEDIUM_VEG): int(classes["medium_veg"]),
        str(INTERNAL_HIGH_VEG): int(classes["high_veg"]),
        str(INTERNAL_BUILDING): int(classes["building"]),
        str(INTERNAL_LOW_NOISE): int(classes["low_noise"]),
        str(INTERNAL_HIGH_NOISE): int(classes["high_noise"]),
        str(INTERNAL_WATER): int(classes["unclassified"]),
        str(INTERNAL_WIRE): int(classes["wire"]),
        str(INTERNAL_POLE): int(classes["pole"]),
    }

def build_contract(mapping: dict, power_lines=None) -> dict:
    """Turn the host mapping into the rule-engine contract the vendored step 3 expects.

    The internal codes are the rule engine's vocabulary and never change; ptc_class_mapping
    carries the host's delivered codes. See _build_ptc_mapping for the Bridge note.
    """
    with open(CONFIG_PATH, encoding="utf-8") as handle:
        contract = json.load(handle)

    classes = dict(mapping["classes"])
    enabled = bool(mapping["power_lines_enabled"]) if power_lines is None else bool(power_lines)
    ptc = _build_ptc_mapping(classes)

    contract["ptc_class_mapping"] = ptc
    contract["contract_name"] = "Naksha_AI3_contract"
    contract["output_class_set"] = {
        "enforce": True,
        "allowed": sorted(set(ptc.values())),
        "fallback_class": int(classes["unclassified"]),
        "noise_fallback_class": int(classes["low_noise"]),
    }
    contract["network_extraction"]["enabled"] = enabled
    return contract


def write_resolved_contract(contract: dict, output_path: str) -> str:
    """Write the contract beside the output so a delivered file can always be explained."""
    path = output_path + ".contract_resolved.json"
    with open(path, "w", encoding="utf-8") as handle:
        json.dump({k: v for k, v in contract.items() if not k.startswith("_")},
                  handle, indent=2, sort_keys=True)
    return path

# ----------------------------------------------------------------------- legacy LAS formats
def _las14_point_format_for(source_format_id: int) -> int:
    """A LAS 1.4 point format that stores 8-bit classification (0-255)."""
    source_format_id = int(source_format_id)
    if source_format_id in (0, 1):
        return 6
    if source_format_id in (2, 3):
        return 7
    if source_format_id == 4:
        return 9
    if source_format_id == 5:
        return 10
    if source_format_id in (6, 7, 8, 9, 10):
        return source_format_id
    return 7


def ensure_extended_classification(input_path: str, mapping: dict, work_dir: str) -> str:
    """Return a path whose point format can hold every delivered class code.

    LAS point formats 0-5 store classification in 5 bits (0-31). A host schema with a code above
    31 (the ENEL schema's 51 for low points, for one) cannot be written to them, and the rule
    engine's own round-trip check would fail. Such an input is first rewritten as LAS 1.4 with the
    matching extended format; every other dimension and the CRS are carried over unchanged, and
    the delivered file is therefore LAS 1.4 too.  Inputs that already fit are used as they are.
    """
    import laspy
    import numpy as np

    delivered_max = max(int(v) for v in mapping["classes"].values())
    with laspy.open(input_path) as reader:
        source_pf = int(reader.header.point_format.id)
    if source_pf >= 6 or delivered_max <= 31:
        return input_path

    os.makedirs(work_dir, exist_ok=True)
    target = os.path.join(work_dir, "00_input_las14.laz")
    with laspy.open(input_path) as reader:
        src = reader.header
        header = laspy.LasHeader(version="1.4", point_format=_las14_point_format_for(source_pf))
        header.scales = np.asarray(src.scales).copy()
        header.offsets = np.asarray(src.offsets).copy()
        for attr in ("system_identifier", "generating_software", "global_encoding"):
            try:
                setattr(header, attr, getattr(src, attr))
            except Exception:
                pass
        try:
            header.vlrs.extend(src.vlrs)          # CRS and any other VLRs travel with the data
        except Exception:
            pass
        with laspy.open(target, mode="w", header=header, do_compress=True) as writer:
            for chunk in reader.chunk_iterator(2_000_000):
                converted = laspy.ScaleAwarePointRecord.zeros(len(chunk), header=header)
                for name in chunk.point_format.dimension_names:
                    if name in converted.point_format.dimension_names:
                        converted[name] = chunk[name]
                writer.write_points(converted)
    return target

# --------------------------------------------------------------------------------- progress
class _Heartbeat:
    """Keeps the host's progress bar moving while one long step runs.

    Steps 1 and 2 are single calls with no progress of their own, so without this the bar would
    sit still for minutes. A daemon thread emits one PROGRESS line every few seconds, easing from
    the start of the step's band toward - never reaching - its end; the real end-of-step line then
    jumps to the band edge. Pacing uses the measured seconds-per-point of the step.
    """

    def __init__(self, band, message: str, expected_seconds: float, period: float = 4.0):
        self.band = band
        self.message = message
        self.expected = max(float(expected_seconds), 5.0)
        self.period = period
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._t0 = time.time()

    def _run(self):
        lo, hi = self.band
        while not self._stop.wait(self.period):
            frac = 1.0 - math.exp(-(time.time() - self._t0) / self.expected)
            _progress(lo + (hi - lo) * 0.95 * frac,
                      f"{self.message} ({int(time.time() - self._t0)} s)")

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join(timeout=1.0)
        return False

# -------------------------------------------------------------------------------- preflight
def _probe_pdal_filters() -> dict:
    """Run the PDAL stages the pipeline depends on against a tiny synthetic cloud.

    An environment can have python-pdal installed and still lack the filters (a mismatched
    PDAL install), and that only shows up when the pipeline runs - so the self-test runs them.
    """
    import json as _json
    import numpy as np
    import laspy
    import pdal

    header = laspy.LasHeader(version="1.2", point_format=3)
    header.scales = np.array([0.01, 0.01, 0.01])
    header.offsets = np.array([0.0, 0.0, 0.0])
    las = laspy.LasData(header)
    rng = np.zeros(0)
    import numpy as _np
    rng = _np.random.default_rng(0)
    n = 600
    las.x = rng.uniform(0.0, 20.0, n)
    las.y = rng.uniform(0.0, 20.0, n)
    las.z = rng.uniform(0.0, 1.0, n) + 0.05 * (las.x // 5)
    las.classification = _np.zeros(n, dtype=_np.uint8)

    with tempfile.TemporaryDirectory(prefix="naksha_ai3_probe_") as tmp:
        src = os.path.join(tmp, "probe.las")
        dst = os.path.join(tmp, "probe_out.las")
        las.write(src)
        stages = [
            {"type": "readers.las", "filename": src},
            {"type": "filters.outlier", "method": "statistical", "mean_k": 8,
             "multiplier": 3.0},
            {"type": "filters.smrf", "ignore": "Classification[7:7]", "slope": 0.2,
             "threshold": 0.45, "window": 16.0, "scalar": 1.2},
            {"type": "filters.hag_delaunay", "count": 10, "allow_extrapolation": True},
            {"type": "filters.assign",
             "value": "HeightAboveGround = 0 WHERE HeightAboveGround < 0 "
                      "&& HeightAboveGround > -0.5"},
            {"type": "writers.las", "filename": dst,
             "extra_dims": "HeightAboveGround=float32", "forward": "all"},
        ]
        pipeline = pdal.Pipeline(_json.dumps(stages))
        pipeline.execute()
        with laspy.open(dst) as reader:
            count = int(reader.header.point_count)
            dims = set(reader.header.point_format.dimension_names)
        if count != n:
            raise RuntimeError(f"probe wrote {count} of {n} points")

        if "HeightAboveGround" not in dims:
            raise RuntimeError("probe produced no HeightAboveGround dimension")
    return {"probe_points": n, "outlier": True, "smrf": True, "hag_delaunay": True,
            "assign": True, "writers_las": True, "readers_las": True}


def _resolve_device(device: str, cuda_available: bool) -> str:
    value = str(device or "auto").lower()
    if value == "cuda" and not cuda_available:
        raise RuntimeError("device=cuda was requested but CUDA is not available to PyTorch")
    if value == "auto":
        return "cuda" if cuda_available else "cpu"
    return value

def check_environment(device: str = "auto") -> dict:
    """Import everything the pipeline needs and report what was found.

    Raises RuntimeError with an actionable message when a requirement is missing, so the host
    can offer to rebuild the runtime environment instead of showing a traceback.
    """
    report = {"engine_version": ENGINE_VERSION, "python": sys.version.split()[0]}
    problems = []

    for name in ("numpy", "scipy", "laspy"):
        try:
            module = __import__(name)
            report[name] = str(getattr(module, "__version__", "unknown"))
        except Exception as exc:
            problems.append(f"{name} import failed: {exc}")

    try:
        import torch
        report["torch"] = torch.__version__
        cuda = bool(torch.cuda.is_available())
        report["cuda_available"] = cuda
        if cuda:
            report["cuda_device"] = torch.cuda.get_device_name(0)
        report["device"] = _resolve_device(device, cuda)
    except Exception as exc:
        problems.append(f"torch import failed: {exc}")

    try:
        import torch_geometric
        report["torch_geometric"] = torch_geometric.__version__
    except Exception as exc:
        problems.append(f"torch_geometric import failed: {exc}")
    try:
        import torch_cluster  # noqa: F401
        report["torch_cluster"] = True
    except Exception as exc:
        problems.append(f"torch_cluster import failed: {exc}")

    for path, label in ((CONFIG_PATH, "contract spec"),
                        (CHECKPOINT_PATH, "model checkpoint"),
                        (INFERENCE_CONFIG_PATH, "inference config"),
                        (os.path.join(MYRIA3D_DIR, "run.py"), "Myria3D entry point")):
        if not os.path.isfile(path):
            problems.append(f"{label} missing: {path}")

    if problems:
        raise RuntimeError("; ".join(problems))
    return report

# --------------------------------------------------------------------------------- pipeline
def run_pipeline(input_path: str, output_path: str, mapping: dict, work_dir: str = None,
                 power_lines=None, device: str = "auto", keep_work: bool = False,
                 ground_unclass_pct=None) -> dict:
    """Run steps 1-3 and write the delivered LAZ. Returns the run report."""
    import step1_preprocess
    import step2_inference
    import step3_rule_engine

    contract = build_contract(mapping, power_lines)
    work_dir = os.path.abspath(work_dir or os.path.join(os.path.dirname(output_path),
                                                        "_ai3_work"))
    os.makedirs(work_dir, exist_ok=True)
    # The files handed between the steps are throw-away: uncompressed .las skips a LAZ compress and
    # decompress at every hand-over (about 10 % of the run on a large tile) and holds the same data.
    step1_out = os.path.join(work_dir, "01_normalized.las")
    step2_out = os.path.join(work_dir, "02_ai_predicted.las")
    scratch = os.path.join(work_dir, "scratch")

    timings = {}
    started = time.time()

    pipeline_input = ensure_extended_classification(input_path, mapping, work_dir)
    n_points = _point_count(pipeline_input)

    _progress(BAND_STEP1[0], "Step 1/3: PDAL ground filter and height normalization "
                             f"({n_points:,} points)...")
    t0 = time.time()
    with _Heartbeat(BAND_STEP1, "Step 1/3: ground filter and height normalization",
                    n_points * SECONDS_PER_POINT["step1"]):
        step1_preprocess.run_step1_preprocessing(pipeline_input, step1_out)
    timings["step1"] = time.time() - t0
    _progress(BAND_STEP1[1], f"Step 1/3 complete in {timings['step1']:.1f} s")

    warnings_out = list(mapping.get("warnings", []))
    for text in warnings_out:
        print("WARNING " + text, flush=True)
    try:
        missing_colour, _present = step2_inference.missing_color_dims(step1_out)
    except Exception:
        missing_colour = []
    if missing_colour:
        # The checkpoint was trained on coloured clouds. Step 2 fills missing channels with a
        # neutral constant, which runs, but is far less reliable - the host should say so.
        message = ("Input has no usable colour channel(s) " + ", ".join(missing_colour)
                   + "; the model was trained on coloured point clouds, so results will be "
                     "less reliable")
        warnings_out.append(message)
        print("WARNING " + message, flush=True)

    _progress(BAND_STEP2[0], "Step 2/3: Myria3D / RandLA-Net inference...")
    t0 = time.time()
    with _Heartbeat(BAND_STEP2, "Step 2/3: AI inference", n_points * SECONDS_PER_POINT["step2"]):
        step2_inference.run_step2_inference(step1_out, step2_out, work_dir=scratch)
    timings["step2"] = time.time() - t0
    _progress(BAND_STEP2[1], f"Step 2/3 complete in {timings['step2']:.1f} s")

    _progress(BAND_STEP3[0], "Step 3/3: contract rule engine and class mapping...")
    contract_path = write_resolved_contract(contract, output_path)
    t0 = time.time()
    with _Heartbeat(BAND_STEP3, "Step 3/3: rule engine", n_points * SECONDS_PER_POINT["step3"]):
        result = step3_rule_engine.run_step3_rule_engine(step2_out, output_path, contract_path,
                                                    ground_unclass_pct=ground_unclass_pct)
    timings["step3"] = time.time() - t0
    _progress(BAND_STEP3[1], f"Step 3/3 complete in {timings['step3']:.1f} s")

    _progress(BAND_WRITE[0], "Verifying the delivered file...")
    histogram = _class_histogram(output_path)
    total = _point_count(output_path)
    allowed = contract["output_class_set"]["allowed"]
    stray = [c for c in histogram if c not in allowed]
    if stray:
        raise RuntimeError(f"delivered class(es) {stray} are outside the host's target set "
                           f"{allowed}")
    _progress(BAND_WRITE[1], f"Delivered {total:,} points, {len(histogram)} classes")

    timings["total"] = time.time() - started
    if not keep_work:
        shutil.rmtree(work_dir, ignore_errors=True)
    report = {
        "engine": ENGINE_NAME,
        "engine_version": ENGINE_VERSION,
        "status": "COMPLETED",
        "warnings": warnings_out,
        "input": os.path.abspath(input_path),
        "output": os.path.abspath(output_path),
        "points": int(total),
        "class_histogram": histogram,
        "timings": {k: round(v, 2) for k, v in timings.items()},
        "power_lines_enabled": bool(contract["network_extraction"]["enabled"]),
        "mapping": {k: v for k, v in mapping.items() if k not in ("ptc", "warnings")},
        "ptc": mapping.get("ptc"),
        "contract": contract_path,
        "contract_md5": _md5(contract_path),
        "finished": datetime.datetime.now().isoformat(timespec="seconds"),
        "rule_timings": result.get("rule_timings"),
        "network_extraction": result.get("network_extraction"),
        "class_0_breakdown": result.get("class_0_breakdown"),
        "output_class_set": result.get("output_class_set"),
    }
    print("RESULT " + json.dumps(report, sort_keys=True), flush=True)
    return report

# ------------------------------------------------------------------------------------- main
def self_test(device: str = "auto") -> int:
    """Verify the environment without touching any data."""
    try:
        report = check_environment(device)
    except Exception as exc:
        return _fail(str(exc), EXIT_ENVIRONMENT)

    for key in sorted(report):
        value = report[key]
        if isinstance(value, dict):
            value = json.dumps(value, sort_keys=True)
        print(f"SELFTEST {key}={value}", flush=True)
    try:
        probe = _probe_pdal_filters()
    except Exception as exc:
        return _fail(f"PDAL probe failed: {type(exc).__name__}: {exc}", EXIT_ENVIRONMENT)
    print("SELFTEST pdal_probe=" + json.dumps(probe, sort_keys=True), flush=True)
    files = {
        "contract_spec_md5": _md5(CONFIG_PATH),
        "checkpoint_md5": _md5(CHECKPOINT_PATH),
        "inference_config_md5": _md5(INFERENCE_CONFIG_PATH),
    }
    print("SELFTEST files=" + json.dumps(files, sort_keys=True), flush=True)
    print("SELFTEST OK", flush=True)
    return EXIT_OK

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog=ENGINE_NAME,
        description="Isolated AI model 3.0 LiDAR classification engine")
    parser.add_argument("--input", help="input LAS/LAZ")
    parser.add_argument("--output", help="delivered LAS/LAZ (input with classification changed)")
    parser.add_argument("--ptc", help="class-definition .ptc file: the delivered class schema")
    parser.add_argument("--mapping", help="mapping JSON (alternative to --ptc, for tests)")
    parser.add_argument("--work-dir", default=None,
                        help="scratch folder for intermediate files (default: beside output)")
    parser.add_argument("--no-power-lines", action="store_true",
                        help="disable wire/pole extraction regardless of the mapping")
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    parser.add_argument("--ground-unclass-pct", type=float, default=None,
                        help="share of the Ground points set to Unclassified at random, 0-100 "
                             "(default: the contract's 20)")
    parser.add_argument("--keep-work", action="store_true",
                        help="keep the intermediate files (default: delete them on success)")
    parser.add_argument("--selftest", action="store_true",
                        help="verify the environment and exit without touching data")
    parser.add_argument("--version", action="store_true")
    args = parser.parse_args(argv)

    if args.version:
        print(f"{ENGINE_NAME} {ENGINE_VERSION}", flush=True)
        return EXIT_OK

    # --device cpu has to take effect before torch is imported anywhere (step 2 also runs
    # Myria3D in a child process, which inherits the environment).
    if args.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    # A host cancel arrives as CTRL_BREAK / SIGTERM: unwind through the normal error path so
    # the exit code is 130 and the work folder is cleaned.
    def _on_signal(signum, _frame):
        raise KeyboardInterrupt()
    for sig_name in ("SIGTERM", "SIGBREAK", "SIGINT"):
        sig = getattr(signal, sig_name, None)
        if sig is not None:
            try:
                signal.signal(sig, _on_signal)
            except Exception:
                pass

    if args.selftest:
        return self_test(args.device)

    if not args.input or not args.output or bool(args.ptc) == bool(args.mapping):
        return _fail("--input, --output and exactly one of --ptc / --mapping are required",
                     EXIT_BAD_ARGS)
    if not os.path.isfile(args.input):
        return _fail(f"input file not found: {args.input}", EXIT_BAD_ARGS)
    if os.path.splitext(args.input)[1].lower() not in (".las", ".laz"):
        return _fail(f"input must be .las or .laz: {args.input}", EXIT_BAD_ARGS)
    if os.path.abspath(args.input) == os.path.abspath(args.output):
        return _fail("--output must differ from --input; the engine never edits the source",
                     EXIT_BAD_ARGS)

    try:
        mapping = mapping_from_ptc(args.ptc) if args.ptc else load_mapping(args.mapping)
    except Exception as exc:
        return _fail(str(exc), EXIT_BAD_ARGS)

    try:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    except Exception as exc:
        return _fail(f"cannot create the output folder: {exc}", EXIT_BAD_ARGS)

    power_lines = None if not args.no_power_lines else False
    if args.ground_unclass_pct is not None and not 0.0 <= args.ground_unclass_pct <= 100.0:
        return _fail("--ground-unclass-pct must be between 0 and 100", EXIT_BAD_ARGS)

    try:
        check_environment(args.device)
    except Exception as exc:
        return _fail(f"runtime environment is not usable: {exc}", EXIT_ENVIRONMENT)

    try:
        if os.path.exists(args.output):
            os.remove(args.output)
    except Exception:
        pass

    try:
        run_pipeline(args.input, args.output, mapping, work_dir=args.work_dir,
                     power_lines=power_lines, device=args.device, keep_work=args.keep_work,
                     ground_unclass_pct=args.ground_unclass_pct)
    except KeyboardInterrupt:
        shutil.rmtree(args.work_dir or os.path.join(os.path.dirname(os.path.abspath(args.output)),
                                                    "_ai3_work"), ignore_errors=True)
        return _fail("cancelled by user", EXIT_CANCELLED)
    except Exception as exc:
        print(traceback.format_exc(), file=sys.stderr, flush=True)
        return _fail(f"pipeline failed: {type(exc).__name__}: {exc}", EXIT_STEP_FAILED)
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
