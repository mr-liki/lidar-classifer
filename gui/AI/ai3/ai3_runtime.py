"""Locate, verify and describe the AI model 3.0 runtime environment.

The engine (``engine/naksha_ai_engine.py``) cannot run in the Naksha app's own interpreter - it
needs PDAL, torch 2.4, torch-geometric and numpy 1.26, while the app ships numpy 2.x and
torch 2.7 - so it runs in a separate Python environment that lives on the user's machine and is
created by ``setup_ai3_runtime.py``.  This module is the one place that knows where that
environment is and whether it is usable.  It uses the standard library only, so it is safe to
import from the GUI before any heavy dependency is loaded.

Resolution order for the interpreter that runs the engine:

1. ``NAKSHA_AI3_PYTHON``  - an explicit python.exe (development machines, managed installs).
2. ``<runtime root>/runtime.json`` field ``python`` - written by the setup script.
3. ``<runtime root>/env/python.exe`` - the environment the setup script creates.

The runtime root is ``NAKSHA_AI3_RUNTIME`` if set, else ``%LOCALAPPDATA%\\NakshaAI\\ai3_runtime``
(never the install folder: it may not be writable and the environment is several GB).
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

PACKAGE_DIR = Path(__file__).resolve().parent
ENGINE_DIR = PACKAGE_DIR / "engine"
ENGINE_SCRIPT = ENGINE_DIR / "naksha_ai_engine.py"

ENV_RUNTIME = "NAKSHA_AI3_RUNTIME"
ENV_PYTHON = "NAKSHA_AI3_PYTHON"

RUNTIME_SCHEMA = 1
# Files whose content decides whether a recorded self-test is still valid: if the engine or the
# model changes, the environment is verified again instead of trusting an old result.
_FINGERPRINT_FILES = (
    "naksha_ai_engine.py",
    "step1_preprocess.py",
    "step2_inference.py",
    "step3_rule_engine.py",
    os.path.join("config", "contract_spec.json"),
    os.path.join("models", "inference_config.yaml"),
)


def _no_window() -> int:
    return getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0


def runtime_root() -> Path:
    override = os.environ.get(ENV_RUNTIME)
    if override:
        return Path(override).expanduser()
    base = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
    return Path(base) / "NakshaAI" / "ai3_runtime"


def runtime_json_path(root: Optional[Path] = None) -> Path:
    return (root or runtime_root()) / "runtime.json"


def read_runtime_json(root: Optional[Path] = None) -> dict:
    path = runtime_json_path(root)
    try:
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def write_runtime_json(data: dict, root: Optional[Path] = None) -> None:
    path = runtime_json_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)
    os.replace(tmp, path)


def engine_fingerprint() -> str:
    """Hash of the engine files and the model checkpoint's size (the checkpoint is 14 MB; its
    size plus the config hash is enough to notice a swapped model without reading it each time)."""
    digest = hashlib.sha256()
    for rel in _FINGERPRINT_FILES:
        path = ENGINE_DIR / rel
        digest.update(rel.encode())
        try:
            digest.update(path.read_bytes())
        except OSError:
            digest.update(b"<missing>")
    ckpt = ENGINE_DIR / "models" / "fractal_lidarhd_myria3d.ckpt"
    try:
        digest.update(str(ckpt.stat().st_size).encode())
    except OSError:
        digest.update(b"<missing>")
    return digest.hexdigest()


def python_path(root: Optional[Path] = None) -> Optional[Path]:
    """The interpreter that should run the engine, or None when there is no candidate."""
    override = os.environ.get(ENV_PYTHON)
    if override:
        candidate = Path(override).expanduser()
        return candidate if candidate.is_file() else None
    root = root or runtime_root()
    recorded = read_runtime_json(root).get("python")
    if recorded and Path(recorded).is_file():
        return Path(recorded)
    candidate = root / "env" / ("python.exe" if os.name == "nt" else "bin/python")
    return candidate if candidate.is_file() else None


@dataclass
class RuntimeStatus:
    """What the GUI needs to decide between 'run', 'set up' and 'repair'."""

    ok: bool = False
    state: str = "missing"          # missing | incomplete | unverified | ready | broken
    message: str = ""
    root: Path = field(default_factory=runtime_root)
    python: Optional[Path] = None
    engine_ok: bool = True
    details: dict = field(default_factory=dict)


def engine_files_missing() -> List[str]:
    needed = [
        ENGINE_SCRIPT,
        ENGINE_DIR / "step1_preprocess.py",
        ENGINE_DIR / "step2_inference.py",
        ENGINE_DIR / "step3_rule_engine.py",
        ENGINE_DIR / "config" / "contract_spec.json",
        ENGINE_DIR / "models" / "fractal_lidarhd_myria3d.ckpt",
        ENGINE_DIR / "models" / "inference_config.yaml",
        ENGINE_DIR / "myria3d" / "run.py",
    ]
    return [str(p) for p in needed if not p.is_file()]


def runtime_status(verify_cached: bool = True) -> RuntimeStatus:
    """Cheap status check (no subprocess): is there an interpreter, and was it verified?"""
    status = RuntimeStatus()
    missing = engine_files_missing()
    if missing:
        status.engine_ok = False
        status.state = "broken"
        status.message = "AI model 3.0 engine files are missing: " + ", ".join(missing)
        return status

    py = python_path(status.root)
    status.python = py
    record = read_runtime_json(status.root)
    status.details = record
    if py is None:
        status.state = "missing"
        status.message = ("The AI model 3.0 runtime is not installed on this computer. "
                          "It must be created once (about 8 GB, needs internet).")
        return status
    if os.environ.get(ENV_PYTHON):
        # An explicit interpreter is trusted to be set up by whoever supplied it; it is still
        # verified by verify_runtime() before the first run.
        status.state = "unverified"
        status.message = f"Using {ENV_PYTHON}={py}; not verified yet."
    elif not record.get("setup_complete"):
        status.state = "incomplete"
        status.message = "A previous AI model 3.0 setup did not finish; it can be resumed."
        return status
    else:
        status.state = "unverified"
        status.message = "Runtime found; not verified for this engine version yet."

    if verify_cached and record.get("selftest_ok") and \
            record.get("engine_fingerprint") == engine_fingerprint() and \
            str(record.get("selftest_python", "")).lower() == str(py).lower():
        status.state = "ready"
        status.ok = True
        status.message = "AI model 3.0 runtime is ready."
    return status


def verify_runtime(timeout: float = 600.0, device: str = "auto") -> RuntimeStatus:
    """Run ``naksha_ai_engine.py --selftest`` and record the outcome next to the runtime.

    The self-test imports the full dependency stack, runs the PDAL filters on a tiny synthetic
    cloud and checks the model files, so a True result means the next classification will not
    fail on the environment.  It takes ~20 s the first time and is cached afterwards (keyed on
    the interpreter and the engine fingerprint).
    """
    status = runtime_status(verify_cached=False)
    if status.python is None or not status.engine_ok:
        return status

    cmd = [str(status.python), str(ENGINE_SCRIPT), "--selftest", "--device", device]
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    started = time.time()
    try:
        proc = subprocess.run(cmd, cwd=str(ENGINE_DIR), capture_output=True, text=True,
                              encoding="utf-8", errors="replace", timeout=timeout, env=env,
                              creationflags=_no_window())
        out, err, code = proc.stdout or "", proc.stderr or "", proc.returncode
    except subprocess.TimeoutExpired:
        status.state = "broken"
        status.message = f"The AI model 3.0 self-test did not finish within {timeout:.0f} s."
        return status
    except OSError as exc:
        status.state = "broken"
        status.message = f"Could not start the AI model 3.0 runtime: {exc}"
        return status

    details = {}
    for line in out.splitlines():
        if line.startswith("SELFTEST ") and "=" in line:
            key, _, value = line[len("SELFTEST "):].partition("=")
            details[key.strip()] = value.strip()
    ok = code == 0 and any(line.strip() == "SELFTEST OK" for line in out.splitlines())

    record = read_runtime_json(status.root)
    record.update({
        "schema": RUNTIME_SCHEMA,
        "selftest_ok": bool(ok),
        "selftest_python": str(status.python),
        "selftest_seconds": round(time.time() - started, 1),
        "selftest_details": details,
        "engine_fingerprint": engine_fingerprint() if ok else "",
    })
    try:
        write_runtime_json(record, status.root)
    except OSError:
        pass                                  # a read-only location must not break a good run

    status.details = record
    if ok:
        status.ok = True
        status.state = "ready"
        status.message = "AI model 3.0 runtime is ready."
    else:
        status.ok = False
        status.state = "broken"
        reason = [ln[len("ERROR "):] for ln in err.splitlines() if ln.startswith("ERROR ")]
        status.message = ("The AI model 3.0 runtime failed its self-test: "
                          + ("; ".join(reason) if reason else (err.strip()[-500:] or f"exit {code}")))
    return status


def gpu_description(status: Optional[RuntimeStatus] = None) -> str:
    details = (status.details if status else read_runtime_json()).get("selftest_details", {})
    if details.get("cuda_available", "").lower() == "true":
        return details.get("cuda_device", "CUDA GPU")
    return "CPU (no CUDA GPU found - classification will be slow)"


if __name__ == "__main__":      # python ai3_runtime.py  ->  one-line status for support
    st = verify_runtime() if "--verify" in sys.argv else runtime_status()
    print(f"{st.state}: {st.message}")
    print(f"root={st.root} python={st.python}")
    sys.exit(0 if st.ok else 1)
