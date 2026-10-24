"""Create the AI model 3.0 runtime environment on this computer.

    python setup_ai3_runtime.py                 # install (resumes an unfinished setup)
    python setup_ai3_runtime.py --check         # report what would be done, change nothing
    python setup_ai3_runtime.py --repair        # delete the environment and build it again
    python setup_ai3_runtime.py --variant cpu   # force the CPU build (default: auto-detect)

The environment holds PDAL, torch 2.4, torch-geometric and the rest of the engine's stack (see
ai3_environment.yml and ai3_requirements.txt).  It is built where ``ai3_runtime.runtime_root()``
points - ``%LOCALAPPDATA%\\NakshaAI\\ai3_runtime`` unless NAKSHA_AI3_RUNTIME says otherwise - and
needs about 8 GB of disk and an internet connection while it is built.  Nothing is installed
outside that folder and the app's own interpreter is never touched.

Stages (each recorded in runtime.json, so an interrupted setup resumes where it stopped):
    1. preflight   disk space, network, GPU / driver detection
    2. micromamba  pinned release, SHA-256 verified
    3. conda       python 3.10 + PDAL from conda-forge
    4. pip         pinned Python packages (CUDA 12.1 or CPU torch)
    5. selftest    the engine's own --selftest, with a PDAL probe on synthetic data

Standard library only: it runs under any Python 3.8+, and the GUI imports it directly.
Progress is reported as ``SETUP <percent> <message>`` lines on stdout, or through ``progress``
when called from Python.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable, Optional

try:                                              # imported by the GUI as a package module
    from . import ai3_runtime as rt
except ImportError:                               # run as a script
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import ai3_runtime as rt                      # type: ignore

PACKAGE_DIR = Path(__file__).resolve().parent
ENVIRONMENT_YML = PACKAGE_DIR / "ai3_environment.yml"
REQUIREMENTS_TXT = PACKAGE_DIR / "ai3_requirements.txt"

MICROMAMBA_VERSION = "2.3.3-0"
MICROMAMBA_URL = ("https://github.com/mamba-org/micromamba-releases/releases/download/"
                  f"{MICROMAMBA_VERSION}/micromamba-win-64")
MICROMAMBA_SHA256 = "90085767d811e08a0fe240f7cb7a713915e1e5f24e3b0acfad9b37daa64c98fe"
ENV_MICROMAMBA = "NAKSHA_AI3_MICROMAMBA"        # use an existing micromamba.exe instead

# Disk needed on the runtime drive while building (GB). The CUDA torch wheel alone is ~2.5 GB
# and unpacks to ~4.5 GB; conda adds ~1.5 GB and pip needs room for the wheel while it installs.
MIN_FREE_GB = {"cuda": 11.0, "cpu": 6.0}
# NVIDIA driver needed by the CUDA 12.1 wheels on Windows.
MIN_DRIVER_CUDA121 = (527, 41)

PROBE_URLS = (
    "https://conda.anaconda.org/conda-forge/noarch/repodata_version.json",
    "https://pypi.org/simple/pip/",
    "https://download.pytorch.org/whl/cu121/",
)

Progress = Callable[[int, str], None]


class SetupError(RuntimeError):
    """A setup failure with a message fit to show the user."""


class SetupCancelled(SetupError):
    pass


def _no_window() -> int:
    return getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0


def _default_progress(percent: int, message: str) -> None:
    print(f"SETUP {int(percent)} {message}", flush=True)


# ------------------------------------------------------------------------------- preflight
def detect_variant() -> tuple[str, str]:
    """('cuda' | 'cpu', reason). CUDA only when an NVIDIA driver new enough for CUDA 12.1 exists."""
    exe = shutil.which("nvidia-smi")
    if not exe and os.name == "nt":
        candidate = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32" / "nvidia-smi.exe"
        exe = str(candidate) if candidate.is_file() else None
    if not exe:
        return "cpu", "no NVIDIA driver found (nvidia-smi is missing)"
    try:
        out = subprocess.run([exe, "--query-gpu=driver_version,name", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=20,
                             creationflags=_no_window()).stdout.strip().splitlines()
    except (OSError, subprocess.SubprocessError):
        return "cpu", "nvidia-smi could not be run"
    if not out:
        return "cpu", "nvidia-smi reported no GPU"
    driver, _, name = out[0].partition(",")
    try:
        parts = tuple(int(p) for p in driver.strip().split(".")[:2])
    except ValueError:
        return "cpu", f"unreadable NVIDIA driver version {driver!r}"
    if parts < MIN_DRIVER_CUDA121:
        return "cpu", (f"NVIDIA driver {driver.strip()} is older than "
                       f"{MIN_DRIVER_CUDA121[0]}.{MIN_DRIVER_CUDA121[1]} (needed for CUDA 12.1)")
    return "cuda", f"{name.strip()} (driver {driver.strip()})"


def free_gb(path: Path) -> float:
    probe = path
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    return shutil.disk_usage(probe).free / (1024 ** 3)


def check_network(timeout: float = 15.0) -> Optional[str]:
    """None when every download host answers, else a message naming the first that does not."""
    for url in PROBE_URLS:
        try:
            request = urllib.request.Request(url, method="HEAD",
                                             headers={"User-Agent": "naksha-ai3-setup"})
            with urllib.request.urlopen(request, timeout=timeout):
                pass
        except urllib.error.HTTPError as exc:
            if exc.code >= 500:
                return f"{url} answered with HTTP {exc.code}"
        except Exception as exc:                         # offline, DNS, proxy, TLS ...
            return f"cannot reach {url.split('/')[2]} ({type(exc).__name__}: {exc})"
    return None


# ------------------------------------------------------------------------------- helpers
def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _kill_tree(proc: subprocess.Popen) -> None:
    try:
        if os.name == "nt":
            subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                           capture_output=True, creationflags=_no_window())
        else:
            proc.kill()
    except Exception:
        pass


class _Runner:
    """Runs one long command, logging every line and reporting progress."""

    def __init__(self, log_path: Path, cancel: threading.Event):
        self.log_path = log_path
        self.cancel = cancel

    def run(self, cmd, cwd=None, env=None, on_line: Optional[Callable[[str], None]] = None) -> int:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.log_path, "a", encoding="utf-8", errors="replace") as log:
            log.write(f"\n$ {' '.join(map(str, cmd))}\n")
            log.flush()
            proc = subprocess.Popen(list(map(str, cmd)), cwd=cwd, env=env,
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                    encoding="utf-8", errors="replace",
                                    creationflags=_no_window())
            try:
                assert proc.stdout is not None
                for raw in proc.stdout:
                    line = raw.rstrip()
                    log.write(line + "\n")
                    if on_line and line:
                        on_line(line)
                    if self.cancel.is_set():
                        _kill_tree(proc)
                        raise SetupCancelled("Setup was cancelled.")
                return proc.wait()
            finally:
                if proc.poll() is None:
                    _kill_tree(proc)


# ------------------------------------------------------------------------------- stages
def ensure_micromamba(root: Path, progress: Progress, cancel: threading.Event) -> Path:
    provided = os.environ.get(ENV_MICROMAMBA)
    if provided:
        exe = Path(provided)
        if not exe.is_file():
            raise SetupError(f"{ENV_MICROMAMBA} points to a missing file: {exe}")
        return exe

    exe = root / "bin" / "micromamba.exe"
    if exe.is_file() and _sha256(exe) == MICROMAMBA_SHA256:
        return exe

    exe.parent.mkdir(parents=True, exist_ok=True)
    tmp = exe.with_suffix(".download")
    progress(4, f"Downloading micromamba {MICROMAMBA_VERSION}...")
    last_error = None
    for attempt in range(1, 4):
        try:
            request = urllib.request.Request(MICROMAMBA_URL,
                                             headers={"User-Agent": "naksha-ai3-setup"})
            with urllib.request.urlopen(request, timeout=60) as response, open(tmp, "wb") as out:
                while True:
                    if cancel.is_set():
                        raise SetupCancelled("Setup was cancelled.")
                    chunk = response.read(1 << 20)
                    if not chunk:
                        break
                    out.write(chunk)
            if _sha256(tmp) != MICROMAMBA_SHA256:
                raise SetupError("The downloaded micromamba does not match its expected "
                                 "SHA-256 checksum; refusing to run it.")
            os.replace(tmp, exe)
            return exe
        except SetupError:
            tmp.unlink(missing_ok=True)
            raise
        except SetupCancelled:
            tmp.unlink(missing_ok=True)
            raise
        except Exception as exc:
            last_error = exc
            time.sleep(2 * attempt)
    tmp.unlink(missing_ok=True)
    raise SetupError(f"Could not download micromamba: {last_error}")


def _clean_env() -> dict:
    env = os.environ.copy()
    for key in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV", "PIP_REQUIRE_VIRTUALENV", "PIP_USER",
                "PIP_TARGET", "CONDA_PREFIX", "CONDA_DEFAULT_ENV"):
        env.pop(key, None)
    env["PYTHONNOUSERSITE"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    env["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
    return env


def stage_conda(root: Path, mamba: Path, progress: Progress, runner: _Runner) -> Path:
    env_dir = root / "env"
    env = _clean_env()
    env["MAMBA_ROOT_PREFIX"] = str(root / "mamba")
    env["MAMBA_NO_BANNER"] = "1"
    progress(8, "Creating the Python environment (PDAL, Python 3.10)...")

    seen = {"n": 0}

    def on_line(line: str) -> None:
        seen["n"] += 1
        # micromamba prints one line per package as it downloads / links
        pct = 8 + min(26, seen["n"] // 4)
        if re.search(r"(Downloading|Extracting|Linking|Transaction|Preparing)", line):
            progress(pct, "Installing PDAL and Python: " + line.strip()[:70])

    code = runner.run([mamba, "create", "-y", "-p", env_dir, "-f", ENVIRONMENT_YML,
                       "--override-channels", "-c", "conda-forge", "--no-rc"],
                      env=env, on_line=on_line)
    py = env_dir / "python.exe"
    if code != 0 or not py.is_file():
        raise SetupError("Creating the conda environment failed "
                         f"(exit {code}). See {runner.log_path} for details.")
    return py


def _requirements_for(variant: str, tmp_dir: Path) -> Path:
    if variant == "cuda":
        return REQUIREMENTS_TXT
    text = REQUIREMENTS_TXT.read_text(encoding="utf-8")
    text = text.replace("whl/cu121", "whl/cpu").replace("torch-2.4.0+cu121", "torch-2.4.0+cpu")
    text = text.replace("pt24cu121", "pt24cpu").replace("+cu121", "+cpu")
    out = tmp_dir / "ai3_requirements_cpu.txt"
    out.write_text(text, encoding="utf-8")
    return out


def stage_pip(root: Path, py: Path, variant: str, progress: Progress, runner: _Runner) -> None:
    requirements = _requirements_for(variant, root)
    total = sum(1 for ln in requirements.read_text(encoding="utf-8").splitlines()
                if ln.strip() and not ln.lstrip().startswith(("#", "-")))
    progress(35, f"Installing the Python packages ({total} packages, {variant.upper()} torch)...")
    state = {"seen": 0}

    def on_line(line: str) -> None:
        if line.startswith(("Collecting ", "Downloading ", "Using cached ")):
            state["seen"] += 1 if line.startswith("Collecting ") else 0
            pct = 35 + min(40, int(40 * state["seen"] / max(total, 1)))
            progress(pct, line.split(" (from")[0].strip()[:80])
        elif line.startswith("Installing collected packages"):
            progress(78, "Installing the downloaded packages...")
        elif line.startswith("Successfully installed"):
            progress(92, "Python packages installed.")

    code = runner.run([py, "-m", "pip", "install", "--no-cache-dir", "--retries", "5",
                       "--timeout", "60", "-r", requirements],
                      env=_clean_env(), on_line=on_line)
    if code != 0:
        raise SetupError("Installing the Python packages failed "
                         f"(exit {code}). See {runner.log_path} for details.")


# ------------------------------------------------------------------------------- driver
def plan(root: Optional[Path] = None, variant: str = "auto") -> dict:
    """What a setup would do here; changes nothing."""
    root = root or rt.runtime_root()
    chosen, reason = detect_variant() if variant == "auto" else (variant, "chosen by the user")
    record = rt.read_runtime_json(root)
    return {
        "root": str(root),
        "variant": chosen,
        "variant_reason": reason,
        "free_gb": round(free_gb(root), 1),
        "needed_gb": MIN_FREE_GB[chosen],
        "enough_space": free_gb(root) >= MIN_FREE_GB[chosen],
        "conda_done": bool(record.get("conda_done")),
        "pip_done": bool(record.get("pip_done")),
        "setup_complete": bool(record.get("setup_complete")),
        "engine_files_missing": rt.engine_files_missing(),
        "network_problem": check_network(),
    }


def run_setup(root: Optional[Path] = None, variant: str = "auto", repair: bool = False,
              progress: Optional[Progress] = None,
              cancel: Optional[threading.Event] = None,
              min_free_gb: Optional[float] = None) -> rt.RuntimeStatus:
    """Create (or resume, or repair) the runtime. Raises SetupError with a user-facing message."""
    progress = progress or _default_progress
    cancel = cancel or threading.Event()
    root = Path(root) if root else rt.runtime_root()
    # The runtime root is where rt.runtime_root() looks; keep both views consistent.
    os.environ[rt.ENV_RUNTIME] = str(root)

    missing = rt.engine_files_missing()
    if missing:
        raise SetupError("The AI model 3.0 engine files are missing from this installation: "
                         + ", ".join(missing))

    progress(0, "Checking this computer...")
    chosen, reason = detect_variant() if variant == "auto" else (variant, "chosen by the user")
    need = MIN_FREE_GB[chosen] if min_free_gb is None else float(min_free_gb)
    root.mkdir(parents=True, exist_ok=True)
    have = free_gb(root)
    record = rt.read_runtime_json(root)
    if repair:
        progress(1, "Removing the previous environment...")
        shutil.rmtree(root / "env", ignore_errors=True)
        record = {}
    already = bool(record.get("conda_done")) or bool(record.get("pip_done"))
    if have < need and not already:
        raise SetupError(f"Not enough free disk space on {root.anchor or root}: {have:.1f} GB free, "
                         f"{need:.0f} GB needed for the {chosen.upper()} build. Free some space, "
                         f"or set {rt.ENV_RUNTIME} to a folder on another drive.")
    progress(2, f"Using the {chosen.upper()} build: {reason}.")

    problem = check_network()
    if problem and not (record.get("conda_done") and record.get("pip_done")):
        raise SetupError("The AI model 3.0 runtime has to be downloaded once, but the internet "
                         f"is not reachable: {problem}. Connect to the internet and try again.")

    runner = _Runner(root / "logs" / "setup.log", cancel)
    mamba = ensure_micromamba(root, progress, cancel)
    record.update({"schema": rt.RUNTIME_SCHEMA, "variant": chosen, "variant_reason": reason,
                   "micromamba": str(mamba), "micromamba_version": MICROMAMBA_VERSION})
    rt.write_runtime_json(record, root)

    py = root / "env" / "python.exe"
    if not (record.get("conda_done") and py.is_file()):
        py = stage_conda(root, mamba, progress, runner)
        record.update({"conda_done": True, "python": str(py), "pip_done": False,
                       "setup_complete": False})
        rt.write_runtime_json(record, root)

    if not record.get("pip_done"):
        stage_pip(root, py, chosen, progress, runner)
        record.update({"pip_done": True, "setup_complete": False})
        rt.write_runtime_json(record, root)

    progress(94, "Verifying the installation (PDAL probe, GPU, model files)...")
    # Release the downloaded conda archives: the environment keeps its own hard-linked copy.
    try:
        env = _clean_env()
        env["MAMBA_ROOT_PREFIX"] = str(root / "mamba")
        subprocess.run([str(mamba), "clean", "--all", "-y", "--no-rc"], env=env,
                       capture_output=True, creationflags=_no_window(), timeout=600)
    except Exception:
        pass

    status = rt.verify_runtime(device="cpu" if chosen == "cpu" else "auto")
    if not status.ok:
        raise SetupError(status.message + f" (see {runner.log_path})")
    record = rt.read_runtime_json(root)
    record.update({"setup_complete": True, "python": str(py),
                   "completed": time.strftime("%Y-%m-%dT%H:%M:%S")})
    rt.write_runtime_json(record, root)
    progress(100, "AI model 3.0 runtime is ready.")
    return rt.runtime_status()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Create the AI model 3.0 runtime environment")
    parser.add_argument("--root", help="runtime folder (default: NAKSHA_AI3_RUNTIME or "
                                       "%%LOCALAPPDATA%%\\NakshaAI\\ai3_runtime)")
    parser.add_argument("--variant", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--repair", action="store_true", help="delete the environment, rebuild")
    parser.add_argument("--check", action="store_true", help="report the plan, change nothing")
    parser.add_argument("--min-free-gb", type=float, default=None)
    args = parser.parse_args(argv)

    root = Path(args.root) if args.root else rt.runtime_root()
    if args.check:
        info = plan(root, args.variant)
        for key, value in info.items():
            print(f"{key}: {value}")
        return 0 if info["enough_space"] and not info["network_problem"] \
            and not info["engine_files_missing"] else 1
    try:
        status = run_setup(root, args.variant, args.repair, min_free_gb=args.min_free_gb)
    except SetupCancelled as exc:
        print(f"ERROR {exc}", file=sys.stderr)
        return 130
    except SetupError as exc:
        print(f"ERROR {exc}", file=sys.stderr)
        return 3
    print(f"READY {status.python}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
