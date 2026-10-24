# AI model 3.0

"Use AI model 3.0" in the AI Classification dialog. A PDAL ground/height step, Myria3D RandLA-Net
inference and a rule engine (vegetation bands, buildings, wires, poles), run as a **separate
process** in its own Python environment that lives on the user's machine.

## Why a separate environment

The engine needs PDAL, torch 2.4 (CUDA 12.1), torch-geometric and numpy 1.26. The Naksha app runs
numpy 2.x and torch 2.7 and has no PDAL, so the two cannot share an interpreter. The GUI therefore
never imports the pipeline: it starts `engine/naksha_ai_engine.py` with the runtime's interpreter
and reads its progress, the same pattern as the Premium engine.

```
ai_dialog.py  --"ai3"-->  ai3_worker.Ai3Worker (QThread)
                              |  ai3_controller.Ai3Controller   (Popen, progress, cancel)
                              v
            <runtime>\env\python.exe  engine\naksha_ai_engine.py
                              step1 PDAL -> step2 Myria3D -> step3 rule engine
```

## Files

| file | role |
|---|---|
| `engine/` | the engine, vendored: `naksha_ai_engine.py`, `step1/2/3`, `config/contract_spec.json`, `models/` (checkpoint + config), `myria3d/` (V3.8.2 sources) |
| `ai3_runtime.py` | where the runtime is, whether it is usable (`runtime_status`, `verify_runtime`) |
| `setup_ai3_runtime.py` | creates the runtime on the user's machine (also a CLI) |
| `ai3_setup_dialog.py` | Qt progress window; `ensure_runtime(parent)` |
| `ai3_controller.py` | runs the engine, parses `PROGRESS` / `RESULT` / `WARNING` lines, cancels |
| `ai3_worker.py` | full-file and fence flows, viewer update, classified copy, report |
| `ai3_environment.yml`, `ai3_requirements.txt` | pinned conda + pip contents of the runtime (the CPU variant is generated from the pip list) |

## The runtime

Created once per machine by `setup_ai3_runtime.py` (the dialog offers it on first use): micromamba
(pinned, SHA-256 checked) -> conda-forge Python 3.10 + PDAL -> pip packages (CUDA 12.1 torch if an
NVIDIA driver >= 527.41 is present, otherwise CPU torch) -> the engine's own `--selftest`.
About 3 GB download, 8 GB on disk, 10-30 minutes, internet needed only while it runs.

* Location: `NAKSHA_AI3_RUNTIME`, else `%LOCALAPPDATA%\NakshaAI\ai3_runtime`.
* `NAKSHA_AI3_PYTHON` points at an existing python.exe instead (managed installs, developers).
* `python setup_ai3_runtime.py --check` shows the plan, `--repair` rebuilds, `--variant cpu` forces CPU.
* Interrupted setups resume; `logs/setup.log` has the full output.

## Engine contract (`naksha_ai_engine.py`)

```
python naksha_ai_engine.py --input IN.laz --output OUT.laz --ptc CLASSES.ptc
       [--work-dir DIR] [--no-power-lines] [--ground-unclass-pct N] [--device auto|cuda|cpu]
       [--keep-work] [--selftest]
```
`--ground-unclass-pct`: share of Ground set to class 0 at random (default 20; the dialog's
"Ground -> Uncategorized" box).
stdout: `PROGRESS <0-100> <message>`, `WARNING <text>`, `RESULT {json}`. stderr: `ERROR <text>`.
Exit: 0 ok, 1 bad arguments, 2 pipeline failed, 3 environment, 130 cancelled.

**The .ptc is the input schema.** AI model 3.0 only runs when a class-definition `.ptc` has been
loaded in Display Mode and applied (`app.active_ptc_path` / `active_ptc_schema`); the dialog, the
fence dialog and the worker all refuse otherwise (`ai3_ptc.applied_ptc_path`). The host passes a
copy of that file as `--ptc`; the engine resolves every internal class to a class id of the file
(exact name, then aliases such as *Bare Conductors* / *Pylons or Poles* / *Low point*, then the
file's unnamed default class, normally `0`) and derives the allowed output set from it. Ground, Low /
Medium / High vegetation and Building must exist in the file, otherwise the run is refused. Without
wire / pole classes the power-line rules are turned off with a warning. Water is never delivered
(it stays in the default class). `--mapping MAP.json` (`{"classes": {ground, low_veg, ...}}`) remains
as a test-only alternative to `--ptc`. Inputs in a legacy LAS point format are converted to LAS 1.4
first when a delivered code exceeds 31 (e.g. 51).

## Behaviour worth knowing

* **Output.** The classification is written into the viewer, and a classified copy of the *source*
  file (all other attributes unchanged) goes to `AI3_Classified/<name>_ai3_classified.laz`, with a
  JSON report beside it.
* **Fence mode.** The pipeline needs context (16 m ground window, 50 m inference tiles), so it runs
  on the fence's bounding box plus a context buffer (`NAKSHA_AI3_FENCE_BUFFER_M`, default 50 m) and
  merges back **only the fence points**. A fence result is not guaranteed identical to a full-file
  run; the full file is the reference.
* **No Center Line corridor** for this model: the option is disabled when it is selected.
* **Colour.** The checkpoint was trained on coloured clouds. Without RGB the engine runs with a
  neutral fill and says so (a note in the result box); expect weaker results.
* The engine never edits its input and never touches the network at run time.

## Tests

`python tests/test_ai3_integration.py` (or pytest). Unit tests always run. The engine tests need
`NAKSHA_AI3_TEST_LAS` (a small LAS/LAZ, ~250 k points is enough) and a ready runtime; they cover
full-file, fence, legacy format with code 51, too-small fences, cancel and a missing runtime.

## Updating the engine

Replace the files under `engine/` (keep the names), run `python ai3_runtime.py --verify`. The
runtime re-verifies automatically because its recorded self-test is keyed on a fingerprint of the
engine files. Dependency changes go in `ai3_environment.yml` / `ai3_requirements.txt`; users
then repair the runtime (`--repair`).
