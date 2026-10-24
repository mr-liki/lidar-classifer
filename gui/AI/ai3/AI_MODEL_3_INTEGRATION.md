# AI model 3.0 in Naksha LiDAR: integration, setup and requirements

This document describes how the "AI model 3.0" classification pipeline was integrated into the
Naksha LiDAR app, how it is set up on a user's machine, what it needs, and how it behaves. For the
short developer reference see `README.md` in this folder.

---

## 1. What it is

AI model 3.0 is a three-step LiDAR classification pipeline that was developed as a standalone
project (`E:\AI model_3.0`):

| Step | File | What it does |
|---|---|---|
| 1 | `step1_preprocess.py` | PDAL SMRF ground filter and height-above-ground (Delaunay) normalisation |
| 2 | `step2_inference.py` | Myria3D / RandLA-Net inference (checkpoint `fractal_lidarhd_myria3d.ckpt`) |
| 3 | `step3_rule_engine.py` | Contract-driven rule engine: vegetation height bands, buildings, noise, wires, poles (`config/contract_spec.json`) |

In Naksha it appears as the checkbox **"Use AI model 3.0"** in the AI Classification dialog.

---

## 2. How it was integrated (design)

### 2.1 Separate engine, separate process

The pipeline cannot run inside the Naksha interpreter:

| | Naksha app | AI model 3.0 pipeline |
|---|---|---|
| Python | 3.10 | 3.10 |
| numpy | 2.2.6 | 1.26.4 |
| torch | 2.7.1 + cu118 | 2.4.1 + cu121 |
| PDAL | not present | python-pdal 3.5.3 / libpdal-core 2.9.2 |
| Other | | torch_geometric 2.5.3, lightning, hydra-core, lazrs, gdal, pyproj |

So the GUI never imports the pipeline. It starts `engine/naksha_ai_engine.py` as a **child
process** using the engine's own Python and talks to it over a plain text protocol (the same
pattern the Premium engine uses).

```
ai_dialog.py ──"ai3"──> ai3_worker.Ai3Worker (QThread)
                            │  ai3_controller.Ai3Controller (Popen, progress, cancel)
                            ▼
      <runtime>\env\python.exe  engine\naksha_ai_engine.py
                            step1 PDAL ─> step2 Myria3D ─> step3 rule engine
```

### 2.2 Self-contained

Everything the engine needs is copied into the Naksha project under `gui/AI/ai3/engine/`. Nothing
refers to `E:\AI model_3.0`, and there are no hard-coded `E:\` paths. At run time the engine never
edits its input and never uses the network.

### 2.3 The runtime lives on the user's machine

The Python environment is **not** shipped inside the app. It is created on first use (see
section 4) at:

- default: `%LOCALAPPDATA%\NakshaAI\ai3_runtime`
- override: environment variable `NAKSHA_AI3_RUNTIME`
- use an existing interpreter instead: `NAKSHA_AI3_PYTHON` (managed installs, developers)

---

## 3. Files

All under `E:\naksha-lidar\gui\AI\ai3\` unless noted.

| Path | Role |
|---|---|
| `engine/naksha_ai_engine.py` | The engine entry point: argument handling, `.ptc` resolution, contract building, runs steps 1-3, progress protocol, `--selftest` |
| `engine/step1_preprocess.py`, `step2_inference.py`, `step3_rule_engine.py` | Vendored pipeline steps (step 3 identical to the 3.0 project; step 2 differs only by benign offline guards) |
| `engine/config/contract_spec.json` | Base rule-engine contract |
| `engine/models/` | `fractal_lidarhd_myria3d.ckpt` + `inference_config.yaml` |
| `engine/myria3d/` | Myria3D V3.8.2 sources |
| `ai3_runtime.py` | Where the runtime is, whether it is usable (`runtime_status`, `verify_runtime`, `engine_fingerprint`) |
| `setup_ai3_runtime.py` | Builds the runtime on the user's machine (also a CLI) |
| `ai3_setup_dialog.py` | Qt progress window; `ensure_runtime(parent)` offers the setup on first use |
| `ai3_controller.py` | Starts the engine, parses `PROGRESS` / `WARNING` / `RESULT`, cancels, explains failures |
| `ai3_worker.py` | `Ai3Worker`: full-file and fence flows, viewer update, classified copy, report |
| `ai3_ptc.py` | `applied_ptc_path(app)`: the `.ptc` applied in Display Mode; messages |
| `ai3_environment.yml`, `ai3_requirements.txt` | Pinned conda and pip contents of the runtime |
| `README.md` | Short developer reference |
| `tests/test_ai3_integration.py` | 17 tests (the `tests/` folder is git-ignored: use `git add -f`) |

Host files changed:

- `gui/ai_dialog.py` (CRLF file; line endings were preserved): the dialog changes in section 6.
- `naksha.spec` (CRLF): adds the engine folder, `ai3_environment.yml`, `ai3_requirements.txt` as
  data files and `gui.AI.ai3*` as hidden imports so a PyInstaller build includes them.

---

## 4. Setup and requirements

### 4.1 Requirements on the user's machine

| Item | Requirement |
|---|---|
| OS | Windows 10 / 11, 64-bit |
| Disk | about 8 GB free for the runtime (plus working space for the data) |
| Download | about 3 GB, **internet only while the setup runs** |
| Time | 10-30 minutes, once |
| GPU (optional) | NVIDIA GPU with driver >= 527.41 gives the CUDA 12.1 build; otherwise the CPU build is installed (much slower) |
| Input data | LAS / LAZ. **Colour (RGB) is strongly recommended**: the model was trained on coloured clouds. Without it the engine fills neutral colour, runs, and reports a warning; expect weaker results |
| Class file | A class-definition `.ptc` loaded and applied in Display Mode (section 5) |

### 4.2 What the setup installs

Created by `setup_ai3_runtime.py`, in stages that resume if interrupted:

1. **micromamba** (pinned version, SHA-256 checked).
2. **conda-forge** packages: Python 3.10.21, python-pdal 3.5.3, libpdal-core 2.9.2, gdal 3.11.4, pyproj.
3. **pip** packages: numpy 1.26.4, torch 2.4.1+cu121 (or CPU build), torch_geometric 2.5.3,
   lazrs 0.8.2, lightning 2.6.6, hydra-core and the rest of `ai3_requirements.txt`.
4. **Self-test**: the engine's own `--selftest` imports the whole stack, runs the PDAL filters on a
   tiny synthetic cloud and checks the model files. The result is recorded in `runtime.json` together
   with a fingerprint of the engine files, so a changed engine is re-verified automatically.

### 4.3 How a user triggers it

Tick "Use AI model 3.0" and press OK. If the runtime is missing, Naksha offers to build it and shows
a progress window. Nothing else is needed.

### 4.4 Command line

```
python setup_ai3_runtime.py --check            # show the plan and state
python setup_ai3_runtime.py                    # build
python setup_ai3_runtime.py --repair           # rebuild
python setup_ai3_runtime.py --variant cpu      # force the CPU build
python ai3_runtime.py --verify                 # re-run the self-test
```

Logs: `<runtime>\logs\setup.log`.

### 4.5 Developer machine note

On the development machine the runtime lives at `E:\ai3_test_runtime`, reached through a junction
`C:\Users\NT01173\AppData\Local\NakshaAI\ai3_runtime`. Conda environments are not relocatable, so
that folder must not be moved or deleted.

---

## 5. The .ptc file is the input schema

AI model 3.0 **cannot run unless a class-definition `.ptc` file has been loaded in Display Mode and
Apply was pressed.** Naksha then uses that file as the engine's input `.ptc`.

### 5.1 Blocking

| Where | Behaviour |
|---|---|
| AI Classification dialog | A red hint under the checkbox ("Requires a .ptc: load it in Display Mode and press Apply."). OK shows "PTC file required" and stops. Once applied, the hint turns green: "Input PTC: <name>" |
| Fence-mode dialog | Choosing AI model 3.0 without an applied `.ptc` shows the same message |
| `Ai3Worker` | Last-resort check; raises an error with the same instruction |

"Applied" means `app.active_ptc_schema` is set and `app.active_ptc_path` points to an existing file
(Display Mode sets both only on Apply): `ai3_ptc.applied_ptc_path(app)`.

### 5.2 How the engine reads it

The host copies the file into the work folder and passes `--ptc`. The engine's parser
understands TerraScan-style tables (`<id>\t\t<name>` plus `*` view records, including the variants
with a trailing numeric column or a code prefix such as `GRD Ground`) and plain text tables.
Each internal class is resolved in this order: contracted fold, exact normalised name, auditable
alias, then the file's unnamed default class (normally `0`).

| Internal class | Resolved from the `.ptc` |
|---|---|
| Ground, Low / Medium / High vegetation, Building | by name (**required**; the run is refused if missing) |
| Wire | "Bare Conductor(s)" |
| Pole | "Pylon(s) or Pole(s)" |
| Low / high noise | "Low point" (both noise classes go to this one container) |
| Unclassified | the default class (e.g. `0`) |
| Water | never delivered; stays in the default class |
| Bridge | not delivered; the rule engine re-files those points |

Further rules:

- The allowed output set, fallback class and noise fallback are derived from the resolved mapping
  (rule engine "Rule H" then enforces them), so a `.ptc` with different numbering works.
- If the file has no wire or pole class, wire/pole extraction is **turned off** and a warning is shown.
- A binary `.ptc` (Leica Cyclone point cloud) or an unreadable file is refused with a clear message.
- A delivered code above 31 (for example 51) cannot be stored in legacy LAS point formats 0-5. Such
  an input is first rewritten as **LAS 1.4** (CRS, scale, offset and other attributes carried over).

Example, `Class_ENEL_2025_connect.ptc`: unclassified 0, ground 1, low veg 2, medium veg 3, high veg 4,
building 5, noise 51, wire 14, pole 16.

---

## 6. Dialog behaviour

- **AI Engine** group is at the top of the AI Model tab, with three mutually exclusive options
  (Advanced, Premium, **Use AI model 3.0**).
- The old "Class → Output Code Mapping" table and its presets are removed from view (codes come
  from the applied `.ptc`). The hidden code fields still exist because other engines read them.
- The **Power Line Detection** block stays below the AI Engine group, but is **hidden when AI model 3.0
  is selected**: the engine then always runs the wire/pole rules (subject to the `.ptc` having those classes).
- The **Center Line corridor** option is disabled for this model.
- Class summary and notes (warnings from the engine, for example missing colour) are shown at the end.

---

## 7. Fence mode

The pipeline needs context (16 m ground window, 50 m inference tiles), so in fence mode:

1. Crop the bounding box of the fence plus a context buffer (`NAKSHA_AI3_FENCE_BUFFER_M`, default 50 m).
2. Run the engine on the crop.
3. Merge back **only the points inside the fence**; points outside are untouched.

Notes: a fence needs at least 5,000 engine points (too small is refused with a message). A fence
result is not guaranteed identical to a full-file run; the full file is the reference.

---

## 8. Output

- The classification is written into the viewer.
- A classified copy of the **source** file (all other attributes unchanged) is written to
  `AI3_Classified/<name>_ai3_classified.laz`, next to the source, with `<name>_ai3_report.json`
  (source, `.ptc`, points, warnings, engine report).
- The engine also writes `<output>.contract_resolved.json` (the exact contract used) in its work
  area. The work folder is deleted on success and on cancel.
- A "common post-process" used by other engines is skipped for this model.

---

## 9. Engine command-line contract

```
python naksha_ai_engine.py --input IN.laz --output OUT.laz --ptc CLASSES.ptc
       [--work-dir DIR] [--no-power-lines] [--ground-unclass-pct N] [--device auto|cuda|cpu]
       [--keep-work] [--selftest]
```

`--ground-unclass-pct` (0-100, default 20) is the share of the final Ground points set at random to
class 0 (Uncategorized). The dialog's **Ground -> Uncategorized** box sets it: it starts at 20 % when
"Use AI model 3.0" is ticked, and the user can change it. Leaving the model restores the value the
other engines had (50 %).

- stdout: `PROGRESS <0-100> <message>`, `WARNING <text>`, `RESULT {json}`; stderr: `ERROR <text>`.
- Exit codes: 0 ok, 1 bad arguments, 2 pipeline step failed, 3 environment unusable, 130 cancelled.
- `--mapping MAP.json` is a test-only alternative to `--ptc`.
- The host passes a clean environment (no `PYTHONPATH`, `PYTHONHOME`, `CONDA_PREFIX`; user site
  disabled, UTF-8 output) so the engine only sees its own packages.
- Cancel kills the engine and its Myria3D child (`taskkill /T`).

---

## 10. Performance

Measured on a 14.9 M point tile (RTX GPU, 20 cores): step 3 takes about 121 s (it was 370 s before
optimisation, with byte-identical output). Progress is paced by the measured per-point cost of each
step so the bar keeps moving during the long steps.

---

### Low vegetation (Rule V)

The model labels almost nothing below 1-2 m as vegetation (grass, crops and shrubs come out as
Unclassified), and the canopy rule only fills from 2 m up, so open terrain had medium and high
vegetation but almost no low vegetation. Rule V (after the outlier rule) files points still
Unclassified between 0.15 m and 3 m above the terrain into the vegetation band of their height
(<= 1 m low, <= 3 m medium), except within 0.75 m (3D) of a building, pole or wire point - 3D so
the bushes under an overhead line are filled too. Higher up, Unclassified points inside a crown
(at least half of the classified points within 1.5 m are vegetation, up to 6 passes inwards)
become vegetation of their band: these were the grey patches inside the canopy, mostly under
wires whose corridor the canopy rule must leave alone. Contract block `low_vegetation_fill`
(`enabled`, `source_classes`, `max_hag`, `structure_buffer`, `canopy_radius`, `canopy_veg_share`,
`canopy_min_veg`, `canopy_passes`).
Measured: CEDRASCO low vegetation 34.8 k -> 126 k points, FUNIVIA 7.2 k -> 61 k; the 123 tile
(where the model already labels its low layer) +752.

### Wire clumps in crowns (Rule X)

Small clumps of "wire" left inside crowns (2-4 m wide, mostly vegetation around them) showed as red
rings in the canopy with grey (Unclassified) points inside. Before the outlier rule, wire points
are linked in 3D (1.5 m); a component narrower than 6 m, with at least 50 % vegetation within 4 m
of its centre and no pole point within 2 m, becomes vegetation of its height band, and Rule V then
fills the points inside it. Contract block `wire_blob_purge`. CEDRASCO: 46 clumps, 3,610 points;
every wire component 8 m or longer is unchanged; the 123 tile is unchanged.

### Run time (3.6 M point FUNIVIA tile, 4 GB laptop-class GPU)

| | total | step 1 / 2 / 3 |
|---|---|---|
| exact CUDA kernels, LAZ hand-over files | 108 s | 23 / 54 / 31 s |
| LAS hand-over files | 105 s | 22 / 51 / 31 s |
| + fast CUDA kernels (default) | 86 s | 22 / 33 / 30 s |

- The files handed between the steps are uncompressed `.las` (same data, no LAZ compress / decompress).
- The model's GPU scatter-adds run without deterministic kernels by default. All random choices are
  still seeded; on this tile the classes were identical to the exact run and between two fast runs.
  Set `NAKSHA_AI3_DETERMINISTIC=1` for bit-exact kernels.
- Rule T (network trace, see the engine) is **off** by default (`network_trace.enabled`): it is
  experimental and cost 330 s on this tile.

## 11. Testing

```
set NAKSHA_AI3_TEST_LAS=path\to\small.laz      # about 250 k points is enough
venv\Scripts\python.exe tests\test_ai3_integration.py
```

Unit tests always run: `.ptc` resolution, the gate, headers, runtime resolution, missing runtime. The
engine tests need the test LAS and a ready runtime and cover full file, fence, legacy format with
code 51, too-small fence, cancel and the self-test. Last result: 17 of 17 passed.

---

## 12. Updating

- **Engine code or model:** replace files under `engine/` (keep the names), then run
  `python ai3_runtime.py --verify`. The runtime re-verifies automatically when the fingerprint changes.
- **Dependencies:** edit `ai3_environment.yml` / `ai3_requirements.txt`; users then run `--repair`.

---

## 13. Known limits and not yet verified

- A PyInstaller build, a launch of the complete app, the CPU-only install path and a clean-machine
  install at the default location have not been tested yet. Dialog and worker were exercised offscreen.
- Without RGB in the input the results are weaker.
- Fence results can differ slightly from a full-file run.
- Water is never delivered, Bridge is not delivered.
- The runtime is about 8 GB on disk and needs internet once for the setup.
