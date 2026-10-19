# Naksha LiDAR - Stage 3: VTK-parity Curve and Freehand in Vulkan

## Download and install

This bundle is **only** for the Stage 1 + Stage 2 patched Windows worktree:

`E:\NAKSHA_LIDAR\lidar-classifer-parity-stage1`

It does **not** apply directly to `Sub_master`, the untouched original project, or the unpatched Stage 1 export. It does not alter production Vulkan DLLs.

**Bundle contents**:

- `Naksha_Vulkan_VTK_Parity_Stage3_Integration.patch` - changes two existing Python GUI files; adds one new native stroke geometry helper.
- `Naksha_Vulkan_VTK_Parity_Stage3_Tests.patch` - adds focused mouse/geometry tests.
- `Naksha_Vulkan_VTK_Parity_Stage3_Install.ps1` - dry-run installer, backup, apply, test and rollback.
- `Naksha_Vulkan_VTK_Parity_Stage3_Validation.md` - test results, limitations and reasoning.
- `Naksha_Vulkan_VTK_Parity_Stage3_Manual_Gui_Checklist.md` - Windows acceptance checklist.

### Exact Windows PowerShell commands

1. Download the **Stage 3 Patch Bundle ZIP** into `C:\Users\NT01231\Downloads`.
2. Extract all contents into the **same Downloads directory**. The `.ps1` and both `.patch` files must be in that directory, not only inside an unopened ZIP.
3. Open a new PowerShell window. Run **only the dry run first**:

```powershell
$script = 'C:\Users\NT01231\Downloads\Naksha_Vulkan_VTK_Parity_Stage3_Install.ps1'
powershell.exe -NoProfile -ExecutionPolicy Bypass -File $script -Action Check
```

Expected: `DRY RUN PASS: patches fit; no files changed.`

4. If the dry run passes, **apply Stage 3 once**:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File $script -Action Apply
```

This creates a timestamped backup under `E:\NAKSHA_LIDAR\VTK_VULKAN_REVIEW\Stage3_Backups`, applies both patches, verifies production-source hashes, runs `git diff --check`, and runs all 77 focused Stage 1/2/3 tests with the worktree-local Python interpreter.

5. To launch the application once tests pass:

```powershell
cd 'E:\NAKSHA_LIDAR\lidar-classifer-parity-stage1'
& '.\venv\Scripts\python.exe' main.py
```

### Safe rollback

If interactive behavior is wrong, close Naksha LiDAR first. Then run:

```powershell
$script = 'C:\Users\NT01231\Downloads\Naksha_Vulkan_VTK_Parity_Stage3_Install.ps1'
powershell.exe -NoProfile -ExecutionPolicy Bypass -File $script -Action Rollback
```

Rollback verifies the Stage 3 source hashes, validates reverse application, removes only Stage 3 modifications and its new files, and checks that Stage 2 `native_draw.py` and `curve_tools.py` are restored. It does **not** reset, checkout or discard other project changes. Do not edit Stage 3 source files before running rollback; reverse-apply compatibility must pass. If rollback refuses, **stop and send the error**. The untouched pre-Stage 3 copies are also in the timestamped backup folder; do not overwrite source blindly.

### Extra safety notes

- The script explicitly checks the branch name `feat/vulkan-vtk-parity-stage1`, the two modified production files' expected Stage 2 SHA256 values, the exact patch SHA256 values, and dry-run `git apply --check` before any modifications. These checks intentionally **fail closed** if the local source has diverged.
- Use the scripts from the Downloads folder as shown; the installer resolves patches relative to itself.
- The script leaves the Stage 1/2 patches already installed: **do not reapply Stage 1 or Stage 2**.
- If any patch or test fails, **do not merge**. Never use `git reset --hard`, `git apply --reject`, force, or a manual cleanup in your working folder to bypass the safety check.
- Your Git repository may ignore files under `tests/`. Before any Git commit, inspect `git status --short`, and if the new test does not appear run `git add -f tests/test_stage3_curve_freehand.py` on your new branch. Do not commit or merge before Vulkan runtime acceptance.
- The PowerShell PSReadLine paste/Backspace bug in your earlier logs is unrelated. The installer launches in a fresh `powershell.exe -NoProfile` process so it does not rely on interactive PSReadLine editing.

## Precisely what Stage 3 changes

**Vulkan Curve (`gui/native_draw.py` plus helper):** samples the exact original VTK Catmull-Rom algorithm (20 subdivisions per segment); supports two-point curves, keeps committed spline preview steady while cursor movement draws a separate straight guide, preserves editable controls distinct from sampled render vertices, ignores duplicate control points, prefers the curve's own starting point within the closure tolerance, and supports genuine scene endpoint snapping. Mouse completion, cancel, repeated figures and Undo/Redo stay on the established native routing.

**Vulkan Freehand (`gui/native_draw.py`):** left drag samples actual cursor motion with VTK's 2-screen-pixel movement threshold; left release **pauses** without creating a document object; another left drag resumes the same stroke; right-click **finishes and closes** the stroke (for a real multi-point boundary); and right-click leaves Freehand active for another stroke. Only first and last points may magnetically snap; intermediate Freehand samples stay raw. Own-start snap takes precedence when near closure. An intended snapped endpoint uses exact world coordinates, with preview feedback, 50ms scene-pick throttling, and a final unthrottled snap. Undo/Redo handles paused draft drag checkpoints and committed objects.

**Original VTK protection (`gui/curve_tools.py`):** restores its original code byte-for-byte from your VTK reference (undoes an unrelated Stage 2 modification to its own closure logic). Original VTK Freehand is unchanged.

**No C++ changes**, native DLL replacement, SNT imports/modifications, LAS/LAZ/LNT pipeline changes or modifications to unrelated tool modules.

## Verification

Automated result from a clean recreation of your uploaded Stage 1 source, applying Stage 2 then Stage 3 patches:

```
77 passed in 0.99s
```

All edited Python files parsed successfully. These tests simulate the real `NativeDraw` input handlers using lightweight Qt stubs and compare the spline mathematically with the extracted original VTK geometry method; they **do not** prove GUI focus, actual GPU preview latency, native DLL interaction, Windows coordinate projection, or pixel-perfect parity. These must be checked interactively in the supplied Windows app with the two reference videos side by side. Read the manual checklist before merging.
