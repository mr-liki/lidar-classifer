# Naksha LiDAR - Vulkan / VTK Stage 2 integration

## What these files are

- `Naksha_Vulkan_VTK_Parity_Stage2_Integration.patch`: production implementation patch.
- `Naksha_Vulkan_VTK_Parity_Stage2_Tests.patch`: adds Stage 2 tests and updates two Stage 1 acceptance tests to match the user's new requirements.
- `Naksha_Vulkan_VTK_Parity_Stage2_Validation.md`: verification details, caveats and manual checks.

**Apply Stage 2 ONLY to the existing Stage 1 patched worktree** at:

`E:\NAKSHA_LIDAR\lidar-classifer-parity-stage1`

This patch was generated against the **17 Python source files the user exported from that Stage 1 worktree** in `Vulkan_Stage2_Source.zip`, not against upstream `Sub_master`. Do NOT apply to the original project or reapply Stage 1.

## Windows PowerShell - step-by-step

1. Download the Stage 2 bundle ZIP to `C:\Users\NT01231\Downloads`.
2. Extract ZIP contents **directly into the Downloads directory**, or simply download the two `.patch` files separately into Downloads. These must be actual files, not paths inside an unopened ZIP.
3. Open PowerShell and copy the following complete block:

```powershell
$ErrorActionPreference = 'Stop'
$work = 'E:\NAKSHA_LIDAR\lidar-classifer-parity-stage1'
$downloads = 'C:\Users\NT01231\Downloads'
$integration = Join-Path $downloads 'Naksha_Vulkan_VTK_Parity_Stage2_Integration.patch'
$tests = Join-Path $downloads 'Naksha_Vulkan_VTK_Parity_Stage2_Tests.patch'

if (!(Test-Path $work)) { throw 'Stage 1 worktree not found' }
if (!(Test-Path $integration)) { throw 'Integration patch not found' }
if (!(Test-Path $tests)) { throw 'Tests patch not found' }
Set-Location $work
Write-Host 'Branch:' (git branch --show-current)
if ((git branch --show-current) -ne 'feat/vulkan-vtk-parity-stage1') {
    throw 'Wrong Git branch; stop and investigate.'
}
Write-Host 'Current changes (Stage 1 must already be present):'
git status --short

# Verify BOTH patches without changing files.
git -c core.whitespace=cr-at-eol apply --check $integration
if ($LASTEXITCODE -ne 0) { throw 'Stage 2 integration does not fit this source; STOP.' }
git -c core.whitespace=cr-at-eol apply --check $tests
if ($LASTEXITCODE -ne 0) { throw 'Stage 2 test patch does not fit this source; STOP.' }

# Apply each ONCE.
git -c core.whitespace=cr-at-eol apply $integration
if ($LASTEXITCODE -ne 0) { throw 'Integration patch failed; STOP.' }
git -c core.whitespace=cr-at-eol apply $tests
if ($LASTEXITCODE -ne 0) { throw 'Tests patch failed; STOP.' }
Write-Host 'Stage 2 applied successfully' -ForegroundColor Green

# Run both regression suites with the worktree-local venv.
& '.\venv\Scripts\python.exe' -m pytest -q tests/test_stage1_parity.py tests/test_stage2_parity.py
if ($LASTEXITCODE -ne 0) { throw 'Tests failed; do not merge.' }
git -c core.whitespace=cr-at-eol diff --check
if ($LASTEXITCODE -ne 0) { throw 'Whitespace check failed.' }
git status --short
```

**Do not rerun `git apply` after success.** The same patch cannot normally be applied twice. If either `--check` fails, STOP and share the complete PowerShell output; do not run `git reset`, `git checkout`, `git stash`, `--force`, or `--reject` to make it apply.

4. If the tests pass, launch:

```powershell
cd 'E:\NAKSHA_LIDAR\lidar-classifer-parity-stage1'
& '.\venv\Scripts\python.exe' main.py
```

## User-visible changes

- **Curve:** when final cursor/control point is within the endpoint snap radius, the final point becomes exactly the first point; an ordinary curve may remain open.
- **Freehand:** a legitimate region with at least three distinct XY points automatically closes to its first point on release/right-click, even when released far away. Short accidental two-point strokes stay open.
- **Edit selection:** Individual/Block/Shape/Line, Select All and Invert only pick eligible user Draw items and completed Measure Line/Path objects. SNT block/cell reference geometry no longer enters ordinary editing selection. The dedicated SNT block context menu is unchanged.
- **Measure Line/Path:** selectable, highlighted, measured total available in Element Selection, and kept non-destructively read-only for Move/Delete. Copy can make a separate Draw overlay.
- **SNT Text mode:** click or drag a selection rectangle to select visible SNT text; highlight selection by bounding rectangles; clipboard Copy reads text. Imported SNT remains read-only; Move/Rotate/Mirror/Extend do not alter SNT text. Native hit boxes are approximate if a file does not provide explicit world-space glyph bounds.
- **Stage 1:** existing Intersect, Rectangle editing and Circle vertex suppression remain in the project and their tests still run.

## Limits and cautions

This is an **implementation patch with unit tests**, not verification on the user's Windows GPU. After installing, perform interactive testing in the Vulkan viewport with SNT + LAZ/LNT loaded; specifically inspect text drag precision, selection ownership, snapping and closures, undo/redo, and SNT right-click navigation. Original SNT file bytes must remain unchanged after selection.

The Stage 2 tests patch *updates two Stage 1 expectations*: it replaces the old Freehand far-end-open expectation with the requested auto-close behavior and replaces an old SNT-as-Intersect-input assumption with explicit imported-SNT selection exclusion. Other Stage 1 tests are preserved. Stage 1 modules not included in this patch are not overwritten.

**Before committing:** the repo's `.gitignore` may hide files under `tests/`. If `git status --short` does not show the new Stage 2 test file, add it with `git add -f tests/test_stage2_parity.py` (and, if needed, `git add -f tests/test_stage1_parity.py`) after testing; use normal `git add` for changed production files. Do not merge to `Sub_master` before runtime review.
