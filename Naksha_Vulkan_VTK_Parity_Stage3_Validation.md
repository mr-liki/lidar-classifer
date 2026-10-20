# Naksha LiDAR Stage 3 - Root cause, implementation and validation

## Inputs examined

1. Freehand reference video: `NakshaAI-Lidar 2026-10-19 14-32-44(2).mp4` (approximately 20 seconds, 1920x1020).
2. Curve reference video: `NakshaAI-Lidar 2026-10-19 14-35-04(1).mp4` (approximately 17 seconds, 1920x1020).
3. The uploaded `VTK_Sub_master_GUI.zip` and original VTK `gui/digitize_tools.py` / `gui/curve_tools.py`.
4. The uploaded `Vulkan_Stage2_Source.zip` (17 files) plus the published Stage 2 patch, representing the latest shared source available for offline patching. Changes made locally **after** this export are not included; installer hash checks reject an incompatible worktree.

The videos show a natural continuous Freehand stroke and smooth, control-point-driven Curve gestures. They do not by themselves prove numerical coordinates, exact pixels-per-snap tolerance or every keyboard shortcut; geometry/control data is grounded in the supplied source.

## Specific source-level root causes

| Behavior | Original VTK source | Before Stage 3: native Vulkan | Stage 3 correction |
|---|---|---|---|
| Freehand mouse release | `digitize_tools.py`: left release pauses stroke, keeps existing samples | `native_draw.py`: left release called `_finish_freehand`, committing prematurely | Release pauses; subsequent left drag resumes same draft; right-click finalizes |
| Freehand stroke sampling | Raw mouse movements at a 2-screen-pixel jitter threshold | 0.5px threshold and terminal `_point()` snapping, which could change sampled geometry | 2px threshold, raw intermediate samples, exact endpoint magnetism only at start/finish |
| Freehand close | VTK right-click auto-closes a valid region | Stage 2 early completion/closure tied to left release | Right-click closes via established `close_freehand` semantics; snapping to start wins when near |
| Curve tessellation | VTK Catmull-Rom, 20 samples per control-pair interval + true end | Native 16 samples; under 3 controls returned a straight two-point line | Numerically matched VTK 20-sample Catmull-Rom for 2+ points, with 2D/3D support |
| Curve moving preview | Fixed smoothed curve through clicked controls, moving straight cursor guide | Smoothed interpolation included current cursor point, deforming the already-placed spline | Independent steady spline + cursor guide, with snap indication |
| Curve editable vertices | Original control points, not spline samples | Native stored controls, but could snap to all tessellation vertices of existing native Curve | Only saved controls drive editing; endpoint magnet limited to true endpoint controls |
| Existing VTK Curve code | Original VTK implementation | Stage 2 altered VTK closure logic | Original `curve_tools.py` restored verbatim from supplied VTK archive |

## Exact modified/added files

**Modified**:
- `gui/native_draw.py` - Vulkan event routing, pause/resume, snaps, previews, history, interpolation delegation.
- `gui/curve_tools.py` - restore unchanged VTK Curve code after a Stage 2 intrusion.

**Added**:
- `gui/native_stroke_geometry.py` - GUI-independent exact-VTK Catmull-Rom and safe stroke helpers.
- `tests/test_stage3_curve_freehand.py` - numerical VTK comparison and simulated event tests.

No C++ file, `.dll`, rendering backend, digitizer, SNT file or point-cloud loader was changed.

## Automated tests and clean replay

- Extracted all 17 files from `Vulkan_Stage2_Source.zip` into a new clean directory.
- Applied the published `Naksha_Vulkan_VTK_Parity_Stage2_Integration.patch` and Stage 1/2 test patches.
- Ran `git -c core.whitespace=cr-at-eol apply --check` for **both** Stage 3 patches; both passed.
- Applied both patches (no rejected hunks).
- Ran tests:

```
python -m pytest -q tests/test_stage1_parity.py tests/test_stage2_parity.py tests/test_stage3_curve_freehand.py
........................................................................ [ 93%]
.....                                                                    [100%]
77 passed in 0.99s
```

- All four edited/new Python files passed `ast.parse` syntax checks. Original VTK Curve file matches supplied original SHA256 exactly: `ead21ff1706487e2fa55d5133b9b9fb189d871d8e7243dcedce13ac5fd50d5e4`.
- Stage 3 adds 23 focused checks; Stage 1/2 existing 54 tests still pass.
- Focused checks include original VTK curve sampler numerical equivalence, two-control-point interpolation, 3D sampling, boundary closure priority, cursor guide, no tessellation editing points, endpoint snapping, duplicate-control removal, Freehand held-mouse sampling, pause/resume, right-click completion, endpoint snap, self-close, throttled preview, avoiding intermediate sample snap, repeated strokes, history and Escape cancel.
- The tests use synthetic position events and lightweight PySide6 mocks; they are stronger than trivial helper-only tests, but they are **not** actual Windows Qt/Vulkan viewport interaction tests.

## Performance observations and limitations

- Freehand intermediate samples no longer invoke `_point` or pick the SNT scene; scene magnetic searches are throttled to once per ~50ms during a moving stroke, then recomputed at final commit. This is a **source property**, not a measured GPU frame-rate increase.
- Freehand self-closure unique-point detection stops after three unique samples (bounded common-case processing). The saved stroke can still contain large numbers of samples; GPU overlay update cost and real-world RAM behavior have **not** been measured.
- Curve interpolation has bounded 16-entry LRU with a 128-control-point admission limit reuse of repeated static control sets. This does not bound total SNT/LAS/LAZ or renderer memory.
- No Stage 3 frame-time, latency, GPU upload, point-cloud FPS or million-point scalability measurements were collected; do not infer a performance improvement from the 0.99s unit-test runtime.
- Correct behavior of Qt/native GPU mouse event routing, exact hover pixel positions, text/overlay z-order, `Edit/Undo` command interactions with real dialog focus, big-scene snapping and actual file save/reopen **remains unverified**.
- No explicit closed native spline primitive was introduced. Curves remain stored as control points and are tessellated for display using the original VTK mathematics; full geometric equality of CAD serialization/splines must be checked in the application.
- Freehand valid strokes close on right-click, matching the original VTK and Stage 2 expectation. An accidental two-point stroke is left open rather than forming a zero-area polygon.

## Mandatory Windows runtime acceptance (NOT EXECUTED HERE)

Use the manual checklist included in the bundle. Exercise both VTK and Vulkan backends with the same coordinate scene, screen dimensions and snapping settings, comparing actual vertices/saved geometry and screenshots. Test SNT-only, LAZ-only and combined SNT+LAZ; verify import read-only state and loading/LOD performance. Check undo/redo for draft and completed objects; compare snap coordinates and cursor highlight; test all other draw/measure/select features as regressions. Record renderer/preview logs and any failed steps. **Do not merge into `Sub_master` before these pass.**
