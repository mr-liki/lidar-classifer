# Naksha LiDAR Stage 2 - validation report

**Input:** `Vulkan_Stage2_Source.zip`, 17 production Python files exported by user from their Stage 1 patched worktree.

**Generated:** Two Git patches, one implementation and one regression test patch. GitHub and user's Windows checkout were not modified by this build.

## Automated verification

- Extracted all 17 files; verified ZIP integrity.
- Generated Git patches relative to an exact snapshot of the uploaded Stage 2 source; retained original Windows CRLF in modified production files.
- Fresh clone of baseline snapshot:
  - `git -c core.whitespace=cr-at-eol apply --check` passes for **both** patches.
  - Applied integration, then test patch with `git apply`: **passes**.
  - `python -m pytest -q tests/test_stage1_parity.py tests/test_stage2_parity.py`: **54 passed**.
  - `git -c core.whitespace=cr-at-eol diff --check`: **passes**.
  - Parsed all supplied GUI production Python files using `ast.parse`: **passes**.

## Modified production files

- `gui/native_draw.py`
- `gui/curve_tools.py`
- `gui/native_selection_bridge.py`
- `gui/selection_popup.py`
- **New** `gui/native_selection_targets.py` (dependency-free selection, closure, label hit-box helpers)

## Test files

- `tests/test_stage1_parity.py` (two Stage 1 acceptance expectations updated for the new, explicit user requirements)
- **New** `tests/test_stage2_parity.py` (29 focused Stage 2 tests)

## Runtime validation not completed

- No actual Windows GUI, Qt mouse gestures, native Vulkan renderer, GPU frame or large LAZ/LNT data was available in this test environment.
- Text picking estimates bounding boxes using provided world bounds when present, otherwise label height, width factor, rotation and text length. Font metrics, font justification, screen offset and some SNT text record types may need real-data adjustments.
- Measurements are selectable and highlighted as overlays but Move/Delete are deliberately read-only to prevent corrupting measurement records. Native measurement clipboard/copy path is a separate Draw overlay; no destructive modification of Measure or SNT is intended.
- Intersect's imported-SNT handling has been narrowed to match the user's instruction that ordinary selection only affects editable Draw/Measure objects, and requires manual acceptance in real scenes.
- Native Curve and Freehand closure require visual checks at different zoom levels, while moving cursor near start, and on right-click completion. Open Curve remains supported.
- Check SNT block right-click loading and label rendering were not regressed.
- Test on Windows before committing or merging to `Sub_master`.

## Suggested manual acceptance sequence

1. Start Vulkan viewport, attach SNT and load the corresponding LAZ/LNT.
2. Draw Curve: finish near its start, verify exact closure; finish far from start, verify open Curve.
3. Draw Freehand: release near start and far from start, verify a single closed geometry with no visible gap; right-click completion.
4. Use normal Block and Shape editing selection over an imported SNT block: it should **not** select the block boundary; authored Draw and finished Measure Line/Path should select.
5. Dedicated SNT grid/block right-click context menu still loads the correct LAZ/LNT.
6. SNT Text mode: drag across real text glyphs, confirm selection count/highlight; click a glyph and Copy text. Selection must not change imported SNT bytes or enable destructive text moves.
7. Try Move/Delete on Measure Line/Path selections, confirm originals unchanged; check measured total in Element Selection.
8. Regression: Stage 1 Intersect/rectangle/circle/undo-redo, existing polyline/orthogonal drawing, selection modifiers, tool focus and popup behavior.

**Status:** clean patch application + unit/regression tests passed. Real Windows Vulkan/VTK interaction parity is still unverified.
