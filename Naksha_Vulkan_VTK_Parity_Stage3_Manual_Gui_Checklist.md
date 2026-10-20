# Naksha LiDAR Stage 3 - Windows Vulkan and VTK manual acceptance checklist

This list is **not** a report of performed GUI tests. It is the acceptance procedure required before merging.

## Baseline

- [ ] Application launched with `E:\NAKSHA_LIDAR\lidar-classifer-parity-stage1\venv\Scripts\python.exe`.
- [ ] Vulkan active and visible main viewport, not merely a loaded DLL.
- [ ] Reference video: Freehand `NakshaAI-Lidar 2026-10-19 14-32-44(2).mp4`; Curve `NakshaAI-Lidar 2026-10-19 14-35-04(1).mp4`.
- [ ] Open a SNT-only scene and repeat below.
- [ ] Open a LAZ/LNT-only scene and repeat below.
- [ ] Open combined SNT + LAZ/LNT and repeat below.

## Curve

- [ ] Place two control points: visible smooth (sampled) curve or valid two-control shape; exact endpoints retained.
- [ ] Add three, four or more points: maintain smooth curvature and update preview without moving already committed segments.
- [ ] Hover cursor while placing: smooth fixed preview plus straight active cursor guide, snap highlight when applicable.
- [ ] Create a tight bend, loop and intentional self crossing without spurious extra points or unwanted closure.
- [ ] Snap near a separate Line/SmartLine/Polyline endpoint; verify exact saved XY coordinates, not just screen alignment.
- [ ] With at least three control points, approach first control point within tolerance; snap to precise first XY coordinates.
- [ ] Move away from closure radius: no unwanted closure.
- [ ] Confirm right-click/Enter completion and Escape cancellation; draw another independent curve.
- [ ] Undo/Redo changes to individual control points and completed figure; inspect history.
- [ ] Vertex-edit: ONLY real Curve controls, not 20x interpolation/tessellation samples.
- [ ] Compare original VTK Curve rendering, geometry XY/XYZ, zoom and screenshots.

## Freehand

- [ ] Press/hold left mouse, move continuously: stroke follows all movements, not click-to-click polyline.
- [ ] Smooth bends, loops, overlap/crossings; intermediate stroke points not pulled toward nearby SNT/Draw lines.
- [ ] Release left mouse: drawing remains pending and is **not** committed to document.
- [ ] Press left again and resume same stroke; no separate/duplicated previous stroke.
- [ ] Near an eligible existing endpoint at first/final point, snap to exact world coordinates.
- [ ] Test Nearby Snap to line/curve geometry and own-start preference; visible snap indicator before commit.
- [ ] Return within closure tolerance of own start; right-click: final point exactly equals start.
- [ ] Finish far from origin: a multi-point closed region joins back to origin as original VTK does.
- [ ] Short accidental two-point stroke does not create zero-area closed polygon.
- [ ] Right-click commits once; tool stays active, second Freehand stroke is independent and previous one persists.
- [ ] Escape cancels only current draft; draft Undo/Redo and document Undo/Redo preserve existing drawings.
- [ ] Compare original VTK mouse gestures, visual latency, endpoints and saved geometry.

## Regression and performance

- [ ] Line, SmartLine, Polyline, Orthogonal Polygon, Rectangle and Circle drawing.
- [ ] Text, Hatch, Element Selection, vertex Add/Move/Delete, Parallel, Centerline.
- [ ] Measure Line/Path, freehand/curve selection, class operations and undo/redo.
- [ ] SNT blocks still open grids on right-click; imported SNT geometry has NOT changed or become accidentally editable.
- [ ] No unnecessary full SNT or LiDAR scan on each mouse move; record draw latency and render FPS if telemetry available.
- [ ] Test big SNT + 7.5-million-point LAZ/LNT workload and zoom/pan while drawing.
- [ ] Save, reopen and compare Curve control coords and Freehand closed boundary; no precision drift or duplicates.
- [ ] Switch between original VTK and native Vulkan; ensure each tool remains functional and two input handlers never both act on one click.
- [ ] Verify GUI test outcomes and attach console logs/screenshots to review before merging to `Sub_master`.
