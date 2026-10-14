---
name: lnt-surface-lod
description: "Persistent viewport-dependent native .lnt surface tiles, deployment and verified behaviour"
metadata:
  node_type: memory
  type: project
  modified: 2026-10-14
---

Native class-coloured Shading now uses persistent viewport surface LOD by default
when the compact tile arena is available and the root sample has at least two
visible classes. The user explicitly requested varied display LOD after earlier
full-detail requirements: retain all original points/full-resolution tiles, while
coarse representatives cover the source hierarchy. Report source coverage and
representatives drawn separately. Single-class/elevation modes retain the legacy
cut path; NAKSHA_LNT_SURFACE_LOD=0 disables the new path.

Main sources: native/lnt/include/LNTSurfaceLod.h, LNTSurfaceArchive.h,
LNTEmbeddedSurface.h, LNTVulkanEngine.h, native/lnt/lnt_view_api.cpp, and the
SurfaceRenderer tile-arena/C API changes under native/naksha_vulkan. Python
bindings/status are gui/lnt_native.py and gui/lnt_open.py. Documentation:
native/lnt/EMBEDDED_SHADING.md.

Tiles have camera-independent (depth,ix,iy) keys. They use octree representatives
and exact leaf records, same-level eight-neighbour halos and conditional fringe
groups. Parents remain until visible children are GPU-resident. Workers build or
load individual tiles; bounded CPU/GPU LRU caches reuse them on pan/zoom. Palette
recolour avoids triangulation; visibility settings use up to eight archive slots.
Source edits invalidate/re-key the archive. Read failures now expose the node ID
and Reader::last_error(), with a coarser resident tile retained for coverage.

The LNTSLA1 archive appends independently compressed CRC32C tile records,
SHA-256-chained index segments and a 448-byte footer after the v2 point directory.
Original bytes remain unchanged. Commit batches are 16 MB / two seconds, flushed
on close. An unchanged committed size/mtime stamp avoids source re-hashing on
reopen. Torn-tail recovery, growth bounds and stale-writer refusal are tested.
UNC saves are allowed; real Win32 errors are reported. No tests write the real
network dataset. Read-only files continue with session-only caches.

Legacy LNTSHD1/2/3 caches remain readable in the cut path, but the tile archive
replaces their derived tail on its first commit. Their geometry is NOT migrated
because partition/ownership differs; requested tiles build once then persist.
Removing derived data means truncating only to index_offset+16+52*node_count,
with the file closed and original point bytes preserved.

Environment: NAKSHA_LNT_SURFACE_LOD_PX (default 2, range .25-64),
NAKSHA_LNT_SURFACE_GPU_MB, NAKSHA_LNT_SURFACE_CPU_MB (default 768),
NAKSHA_LNT_SURFACE_PERSIST=0. Exact detailed geometry is available at fine views.

Deployed 2026-10-14: main_shading_lod.dll -> native/lnt/main_shading.dll;
build_refresh/naksha_vulkan.dll -> both build_msvc/naksha_vulkan.dll and
build_msvc/naksha_vulkan_surface_only.dll. Byte equality, ctypes loads, tile
exports, shader equality and the exact main.py stale-source scan all passed.
Backup: native/lnt/backup_before_surface_lod_deploy_20261014/.
Do not kill a running user app. If DLL replacement is locked, report it.
Make final native source edits BEFORE the final build/deployment: startup checks
source mtimes and even a later comment edit can make the deployed DLL stale.

Checks passed: lnt_surface_lod_test.cpp, full lnt_surface_lod_view_test.py against
candidates AND deployed DLLs, lnt_embedded_surface_test.cpp,
lnt_embedded_reopen_test.py, lnt_shading_navigation_test.py,
lnt_surface_error_test.py, lnt_surface_full_detail_cpu_test.cpp and
lnt_surface_barycentric_test.cpp (57/57, one optional check skipped).
Candidate log: audit/lnt_surface_lod_final_candidate.log; installed log:
audit/lnt_surface_lod_final_deployed.log. GPU suite includes build/reopen/corrupt/
evict/edit/read-only/bad-leaf phases. A prior test incorrectly demanded new
triangulation when panning across an already fully cached area; it now requires
reuse and a nonempty exact cover instead.

Local 4M fixture this run: overview about .2 s, reopen .09 s, overview 889k
triangles versus 8.0M full detail; all 4M source records covered. Cached revisit
did no triangulation; mixed-LOD and overview hole checks passed. Previous handoff
50M measurements (not rerun here): overview .72 s first tiles / 1.78 s settled,
reopen .08 s, close zoom .3-.5 s first visit / .07-.11 s on reopen, pan p95 ~10 ms.
These are local fixture measurements, not promised UNC performance.

User should manually check MARCENA000018.lnt over UNC. First commit may hash about
520 MB over that network; tiles/settings persist in the same .lnt. Avoid the full
Qt/VTK harness (prior NVIDIA OpenGL teardown fault); native-only tests are safer.

Build viewer from native/lnt with g++ -O3 -std=c++17 -static -shared
-DLNT_EXTERN_BUILDER, lnt_view_api.cpp and LNTReader.cpp plus src/LNTStreamer.cpp,
src/LNTLASReader.cpp, src/LNTCameraController.cpp, src/LNTInteractiveLoop.cpp and
src/LNTBuilder.cpp, output main_shading_lod.dll. Build engine via
E:/devtools/cmake/current/bin/cmake.exe --build native/naksha_vulkan/build_refresh
--target naksha_vulkan -j 6. For the barycentric harness link the same helper
translation units except Builder.cpp (its harness includes the implementation).
