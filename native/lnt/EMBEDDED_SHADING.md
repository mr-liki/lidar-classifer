# Persistent viewport surface LOD

Native class-coloured Shading uses a persistent tile archive when the compact
renderer is available and the root sample has at least two visible classes.
Single-class and elevation-coloured surfaces retain the legacy cut-based path.

Tiles follow an XY quadtree aligned with the point octree. Coarse tiles use
existing octree representatives; exact tiles retain every original leaf record
in their footprint. Screen-space spacing, a pixel threshold and hysteresis choose
the viewport cover. A parent stays visible until its visible children are GPU
resident. Pan and zoom select and reuse tiles instead of retriangulating camera
cuts. CPU and GPU caches use bounded LRU eviction. Palette RGB changes recolour
resident tiles; visibility and geometry-setting changes use separate archive
settings slots. Point/class edits invalidate the source identity.

Coarse rendering represents original records through the hierarchy; it does not
draw each original record individually. The status reports source coverage,
representatives drawn and exact coverage separately. Exact tiles account for
all visible original records through triangles or coverage geometry. Eight-neighbour
halos and conditional fringe groups close tile and mixed-LOD borders.

The `LNTSLA1` archive is appended after the original v2 node directory. Original
headers, point records and node entries remain unchanged. Each independently
readable tile has CRC32C checks. Lossless node-relative positions use an exact
16-bit representation when possible, otherwise raw doubles; faces use zigzag
index deltas, byte shuffling and zstd level 3. The append-only index has SHA-256
chaining and periodic snapshots. Its footer stores source/directory identity,
up to eight geometry settings and a committed size/mtime stamp. Unchanged stamps
avoid re-hashing source points on reopen; changed stamps trigger validation.
Corrupted tiles are rejected and rebuilt individually. Torn commits recover the
last good index. Archive growth is bounded and may reset the derived tail.

UNC saves are allowed through Win32 wide-path APIs. Real permission, sharing and
other save errors are reported; read-only files still shade using session caches.
The first commit hashes the original source bytes in the writer thread, which
can take time over a network. Commits batch at 16 MB or two seconds and flush on
shutdown. Persistence is inside the same `.lnt`, with no required sidecar.

Point-only files remain compatible. Legacy `LNTSHD1/2/3` monolithic caches are
replaced by the tile archive on its first commit. Their meshes are not migrated:
the spatial partition and ownership differ. Only requested regions/levels are
triangulated, then reused on later visits and opens. The legacy mesh writer
refuses to overwrite a tile archive. Removing shading data means truncating
only the derived tail back to `index_offset + 16 + 52 * node_count`, with the file
closed; original point bytes must never be truncated or rewritten.

Environment controls:

- `NAKSHA_LNT_SURFACE_LOD=0`: disable the tile path and use legacy shading.
- `NAKSHA_LNT_SURFACE_LOD_PX`: representative spacing threshold, default 2 px;
  accepted range 0.25–64. Lower values request finer geometry.
- `NAKSHA_LNT_SURFACE_GPU_MB`: GPU arena budget, accepted range 64–16384 MiB.
- `NAKSHA_LNT_SURFACE_CPU_MB`: decoded tile cache, default 768 MiB;
  accepted range 64–65536 MiB.
- `NAKSHA_LNT_SURFACE_PERSIST=0`: session caching only; do not append tiles.

The point streamer resumes when leaving Shading. Compact rendering keeps the
nonblocking frame policy: busy GPU attempts defer submission while inputs keep
updating, and the next submission uses the latest camera. Matching shaders and
renderer DLLs must be deployed together. Rebuild after all native source edits;
startup rejects a canonical renderer DLL older than any scanned native source.

Verification uses private local fixtures. Native and GPU acceptance cover exact
record accounting, LOD borders, viewport refinement, revisit/reopen without
triangulation, cache eviction, recolouring, class edits, visibility settings,
corruption, torn commits and source-read/save failures. Legacy checks remain
available with surface LOD disabled. Large UNC timings depend on storage and
server behaviour; local timings are not guaranteed network performance.
