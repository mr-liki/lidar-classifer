// LNTFormat.h - the .lnt single-file binary layout.
//
// EVERY struct here is byte-exact and packed. Each one carries a static_assert
// on BOTH its size and its field offsets, so a layout change fails the build
// instead of silently reinterpreting a file. That matters more than usual here:
// the reader computes the trailer address arithmetically
// (index_offset + 16 + 52 * node_count) rather than by parsing, so a single
// stray padding byte would make every node record off by that byte - and the
// points would still decode into plausible-looking coordinates.
//
// File layout:
//
//   offset 0                LNTHeader                        (64 B)
//   offset 64                LNTPoint chunk 0                (11 B/pt)
//                            LNTPoint chunk 1
//                            ...
//                            LNTPoint chunk N-1
//   offset index_offset      LNTDirectoryHeader               (16 B)
//   offset index_offset + 16 LNTNodeEntry[node_count]         (52 B each)
//   after node directory     optional embedded shading mesh and LNTSHD1 footer
//                            (see LNTEmbeddedSurface.h; derived data only)
//
// index_offset is written LAST (the header is patched after the trailer is in
// place), so a partially written file can never advertise a valid trailer.

#ifndef LNT_FORMAT_H
#define LNT_FORMAT_H

#include <cstddef>
#include <cstdint>

namespace lnt {

// ---- constants ----------------------------------------------------------
static constexpr char     kMagic[4]      = {'L', 'N', 'T', '\0'};
static constexpr uint16_t kFormatVersion = 2;   // 2: LNTPoint gained `classification` (11 B/pt)

// 21 bits/axis: 3 * 21 = 63 <= 64, so a 3-axis Morton key fits uint64 exactly
// with one bit to spare. 22 bits/axis would be 66 bits and overflow silently.
static constexpr int kAxisBits   = 21;
static constexpr int kAxisLevels = 1 << kAxisBits;   // 2,097,152 bins
static constexpr int kAxisMax    = kAxisLevels - 1;  // 2,097,151

// 16 bits/axis for a point's position relative to its own node cell.
//
// BE PRECISE ABOUT BINS vs POSITIONS here, because the two 16-bit conventions
// differ and conflating them is what produced the 0.99998-of-a-step error the
// quantiser used to carry. The field holds kLocalLevels = 65,536 distinct
// VALUES (0 .. kLocalMax); the reconstruction divides by kLocalMax, so the cell
// is divided into kLocalMax = 65,535 INTERVALS, one per step of
// node_cell_step(). quantize_to_16bit rounds onto those 65535 intervals rather
// than flooring into 65536 bins, which is what makes forward and inverse exact
// inverses and bounds the error to half a step.
static constexpr int kLocalBits   = 16;
static constexpr int kLocalLevels = 1 << kLocalBits;  // 65,536 representable values
static constexpr int kLocalMax    = kLocalLevels - 1; // 65,535 intervals / steps

#pragma pack(push, 1)

// ---- LNTHeader: EXACTLY 64 bytes ---------------------------------------
struct LNTHeader {
    char     magic[4];       //  0  'L','N','T','\0'
    uint16_t version;        //  4
    uint16_t reserved;       //  6  must be 0
    uint64_t index_offset;   //  8  absolute offset of LNTDirectoryHeader
    double   bounds_min[3];  // 16  cubic envelope: min corner (X, Y, Z)
    double   bounds_max[3];  // 40  cubic envelope: max corner (X, Y, Z)
};
// 8 + 24 + 24 = 56... the two arrays are 24 B each, so 16 + 24 + 24 = 64.
static_assert(sizeof(LNTHeader) == 64, "LNTHeader must be exactly 64 bytes");
static_assert(offsetof(LNTHeader, magic) == 0, "LNTHeader.magic offset");
static_assert(offsetof(LNTHeader, version) == 4, "LNTHeader.version offset");
static_assert(offsetof(LNTHeader, reserved) == 6, "LNTHeader.reserved offset");
static_assert(offsetof(LNTHeader, index_offset) == 8, "LNTHeader.index_offset offset");
static_assert(offsetof(LNTHeader, bounds_min) == 16, "LNTHeader.bounds_min offset");
static_assert(offsetof(LNTHeader, bounds_max) == 40, "LNTHeader.bounds_max offset");

// ---- LNTDirectoryHeader: EXACTLY 16 bytes, written at index_offset -----
struct LNTDirectoryHeader {
    uint64_t total_points;   //  0  dataset size (leaf points, no LOD copies)
    uint64_t total_nodes;    //  8  number of LNTNodeEntry records that follow
};
static_assert(sizeof(LNTDirectoryHeader) == 16, "LNTDirectoryHeader must be exactly 16 bytes");
static_assert(offsetof(LNTDirectoryHeader, total_points) == 0, "LNTDirectoryHeader.total_points offset");
static_assert(offsetof(LNTDirectoryHeader, total_nodes) == 8, "LNTDirectoryHeader.total_nodes offset");

// ---- LNTPoint: EXACTLY 11 bytes (format version 2) --------------------
// world = (node.center - node.half_size) + rel * (2 * node.half_size) / 65535
// The node cell is a CUBE, so one half_size covers all three axes and the
// point carries no origin of its own. A point is therefore decodable from its
// LNTNodeEntry alone, with no per-point header anywhere.
struct LNTPoint {
    uint16_t x_rel;          //  0  0..65535 within the node cell
    uint16_t y_rel;          //  2
    uint16_t z_rel;          //  4
    uint8_t  r;              //  6  8-bit RGB as stored in the .lnt
    uint8_t  g;              //  7
    uint8_t  b;              //  8
    uint8_t  intensity;      //  9
    uint8_t  classification; // 10  ASPRS class code, 0..255 (version 2)
};
// The pack(1) is what makes 11 a guarantee: unpacked, the uint16 members would
// pad this to 12.
static_assert(sizeof(LNTPoint) == 11, "LNTPoint must be exactly 11 bytes");
static_assert(offsetof(LNTPoint, x_rel) == 0, "LNTPoint.x_rel offset");
static_assert(offsetof(LNTPoint, y_rel) == 2, "LNTPoint.y_rel offset");
static_assert(offsetof(LNTPoint, z_rel) == 4, "LNTPoint.z_rel offset");
static_assert(offsetof(LNTPoint, r) == 6, "LNTPoint.r offset");
static_assert(offsetof(LNTPoint, g) == 7, "LNTPoint.g offset");
static_assert(offsetof(LNTPoint, b) == 8, "LNTPoint.b offset");
static_assert(offsetof(LNTPoint, intensity) == 9, "LNTPoint.intensity offset");
static_assert(offsetof(LNTPoint, classification) == 10, "LNTPoint.classification offset");

// ---- LNTNodeEntry: EXACTLY 52 bytes -----------------------------------
struct LNTNodeEntry {
    uint64_t node_id;        //  0  Morton prefix + level tag (see node_id_at)
    uint64_t byte_offset;    //  8  absolute offset of this node's LNTPoint chunk
    uint32_t byte_length;    // 16  chunk size in bytes (point_count * 10)
    uint32_t point_count;    // 20  points in this chunk
    // 24  Node cell centre, as a float32 OFFSET FROM LNTHeader::bounds_min.
    //     It is deliberately NOT an absolute world coordinate: a float32 ulp at
    //     a UTM-like 6.8e5 m is 0.0625 m, which is coarser than the 16-bit cell
    //     step of a shallow leaf (9.16e-3 m) and would make the payload's
    //     precision unreachable. Relative to bounds_min the offset is bounded
    //     by the cube side, where the ulp is 1.22e-4 m. See LNTMath.h.
    float    center[3];
    float    half_size;      // 36  node cell half-extent; cell is a cube
    uint8_t  children_mask;  // 40  bit k set => octant k exists (bit0=x, bit1=y, bit2=z)
    uint8_t  tree_level;     // 41  0 = root, max 21
    uint8_t  reserved[10];   // 42  must be 0
};
// WITHOUT pack(1) this struct would be 56 bytes: the 8-byte node_id/byte_offset
// force 8-byte alignment, so the trailing 42..51 payload would be padded to 48
// and the struct to 56. The pack is what makes 52 possible, which is exactly
// why the static_assert matters here.
static_assert(sizeof(LNTNodeEntry) == 52, "LNTNodeEntry must be exactly 52 bytes");
static_assert(offsetof(LNTNodeEntry, node_id) == 0, "LNTNodeEntry.node_id offset");
static_assert(offsetof(LNTNodeEntry, byte_offset) == 8, "LNTNodeEntry.byte_offset offset");
static_assert(offsetof(LNTNodeEntry, byte_length) == 16, "LNTNodeEntry.byte_length offset");
static_assert(offsetof(LNTNodeEntry, point_count) == 20, "LNTNodeEntry.point_count offset");
static_assert(offsetof(LNTNodeEntry, center) == 24, "LNTNodeEntry.center offset");
static_assert(offsetof(LNTNodeEntry, half_size) == 36, "LNTNodeEntry.half_size offset");
static_assert(offsetof(LNTNodeEntry, children_mask) == 40, "LNTNodeEntry.children_mask offset");
static_assert(offsetof(LNTNodeEntry, tree_level) == 41, "LNTNodeEntry.tree_level offset");
static_assert(offsetof(LNTNodeEntry, reserved) == 42, "LNTNodeEntry.reserved offset");

#pragma pack(pop)

// ---- derived layout facts the reader relies on -------------------------
static constexpr uint64_t kHeaderBytes    = sizeof(LNTHeader);           // 64
static constexpr uint64_t kDirHeaderBytes = sizeof(LNTDirectoryHeader);  // 16
static constexpr uint64_t kNodeEntryBytes = sizeof(LNTNodeEntry);        // 52
static constexpr uint64_t kPointBytes     = sizeof(LNTPoint);            // 11

// A node chunk's length is recorded as uint32, so the chunk must stay under
// 4 GiB. At 10 B/point that caps a node at ~429M points.
static constexpr uint32_t kMaxChunkPoints = 429496729u / kPointBytes;

// Magic comparison without a NUL-termination assumption on either side.
inline bool magic_matches(const char m[4]) {
    return m[0] == 'L' && m[1] == 'N' && m[2] == 'T' && m[3] == '\0';
}

}  // namespace lnt

#endif  // LNT_FORMAT_H
