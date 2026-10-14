// LNTReader.cpp - the reader as its own translation unit.
//
// This file exists for the split build:
//     g++ -O3 -std=c++17 main.cpp LNTReader.cpp -o main
//
// The Reader class itself is defined INLINE in include/LNTReader.h on purpose,
// because the .lnt pipeline also supports a single-command build
// (`g++ -O3 -std=c++17 main.cpp -o main`) which has no second object file to
// link against. Putting the class only here would break that build; defining it
// inline keeps one source of truth and makes both forms work, exactly as
// LNTBuilderImpl.h does for the builder.
//
// What DOES live here is the two dataset-wide helpers declared in the header
// but defined non-inline, because they are only ever called by tools that are
// always compiled together with this file (lnt_view.cpp). main.cpp does not
// reference them, so the single-command harness build still links with one TU.

#include <cstdio>
#include <string>
#include <vector>

#include "include/LNTReader.h"

namespace lnt {

std::string describe_file(const Reader& r) {
    if (!r.ok()) return "unreadable: " + r.error();

    const LNTHeader& h = r.header();
    const BoundingBox env = r.get_bounds();

    uint64_t leaves = 0, interior = 0;
    for (const LNTNodeEntry& e : r.nodes())
        (is_leaf_node(e) ? leaves : interior)++;

    char buf[512];
    std::string s;
    // The magic is written with its explicit NUL so the string is the real
    // 4-byte field rather than a 3-character prefix that happens to compare
    // equal to most of it.
    s += "  magic              : { 'L', 'N', 'T', '\\0' }   version ";
    s += std::to_string(h.version);
    s += "   reserved ";
    s += std::to_string(h.reserved);
    s += "\n";

    std::snprintf(buf, sizeof(buf), "  file bytes         : %llu\n",
                  (unsigned long long)r.get_file_bytes());
    s += buf;
    std::snprintf(buf, sizeof(buf), "  index_offset       : %llu\n",
                  (unsigned long long)r.get_index_offset());
    s += buf;
    std::snprintf(buf, sizeof(buf),
                  "  dataset points     : %llu   (leaf points; LOD copies not counted)\n",
                  (unsigned long long)r.get_total_points());
    s += buf;
    std::snprintf(buf, sizeof(buf),
                  "  nodes              : %llu   (%llu leaves, %llu interior)\n",
                  (unsigned long long)r.get_total_nodes(),
                  (unsigned long long)leaves, (unsigned long long)interior);
    s += buf;
    std::snprintf(buf, sizeof(buf),
                  "  envelope min       : %.3f %.3f %.3f\n",
                  h.bounds_min[0], h.bounds_min[1], h.bounds_min[2]);
    s += buf;
    std::snprintf(buf, sizeof(buf),
                  "  envelope max       : %.3f %.3f %.3f\n",
                  h.bounds_max[0], h.bounds_max[1], h.bounds_max[2]);
    s += buf;
    std::snprintf(buf, sizeof(buf),
                  "  envelope cube      : %s (%.3f x %.3f x %.3f m)\n",
                  env.is_cube(1e-3) ? "yes" : "NO", env.extent(0), env.extent(1),
                  env.extent(2));
    s += buf;
    return s;
}

bool read_all_leaf_points(Reader& r, std::vector<RawPoint>& out_points,
                          uint64_t& leaf_points, uint64_t& lod_points,
                          uint64_t& leaf_nodes, uint64_t& lod_nodes) {
    out_points.clear();
    leaf_points = lod_points = 0;
    leaf_nodes = lod_nodes = 0;
    if (!r.ok()) return false;

    std::vector<RawPoint> pts;
    for (const LNTNodeEntry& e : r.nodes()) {
        if (is_leaf_node(e)) {
            // Full-detail geometry: decode and keep.
            if (!r.read_node_points(e, pts)) return false;
            out_points.insert(out_points.end(), pts.begin(), pts.end());
            leaf_points += e.point_count;
            ++leaf_nodes;
        } else {
            // A strided LOD sample. Not decoded here and deliberately not
            // appended: every one of its points is already present in some
            // descendant leaf, so merging would duplicate geometry.
            lod_points += e.point_count;
            ++lod_nodes;
        }
    }
    return true;
}

}  // namespace lnt
