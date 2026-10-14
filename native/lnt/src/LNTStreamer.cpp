// LNTStreamer.cpp - the streamer as its own translation unit.
//
// This file exists for the split build:
//     g++ -O3 -std=c++17 main.cpp LNTReader.cpp src/LNTStreamer.cpp -o main
//
// The streamer itself is defined INLINE in include/LNTStreamer.h on purpose,
// because the .lnt pipeline also supports a single-command build
// (`g++ -O3 -std=c++17 main.cpp -o main`) with no second object file to link
// against. Declaring the class only here would break that build; defining it
// inline keeps one source of truth and makes both forms work - the same rule
// LNTReader.h and LNTBuilderImpl.h follow.
//
// What DOES live here is the tool-facing report formatter, declared in the
// header but defined non-inline, for the same reason `describe_file()` lives in
// LNTReader.cpp: it is called by tools that are always compiled together with
// this file, and main.cpp must not reference it or the single-command harness
// build stops linking. It is NOT part of the streaming hot path - it exists so a
// camera flight can be read by a human instead of inferred from counters.

#include <cstdio>
#include <string>

#include "../include/LNTStreamer.h"

namespace lnt {

// A one-screen summary of a streamer's live state. Reports the tree it recovered,
// the last frame's cut, the cache's budget compliance and the lazy-upload
// saving in the same units as the file, so nothing here has to be recomputed by
// the caller (and therefore cannot disagree with what the streamer did).
std::string describe_streaming(const LntStreamer& streamer, const LntCacheManager& cache) {
    if (!streamer.attached()) return "streamer: not attached\n";

    const LntNodeTree& tree = streamer.tree();
    const LntLodParams& p = streamer.params();
    const LntStreamStats& s = streamer.stats();
    const LntCacheStats cs = cache.stats();

    std::string out;
    char buf[512];

    out += "  tree            : " + tree.describe() + "\n";

    std::snprintf(buf, sizeof(buf),
                  "  sse threshold   : %.1f px   max nodes %u   max level %d\n",
                  p.sse_threshold, p.max_nodes, p.max_level);
    out += buf;

    // "drawable" is the count the last ENGINE frame pushed, so it is 0 for a
    // selection-only caller that never touched the engine. Saying so here is
    // cheaper than letting a tool's reader conclude that a 100000-point cut is
    // un-drawable when it simply has not been drawn.
    std::snprintf(buf, sizeof(buf),
                  "  last frame      : %llu nodes cut, %llu points wanted, %llu points "
                  "pushed by the last engine frame\n",
                  (unsigned long long)s.cut_nodes,
                  (unsigned long long)s.wanted_points,
                  (unsigned long long)s.cut_points);
    out += buf;

    std::snprintf(buf, sizeof(buf),
                  "  cull            : %llu tested, %llu culled, %llu inside a culled "
                  "subtree, %llu descended\n",
                  (unsigned long long)s.last_selection.visited,
                  (unsigned long long)s.last_selection.culled,
                  (unsigned long long)s.last_selection.subtree_skipped,
                  (unsigned long long)s.last_selection.split);
    out += buf;

    std::snprintf(buf, sizeof(buf),
                  "  cache           : %s\n", cs.describe().c_str());
    out += buf;

    std::snprintf(buf, sizeof(buf),
                  "  budget          : %s (peak %.1f MiB of %.1f MiB)\n",
                  cs.within_budget() ? "within" : "EXCEEDED",
                  double(cs.peak_bytes) / (1024.0 * 1024.0),
                  double(cs.budget_bytes) / (1024.0 * 1024.0));
    out += buf;

    std::snprintf(buf, sizeof(buf),
                  "  lazy upload     : %llu of %llu nodes written; %llu never touched\n",
                  (unsigned long long)s.tiles_uploaded,
                  (unsigned long long)tree.nodes.size(),
                  (unsigned long long)streamer.nodes_never_uploaded());
    out += buf;

    std::snprintf(buf, sizeof(buf),
                  "  main-thread cost: last %llu ns, worst %llu ns over %llu frames\n",
                  (unsigned long long)s.main_ns, (unsigned long long)s.max_main_ns,
                  (unsigned long long)s.frames);
    out += buf;

    if (s.truncated_frames)
        out += "  NOTE            : " + std::to_string(s.truncated_frames) +
               " frame(s) hit the node cap; the cut was clipped\n";
    return out;
}

}  // namespace lnt
