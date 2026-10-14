// LNTBoundaryScanner.h - parallel global boundary scan.
//
// Stage 1 of the .lnt pipeline. Reads every raw point once and reduces to the
// absolute global bounds, then forces the envelope into a perfect cube.
//
// Parallelism: the input is split into contiguous chunks, one per worker. Each
// worker owns a private Bounds3d and never touches another worker's memory, so
// there is no atomics contention and no false sharing on the hot loop - the
// reduction is a single merge pass over `threads` results at the end. Chunking
// by index (not by striding) keeps each worker's reads sequential, which
// matters because the input may be a memory-mapped file.
//
// Threading is std::thread only: no TBB, no OpenMP, no external dependency.

#ifndef LNT_BOUNDARY_SCANNER_H
#define LNT_BOUNDARY_SCANNER_H

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <thread>
#include <vector>

#include "LNTTypes.h"

namespace lnt {

namespace detail {

// Run `body(t)` on `n` threads, where body receives the worker index.
// Falls back to running inline when n == 1 so a single-threaded build has no
// thread-creation overhead at all.
template <typename Body>
inline void run_workers(unsigned n, const Body& body) {
    if (n <= 1) {
        body(0u);
        return;
    }
    std::vector<std::thread> pool;
    pool.reserve(n);
    for (unsigned t = 1; t < n; ++t) pool.emplace_back([&body, t] { body(t); });
    body(0u);                                  // the calling thread is worker 0
    for (auto& th : pool) th.join();
}

inline unsigned resolve_threads(unsigned requested) {
    if (requested > 0) return requested;
    const unsigned hw = std::thread::hardware_concurrency();
    return hw > 0 ? hw : 1u;
}

}  // namespace detail

class LNTBoundaryScanner {
public:
    // Scan `n` points with `threads` workers (0 => hardware concurrency).
    // Does NOT modify the bounds - the caller decides about cubicity.
    static Bounds3d scan(const RawPoint* pts, size_t n, unsigned threads = 0) {
        Bounds3d out;
        out.set_empty();
        if (pts == nullptr || n == 0) return out;

        const unsigned T = detail::resolve_threads(threads);
        const unsigned workers = static_cast<unsigned>(
            std::min<size_t>(T, std::max<size_t>(n, 1)));

        std::vector<Bounds3d> partial(workers);
        for (auto& b : partial) b.set_empty();

        const size_t chunk = (n + workers - 1) / workers;
        detail::run_workers(workers, [&](unsigned t) {
            const size_t begin = static_cast<size_t>(t) * chunk;
            const size_t end = std::min(n, begin + chunk);
            Bounds3d local;
            local.set_empty();
            // A tight loop over a POD array: the compiler vectorises this, and
            // the branch-free min/max keeps it predictable on random data.
            for (size_t i = begin; i < end; ++i) {
                const RawPoint& p = pts[i];
                local.expand(p.x, p.y, p.z);
            }
            partial[t] = local;
        });

        for (const auto& b : partial) out.expand(b);
        return out;
    }

    // Scan and force a cubic envelope. This is what the builder calls.
    static ScanResult scan_cube(const RawPoint* pts, size_t n, unsigned threads = 0) {
        const auto t0 = std::chrono::steady_clock::now();
        ScanResult r;
        r.threads = detail::resolve_threads(threads);
        r.points = n;
        r.bounds = scan(pts, n, threads);
        r.bounds.make_cube();
        r.seconds = std::chrono::duration<double>(
                        std::chrono::steady_clock::now() - t0).count();
        return r;
    }
};

}  // namespace lnt

#endif  // LNT_BOUNDARY_SCANNER_H
