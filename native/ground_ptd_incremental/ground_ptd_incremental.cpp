// Naksha Ground PTD incremental engine (exact, Qhull-authoritative).
//
// SciPy/Qhull still builds every TIN exactly as the reference does. This
// engine replaces only the work done AFTER each Delaunay build:
//   * point-to-simplex lookup with SciPy 1.15.3 find_simplex semantics,
//   * per-candidate PTD tests (slope, distance, iteration angle, rating),
//   * one-best-candidate-per-simplex ranking or all-passing selection.
//
// Exactness rules
//   * EVERY query runs the ordered SciPy walker with its carried start state.
//     Cached/grid interior certification is quarantined: real Qhull output
//     contains non-adjacent overlapping triangles (mismatch1.npz).
//   * Candidate tests reproduce NumPy's operation order bit-for-bit
//     (measured: einsum = (p0+p2)+p1, norm = sqrt((s0+s1)+s2), UCRT acos and
//     atan2, degrees = x*(180/pi)). The Python wrapper verifies this replica
//     against NumPy on this machine before the engine is ever used.
//   * Cached evaluations are reused only when the candidate's simplex has the
//     identical ordered vertex triple (hence identical coordinates and
//     identical NumPy inputs).
//
// Compile with /fp:strict (no FMA contraction or reassociation).
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <new>
#include <vector>

#if defined(_WIN32)
#define NOMINMAX
#include <windows.h>
#define NK_EXPORT extern "C" __declspec(dllexport)
#else
#include <dlfcn.h>
#define NK_EXPORT extern "C" __attribute__((visibility("default")))
#endif

namespace {

constexpr double EPS = 100.0 * std::numeric_limits<double>::epsilon();
constexpr double CERT_INTERIOR = 1e-7;
// numpy: npy_degrees(x) = x * (180.0/NPY_PI); verified bit-identical.
constexpr double RAD2DEG = 180.0 / 3.141592653589793238462643383279502884;

using UnaryFn = double (*)(double);
using BinaryFn = double (*)(double, double);
UnaryFn g_acos = nullptr;
BinaryFn g_atan2 = nullptr;

bool load_math() {
    if (g_acos && g_atan2) return true;
#if defined(_WIN32)
    // NumPy calls the Universal CRT's acos/atan2; use the very same entry
    // points instead of any compiler intrinsic or vector math library.
    HMODULE crt = GetModuleHandleW(L"ucrtbase.dll");
    if (!crt) crt = LoadLibraryW(L"ucrtbase.dll");
    if (!crt) return false;
    g_acos = reinterpret_cast<UnaryFn>(GetProcAddress(crt, "acos"));
    g_atan2 = reinterpret_cast<BinaryFn>(GetProcAddress(crt, "atan2"));
#else
    g_acos = static_cast<UnaryFn>(&::acos);
    g_atan2 = static_cast<BinaryFn>(&::atan2);
#endif
    return g_acos && g_atan2;
}

struct Params {
    double terrain_rad;
    double iteration_angle;
    double iteration_distance;
    double edge_length_threshold;
    double stop_edge_length;
    double rating_w;           // distance_weight / 100.0, computed in Python
    double rating_one_minus_w; // 1 - w, computed in Python
    int32_t reduce_angle_edge;
    int32_t stop_triangulation;
    int32_t use_distance_as_rating;
    int32_t add_only_upward;
    int32_t one_per_simplex;
    int32_t reserved;
};

struct Tri {
    const int32_t* simplices = nullptr;
    const int32_t* neighbors = nullptr;
    const double* equations = nullptr;
    const double* transform = nullptr;
    const double* points = nullptr;  // SciPy tri.points (npts x 2)
    int64_t npoints = 0;
    int64_t ntri = 0;
    double low_x = 0, low_y = 0, high_x = 0, high_y = 0;
    double scale = 0, shift = 0;
};

inline double tf(const Tri& t, int32_t id, int row, int col) {
    return t.transform[static_cast<int64_t>(id) * 6 + row * 2 + col];
}
inline int32_t nb(const Tri& t, int32_t id, int col) {
    return t.neighbors[static_cast<int64_t>(id) * 3 + col];
}
inline double eqn(const Tri& t, int32_t id, int col) {
    return t.equations[static_cast<int64_t>(id) * 4 + col];
}

// ---- Exact port of the SciPy 1.15.3 ordered walker (from ground_ptd_exact.cpp)
int32_t brute(double x, double y, const Tri& t) {
    const double broad = std::sqrt(EPS);
    const int32_t count = static_cast<int32_t>(t.ntri);
    double bar[3];
    for (int32_t sid = 0; sid < count; ++sid) {
        if (!std::isnan(tf(t, sid, 0, 0))) {
            bar[2] = 1.0;
            bool inside = true;
            for (int k = 0; k < 2; ++k) {
                bar[k] = 0.0;
                bar[k] += tf(t, sid, k, 0) * (x - tf(t, sid, 2, 0));
                bar[k] += tf(t, sid, k, 1) * (y - tf(t, sid, 2, 1));
                bar[2] -= bar[k];
                if (!(bar[k] >= -EPS && bar[k] <= 1.0 + EPS)) { inside = false; break; }
            }
            if (inside && bar[2] >= -EPS && bar[2] <= 1.0 + EPS) return sid;
        } else {
            for (int k = 0; k < 3; ++k) {
                const int32_t other = nb(t, sid, k);
                if (other == -1 || std::isnan(tf(t, other, 0, 0))) continue;
                bar[2] = 1.0;
                for (int m = 0; m < 2; ++m) {
                    bar[m] = 0.0;
                    bar[m] += tf(t, other, m, 0) * (x - tf(t, other, 2, 0));
                    bar[m] += tf(t, other, m, 1) * (y - tf(t, other, 2, 1));
                    bar[2] -= bar[m];
                }
                bool inside = true;
                for (int m = 0; m < 3; ++m) {
                    const double low = (nb(t, other, m) == sid) ? -broad : -EPS;
                    if (!(bar[m] >= low && bar[m] <= 1.0 + EPS)) { inside = false; break; }
                }
                if (inside) return other;
            }
        }
    }
    return -1;
}

struct WalkTrace {
    int64_t paraboloid = 0;
    int64_t directed = 0;
};
// One ordered query. Returns 0 ok, 1 incomplete (caller must use SciPy).
int walk_one(double x, double y, const Tri& t, int32_t& start, int32_t& out,
             int64_t* brute_count, WalkTrace* trace = nullptr) {
    const int64_t ntri = t.ntri;
    const int32_t count = static_cast<int32_t>(ntri);
    int32_t sid = start;
    if (sid < 0 || sid >= count) sid = 0;
    double z = 0.0;
    z += x * x;
    z += y * y;
    z *= t.scale;
    z += t.shift;
    double best = eqn(t, sid, 3);
    best += eqn(t, sid, 0) * x;
    best += eqn(t, sid, 1) * y;
    best += eqn(t, sid, 2) * z;
    bool changed = true;
    int64_t steps = 0;
    while (changed) {
        if (best > 0.0) break;
        changed = false;
        for (int k = 0; k < 3; ++k) {
            const int32_t n = nb(t, sid, k);
            if (n == -1) continue;
            double dist = eqn(t, n, 3);
            dist += eqn(t, n, 0) * x;
            dist += eqn(t, n, 1) * y;
            dist += eqn(t, n, 2) * z;
            if (dist > best + EPS * (1.0 + std::fabs(best))) {
                sid = n;
                best = dist;
                changed = true;
            }
        }
        ++steps;
        if (trace) ++trace->paraboloid;
        if (steps > ntri + 1) return 1;
    }
    start = sid;
    bool finished = false;
    for (int64_t cycle = 0; cycle < 1 + ntri / 4; ++cycle) {
        if (trace) ++trace->directed;
        int inside = 1;
        double bar[3];
        for (int k = 0; k < 3; ++k) {
            if (k == 2) {
                bar[2] = 1.0;
                bar[2] -= bar[0];
                bar[2] -= bar[1];
            } else {
                bar[k] = 0.0;
                bar[k] += tf(t, sid, k, 0) * (x - tf(t, sid, 2, 0));
                bar[k] += tf(t, sid, k, 1) * (y - tf(t, sid, 2, 1));
            }
            if (bar[k] < -EPS) {
                const int32_t n = nb(t, sid, k);
                if (n == -1) { start = sid; sid = -1; inside = 2; break; }
                sid = n;
                inside = -1;
                break;
            } else if (bar[k] <= 1.0 + EPS) {
                // accepted coordinate
            } else {
                inside = 0;
            }
        }
        if (inside == -1) continue;
        if (inside == 2) { finished = true; break; }
        if (inside == 1) { start = sid; finished = true; break; }
        sid = brute(x, y, t);
        if (brute_count) ++*brute_count;
        start = sid;
        finished = true;
        break;
    }
    if (!finished) {
        sid = brute(x, y, t);
        if (brute_count) ++*brute_count;
        start = sid;
    }
    out = sid;
    return 0;
}

// Strict interior certificate in SciPy arithmetic (Stage 3/4 rule).
inline bool certified_in(const Tri& t, int32_t sid, double x, double y) {
    if (std::isnan(tf(t, sid, 0, 0))) return false;
    double b0 = 0.0, b1 = 0.0, b2 = 1.0;
    b0 += tf(t, sid, 0, 0) * (x - tf(t, sid, 2, 0));
    b0 += tf(t, sid, 0, 1) * (y - tf(t, sid, 2, 1));
    b2 -= b0;
    if (!(b0 > CERT_INTERIOR && b0 < 1.0 - CERT_INTERIOR)) return false;
    b1 += tf(t, sid, 1, 0) * (x - tf(t, sid, 2, 0));
    b1 += tf(t, sid, 1, 1) * (y - tf(t, sid, 2, 1));
    b2 -= b1;
    return b1 > CERT_INTERIOR && b1 < 1.0 - CERT_INTERIOR &&
           b2 > CERT_INTERIOR && b2 < 1.0 - CERT_INTERIOR;
}

// The walker can stop in an adjacent simplex only if SciPy's arithmetic puts
// the query within [-eps, 1+eps] of it; a degenerate neighbour can trigger the
// brute-force path with a broadened tolerance. Refuse both situations.
inline bool neighbours_clear(const Tri& t, int32_t sid, double x, double y) {
    for (int k = 0; k < 3; ++k) {
        const int32_t n = nb(t, sid, k);
        if (n == -1) continue;
        if (std::isnan(tf(t, n, 0, 0))) return false;
        double b0 = 0.0, b1 = 0.0, b2 = 1.0;
        b0 += tf(t, n, 0, 0) * (x - tf(t, n, 2, 0));
        b0 += tf(t, n, 0, 1) * (y - tf(t, n, 2, 1));
        b1 += tf(t, n, 1, 0) * (x - tf(t, n, 2, 0));
        b1 += tf(t, n, 1, 1) * (y - tf(t, n, 2, 1));
        b2 -= b0;
        b2 -= b1;
        // Clearly outside only if some coordinate is decisively below -eps.
        if (!(b0 < -2.0 * EPS - CERT_INTERIOR || b1 < -2.0 * EPS - CERT_INTERIOR ||
              b2 < -2.0 * EPS - CERT_INTERIOR)) return false;
    }
    return true;
}

// ---- Stage 4 style spatial proposal grid (never guesses; only certifies).
constexpr int64_t MAX_BIN_ENTRIES = 8000000;
constexpr int MAX_CELLS_PER_TRIANGLE = 64;
struct Grid {
    int nx = 0, ny = 0;
    double minx = 0, miny = 0, scale_x = 0, scale_y = 0;
    std::vector<int64_t> offsets;
    std::vector<int32_t> ids;
    struct Box { double x0, x1, y0, y1; };
    std::vector<Box> boxes;
    bool ok = false;

    int cell_x(double v) const noexcept {
        const double z = (v - minx) * scale_x;
        if (!std::isfinite(z) || z <= 0) return 0;
        if (z >= nx) return nx - 1;
        return static_cast<int>(z);
    }
    int cell_y(double v) const noexcept {
        const double z = (v - miny) * scale_y;
        if (!std::isfinite(z) || z <= 0) return 0;
        if (z >= ny) return ny - 1;
        return static_cast<int>(z);
    }
    void build(const Tri& t) {
        ok = false;
        const double span_x = t.high_x - t.low_x, span_y = t.high_y - t.low_y;
        if (!(span_x > 0 && span_y > 0) || t.ntri <= 0) return;
        const double aspect = std::max(0.125, std::min(8.0, span_x / span_y));
        const double target = std::max(1024.0, std::min(262144.0, double(t.ntri) / 4.0));
        nx = std::max(16, std::min(512, static_cast<int>(std::sqrt(target * aspect))));
        ny = std::max(16, std::min(512, static_cast<int>(std::sqrt(target / aspect))));
        minx = t.low_x; miny = t.low_y;
        scale_x = nx / span_x; scale_y = ny / span_y;
        if (!(std::isfinite(scale_x) && std::isfinite(scale_y))) return;
        const size_t ncell = size_t(nx) * size_t(ny);
        std::vector<int64_t> counts(ncell, 0);
        boxes.assign(size_t(t.ntri), Box{0, -1, 0, -1});
        std::vector<int32_t> span(size_t(t.ntri) * 4, -1);
        int64_t total = 0;
        for (int64_t s = 0; s < t.ntri; ++s) {
            double xs[3], ys[3];
            bool bad = false;
            for (int k = 0; k < 3; ++k) {
                const int32_t v = t.simplices[s * 3 + k];
                if (v < 0 || v >= t.npoints) { bad = true; break; }
                xs[k] = t.points[int64_t(v) * 2];
                ys[k] = t.points[int64_t(v) * 2 + 1];
            }
            if (bad) return;
            Box b{std::min(xs[0], std::min(xs[1], xs[2])), std::max(xs[0], std::max(xs[1], xs[2])),
                  std::min(ys[0], std::min(ys[1], ys[2])), std::max(ys[0], std::max(ys[1], ys[2]))};
            boxes[size_t(s)] = b;
            const int x0 = cell_x(b.x0), x1 = cell_x(b.x1), y0 = cell_y(b.y0), y1 = cell_y(b.y1);
            const int64_t cnt = int64_t(x1 - x0 + 1) * int64_t(y1 - y0 + 1);
            if (cnt <= 0 || cnt > MAX_CELLS_PER_TRIANGLE || total + cnt > MAX_BIN_ENTRIES) continue;
            span[size_t(s) * 4 + 0] = x0; span[size_t(s) * 4 + 1] = x1;
            span[size_t(s) * 4 + 2] = y0; span[size_t(s) * 4 + 3] = y1;
            total += cnt;
            for (int cy = y0; cy <= y1; ++cy)
                for (int cx = x0; cx <= x1; ++cx) ++counts[size_t(cy) * nx + cx];
        }
        offsets.assign(ncell + 1, 0);
        for (size_t i = 0; i < ncell; ++i) offsets[i + 1] = offsets[i] + counts[i];
        ids.assign(size_t(total), -1);
        std::vector<int64_t> cursor(offsets.begin(), offsets.end() - 1);
        for (int64_t s = 0; s < t.ntri; ++s) {
            const int32_t* sp = &span[size_t(s) * 4];
            if (sp[0] < 0) continue;
            for (int cy = sp[2]; cy <= sp[3]; ++cy)
                for (int cx = sp[0]; cx <= sp[1]; ++cx)
                    ids[size_t(cursor[size_t(cy) * nx + cx]++)] = static_cast<int32_t>(s);
        }
        ok = true;
    }
    // Unique certified simplex or -1 (ambiguous overlaps refuse to guess).
    int32_t certify(const Tri& t, double x, double y) const noexcept {
        if (!ok) return -1;
        const size_t bin = size_t(cell_y(y)) * nx + cell_x(x);
        int32_t found = -1;
        for (int64_t p = offsets[bin]; p < offsets[bin + 1]; ++p) {
            const int32_t s = ids[size_t(p)];
            const Box& b = boxes[size_t(s)];
            if (x < b.x0 || x > b.x1 || y < b.y0 || y > b.y1) continue;
            if (!certified_in(t, s, x, y)) continue;
            if (found >= 0 && found != s) return -1;
            found = s;
        }
        return found;
    }
};

// ---- Open-addressing map: sorted vertex triple -> simplex id.
struct TripleMap {
    std::vector<int32_t> slot_sid;
    std::vector<int32_t> keys;  // 3 per slot
    uint64_t mask = 0;
    static uint64_t hash(int32_t a, int32_t b, int32_t c) {
        uint64_t h = uint64_t(uint32_t(a)) * 0x9E3779B97F4A7C15ULL;
        h ^= uint64_t(uint32_t(b)) * 0xC2B2AE3D27D4EB4FULL + (h << 6) + (h >> 2);
        h ^= uint64_t(uint32_t(c)) * 0x165667B19E3779F9ULL + (h << 6) + (h >> 2);
        h ^= h >> 31;
        return h;
    }
    void reset(int64_t n) {
        uint64_t cap = 16;
        while (cap < uint64_t(n) * 2 + 16) cap <<= 1;
        slot_sid.assign(size_t(cap), -1);
        keys.assign(size_t(cap) * 3, 0);
        mask = cap - 1;
    }
    void insert(int32_t a, int32_t b, int32_t c, int32_t sid) {
        uint64_t i = hash(a, b, c) & mask;
        while (slot_sid[size_t(i)] != -1) {
            const int32_t* k = &keys[size_t(i) * 3];
            if (k[0] == a && k[1] == b && k[2] == c) { slot_sid[size_t(i)] = -2; return; }  // duplicate key: unusable
            i = (i + 1) & mask;
        }
        slot_sid[size_t(i)] = sid;
        int32_t* k = &keys[size_t(i) * 3];
        k[0] = a; k[1] = b; k[2] = c;
    }
    int32_t find(int32_t a, int32_t b, int32_t c) const {
        uint64_t i = hash(a, b, c) & mask;
        while (slot_sid[size_t(i)] != -1) {
            const int32_t* k = &keys[size_t(i) * 3];
            if (k[0] == a && k[1] == b && k[2] == c) return slot_sid[size_t(i)];
            i = (i + 1) & mask;
        }
        return -1;
    }
};

inline void sort3(int32_t& a, int32_t& b, int32_t& c) {
    if (a > b) std::swap(a, b);
    if (b > c) std::swap(b, c);
    if (a > b) std::swap(a, b);
}

inline double norm3(double a, double b, double c) {
    double s = a * a;
    s += b * b;
    s += c * c;
    return std::sqrt(s);
}

struct Facet {
    double me;
    double n[3];
    uint8_t tok;
};

struct Engine {
    const double* pts = nullptr;  // n x 3, owned by Python for the engine lifetime
    int64_t n = 0;
    Params prm{};
    // Per-candidate cache, indexed by local working-point id.
    std::vector<int32_t> ctri;   // 3 ordered vertex ids (local working ids)
    std::vector<uint8_t> cvalid; // location hint present
    std::vector<uint8_t> evalid; // evaluation stored for ctri
    std::vector<uint8_t> eact;
    std::vector<double> ealpha, epd;
    // Per-iteration scratch.
    std::vector<int32_t> gtri;   // ntri x 3 ordered local ids
    std::vector<Facet> facets;
    std::vector<int32_t> facet_gen;
    int32_t generation = 0;
    TripleMap map;
    Grid grid;
    std::vector<int32_t> sids;
    std::vector<int32_t> best;

    bool init(const double* p, int64_t count, const Params& params) {
        pts = p; n = count; prm = params;
        ctri.assign(size_t(n) * 3, -1);
        cvalid.assign(size_t(n), 0);
        evalid.assign(size_t(n), 0);
        eact.assign(size_t(n), 0);
        ealpha.assign(size_t(n), 0.0);
        epd.assign(size_t(n), 0.0);
        return true;
    }

    const Facet& facet(int32_t s) {
        if (facet_gen[size_t(s)] == generation) return facets[size_t(s)];
        const int32_t* g = &gtri[size_t(s) * 3];
        const double* a = pts + int64_t(g[0]) * 3;
        const double* b = pts + int64_t(g[1]) * 3;
        const double* c = pts + int64_t(g[2]) * 3;
        // cpu_facets(a, b, c), NumPy operation order.
        const double ba0 = b[0] - a[0], ba1 = b[1] - a[1], ba2 = b[2] - a[2];
        const double cb0 = c[0] - b[0], cb1 = c[1] - b[1], cb2 = c[2] - b[2];
        const double ac0 = a[0] - c[0], ac1 = a[1] - c[1], ac2 = a[2] - c[2];
        const double ca0 = c[0] - a[0], ca1 = c[1] - a[1], ca2 = c[2] - a[2];
        double me = std::max(norm3(ba0, ba1, ba2), norm3(cb0, cb1, cb2));
        me = std::max(me, norm3(ac0, ac1, ac2));
        // np.cross(u, v): cp0 = u1*v2 - u2*v1, cp1 = u2*v0 - u0*v2, cp2 = u0*v1 - u1*v0
        double n0 = ba1 * ca2; n0 -= ba2 * ca1;
        double n1 = ba2 * ca0; n1 -= ba0 * ca2;
        double n2 = ba0 * ca1; n2 -= ba1 * ca0;
        const double nl = norm3(n0, n1, n2);
        const uint8_t good = nl > 1e-12 ? 1 : 0;
        const double div = nl > 0 ? nl : 1.0;
        n0 = n0 / div; n1 = n1 / div; n2 = n2 / div;
        if (n2 < 0) { n0 = n0 * -1.0; n1 = n1 * -1.0; n2 = n2 * -1.0; }
        double clipped = n2;
        if (clipped < -1.0) clipped = -1.0;
        if (clipped > 1.0) clipped = 1.0;
        const double sl = g_acos(clipped);
        Facet& f = facets[size_t(s)];
        f.me = me; f.n[0] = n0; f.n[1] = n1; f.n[2] = n2;
        f.tok = (good && sl <= prm.terrain_rad) ? 1 : 0;
        facet_gen[size_t(s)] = generation;
        return f;
    }

    // Reference PTD tests for candidate q in simplex s; returns act.
    uint8_t evaluate(int64_t q, int32_t s, double& alpha_out, double& pd_out) {
        const Facet& f = facet(s);
        const int32_t* g = &gtri[size_t(s) * 3];
        const double* pa = pts + q * 3;
        const double* A = pts + int64_t(g[0]) * 3;
        const double diff0 = pa[0] - A[0], diff1 = pa[1] - A[1], diff2 = pa[2] - A[2];
        // np.einsum('ij,ij->i'): AVX2 lanes fma(a,b,0), horizontal (l0+l2)+(l1+l3)
        const double l0 = f.n[0] * diff0 + 0.0;
        const double l1 = f.n[1] * diff1 + 0.0;
        const double l2 = f.n[2] * diff2 + 0.0;
        double acc = l0 + l2;
        acc = acc + (l1 + 0.0);
        const double d = 0.0 + acc;
        const double pd = std::fabs(d);
        uint8_t act = f.tok;
        if (prm.stop_triangulation) act &= (f.me >= prm.stop_edge_length) ? 1 : 0;
        act &= (pd <= prm.iteration_distance) ? 1 : 0;
        if (prm.add_only_upward) act &= (d >= 0) ? 1 : 0;
        // pp = pa - d[:, None] * pn
        const double pp0 = pa[0] - d * f.n[0];
        const double pp1 = pa[1] - d * f.n[1];
        const double pp2 = pa[2] - d * f.n[2];
        double nr = 0;
        for (int k = 0; k < 3; ++k) {
            const double* V = pts + int64_t(g[k]) * 3;
            const double v = norm3(V[0] - pp0, V[1] - pp1, V[2] - pp2);
            nr = (k == 0) ? v : std::min(nr, v);
        }
        const double alpha = (nr < 1e-12) ? 90.0 : g_atan2(pd, nr) * RAD2DEG;
        double ea = prm.iteration_angle;
        if (prm.reduce_angle_edge && f.me < prm.edge_length_threshold)
            ea = prm.iteration_angle * f.me / prm.edge_length_threshold;
        if (prm.use_distance_as_rating) {
            const double ratio = (ea > 0) ? alpha / ea : 1.0;
            const double dist = (prm.iteration_distance > 0) ? pd / prm.iteration_distance : 1.0;
            act &= ((prm.rating_one_minus_w * ratio + prm.rating_w * dist) <= 1.0) ? 1 : 0;
        } else {
            act &= (alpha <= ea) ? 1 : 0;
        }
        alpha_out = alpha;
        pd_out = pd;
        return act;
    }
};

}  // namespace

// Version 2 refuses stale binaries that still use the unsafe certificates.
NK_EXPORT int nk_ptd_api_version() noexcept { return 2; }

NK_EXPORT void* nk_ptd_create(const double* pts, int64_t n, const Params* params) noexcept {
    if (!pts || n <= 0 || !params || n > std::numeric_limits<int32_t>::max()) return nullptr;
    if (!load_math()) return nullptr;
    try {
        auto* e = new Engine();
        e->init(pts, n, *params);
        return e;
    } catch (...) {
        return nullptr;
    }
}

NK_EXPORT void nk_ptd_destroy(void* handle) noexcept {
    delete static_cast<Engine*>(handle);
}

// mode 0: PTD iteration (per-simplex best or all passing per params).
// mode 1: final surface completion (all passing).
// Returns 0 ok; 2 none of the queries is inside the TIN; 1 incomplete walk
// (caller must run the original SciPy path for this call); <0 invalid input.
// stats[0..15]: queries, bbox_out, cache_cert, grid_cert, walked, brute, outside,
//               eval_reused, eval_new, passing, chosen, grid_built, guard_refusals
// times[0..3]: map, locate, evaluate, select seconds.
NK_EXPORT int nk_ptd_iterate(
        void* handle, int mode,
        const int64_t* tin_ids, int64_t ntin,
        const int32_t* simplices, const int32_t* neighbors,
        const double* equations, const double* transform, int64_t ntri,
        const double* tri_points,
        double low_x, double low_y, double high_x, double high_y,
        double parab_scale, double parab_shift,
        const int64_t* queries, int64_t nquery,
        int64_t* out_selected, int64_t* nselected,
        int32_t* out_sids, uint8_t* out_act, double* out_alpha, double* out_pd,
        int64_t* stats, double* times) noexcept {
    auto* e = static_cast<Engine*>(handle);
    if (!e || !tin_ids || !simplices || !neighbors || !equations || !transform ||
        !tri_points || !queries || !out_selected || !nselected || !stats || !times ||
        ntri <= 0 || ntin <= 0 || nquery < 0 ||
        ntri > std::numeric_limits<int32_t>::max()) return -1;
    if (!std::isfinite(parab_scale) || !std::isfinite(parab_shift)) return -1;
    for (int k = 0; k < 16; ++k) stats[k] = 0;
    for (int k = 0; k < 4; ++k) times[k] = 0;
    *nselected = 0;
    using Clock = std::chrono::steady_clock;
    try {
        auto t0 = Clock::now();
        Tri t;
        t.simplices = simplices; t.neighbors = neighbors; t.equations = equations;
        t.transform = transform; t.points = tri_points; t.npoints = ntin; t.ntri = ntri;
        t.low_x = low_x; t.low_y = low_y; t.high_x = high_x; t.high_y = high_y;
        t.scale = parab_scale; t.shift = parab_shift;
        ++e->generation;
        if (e->generation == std::numeric_limits<int32_t>::max()) {
            e->generation = 1;
            std::fill(e->facet_gen.begin(), e->facet_gen.end(), 0);
        }
        e->gtri.resize(size_t(ntri) * 3);
        if (e->facets.size() < size_t(ntri)) {
            e->facets.resize(size_t(ntri));
            e->facet_gen.resize(size_t(ntri), 0);
        }
        for (int64_t s = 0; s < ntri; ++s) {
            int32_t g[3];
            for (int k = 0; k < 3; ++k) {
                const int32_t v = simplices[s * 3 + k];
                if (v < 0 || v >= ntin) return -2;
                const int64_t id = tin_ids[v];
                if (id < 0 || id >= e->n) return -2;
                g[k] = static_cast<int32_t>(id);
                e->gtri[size_t(s) * 3 + k] = g[k];
            }
        }
        auto t1 = Clock::now();
        times[0] = std::chrono::duration<double>(t1 - t0).count();

        // ---- ordered lookup with exact SciPy semantics
        e->sids.assign(size_t(nquery), -1);
        int32_t start = 0;
        bool grid_built = false;
        int64_t cache_cert = 0, grid_cert = 0, walked = 0, brute_n = 0, bbox_out = 0,
                outside = 0, guard = 0;
        for (int64_t i = 0; i < nquery; ++i) {
            const int64_t q = queries[i];
            if (q < 0 || q >= e->n) return -3;
            const double x = e->pts[q * 3], y = e->pts[q * 3 + 1];
            if (!std::isfinite(x) || !std::isfinite(y)) return -3;
            if (x < low_x - EPS || x > high_x + EPS || y < low_y - EPS || y > high_y + EPS) {
                ++bbox_out; ++outside;
                continue;  // -1, walker start state unchanged
            }
            int32_t sid;
            ++walked;
            if (walk_one(x, y, t, start, sid, &brute_n) != 0) return 1;
            e->sids[size_t(i)] = sid;
            if (sid < 0) ++outside;
        }
        auto t2 = Clock::now();
        times[1] = std::chrono::duration<double>(t2 - t1).count();
        stats[0] = nquery; stats[1] = bbox_out; stats[2] = cache_cert; stats[3] = grid_cert;
        stats[4] = walked; stats[5] = brute_n; stats[6] = outside; stats[11] = grid_built;
        stats[12] = guard;
        if (out_sids) std::memcpy(out_sids, e->sids.data(), size_t(nquery) * sizeof(int32_t));
        if (outside == nquery) {
            times[2] = 0;
            return 2;
        }

        // ---- PTD tests (cached when the ordered vertex triple is unchanged)
        int64_t reused = 0, fresh = 0, passing = 0;
        std::vector<uint8_t> act(size_t(nquery), 0);
        std::vector<double> alpha(size_t(nquery), 0.0), pdv(size_t(nquery), 0.0);
        for (int64_t i = 0; i < nquery; ++i) {
            const int32_t s = e->sids[size_t(i)];
            if (s < 0) continue;
            const int64_t q = queries[i];
            const int32_t* g = &e->gtri[size_t(s) * 3];
            int32_t* c = &e->ctri[size_t(q) * 3];
            const bool same = e->cvalid[size_t(q)] && c[0] == g[0] && c[1] == g[1] && c[2] == g[2];
            if (same && e->evalid[size_t(q)]) {
                act[size_t(i)] = e->eact[size_t(q)];
                alpha[size_t(i)] = e->ealpha[size_t(q)];
                pdv[size_t(i)] = e->epd[size_t(q)];
                ++reused;
            } else {
                double a_, p_;
                act[size_t(i)] = e->evaluate(q, s, a_, p_);
                alpha[size_t(i)] = a_;
                pdv[size_t(i)] = p_;
                c[0] = g[0]; c[1] = g[1]; c[2] = g[2];
                e->cvalid[size_t(q)] = 1;
                e->evalid[size_t(q)] = 1;
                e->eact[size_t(q)] = act[size_t(i)];
                e->ealpha[size_t(q)] = a_;
                e->epd[size_t(q)] = p_;
                ++fresh;
            }
            if (act[size_t(i)]) ++passing;
        }
        auto t3 = Clock::now();
        times[2] = std::chrono::duration<double>(t3 - t2).count();
        if (out_act) std::memcpy(out_act, act.data(), size_t(nquery));
        if (out_alpha) std::memcpy(out_alpha, alpha.data(), size_t(nquery) * sizeof(double));
        if (out_pd) std::memcpy(out_pd, pdv.data(), size_t(nquery) * sizeof(double));

        // ---- selection
        int64_t chosen = 0;
        if (mode == 0 && e->prm.one_per_simplex) {
            e->best.assign(size_t(ntri), -1);
            for (int64_t i = 0; i < nquery; ++i) {
                if (!act[size_t(i)]) continue;
                const int32_t s = e->sids[size_t(i)];
                const int32_t old = e->best[size_t(s)];
                // lexsort((point, distance, angle, sid)): angle, then distance,
                // then point id. Passing keys are finite by construction.
                if (old == -1 || alpha[size_t(i)] < alpha[size_t(old)] ||
                    (alpha[size_t(i)] == alpha[size_t(old)] &&
                     (pdv[size_t(i)] < pdv[size_t(old)] ||
                      (pdv[size_t(i)] == pdv[size_t(old)] && queries[i] < queries[old]))))
                    e->best[size_t(s)] = static_cast<int32_t>(i);
            }
            for (int64_t s = 0; s < ntri; ++s)
                if (e->best[size_t(s)] >= 0) out_selected[chosen++] = queries[e->best[size_t(s)]];
        } else {
            for (int64_t i = 0; i < nquery; ++i)
                if (act[size_t(i)]) out_selected[chosen++] = queries[i];
        }
        *nselected = chosen;
        auto t4 = Clock::now();
        times[3] = std::chrono::duration<double>(t4 - t3).count();
        stats[7] = reused; stats[8] = fresh; stats[9] = passing; stats[10] = chosen;
        return 0;
    } catch (const std::bad_alloc&) {
        return -9;
    } catch (...) {
        return -10;
    }
}

// Bounded diagnostics use the SAME walker as the engine, not a Python replica.
// Trace columns: start before/after, paraboloid cycles, directed cycles, brute.
NK_EXPORT int nk_ptd_trace_walk(
        const double* xy, int64_t rows, const int32_t* neighbors,
        const double* equations, const double* transform, int64_t ntri,
        double low_x, double low_y, double high_x, double high_y,
        double scale, double shift, int32_t* out, int64_t* trace) noexcept {
    if (!xy || !neighbors || !equations || !transform || !out || !trace ||
        rows < 0 || rows > 1000000 || ntri <= 0 || ntri > INT32_MAX) return -1;
    for (int64_t j = 0; j < ntri * 3; ++j)
        if (neighbors[j] < -1 || neighbors[j] >= ntri) return -2;
    Tri t;
    t.neighbors = neighbors; t.equations = equations; t.transform = transform;
    t.ntri = ntri; t.low_x = low_x; t.low_y = low_y;
    t.high_x = high_x; t.high_y = high_y; t.scale = scale; t.shift = shift;
    int32_t start = 0;
    for (int64_t i = 0; i < rows; ++i) {
        const double x = xy[2 * i], y = xy[2 * i + 1];
        if (!std::isfinite(x) || !std::isfinite(y)) return -3;
        int64_t* row = trace + i * 5;
        row[0] = start; row[2] = row[3] = row[4] = 0;
        out[i] = -1;
        if (!(x < low_x - EPS || x > high_x + EPS || y < low_y - EPS || y > high_y + EPS)) {
            WalkTrace steps;
            if (walk_one(x, y, t, start, out[i], &row[4], &steps) != 0) return 1;
            row[2] = steps.paraboloid; row[3] = steps.directed;
        }
        row[1] = start;
    }
    return 0;
}

// Self-test hook: evaluate explicit (candidate, triangle) rows exactly as the
// engine does, for the Python NumPy-parity verification.
NK_EXPORT int nk_ptd_evaluate_rows(
        const double* pa, const double* A, const double* B, const double* C, int64_t rows,
        const Params* params, uint8_t* out_act, double* out_alpha, double* out_pd,
        double* out_me, double* out_nor, uint8_t* out_tok) noexcept {
    if (!pa || !A || !B || !C || !params || rows < 0 || !out_act || !out_alpha || !out_pd ||
        !out_me || !out_nor || !out_tok) return -1;
    if (!load_math()) return -2;
    try {
        for (int64_t r = 0; r < rows; ++r) {
            // Lay each row out as a 4-point working set and reuse Engine code.
            double local[12];
            std::memcpy(local + 0, pa + r * 3, 3 * sizeof(double));
            std::memcpy(local + 3, A + r * 3, 3 * sizeof(double));
            std::memcpy(local + 6, B + r * 3, 3 * sizeof(double));
            std::memcpy(local + 9, C + r * 3, 3 * sizeof(double));
            Engine e;
            e.pts = local; e.n = 4; e.prm = *params;
            e.gtri = {1, 2, 3};
            e.facets.resize(1);
            e.facet_gen.assign(1, 0);
            e.generation = 1;
            double a_, p_;
            out_act[r] = e.evaluate(0, 0, a_, p_);
            out_alpha[r] = a_;
            out_pd[r] = p_;
            const Facet& f = e.facets[0];
            out_me[r] = f.me;
            out_nor[r * 3] = f.n[0]; out_nor[r * 3 + 1] = f.n[1]; out_nor[r * 3 + 2] = f.n[2];
            out_tok[r] = f.tok;
        }
        return 0;
    } catch (...) {
        return -10;
    }
}
