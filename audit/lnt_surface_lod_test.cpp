// Native (CPU-only) checks for the persistent viewport surface LOD.
//   g++ -O2 -std=c++17 -static audit/lnt_surface_lod_test.cpp -o audit/lnt_surface_lod_test.exe
//   audit/lnt_surface_lod_test.exe audit/lod_fixture_4m.lnt
// Works on a private copy of the fixture; never writes the input file.
#define NOMINMAX
#include "../native/lnt/include/LNTSurfaceLod.h"

#include <array>
#include <cstdio>
#include <filesystem>
#include <iostream>
#include <map>
#include <set>

using namespace lnt;
namespace sl = surface_lod;
namespace sa = surface_archive;
namespace es = embedded_surface;

#define CHECK(c) do { if (!(c)) { std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); std::exit(1); } } while (0)

using Tri = std::array<double, 9>;
static Tri tri_of(const sl::TileData& t, size_t f) {
    std::array<std::array<double, 3>, 3> v;
    for (int k = 0; k < 3; ++k)
        for (int a = 0; a < 3; ++a) v[k][a] = t.positions[size_t(t.faces[f * 3 + k]) * 3 + a];
    std::sort(v.begin(), v.end());
    return {v[0][0], v[0][1], v[0][2], v[1][0], v[1][1], v[1][2], v[2][0], v[2][1], v[2][2]};
}

struct Builder {
    sl::Quadtree qt;
    std::unique_ptr<sl::NodePoints> np;
    std::string path;
    SurfaceParams P;
    std::map<uint64_t, sl::TileData> tiles;
    void init(Reader& rd, const std::string& p) {
        path = p;
        CHECK(qt.build(rd));
        np = std::make_unique<sl::NodePoints>(2ull << 30);
        np->clear(1);
        P = sl::lod_params(SurfaceParams(), qt);
    }
    void build(const std::vector<uint64_t>& keys) {
        std::vector<uint64_t> todo;
        for (uint64_t k : keys) if (!tiles.count(k)) todo.push_back(k);
        std::vector<sl::TileData> out(todo.size());
        std::atomic<size_t> next{0};
        std::atomic<bool> bad{false};
        auto work = [&]() {
            Reader rd;
            if (!rd.open(path)) { bad = true; return; }
            sl::BuildIo io{&qt, np.get(), &rd, 1, nullptr};
            for (;;) {
                const size_t i = next.fetch_add(1);
                if (i >= todo.size()) return;
                std::string e;
                if (!sl::build_tile(io, todo[i], P, out[i], &e)) { std::fprintf(stderr, "build %s\n", e.c_str()); bad = true; }
            }
        };
        std::vector<std::thread> pool;
        for (unsigned t = 0; t < std::max(2u, std::thread::hardware_concurrency() / 2); ++t) pool.emplace_back(work);
        for (auto& t : pool) t.join();
        CHECK(!bad);
        for (size_t i = 0; i < todo.size(); ++i) tiles[todo[i]] = std::move(out[i]);
    }
    void exact_cover(uint64_t k, std::vector<uint64_t>& out) const {
        if (qt.exact(k)) { out.push_back(k); return; }
        std::vector<uint64_t> kids;
        qt.children(k, kids);
        for (uint64_t c : kids) exact_cover(c, out);
    }
};

// Triangles a set of drawn same-or-mixed-level tiles shows, with the manager's rule:
// owned always, a fringe group where its owner tile is not drawn.
static std::vector<Tri> drawn_triangles(const Builder& b, const std::set<uint64_t>& drawn) {
    std::vector<Tri> out;
    static const int kDx[4] = {-1, -1, 0, 1}, kDy[4] = {0, -1, -1, -1};
    for (uint64_t k : drawn) {
        const sl::TileData& t = b.tiles.at(k);
        size_t first = 0;
        for (int g = 0; g < 5; ++g) {
            bool show = g == 0;
            if (g) {
                const int64_t x = int64_t(sl::key_x(k)) + kDx[g - 1], y = int64_t(sl::key_y(k)) + kDy[g - 1];
                show = x < 0 || y < 0 || !drawn.count(sl::make_key(sl::key_level(k), uint32_t(x), uint32_t(y)));
            }
            if (show) for (size_t f = first; f < first + t.groups[g]; ++f) out.push_back(tri_of(t, f));
            first += t.groups[g];
        }
    }
    return out;
}

// XY grid index of triangles for point-in-triangle queries.
struct Cover {
    double x0, y0, cell;
    int n;
    std::vector<std::vector<uint32_t>> bins;
    std::vector<Tri> tris;
    Cover(std::vector<Tri> t, double x0_, double y0_, double side, int n_) : x0(x0_), y0(y0_), cell(side / n_), n(n_), bins(size_t(n_) * n_), tris(std::move(t)) {
        for (uint32_t i = 0; i < tris.size(); ++i) {
            const Tri& q = tris[i];
            const double mnx = std::min({q[0], q[3], q[6]}), mxx = std::max({q[0], q[3], q[6]});
            const double mny = std::min({q[1], q[4], q[7]}), mxy = std::max({q[1], q[4], q[7]});
            const int a0 = std::max(0, int((mnx - x0) / cell)), a1 = std::min(n - 1, int((mxx - x0) / cell));
            const int b0 = std::max(0, int((mny - y0) / cell)), b1 = std::min(n - 1, int((mxy - y0) / cell));
            for (int yy = b0; yy <= b1; ++yy) for (int xx = a0; xx <= a1; ++xx) bins[size_t(yy) * n + xx].push_back(i);
        }
    }
    bool covers(double x, double y) const {
        const int xx = int((x - x0) / cell), yy = int((y - y0) / cell);
        if (xx < 0 || yy < 0 || xx >= n || yy >= n) return false;
        for (uint32_t i : bins[size_t(yy) * n + xx]) {
            const Tri& q = tris[i];
            auto side = [&](double ax, double ay, double bx, double by) { return (bx - ax) * (y - ay) - (by - ay) * (x - ax); };
            const double d0 = side(q[0], q[1], q[3], q[4]), d1 = side(q[3], q[4], q[6], q[7]), d2 = side(q[6], q[7], q[0], q[1]);
            const double eps = 1e-9;
            if ((d0 >= -eps && d1 >= -eps && d2 >= -eps) || (d0 <= eps && d1 <= eps && d2 <= eps)) return true;
        }
        return false;
    }
};

int main(int argc, char** argv) {
    const std::string fixture = argc > 1 ? argv[1] : "audit/lod_fixture_4m.lnt";
    const std::string path = "audit/surface_lod_test_copy.lnt";
    std::filesystem::copy_file(fixture, path, std::filesystem::copy_options::overwrite_existing);
    Reader rd;
    CHECK(rd.open(path));
    const uint64_t base = es::base_end(rd);
    CHECK(std::filesystem::file_size(path) == base);
    Builder b;
    b.init(rd, path);
    std::printf("fixture: %llu points, %zu nodes, octree depth %u, xy extent %.1f m\n",
                (unsigned long long)rd.directory().total_points, rd.nodes().size(), b.qt.max_level(), b.qt.xy_extent());

    // ---- 1. exact cover: every original record exactly once, full accounting -------
    std::vector<uint64_t> exact;
    b.exact_cover(sl::make_key(0, 0, 0), exact);
    auto t0 = std::chrono::steady_clock::now();
    b.build(exact);
    const double exact_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    uint64_t source = 0, represented = 0, tris = 0, fringe = 0, coverage = 0, raw_bytes = 0, stored_bytes = 0;
    std::map<unsigned, int> levels;
    for (uint64_t k : exact) {
        const sl::TileData& t = b.tiles[k];
        CHECK(t.exact());
        CHECK(t.referenced + t.coverage() == t.represented);
        source += t.source_points; represented += t.represented; tris += t.groups[0];
        fringe += t.triangles() - t.groups[0]; coverage += t.coverage();
        ++levels[sl::key_level(k)];
        // lossless record round trip
        es::MeshCodec codec;
        sa::PendingRecord rec;
        bool relative = false;
        CHECK(sl::encode_tile(t, sl::settings_digest(b.P), b.qt, codec, rec, &relative));
        CHECK(relative);
        raw_bytes += rec.meta.raw; stored_bytes += rec.stored.size();
        std::vector<char> raw;
        if (rec.meta.codec == 1) { raw.resize(size_t(rec.meta.raw)); codec.decode(raw.data(), raw.size(), rec.stored.data(), rec.stored.size()); }
        else raw = rec.stored;
        sl::TileData back;
        std::string e;
        CHECK(sl::decode_tile(raw, k, b.qt, back, &e));
        CHECK(back.positions == t.positions && back.faces == t.faces && back.vertex_class == t.vertex_class &&
              back.coverage_positions == t.coverage_positions && back.coverage_class == t.coverage_class &&
              std::equal(t.groups, t.groups + 5, back.groups) && back.source_points == t.source_points);
    }
    CHECK(source == rd.directory().total_points);
    CHECK(represented == rd.directory().total_points);   // all classes visible: every record a vertex or coverage point
    std::printf("PASS exact cover: %zu full-resolution tiles (levels:", exact.size());
    for (auto& l : levels) std::printf(" %u x%d", l.first, l.second);
    std::printf("), %llu source records each counted once, %llu owned triangles, %llu fringe, %llu coverage points, built in %.0f ms\n",
                (unsigned long long)source, (unsigned long long)tris, (unsigned long long)fringe, (unsigned long long)coverage, exact_ms);
    std::printf("PASS lossless node-relative records: %.1f MB payload -> %.1f MB stored (%.1f B per source record); positions bit-identical after decode\n",
                raw_bytes / 1e6, stored_bytes / 1e6, double(stored_bytes) / double(source));

    // ---- 2. same triangles as the existing whole-cut mesher on the same tiles -------
    for (unsigned level : {2u, b.qt.max_level()}) {
        std::vector<uint64_t> keys;
        const uint32_t n = 1u << level;
        for (uint32_t y = 0; y < n; ++y) for (uint32_t x = 0; x < n; ++x)
            if (b.qt.exists(sl::make_key(level, x, y))) keys.push_back(sl::make_key(level, x, y));
        if (keys.size() > 400) keys.resize(400);           // a contiguous band is enough
        b.build(keys);
        // Reference: one global run of the whole-cut mesher over every tile of the band.
        std::vector<SurfaceTile> cells(keys.size());
        sl::KeptNodes keep;
        Reader r2;
        CHECK(r2.open(path));
        sl::BuildIo io{&b.qt, b.np.get(), &r2, 1, nullptr};
        for (size_t i = 0; i < keys.size(); ++i) CHECK(sl::gather_tile(io, keys[i], cells[i], keep, nullptr));
        std::vector<const SurfaceTile*> refs;
        for (auto& c : cells) refs.push_back(&c);
        const detail::TileXIndex xi(refs);
        double L[3];
        surface_light_vector(b.P.azimuth_deg, b.P.elevation_deg, L);
        uint64_t compared = 0, interior = 0;
        std::set<uint64_t> in_band(keys.begin(), keys.end());
        for (size_t i = 0; i < keys.size(); ++i) {
            // Only tiles whose 8 neighbours are all in the band see the same halo.
            bool full = true;
            for (int dy = -1; dy <= 1; ++dy) for (int dx = -1; dx <= 1; ++dx) {
                const int64_t x = int64_t(sl::key_x(keys[i])) + dx, y = int64_t(sl::key_y(keys[i])) + dy;
                if (x < 0 || y < 0 || x >= n || y >= n) continue;
                const uint64_t nk = sl::make_key(level, uint32_t(x), uint32_t(y));
                if (b.qt.exists(nk) && !in_band.count(nk)) full = false;
            }
            if (!full) continue;
            ++interior;
            detail::TileMesh m;
            detail::mesh_one_tile(refs, i, xi, b.P, L, m);
            std::multiset<Tri> ref, got;
            for (size_t f = 0; f < m.faces.size() / 3; ++f) {
                std::array<std::array<double, 3>, 3> v;
                for (int k = 0; k < 3; ++k) for (int a = 0; a < 3; ++a) v[k][a] = m.positions[size_t(m.faces[f * 3 + k]) * 3 + a];
                std::sort(v.begin(), v.end());
                ref.insert({v[0][0], v[0][1], v[0][2], v[1][0], v[1][1], v[1][2], v[2][0], v[2][1], v[2][2]});
            }
            const sl::TileData& t = b.tiles[keys[i]];
            for (size_t f = 0; f < t.groups[0]; ++f) got.insert(tri_of(t, f));
            CHECK(ref == got);
            CHECK(m.coverage_point_indices.size() == t.coverage());
            compared += got.size();
        }
        std::printf("PASS level %u: %llu owned triangles in %llu tiles identical to the whole-cut mesher (tile identity is camera-independent)\n",
                    level, (unsigned long long)compared, (unsigned long long)interior);
    }

    // ---- 3. mixed-LOD borders: no crack where both levels have surface -------------
    {
        int checked_pairs = 0;
        uint64_t samples = 0, required = 0;
        for (unsigned d = 1; d + 1 <= b.qt.max_level() && checked_pairs < 6; ++d) {
            const uint32_t n = 1u << d;
            for (uint32_t y = 0; y < n && checked_pairs < 6; ++y)
                for (uint32_t x = 1; x < n && checked_pairs < 6; ++x) {
                    const uint64_t A = sl::make_key(d, x - 1, y), B = sl::make_key(d, x, y);
                    if (!b.qt.exists(A) || !b.qt.exists(B) || b.qt.exact(A) || b.qt.exact(B)) continue;
                    std::vector<uint64_t> ka, kb, need{A, B};
                    b.qt.children(A, ka); b.qt.children(B, kb);
                    // Every same-level neighbour the fringe rule may consult must exist as data.
                    for (uint64_t c : ka) need.push_back(c);
                    for (uint64_t c : kb) need.push_back(c);
                    b.build(need);
                    double mn[2], mx[2];
                    b.qt.cell_xy(B, mn, mx);
                    const double border = mn[0], w = mx[0] - mn[0];
                    const std::set<uint64_t> lvl_d{A, B};
                    std::set<uint64_t> lvl_d1(ka.begin(), ka.end());
                    lvl_d1.insert(kb.begin(), kb.end());
                    std::set<uint64_t> mixed1{A}, mixed2{B};
                    mixed1.insert(kb.begin(), kb.end());
                    mixed2.insert(ka.begin(), ka.end());
                    const double x0 = border - w, y0 = mn[1];
                    const Cover cd(drawn_triangles(b, lvl_d), x0, y0, 2 * w, 64), cd1(drawn_triangles(b, lvl_d1), x0, y0, 2 * w, 64);
                    const Cover m1(drawn_triangles(b, mixed1), x0, y0, 2 * w, 64), m2(drawn_triangles(b, mixed2), x0, y0, 2 * w, 64);
                    const double step = w / 2000.0;
                    for (double yy = mn[1] + step * 0.5; yy < mx[1]; yy += step)
                        for (double xx = border - w * 0.03; xx <= border + w * 0.03; xx += step) {
                            ++samples;
                            if (!cd.covers(xx, yy) || !cd1.covers(xx, yy)) continue;
                            ++required;
                            if (!m1.covers(xx, yy) || !m2.covers(xx, yy)) {
                                std::fprintf(stderr, "crack at %.4f %.4f between level %u and %u\n", xx, yy, d, d + 1);
                                CHECK(false);
                            }
                        }
                    ++checked_pairs;
                }
        }
        CHECK(checked_pairs > 0);
        std::printf("PASS mixed-LOD borders: %d coarse/fine border pairs, %llu of %llu border samples have surface at both levels and every one is covered in both mixed configurations\n",
                    checked_pairs, (unsigned long long)required, (unsigned long long)samples);
    }

    // ---- 4. archive: commit, reopen, integrity, interruption, invalidation ----------
    {
        sa::File f0;
        CHECK(f0.open_read(path, nullptr));
        es::Digest dir{}, src{};
        CHECK(sa::directory_digest(f0, rd.get_index_offset(), base, dir));
        CHECK(sa::source_digest(f0, base, src));
        const int64_t mtime0 = f0.mtime();
        f0.close();
        const es::Digest settings = sl::settings_digest(b.P);
        es::MeshCodec codec;
        auto records_for = [&](const std::vector<uint64_t>& keys, const es::Digest& s) {
            std::vector<sa::PendingRecord> recs;
            for (uint64_t k : keys) { sa::PendingRecord r; sl::encode_tile(b.tiles.at(k), s, b.qt, codec, r); recs.push_back(std::move(r)); }
            return recs;
        };
        std::vector<uint64_t> first(exact.begin(), exact.begin() + exact.size() / 2), second(exact.begin() + exact.size() / 2, exact.end());
        auto recs = records_for(first, settings);
        sa::CommitResult c1 = sa::commit(path, base, rd.get_index_offset(), src, mtime0, recs);
        CHECK(c1.ok);
        auto reopen = [&]() { sa::File f; CHECK(f.open_read(path, nullptr)); return sa::read_state(f, base, dir); };
        sa::State st = reopen();
        CHECK(st.present && st.stamp_ok() && st.entries.size() == first.size() && st.slot_of(settings) == 0);
        {   // records decode to the exact built tiles
            sa::File f; CHECK(f.open_read(path, nullptr));
            for (const auto& e : st.entries) {
                std::vector<char> raw; std::string why; sl::TileData t;
                CHECK(sa::read_record(f, e, settings, raw, codec, &why) && sl::decode_tile(raw, e.key, b.qt, t, &why));
                CHECK(t.positions == b.tiles[e.key].positions && t.faces == b.tiles[e.key].faces);
            }
            es::Digest again{};
            CHECK(sa::source_digest(f, base, again) && again == src);
        }
        std::printf("PASS archive commit: %zu tiles stored after the directory, stamp valid, original bytes [0, %llu) unchanged\n",
                    st.entries.size(), (unsigned long long)base);
        recs = records_for(second, settings);
        sa::CommitResult c2 = sa::commit(path, base, rd.get_index_offset(), src, c1.mtime, recs);
        CHECK(c2.ok);
        st = reopen();
        CHECK(st.present && st.stamp_ok() && st.entries.size() == exact.size() && st.footer.chain_length == 2);
        // Old readers: the original reader opens the file; the legacy cache lookup misses.
        {
            Reader old;
            CHECK(old.open(path) && old.directory().total_points == rd.directory().total_points);
            es::Digest any{};
            CHECK(!es::cached_source(old, settings, any));
        }
        std::printf("PASS append commit: %zu tiles in a 2-segment chain; point-only readers still open the file\n", st.entries.size());
        // Stale writer: the file changed since the hash was taken -> refused, untouched.
        const auto size_before = std::filesystem::file_size(path);
        recs = records_for({exact[0]}, settings);
        sa::CommitResult stale = sa::commit(path, base, rd.get_index_offset(), src, c2.mtime - 10000000, recs);
        CHECK(!stale.ok && stale.source_changed && std::filesystem::file_size(path) == size_before);
        // Corrupt one record: only that record is rejected.
        {
            const auto& e = st.entries[st.entries.size() / 2];
            std::fstream io(path, std::ios::in | std::ios::out | std::ios::binary);
            io.seekg(std::streamoff(e.offset + sizeof(sa::BlobHeader) + 5)); char c; io.read(&c, 1); c ^= 0x5a;
            io.seekp(std::streamoff(e.offset + sizeof(sa::BlobHeader) + 5)); io.write(&c, 1);
        }
        {
            sa::File f; CHECK(f.open_read(path, nullptr));
            int bad = 0, good = 0;
            for (const auto& e : st.entries) { std::vector<char> raw; std::string why; (sa::read_record(f, e, settings, raw, codec, &why) ? good : bad)++; }
            CHECK(bad == 1 && good == int(st.entries.size()) - 1);
        }
        std::printf("PASS corrupted record rejected by CRC (1 of %zu); stale writer refused without touching the file\n", st.entries.size());
        // Torn commit: junk after the footer -> previous footer recovered; next commit truncates it.
        {
            std::ofstream junk(path, std::ios::binary | std::ios::app);
            std::vector<char> j(100000, 'x');
            junk.write(j.data(), j.size());
        }
        st = reopen();
        CHECK(st.present && st.recovered && !st.stamp_ok() && st.entries.size() == exact.size());
        {
            sa::File f; CHECK(f.open_read(path, nullptr));
            es::Digest again{};
            CHECK(sa::source_digest(f, base, again) && again == src);       // original data intact
        }
        // Truncated mid-footer: same recovery.
        std::filesystem::resize_file(path, std::filesystem::file_size(path) - 100000 - 100);
        st = reopen();
        CHECK(st.present && st.recovered && st.footer.chain_length == 1 && st.entries.size() == first.size());
        sa::File mf; CHECK(mf.open_read(path, nullptr)); const int64_t mt = mf.mtime(); mf.close();
        recs = records_for(second, settings);
        sa::CommitResult c3 = sa::commit(path, base, rd.get_index_offset(), src, mt, recs);
        CHECK(c3.ok);
        st = reopen();
        CHECK(st.present && !st.recovered && st.stamp_ok() && st.entries.size() == exact.size());
        std::printf("PASS interrupted commits: torn tail and truncated footer recover the last good index; the next commit truncates the torn bytes\n");
        // Settings slots: up to 8 configurations coexist; a 9th restarts the archive.
        int64_t m = c3.mtime;
        for (int i = 1; i <= 8; ++i) {
            SurfaceParams q = b.P; q.class_visible[100 + i] = 0;
            recs = records_for({exact[0]}, sl::settings_digest(q));
            sa::CommitResult c = sa::commit(path, base, rd.get_index_offset(), src, m, recs);
            CHECK(c.ok); m = c.mtime;
            st = reopen();
            CHECK(st.present && st.footer.slot_count == uint32_t(std::min(i + 1, 8)) ? true : st.footer.slot_count == 1);
        }
        CHECK(st.footer.slot_count == 1 && st.entries.size() == 1);
        std::printf("PASS settings slots: 8 class-visibility configurations coexist; a 9th restarts the derived tail\n");
        // Snapshot: the chain is compacted every 64 commits.
        for (int i = 0; i < 70; ++i) {
            recs = records_for({exact[size_t(i) % exact.size()]}, settings);
            sa::CommitResult c = sa::commit(path, base, rd.get_index_offset(), src, m, recs);
            CHECK(c.ok); m = c.mtime;
        }
        st = reopen();
        CHECK(st.present && st.stamp_ok() && st.footer.chain_length < 64);
        std::printf("PASS 70 more commits: index chain compacted to %u segment(s), %zu tiles reachable\n", st.footer.chain_length, st.entries.size());
        // Growth bound: a commit past the bound restarts the tail with just its records.
        {
            const uint64_t tail = std::filesystem::file_size(path) - base;
            recs = records_for({exact[1]}, settings);
            sa::CommitResult c = sa::commit(path, base, rd.get_index_offset(), src, m, recs, tail / 2);
            CHECK(c.ok); m = c.mtime;
            st = reopen();
            CHECK(st.present && st.entries.size() == 1 && std::filesystem::file_size(path) - base < tail / 2);
            std::printf("PASS growth bound: a commit beyond the bound restarted the derived tail (%.1f MB -> %.1f MB)\n",
                        tail / 1e6, (std::filesystem::file_size(path) - base) / 1e6);
        }
        // Original data change: the stamp breaks and the full hash disagrees.
        {
            std::fstream io(path, std::ios::in | std::ios::out | std::ios::binary);
            io.seekg(64 + 10); char c; io.read(&c, 1); c ^= 1; io.seekp(64 + 10); io.write(&c, 1);
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        st = reopen();
        CHECK(st.present);
        {
            sa::File f; CHECK(f.open_read(path, nullptr));
            es::Digest now{};
            CHECK(sa::source_digest(f, base, now) && now != st.footer.source_hash);
        }
        std::printf("PASS a changed classification byte invalidates the archive (stamp %s, source hash differs)\n", st.stamp_ok() ? "unchanged!" : "broken");
        CHECK(!st.stamp_ok());
    }

    // ---- 5. legacy tail, read-only file, UNC path ------------------------------------
    {
        std::filesystem::copy_file(fixture, path, std::filesystem::copy_options::overwrite_existing);
        Reader r3; CHECK(r3.open(path));
        es::Digest src = es::source_digest(r3);
        SurfaceMesh mesh; mesh.positions = {0, 0, 0, 1, 0, 0, 0, 1, 0}; mesh.faces = {0, 1, 2}; mesh.vertex_class = {2, 5, 2};
        mesh.blocks = {3}; mesh.visible_points = mesh.mesh_referenced_points = 3; mesh.input_points = r3.directory().total_points;
        CHECK(es::save(path, base, src, src, mesh, 2, SurfaceMethod::Facet, true));
        sa::File f; CHECK(f.open_read(path, nullptr));
        es::Digest dir{}; CHECK(sa::directory_digest(f, r3.get_index_offset(), base, dir));
        sa::State st = sa::read_state(f, base, dir);
        CHECK(!st.present && st.legacy_tail);
        const int64_t mt = f.mtime(); f.close();
        es::MeshCodec codec;
        std::vector<sa::PendingRecord> recs(1);
        sl::encode_tile(b.tiles.at(exact[0]), sl::settings_digest(b.P), b.qt, codec, recs[0]);
        CHECK(sa::commit(path, base, r3.get_index_offset(), src, mt, recs).ok);
        CHECK(f.open_read(path, nullptr)); st = sa::read_state(f, base, dir); f.close();
        CHECK(st.present && st.entries.size() == 1);
        std::string why;
        CHECK(!es::save(path, base, src, src, mesh, 2, SurfaceMethod::Facet, true, true, &why) && why.find("tiled surface archive") != std::string::npos);
        CHECK(f.open_read(path, nullptr)); st = sa::read_state(f, base, dir); CHECK(st.present && st.entries.size() == 1); const int64_t mt2 = f.mtime(); f.close();
        std::printf("PASS legacy whole-file mesh replaced by the archive on first commit; a later whole-file save refuses to overwrite the archive (\"%s\")\n", why.c_str());
        // Read-only file: a real, permanent error, nothing written.
        const auto size = std::filesystem::file_size(path);
        SetFileAttributesA(path.c_str(), FILE_ATTRIBUTE_READONLY);
        sa::CommitResult ro = sa::commit(path, base, r3.get_index_offset(), src, mt2, recs);
        SetFileAttributesA(path.c_str(), FILE_ATTRIBUTE_NORMAL);
        CHECK(!ro.ok && ro.permanent && std::filesystem::file_size(path) == size);
        std::printf("PASS read-only file: \"%s\" (permanent; file unchanged)\n", ro.error.c_str());
        // UNC path to this same local copy through the administrative share, when available.
        char full[MAX_PATH]; GetFullPathNameA(path.c_str(), MAX_PATH, full, nullptr);
        std::string unc = std::string("\\\\localhost\\") + char(std::tolower(full[0])) + "$" + (full + 2);
        sa::File uf;
        std::string ue;
        if (uf.open_read(unc, &ue)) {
            const int64_t um = uf.mtime(); uf.close();
            sa::CommitResult cu = sa::commit(unc, base, r3.get_index_offset(), src, um, recs);
            CHECK(cu.ok);
            CHECK(uf.open_read(unc, nullptr)); st = sa::read_state(uf, base, dir);
            std::printf("PASS UNC commit via %s: %zu tiles, stamp %s\n", unc.c_str(), st.entries.size(), st.stamp_ok() ? "valid" : "not preserved by the server (full re-hash on reopen)");
            CHECK(st.present);
            uf.close();
        } else std::printf("SKIP UNC commit: %s\n", ue.c_str());
        r3.close();
    }
    rd.close();
    std::filesystem::remove(path);
    std::printf("ALL SURFACE LOD NATIVE CHECKS PASSED\n");
    return 0;
}
