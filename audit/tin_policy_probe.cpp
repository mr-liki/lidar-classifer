// tin_policy_probe.cpp - dump the NATIVE shaded-class TIN for the parity audit.
//
// The audit (audit/tin_policy_parity.py) builds the same point set through the
// VTK/CPU pipeline and compares the two triangle sets. This probe deliberately
// reuses the production code path (plan_surface -> build_surface_mesh) so the
// measured facets are the ones the viewport uploads, not a re-implementation.
//
// Build (the DLL toolchain, no g++ needed):
//   cl /nologo /O2 /std:c++17 /EHsc /Fe:audit\tin_policy_probe.exe audit\tin_policy_probe.cpp
//
// Usage:
//   tin_policy_probe.exe <points.bin> <out.bin> <azimuth> <elevation> <ambient>
//                        <classes csv> [sharpness] [crisp] [palette.bin]
//
// points.bin  : int64 n, then n * (double x, double y, double z, uint8 class)
// palette.bin : 768 bytes, class id * 3 -> r,g,b (the app's class palette)
// out.bin     : int64 nverts, ntris, visible_points, referenced_points,
//               dropped_long_edges; then positions (3*nverts double),
//               faces (3*ntris int32), vertex_class (nverts uint8),
//               face_rgb (3*ntris uint8)
//
// `crisp` = 1 bakes the VTK crisp cell shade (_crisp_shade_chunk) instead of the plain
// hillshade, which is what makes Azimuth / Sharpness / Ambient each change the pixels the
// way the VTK shaded-classification view does.

#include "../native/lnt/include/LNTSurfaceMesh.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using namespace lnt;

static bool read_points(const char* path, std::vector<SurfacePoint>& pts) {
    FILE* f = std::fopen(path, "rb");
    if (!f) { std::fprintf(stderr, "cannot open %s\n", path); return false; }
    long long n = 0;
    if (std::fread(&n, sizeof(n), 1, f) != 1 || n < 0) { std::fclose(f); return false; }
    pts.resize(static_cast<std::size_t>(n));
    for (long long i = 0; i < n; ++i) {
        double xyz[3];
        unsigned char cls = 0;
        if (std::fread(xyz, sizeof(double), 3, f) != 3) { std::fclose(f); return false; }
        if (std::fread(&cls, 1, 1, f) != 1) { std::fclose(f); return false; }
        pts[static_cast<std::size_t>(i)] = SurfacePoint{xyz[0], xyz[1], xyz[2], cls, 0, 0, 0, 0};
    }
    std::fclose(f);
    return true;
}

int main(int argc, char** argv) {
    if (argc < 7) {
        std::fprintf(stderr,
            "usage: %s <points.bin> <out.bin> <azimuth> <elevation> <ambient> <classes csv>\n",
            argv[0]);
        return 2;
    }
    std::vector<SurfacePoint> pts;
    if (!read_points(argv[1], pts)) return 1;

    SurfaceParams P;
    for (int i = 0; i < 256; ++i) P.class_visible[i] = 0;
    char* cursor = argv[6];
    for (char* tok = std::strtok(cursor, ","); tok; tok = std::strtok(nullptr, ",")) {
        const int c = std::atoi(tok);
        if (c >= 0 && c < 256) P.class_visible[c] = 1;
    }
    P.azimuth_deg = std::atof(argv[3]);
    P.elevation_deg = std::atof(argv[4]);
    P.ambient = std::atof(argv[5]);
    P.method = SurfaceMethod::Facet;
    P.color_source = SurfaceColor::Class;
    if (argc > 7) {
        P.surface.azimuth_deg = float(P.azimuth_deg);
        P.surface.light_elevation_deg = float(P.elevation_deg);
        P.surface.ambient = float(P.ambient);
        P.surface.sharpness = float(std::atof(argv[7]));
        P.crisp_shade = argc > 8 && std::atoi(argv[8]) != 0;
    }

    unsigned visible_classes = 0;
    for (int i = 0; i < 256; ++i) visible_classes += P.class_visible[i] ? 1u : 0u;
    for (int i = 0; i < 256; ++i) {
        P.class_rgb[i][0] = 160;
        P.class_rgb[i][1] = 160;
        P.class_rgb[i][2] = 160;
    }
    if (argc > 9) {
        FILE* pal = std::fopen(argv[9], "rb");
        if (!pal) { std::fprintf(stderr, "cannot open %s\n", argv[9]); return 1; }
        unsigned char table[768];
        if (std::fread(table, 1, sizeof(table), pal) != sizeof(table)) {
            std::fclose(pal);
            std::fprintf(stderr, "short palette %s\n", argv[9]);
            return 1;
        }
        std::fclose(pal);
        for (int i = 0; i < 256; ++i) {
            P.class_rgb[i][0] = table[i * 3 + 0];
            P.class_rgb[i][1] = table[i * 3 + 1];
            P.class_rgb[i][2] = table[i * 3 + 2];
        }
    }

    // One tile: the whole cut. The VTK reference is a single global Delaunay, so a
    // single owner tile is the closest the ownership/halo machinery can get to it.
    std::vector<SurfaceTile> tiles(1);
    SurfaceTile& T = tiles[0];
    double mn[3] = {1e300, 1e300, 1e300}, mx[3] = {-1e300, -1e300, -1e300};
    for (const auto& p : pts) {
        mn[0] = std::min(mn[0], p.x); mx[0] = std::max(mx[0], p.x);
        mn[1] = std::min(mn[1], p.y); mx[1] = std::max(mx[1], p.y);
        mn[2] = std::min(mn[2], p.z); mx[2] = std::max(mx[2], p.z);
        if (P.class_visible[p.cls]) T.classes.set(p.cls);
    }
    for (int a = 0; a < 3; ++a) { T.cmin[a] = mn[a]; T.cmax[a] = mx[a]; }
    T.pts = std::move(pts);

    std::vector<const SurfaceTile*> refs{&T};
    SurfacePlan plan;
    plan_surface(refs, P, visible_classes, plan);
    std::fprintf(stderr,
        "[probe] visible=%llu spacing=%.4f extent=%.3f min_area=%.1e edge_limit=%.3f\n",
        (unsigned long long)plan.visible, plan.spacing, plan.extent, P.min_area, P.max_edge_abs);

    const SurfaceMesh mesh = build_surface_mesh(tiles, P, 1);
    const long long nverts = static_cast<long long>(mesh.positions.size() / 3);
    const long long ntris = static_cast<long long>(mesh.faces.size() / 3);
    std::fprintf(stderr,
        "[probe] vertices=%lld triangles=%lld visible=%llu referenced=%llu dropped=%llu\n",
        nverts, ntris, (unsigned long long)mesh.visible_points,
        (unsigned long long)mesh.mesh_referenced_points,
        (unsigned long long)mesh.dropped_long_edges);

    FILE* out = std::fopen(argv[2], "wb");
    if (!out) { std::fprintf(stderr, "cannot write %s\n", argv[2]); return 1; }
    const long long head[5] = {
        nverts, ntris,
        static_cast<long long>(mesh.visible_points),
        static_cast<long long>(mesh.mesh_referenced_points),
        static_cast<long long>(mesh.dropped_long_edges),
    };
    std::fwrite(head, sizeof(head), 1, out);
    if (nverts) std::fwrite(mesh.positions.data(), sizeof(double), mesh.positions.size(), out);
    if (ntris) std::fwrite(mesh.faces.data(), sizeof(int32_t), mesh.faces.size(), out);
    if (nverts) std::fwrite(mesh.vertex_class.data(), 1, mesh.vertex_class.size(), out);
    if (ntris && mesh.colors.size() >= mesh.faces.size())
        std::fwrite(mesh.colors.data(), 1, mesh.faces.size(), out);
    std::fclose(out);
    return 0;
}
