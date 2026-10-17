// lnt_view_api.cpp - lnt_view.dll: a flat C API over the verified .lnt pipeline,
// for embedding in the Naksha (Qt/Python) application via ctypes.
//
//   lnt_convert_las   LAS -> .lnt (native converter, one call)
//   lnt_view_open     build the streamer + engine, rendering into a CHILD window
//                     of the HWND the caller passes (a Qt native widget)
//   lnt_view_*        input (orbit / pan / zoom / SSE / colour), resize, frame
//
// Every lnt_view_* call must be made on the thread that created the view (the
// GUI thread): the engine's window belongs to it. The child window is made
// mouse-transparent (HTTRANSPARENT), so Qt receives the mouse and wheel events
// and forwards them here as plain numbers - no Win32 message plumbing in Python.
//
// Build:  see build_lnt_view.bat

#define NOMINMAX
#include <windows.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <atomic>
#include <future>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "include/LNTBuilder.h"
#include "include/LNTCacheManager.h"
#include "include/LNTCameraController.h"
#include "include/LNTCrossSection.h"
#include "include/LNTInteractiveLoop.h"
#include "include/LNTLASReader.h"
#include "include/LNTReader.h"
#include "include/LNTRenderBridge.h"
#include "include/LNTSectionQuery.h"
#include "include/LNTShadingModes.h"
#include "include/LNTSurfaceMesh.h"
#include "include/LNTEmbeddedSurface.h"
#include "include/LNTStreamer.h"
#include "include/LNTSurfaceLod.h"
#include "include/LNTVulkanEngine.h"

using namespace lnt;

extern "C" {

struct LntViewStats {
    uint32_t active_nodes;
    uint32_t points_rendered;
    uint32_t points_wanted;
    uint32_t pending_loads;
    uint32_t evictions;
    uint32_t _pad;
    uint64_t cache_used_bytes;
    uint64_t cache_capacity_bytes;
    double frame_ms;
    double update_ms;
    double sse;
    double distance;
};

}  // extern "C"

namespace {

void set_err(char* dst, int cap, const std::string& s) {
    if (!dst || cap <= 0) return;
    std::snprintf(dst, size_t(cap), "%s", s.c_str());
}

LRESULT CALLBACK transparent_proc(HWND h, UINT m, WPARAM w, LPARAM l) {
    if (m == WM_NCHITTEST) return HTTRANSPARENT;   // Qt (the parent) gets the mouse
    if (m == WM_ERASEBKGND) return 1;              // no flicker; Vulkan paints it
    return DefWindowProcA(h, m, w, l);
}

struct View {
    Reader rd;
    VulkanEngine eng;
    LntScene scene;
    LntCacheManager cache;
    LntStreamer streamer;
    LntCameraController ctl;
    LntInteractiveLoop loop;
    BoundingBox env;
    double side = 1.0;
    double origin[3] = {0, 0, 0};
    double init_sse = 8.0;
    uint32_t w = 1, h = 1;
    bool surface_requested = false;
    std::string surface_error;
    bool surface_urgent = false; // first entry to Shading must not wait for pan debounce
    int surface_method = 0;
    int pending_surface_method = -1;
    uint64_t surface_point_counts[3]{}, pending_surface_point_counts[3]{};
    uint64_t surface_revision = 0;
    using SurfaceKey = std::array<double,16>;
    SurfaceKey active_surface_key{}, pending_surface_key{};
    SurfaceKey camera_surface_key() const {
        const auto& c=loop.camera();
        return {c.eye[0],c.eye[1],c.eye[2],c.target[0],c.target[1],c.target[2],
            c.up[0],c.up[1],c.up[2],c.aspect,c.near_clip,c.far_clip,c.fov_y_deg,
            c.ortho_half_height,double(c.ortho),double(surface_gen)};
    }
    SurfaceKey surface_key() const {
        SurfaceKey key{};
        key.back()=double(surface_gen);
        return key;
    }
    SurfaceKey selection_camera_key{};
    std::array<double,7> selection_corridor_key{};
    std::vector<uint64_t> selected_surface_ids;
    bool surface_selection_valid=false;
    const std::vector<uint64_t>& surface_cut() {
        // A complete resident file mesh already contains every camera subset.
        // Vulkan clips it to the camera; do not triangulate subsets on pan/zoom.
        if(surface_requested && !streamer.params().use_corridor &&
           surface_point_counts[0]==rd.directory().total_points &&
           !surface_ids.empty() && surface_key()==active_surface_key)
            return surface_ids;
        const auto key=camera_surface_key();
        const auto& lp=streamer.params();
        std::array<double,7> corridor_key{};
        if(lp.use_corridor) {
            corridor_key[0]=1;
            for(int a=0;a<3;++a) {
                corridor_key[1+a]=lp.corridor.min[a];
                corridor_key[4+a]=lp.corridor.max[a];
            }
        }
        if(!surface_selection_valid || key!=selection_camera_key || corridor_key!=selection_corridor_key) {
            selected_surface_ids.clear();
            const auto& camera=loop.camera();
            for(const auto& node:rd.nodes()) {
                if(!is_leaf_node(node) || !camera.visible(node,origin)) continue;
                if(lp.use_corridor && !rd.bounds_of(node).intersects(lp.corridor)) continue;
                selected_surface_ids.push_back(node.node_id);
            }
            selection_camera_key=key;
            selection_corridor_key=corridor_key;
            surface_selection_valid=true;
        }
        return selected_surface_ids;
    }
    std::vector<uint64_t> surface_ids;
    std::vector<uint64_t> pending_surface_ids;
    std::chrono::steady_clock::time_point surface_cut_changed{};
    ShadingUniforms shading{};
    float last_surface_light[3]{};
    bool has_surface_light = false;
    // Surface feature: the same lit mesh as Shading, coloured by elevation.
    bool surface_elev = false;
    bool surface_gpu_elev = false;
    uint32_t surface_ramp = uint32_t(LNTColorRamp::Hue);
    double surface_z0 = 0.0, surface_z1 = 1.0;
    PackedClassColor class_lut[256]{};
    bool class_lut_uploaded=false;
    // Crisp shaded-classification lighting (LNTShadingModes.h). Defaults match the VTK
    // shaded-classification view, so a caller that never sets them still gets the same look.
    SurfaceShading surface_shading;
    bool has_surface_shading = false;
    bool surface_shading_upload = false;      // the resident mesh went through the crisp path
    uint8_t surface_class_lut[256 * 3]{};     // class id -> RGB, what the engine indexes
    // Light elevation for a cut with one visible class (app.last_shade_angle).
    double surface_single_elevation = 0.0;

    // ---- viewport-LOD surface (LNTSurfaceLod.h) ----
    // The compact crisp multi-class Shading path draws persistent, independently
    // resident surface tiles chosen per frame from the camera; the cut-based builder
    // below remains for the other surface paths (one visible class, legacy elevation colour,
    // engines without the tile arena).
    std::unique_ptr<surface_lod::Manager> lod;
    bool lod_failed = false;                 // arena or archive unusable this session
    bool lod_mode_last = false;              // last frame drew through the LOD path
    bool lod_changed = false;                // last update changed what is drawn
    int root_classes_known = 0;              // 1 = root_classes holds the root sample
    std::bitset<256> root_classes;
    uint8_t lod_lut[768]{};
    float lod_shade = -1.0f;
    unsigned visible_root_classes() {
        if (!root_classes_known) {
            root_classes_known = 1;
            for (const auto& n : rd.nodes()) {
                if (n.tree_level != 0) continue;
                std::vector<RawPoint> pts;
                if (rd.read_node_points(n, pts)) for (const auto& q : pts) root_classes.set(q.classification);
            }
        }
        unsigned count = 0;
        for (int i = 0; i < 256; ++i) count += root_classes.test(i) && class_lut[i].visible;
        return count;
    }
    // The tiled path serves exactly the compact crisp, class-coloured, multi-class case.
    bool lod_eligible() {
        if (lod_failed || (surface_elev && !surface_gpu_elev)) return false;
        if (surface_lod::env_value("NAKSHA_LNT_SURFACE_LOD") == "0") return false;
        if (!eng.has_surface_tiles() || !eng.has_compact_shaded_class_surface()) return false;
        return visible_root_classes() >= 2;
    }
    bool ensure_lod() {
        if (lod) return true;
        lod.reset(new surface_lod::Manager());
        lod->config.from_env();
        std::string e;
        if (!lod->open(rd, lnt_path, &cache.read_mutex(), &e)) {
            std::fprintf(stderr, "[LNT surface] viewport LOD unavailable: %s\n", e.c_str());
            lod.reset();
            lod_failed = true;
            return false;
        }
        return true;
    }
    // Geometry parameters and colours; a palette/light change recolours, never rebuilds.
    void lod_sync() {
        SurfaceParams p = make_surface_params();
        lod->set_params(p);
        uint8_t lut[768];
        for (int i = 0; i < 256; ++i) { lut[i*3] = class_lut[i].r; lut[i*3+1] = class_lut[i].g; lut[i*3+2] = class_lut[i].b; }
        double L[3];
        surface_light_vector(p.azimuth_deg, p.elevation_deg, L);
        const float shade = float(std::min(1.0, std::max(p.ambient, L[2])));
        if (shade != lod_shade || std::memcmp(lut, lod_lut, sizeof(lut)) != 0) {
            std::memcpy(lod_lut, lut, sizeof(lut));
            lod_shade = shade;
            lod->set_palette(lut, shade);
        }
    }

    // Shows class edits at once: surface tiles are recoloured in place (LNTSurfaceLod.h)
    // and resident point tiles of the edited nodes are re-uploaded from their patched
    // records. `pending` = not in the file yet. Returns the surface patch statistics.
    surface_lod::Manager::PatchResult show_class_edits(const std::vector<surface_lod::ClassEdit>& edits,
                                                       bool pending) {
        surface_lod::Manager::PatchResult r;
        if (edits.empty()) return r;
        if (lod) {
            uint8_t visible[256];
            for (int i = 0; i < 256; ++i) visible[i] = class_lut[i].visible;
            r = lod->patch_classes(edits, visible, &eng, pending);
            if (r.rebuilt) lod_changed = true;
        }
        std::unordered_map<uint32_t, std::vector<std::pair<uint32_t, uint8_t>>> by_node;
        for (const auto& e : edits) by_node[e.node].push_back({e.local, e.cls});
        // An uploaded point tile keeps its positions and stays drawn: only the class
        // bytes of the edited run are rewritten in place on the GPU. Re-uploading the
        // whole tile instead dropped it from the draw (a sparse ancestor stood in) until
        // the upload budget brought it back - the whole node visibly refreshed.
        const std::vector<LNTNodeEntry>& ns = rd.nodes();
        std::vector<uint8_t> span;
        for (const auto& kv : by_node) {
            if (kv.first >= ns.size()) continue;
            const uint64_t id = ns[kv.first].node_id;
            uint32_t span_first = 0;
            if (!cache.patch_classes(id, kv.second, &span, &span_first) || span.empty()) continue;
            LntTile* t = scene.tile_mut(id);
            if (!t || !t->uploaded) continue;              // uploaded later from the patched records
            if (uint64_t(span_first) + span.size() > t->count ||
                !eng.upload_point_tile(uint64_t(t->first) + span_first, span.size(), nullptr,
                                       span.data(), nullptr, nullptr))
                t->uploaded = false;
        }
        frame_sig_valid = false;          // redraw now, not after the idle debounce
        return r;
    }

    // ---- point query / edit state (section tools) ----
    std::vector<SectionHit> hits;           // last lnt_query_aabb result
    SectionEditIndex sec_idx;               // built on the first edit

    explicit View(uint64_t budget) : cache(budget) {
        for (auto& c : class_lut) c = {160, 160, 160, 1};
    }

    // Surface rebuild runs on a worker thread, never the render thread. All full-detail
    // points of every leaf are CACHED by node, so a pan or zoom only reads
    // the nodes that are new to the view. The mesh itself is always triangulated over
    // the WHOLE cut at once: that is what gives each tile border its halo of neighbour
    // points, so tiles merge without seams. The result is swapped in at once (the
    // surface on screen stays until then).
    // Build workers run at below-normal priority on at most half the cores so the UI,
    // the renderer and the point loaders are never starved. A rebuild whose cut has been
    // overtaken by the camera is cancelled; what it already built stays in the cache.
    struct NodeMeshCache {
        std::mutex mu;
        std::unordered_map<uint64_t, std::shared_ptr<const SurfaceTile>> meshes;   // complete leaf points per node
        uint64_t bytes = 0;
    };
    struct SurfaceJob {
        SurfaceMesh mesh;
        SurfaceKey key{};
        uint64_t source_points=0;
        std::string error;
        unsigned class_count = 0;
        SurfaceMethod method = SurfaceMethod::Facet;
        bool engine_lit = false;          // compact crisp mesh: no CPU-baked colours
        bool embedded_hit = false;
        bool embedded_legacy = false;
        embedded_surface::Digest source_hash{}, settings_hash{};
        std::vector<uint64_t> ids;
        bool ok = false;
        bool cancelled = false;
        uint64_t gen = 0;
        double read_mesh_ms = 0.0, concat_ms = 0.0;
        std::size_t built = 0, reused = 0;
    };
    std::string lnt_path;
    std::vector<std::unique_ptr<Reader>> surface_readers;   // one per build worker; declared
                                                            // BEFORE the future that uses them
    std::shared_ptr<NodeMeshCache> node_cache = std::make_shared<NodeMeshCache>();
    uint64_t node_cache_gen = 0;
    std::shared_ptr<std::atomic<bool>> surface_cancel = std::make_shared<std::atomic<bool>>(false);
    std::future<SurfaceJob> surface_job;
    std::future<void> surface_save;
    bool surface_job_running = false;
    uint64_t surface_gen = 0;                      // bumped when colouring changes
    std::vector<uint64_t> surface_job_ids;
    std::chrono::steady_clock::time_point surface_last_start{};

    // Idle throttle: the engine presents without waiting for vsync, so an unchanged view
    // would otherwise be redrawn ~125x a second and keep the GPU saturated while the user
    // pans. While nothing changes it is redrawn at most every 100 ms instead.
    double frame_sig[16] = {};
    bool frame_sig_valid = false;
    std::chrono::steady_clock::time_point last_render_at{};
    bool view_is_idle() {
        const double* t = ctl.target();
        const double sig[16] = {ctl.yaw(), ctl.pitch(), ctl.distance(), ctl.goal_distance(),
                                t[0], t[1], t[2], ctl.sse_threshold(),
                                ctl.ortho2d() ? 1.0 : 0.0, double(w), double(h),
                                double(surface_gen), double(shading.display_mode),
                                double(shading.color_ramp), surface_ids.empty() ? 0.0 : 1.0,
                                double(surface_requested)};
        const bool same = frame_sig_valid && std::memcmp(sig, frame_sig, sizeof(sig)) == 0;
        std::memcpy(frame_sig, sig, sizeof(sig));
        frame_sig_valid = true;
        if (!same) return false;
        if (!surface_requested && loop.stats().pending_async_loads != 0) return false;
        if (surface_requested && lod_mode_last) {
            if (lod_changed || (lod && lod->busy())) return false;
            return std::chrono::duration<double, std::milli>(
                       std::chrono::steady_clock::now() - last_render_at).count() < 100.0;
        }
        if (surface_requested &&
            (pending_surface_method >= 0 || (!surface_job_running &&
             (surface_key()!=active_surface_key || surface_cut() != surface_ids))))
            return false;                                               // surface still catching up
        // poll_surface() still runs every GUI tick. An unchanged camera needs
        // only 10 Hz redraws while CPU meshing or hidden point streaming runs.
        return std::chrono::duration<double, std::milli>(
                   std::chrono::steady_clock::now() - last_render_at).count() < 100.0;
    }

    SurfaceParams make_surface_params() const {
        SurfaceParams params;
        for (int i = 0; i < 256; ++i) {
            params.class_rgb[i][0] = class_lut[i].r;
            params.class_rgb[i][1] = class_lut[i].g;
            params.class_rgb[i][2] = class_lut[i].b;
            params.class_visible[i] = class_lut[i].visible;
        }
        // The shading uniform stores the direction light travels; the mesher
        // needs the vector from the terrain toward the source.
        const double lx = -shading.light_direction[0], ly = -shading.light_direction[1];
        const double lz = -shading.light_direction[2];
        if (lx*lx + ly*ly + lz*lz > 0.01) {
            params.azimuth_deg = std::atan2(lx, ly) * 180.0 / 3.14159265358979323846;
            if (params.azimuth_deg < 0.0) params.azimuth_deg += 360.0;
            params.elevation_deg = std::asin(std::clamp(lz, -1.0, 1.0)) *
                                   180.0 / 3.14159265358979323846;
        }
        if (surface_elev && !surface_gpu_elev) {
            params.color_source = SurfaceColor::Elevation;
            params.ramp = LNTColorRamp(surface_ramp);
            params.z_min = surface_z0;
            params.z_max = surface_z1;
        }
        // An explicit shaded-classification configuration is authoritative: the VTK view
        // stores azimuth/sharpness/ambient on the app, and the .lnt view must light the
        // terrain with exactly those values, not with a light vector derived from the
        // legacy ShadingUniforms default (which pointed 90 degrees away).
        if (has_surface_shading) {
            params.azimuth_deg = surface_shading.azimuth_deg;
            params.elevation_deg = surface_shading.light_elevation_deg;
            params.ambient = surface_shading.ambient;
        }
        params.single_elevation_deg = surface_single_elevation;
        params.compact_crisp = (!surface_elev || surface_gpu_elev) && eng.has_compact_shaded_class_surface();
        return params;
    }

    // Rebuild the resident surface (colouring or geometry changed). The node source cache
    // only holds raw points, so it stays valid.
    void invalidate_surface() {
        if (!surface_requested) return;
        if (surface_job_running) surface_cancel->store(true);
        ++surface_gen;
        node_cache_gen = surface_gen;
        surface_ids.clear();
        surface_urgent = true;
    }

    // The engine's shaded-class path indexes a 256*3 RGB table with a vertex's class id.
    void refresh_surface_class_lut() {
        for (int i = 0; i < 256; ++i) {
            surface_class_lut[i * 3 + 0] = class_lut[i].r;
            surface_class_lut[i * 3 + 1] = class_lut[i].g;
            surface_class_lut[i * 3 + 2] = class_lut[i].b;
        }
    }

    // Push-constants only: the sharpness/azimuth/ambient/key/fill derivation the Python side
    // does in _configure_microstation_color_blend_lighting(), evaluated by LNTShadingModes.h.
    bool push_crisp_shading_params() {
        float key = 0.0f, fill = 0.0f;
        shading_key_fill(surface_shading, key, fill);
        return eng.set_crisp_shading_parameters(surface_shading.azimuth_deg,
                                                surface_shading.sharpness,
                                                surface_shading.ambient, key, fill);
    }

    // VTK-parity upload. The engine lights the SAME TIN with the SAME model as the VTK
    // shaded-classification view (surface.frag modes 4/5): a crisp flat facet base plus a
    // barycentric mixed-class overlay. Returns false when the engine DLL has no such path,
    // so the caller falls back to the legacy CPU-baked colours.
    bool upload_crisp_surface(const SurfaceMesh& mesh, uint64_t revision) {
        refresh_surface_class_lut();
        if (!eng.upload_shaded_class_surface(mesh, surface_class_lut, revision)) return false;
        return push_crisp_shading_params();
    }

    // `directory` is the View's own open Reader (alive for the job's whole
    // lifetime: the future is a member declared after it).
    static SurfaceJob build_surface_job(const Reader* directory,
                                        std::vector<uint64_t> ids,
                                        SurfaceParams params, uint64_t gen,
                                        std::shared_ptr<NodeMeshCache> cache,
                                        std::vector<std::unique_ptr<Reader>>* readers,
                                        std::shared_ptr<std::atomic<bool>> cancel, SurfaceKey key) {
        SurfaceJob job;
        job.ids = ids;
        job.gen = gen;
        job.key = key;
        const auto t0 = std::chrono::steady_clock::now();
        try {
            if (ids.empty()) return job;
            // Surface selection is full-detail and independent of the point
            // renderer's SSE, residency gate, node cap and point budget.
            // The caller lends its already-open directory: no re-parse per rebuild.
            for(const auto id:ids) {
                const auto* node=directory->find_node(id);
                if(!node || !is_leaf_node(*node)) {
                    job.error="invalid full-detail surface leaf selection";return job;
                }
                job.source_points+=node->point_count;
            }
            // Only whole-file meshes are persisted, independently of point LOD.
            if(job.source_points==directory->directory().total_points) {
                job.settings_hash=embedded_surface::settings_digest(params,ids);
                embedded_surface::Digest expected{};bool loaded=false;
                const auto cache_start=std::chrono::steady_clock::now();
                if(embedded_surface::cached_source(*directory,job.settings_hash,expected)) {
                    // Overlap original-data validation with independent mesh I/O/decode.
                    // Never publish the mesh until BOTH integrity checks have passed.
                    auto fingerprint=std::async(std::launch::async,[directory](){return embedded_surface::source_digest(*directory);});
                    loaded=embedded_surface::load(*directory,expected,job.settings_hash,
                        job.mesh,job.class_count,job.method,job.engine_lit,&job.embedded_legacy);
                    job.source_hash=fingerprint.get();
                    if(job.source_hash!=expected)
                        std::fprintf(stderr,"[LNT surface] original .lnt data changed; cached geometry rejected\n");
                    loaded=loaded && job.source_hash==expected && job.mesh.input_points==job.source_points;
                } else job.source_hash=embedded_surface::source_digest(*directory);
                if(loaded) {
                    job.embedded_hit=true;job.ok=true;
                    job.read_mesh_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t0).count();
                    std::fprintf(stderr,"[LNT surface] embedded mesh loaded and validated in %.0f ms; triangulation skipped, source %llu, visible %llu, sampled-out 0\n",std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-cache_start).count(),(unsigned long long)job.source_points,(unsigned long long)job.mesh.visible_points);
                    return job;
                }
                job.mesh=SurfaceMesh{}; // stale/corrupt derived bytes never reach upload
            }
            // Host RAM for a full-detail cut, per point: 32 B decoded record + ~25 B mesh
            // vertex (xyz doubles + class) + ~24 B for its two triangles + ~10 B bookkeeping
            // and concat slack. Triangulation scratch is bounded separately by capping the
            // mesher threads (see below). Every point is meshed - none is thinned out.
            MEMORYSTATUSEX memory{};memory.dwLength=sizeof(memory);
            const bool have_memory = GlobalMemoryStatusEx(&memory) != 0;
            const uint64_t available = have_memory ? memory.ullAvailPhys : 0;
            if(have_memory && job.source_points * kSurfaceBytesPerPoint > available) {
                char msg[256];
                std::snprintf(msg, sizeof(msg),
                              "full-detail cut of %llu points needs ~%.1f GB RAM, %.1f GB free; "
                              "zoom in or free memory (previous surface retained)",
                              (unsigned long long)job.source_points,
                              double(job.source_points * kSurfaceBytesPerPoint) / 1e9,
                              double(available) / 1e9);
                job.error=msg;
                return job;
            }
            auto key_of=[](uint64_t id) {return id;};
            std::vector<std::shared_ptr<const SurfaceTile>> sampled(ids.size());
            std::vector<std::size_t> missing;
            {
                std::lock_guard<std::mutex> lk(cache->mu);
                for (std::size_t i = 0; i < ids.size(); ++i) {
                    auto it = cache->meshes.find(key_of(ids[i]));
                    if (it != cache->meshes.end()) sampled[i] = it->second; else missing.push_back(i);
                }
            }
            job.reused = ids.size() - missing.size();
            job.built = missing.size();
            // Keep disk/SMB reads near each other without changing tile identity
            // or record order inside a tile. This avoids index-order seek churn.
            std::sort(missing.begin(),missing.end(),[&](std::size_t a,std::size_t b) {
                return directory->find_node(ids[a])->byte_offset < directory->find_node(ids[b])->byte_offset;
            });

            const unsigned hw = std::thread::hardware_concurrency();
            const unsigned k_read = std::max(1u, std::min({8u, hw == 0 ? 4u : hw,
                                                           unsigned(missing.size())}));
            // Delaunay sweeps are memory-bound; running one per logical CPU
            // oversubscribes caches and raises scratch memory on large cuts.
            unsigned k_mesh = std::max(1u, std::min(12u, hw == 0 ? 4u : hw/2));
            if(const char* e=std::getenv("NAKSHA_LNT_MESH_THREADS")) {
                char* end=nullptr;
                const auto count=std::strtoul(e,&end,10);
                if(end!=e && *end=='\0' && count>=1 && count<=std::max(1u,hw))
                    k_mesh=unsigned(count);
            }

            // 1) read every record in every visible full-detail leaf
            if (!missing.empty()) {
                const unsigned k = k_read;
                while (readers->size() < k) {
                    std::unique_ptr<Reader> r(new Reader());
                    if (!r->open(directory->path())) {job.error="full-detail reader open failed";return job;}
                    readers->push_back(std::move(r));
                }
                std::atomic<std::size_t> next{0};
                std::atomic<bool> read_failed{false};
                auto work = [&](unsigned w) {
                    SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_BELOW_NORMAL);
                    Reader& rd = *(*readers)[w];
                    std::vector<RawPoint> pts;
                    for (;;) {
                        if (cancel->load()) return;
                        const std::size_t n = next.fetch_add(1);
                        if (n >= missing.size()) return;
                        const std::size_t i = missing[n];
                        auto tile = std::make_shared<SurfaceTile>();
                        try {
                            const LNTNodeEntry* node = rd.find_node(ids[i]);
                            const BoundingBox box = node ? rd.bounds_of(*node) : BoundingBox();
                            if (node) {
                                for (int a = 0; a < 3; ++a) {
                                    tile->cmin[a] = box.min[a]; tile->cmax[a] = box.max[a];
                                }
                            }
                            if (node && rd.read_node_points(*node, pts) && pts.size()==node->point_count) {
                                tile->pts.reserve(pts.size());
                                for (const RawPoint& p : pts) {
                                    tile->classes.set(p.classification);
                                    tile->pts.push_back({p.x, p.y, p.z, p.classification,
                                                         p.r, p.g, p.b, p.intensity});
                                }
                            } else { read_failed.store(true); continue; }
                        } catch (...) { read_failed.store(true); continue; }
                        sampled[i] = tile;                      // an empty tile is a result too
                        std::lock_guard<std::mutex> lk(cache->mu);
                        const uint64_t bytes=tile->pts.size()*sizeof(SurfacePoint);
                        if(cache->bytes+bytes>512ull*1024*1024) {
                            cache->meshes.clear();cache->bytes=0;
                        }
                        if(bytes<=512ull*1024*1024) {
                            cache->meshes[key_of(ids[i])] = tile;
                            cache->bytes+=bytes;
                        }
                    }
                };
                std::vector<std::thread> pool;
                for (unsigned w = 1; w < k; ++w) pool.emplace_back(work, w);
                work(0);
                for (auto& t : pool) t.join();
                if (cancel->load()) { job.cancelled = true; return job; }
                if (read_failed.load()) {job.error="full-detail leaf read failed; incomplete mesh withheld";return job;}
            }

            const auto t_read = std::chrono::steady_clock::now();
            // 2) triangulate the WHOLE cut: every tile sees its neighbours' border points
            std::vector<const SurfaceTile*> tiles;
            tiles.reserve(ids.size());
            for (const auto& t : sampled) if (t && !t->pts.empty()) tiles.push_back(t.get());
            if (tiles.empty()) return job;
            std::bitset<256> classes;
            for (const auto* tile : tiles) classes |= tile->classes;
            for (unsigned k=0; k<256; ++k) if (!params.class_visible[k]) classes.reset(k);
            job.class_count = unsigned(classes.count());
            job.method = params.method = job.class_count >= 2
                ? SurfaceMethod::Barycentric : SurfaceMethod::Facet;
            // The compact crisp engine path colours and lights the mesh itself: skip the
            // per-face colours, the mixed-face list and the per-vertex barycentric bake.
            if (params.compact_crisp && job.class_count >= 2 &&
                params.color_source == SurfaceColor::Class) {
                params.bake_colors = false;
                params.method = SurfaceMethod::Facet;
                job.engine_lit = true;
            }
            SurfacePlan plan;
            plan_surface(tiles, params, job.class_count, plan);
            std::vector<SurfaceTile> planar_tiles;
            if(params.compact_crisp && plan.visible>=1000000) {
                partition_surface_plan(tiles,params,plan,planar_tiles);
                tiles.clear();
                for(const auto& tile:planar_tiles)
                    if(tile.point_count()) tiles.push_back(&tile);
                std::fprintf(stderr,"[LNT surface] plan partition: %zu cells, %llu original visible records\n",
                             tiles.size(),(unsigned long long)plan.visible);
            }
            // Triangulation scratch is ~300 B per point of a tile (own + halo). Run only as
            // many tiles at once as the RAM left after the mesh itself allows.
            {
                std::size_t largest = 0;
                for (const auto* t : tiles) largest = std::max(largest, t->point_count());
                const uint64_t per_thread = uint64_t(largest) * 2 * 300 + (64ull << 20);
                const uint64_t mesh_bytes = job.source_points * kSurfaceBytesPerPoint;
                if (have_memory && available > mesh_bytes) {
                    const uint64_t spare = (available - mesh_bytes) * 3 / 4;
                    k_mesh = std::max(1u, std::min<unsigned>(k_mesh, unsigned(std::max<uint64_t>(spare / per_thread, 1))));
                }
            }
            // VTK geometry parity (all-points policy): degenerate and absolute edge filters.
            // Every visible point stays in; none is thinned.
            // One visible class: VTK bakes the plain hillshade (last_shade_angle) into flat
            // cell colours with lighting off - the legacy CPU-baked upload below.
            if (job.class_count == 1 && params.single_elevation_deg > 0.0)
                params.elevation_deg = params.single_elevation_deg;
            double L[3];
            surface_light_vector(params.azimuth_deg, params.elevation_deg, L);
            detail::mesh_timing_reset();
            std::vector<detail::TileMesh> parts(tiles.size());
            const detail::TileXIndex xindex(tiles);
            {
                const unsigned k = std::min<unsigned>(k_mesh, unsigned(tiles.size()));
                std::atomic<std::size_t> next{0};
                std::exception_ptr mesh_error;
                std::mutex mesh_error_mu;
                auto work = [&]() {
                    SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_BELOW_NORMAL);
                    try {
                        for (;;) {
                            if (cancel->load()) return;
                            const std::size_t i = next.fetch_add(1);
                            if (i >= tiles.size()) return;
                            detail::mesh_one_tile(tiles, i, xindex, params, L, parts[i]);
                        }
                    } catch (...) {
                        std::lock_guard<std::mutex> lock(mesh_error_mu);
                        if (!mesh_error) mesh_error = std::current_exception();
                        cancel->store(true);
                    }
                };
                std::vector<std::thread> pool;
                for (unsigned w = 1; w < k; ++w) pool.emplace_back(work);
                work();
                for (auto& t : pool) t.join();
                if (mesh_error) std::rethrow_exception(mesh_error);
                if (cancel->load()) { job.cancelled = true; return job; }
            }
            const auto t1 = std::chrono::steady_clock::now();
            std::fprintf(stderr, "[LNT surface] timing: read %.0f ms, mesh %.0f ms on %u threads\n",
                         std::chrono::duration<double, std::milli>(t_read - t0).count(),
                         std::chrono::duration<double, std::milli>(t1 - t_read).count(),
                         std::min<unsigned>(k_mesh, unsigned(tiles.size())));
            job.read_mesh_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            if (std::getenv("NAKSHA_LNT_MESH_TIMING")) {
                static const char* kPhase[5] = {"gather", "delaunay", "scan", "emit", "coverage"};
                uint64_t sum = 0;
                std::fprintf(stderr, "[LNT mesh timing]");
                for (int i = 0; i < 5; ++i) {
                    const uint64_t v = detail::g_mesh_ns[i].load(std::memory_order_relaxed);
                    sum += v;
                    std::fprintf(stderr, " %s %.0f ms", kPhase[i], double(v) / 1e6);
                }
                std::fprintf(stderr,
                             " (thread-sum %.0f ms vs wall %.0f ms; %llu pts, %llu tris)\n",
                             double(sum) / 1e6,
                             std::chrono::duration<double, std::milli>(t1 - t_read).count(),
                             (unsigned long long)detail::g_mesh_pts.load(std::memory_order_relaxed),
                             (unsigned long long)detail::g_mesh_tris.load(std::memory_order_relaxed));
            }

            // 3) concatenate
            // Count points referenced across tile ownership boundaries before
            // emitting any fallback points.
            detail::reconcile_coverage(parts, tiles, params, L);
            std::size_t verts = 0, tris = 0;
            for (const auto& m : parts) { verts += m.positions.size() / 3; tris += m.faces.size() / 3; }

            job.mesh.positions.reserve(verts * 3);
            job.mesh.faces.reserve(tris * 3);
            job.mesh.vertex_class.reserve(verts);
            if(params.method==SurfaceMethod::Barycentric) job.mesh.vertex_colors.reserve(verts*3);
            if(params.bake_colors) job.mesh.colors.reserve(tris * 3);
            for (auto& m : parts) {
                const int32_t base = int32_t(job.mesh.positions.size() / 3);
                const int32_t face_base = int32_t(job.mesh.faces.size() / 3);
                job.mesh.positions.insert(job.mesh.positions.end(), m.positions.begin(), m.positions.end());
                const std::size_t first=job.mesh.faces.size();
                job.mesh.faces.resize(first+m.faces.size());
                std::transform(m.faces.begin(),m.faces.end(),job.mesh.faces.begin()+first,
                               [base](int32_t f) {return base+f;});
                job.mesh.colors.insert(job.mesh.colors.end(), m.colors.begin(), m.colors.end());
                job.mesh.vertex_colors.insert(job.mesh.vertex_colors.end(), m.vertex_colors.begin(), m.vertex_colors.end());
                // Shaded-classification inputs for the engine's crisp path: one class byte per
                // vertex and the mixed faces, whose local indices become global here.
                job.mesh.vertex_class.insert(job.mesh.vertex_class.end(),
                                             m.vertex_class.begin(), m.vertex_class.end());
                for (int32_t mf : m.mixed_faces) job.mesh.mixed_faces.push_back(face_base + mf);
                job.mesh.coverage_positions.insert(job.mesh.coverage_positions.end(),m.coverage_positions.begin(),m.coverage_positions.end());
                job.mesh.coverage_colors.insert(job.mesh.coverage_colors.end(),m.coverage_colors.begin(),m.coverage_colors.end());
                job.mesh.visible_points+=m.visible_points;
                job.mesh.mesh_referenced_points+=m.mesh_referenced_points;
                if (!m.faces.empty()) job.mesh.blocks.push_back(uint32_t(m.faces.size()));
                m = detail::TileMesh();                       // release this part now
            }
            job.concat_ms = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - t1).count();
            job.mesh.input_points=job.source_points;
            uint64_t decoded=0;
            for(const auto& t:sampled) if(t) decoded+=t->pts.size();
            if(decoded!=job.source_points) {job.error="full-detail count mismatch; mesh withheld";return job;}
            if(job.mesh.visible_points!=plan.visible) {
                job.error="visible source record count mismatch; mesh withheld";return job;
            }
            if(job.mesh.mesh_referenced_points+job.mesh.coverage_positions.size()/3!=job.mesh.visible_points) {
                job.error="visible record conservation failed; mesh withheld";return job;
            }
            job.ok = true;
        } catch (const std::exception& e) {
            job.error=e.what();
            job.ok = false;
        } catch (...) {
            job.error="unknown full-detail surface build failure";
            job.ok = false;
        }
        return job;
    }

    // Start a rebuild for `ids` (no-op while one is running; the next frame retries).
    bool rebuild_surface(const std::vector<uint64_t>& ids) {
        if (ids.empty() || !eng.has_surface() || surface_job_running) return false;
        if (node_cache_gen != surface_gen) {                      // colouring changed: new cache
            node_cache = std::make_shared<NodeMeshCache>();
            node_cache_gen = surface_gen;
        } else {
            std::lock_guard<std::mutex> lk(node_cache->mu);
            if (node_cache->meshes.size() > 6000) {
                node_cache->meshes.clear();node_cache->bytes=0;
            }
        }
        surface_job_ids = ids;
        surface_cancel = std::make_shared<std::atomic<bool>>(false);
        surface_job = std::async(std::launch::async, &View::build_surface_job,
                                 &rd, ids, make_surface_params(), surface_gen,
                                 node_cache, &surface_readers, surface_cancel,surface_key());
        surface_job_running = true;
        surface_error.clear();
        std::fprintf(stderr,"[LNT surface] preparing surface (%zu source leaves; checking embedded mesh first)\n",ids.size());
        return true;
    }

    // The camera moved on to a different cut: stop the running build early.
    void cancel_stale_surface_job(const std::vector<uint64_t>& current_ids) {
        if (surface_job_running && surface_job_ids != current_ids) surface_cancel->store(true);
    }

    // Called every frame: swaps in a finished mesh. Returns true when one was shown.
    bool poll_surface() {
        if (!surface_job_running) return false;
        if (surface_job.wait_for(std::chrono::seconds(0)) != std::future_status::ready) return false;
        SurfaceJob job = surface_job.get();
        surface_job_running = false;
        if (job.cancelled) return false;                          // superseded; retry on next cut
        if (job.gen != surface_gen) return false;                 // built with old colouring
        if (!job.ok) {
            surface_error = job.error.empty() ? "Shading surface build failed" : job.error;
            std::fprintf(stderr,"[LNT surface] incomplete rebuild: %s; previous surface retained\n",job.error.c_str());
            surface_ids = job.ids; active_surface_key=job.key; return false; }     // do not retry the same cut forever
        if (!surface_requested) return false;                     // mode changed meanwhile
        if(job.mesh.visible_points==0) {
            eng.show_surface(false);
            surface_ids=job.ids;active_surface_key=job.key;surface_method=0;
            surface_point_counts[0]=job.source_points;
            surface_point_counts[1]=surface_point_counts[2]=0;
            return true;
        }
        const auto t0 = std::chrono::steady_clock::now();
        const uint64_t revision = ++surface_revision;
        // Shaded Classification (the class-coloured Shading mode) goes through the engine's
        // crisp-hybrid path whenever the engine has it: the engine reproduces
        // gui/shading_display.py's model in-shader, so the .lnt view and the VTK view agree.
        // Surface uses the same crisp mesh; the shader supplies elevation albedo.
        bool crisp = false;
        if ((!surface_elev || surface_gpu_elev) && job.class_count >= 2 && eng.has_shaded_class_surface() &&
            !job.mesh.vertex_class.empty())
            crisp = upload_crisp_surface(job.mesh, revision);
        surface_shading_upload = crisp;
        if (!crisp && job.engine_lit) {
            surface_error = std::string("Shading surface upload failed: ") + eng.last_error();
            // Nothing smaller exists to fall back to: report it instead of retrying forever.
            std::fprintf(stderr, "[LNT surface] %llu-point Shading mesh could not be uploaded "
                                 "(%s); zoom in to shade a smaller cut\n",
                         (unsigned long long)job.source_points, eng.last_error());
            surface_ids = job.ids; active_surface_key = job.key;
            return false;
        }
        if (!crisp && !eng.upload_surface(job.mesh, revision)) {
            surface_error = std::string("Shading surface upload failed: ") + eng.last_error();
            surface_ids = job.ids; active_surface_key = job.key;
            return false;
        }
        if (!eng.show_surface(true)) {
            surface_error = std::string("Shading surface presentation failed: ") + eng.last_error();
            surface_ids = job.ids; active_surface_key = job.key;
            return false;
        }
        surface_error.clear();
        surface_ids = job.ids; active_surface_key=job.key;
        pending_surface_method = eng.has_vertex_surface() ? int(job.method) : 0;
        pending_surface_point_counts[0]=job.source_points;
        pending_surface_point_counts[1]=job.mesh.visible_points;
        pending_surface_point_counts[2]=job.mesh.coverage_positions.size()/3;
        std::fprintf(stderr, "[LNT surface] %zu nodes (%zu read, %zu cached): read+mesh %.0f ms, "
                             "concat %.0f ms, upload %.0f ms, %llu tris, %u classes, %s, source %llu, visible %llu, coverage %llu, sampled-out 0\n",
                     job.built+job.reused, job.built, job.reused, job.read_mesh_ms, job.concat_ms,
                     std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count(),
                     (unsigned long long)job.mesh.triangles(), job.class_count,
                     // "crisp" = the engine lit it with gui/shading_display.py's model (the
                     // VTK-parity path); the other two are the legacy CPU-baked fallbacks.
                     crisp ? "crisp-engine-lit" : (pending_surface_method ? "barycentric" : "facet"),
                     (unsigned long long)job.source_points,(unsigned long long)job.mesh.visible_points,
                     (unsigned long long)(job.mesh.coverage_positions.size()/3));
        if((!job.embedded_hit || job.embedded_legacy) && job.source_points==rd.directory().total_points &&
           (!surface_save.valid() || surface_save.wait_for(std::chrono::seconds(0))==std::future_status::ready)) {
            const auto path=lnt_path;const auto offset=embedded_surface::base_end(rd);
            surface_save=std::async(std::launch::async,[path,offset,job=std::move(job)]() mutable {
                std::string why;
                const bool saved=embedded_surface::save(path,offset,job.source_hash,job.settings_hash,
                    job.mesh,job.class_count,job.method,job.engine_lit,true,&why);
                if(saved) std::fprintf(stderr,"[LNT surface] embedded mesh saved in .lnt (original point records unchanged)\n");
                else std::fprintf(stderr,"[LNT surface] embedded mesh not saved: %s (original point records unchanged)\n",why.c_str());
            });
        }
        return true;
    }

    void reset() {
        ctl.set_target(env.mid(0), env.mid(1), env.mid(2));
        ctl.set_lens(45.0, 1.0, std::max(side, 1.0) * 40.0);
        ctl.set_ortho2d(true);                     // 2D plan view, orthographic
        ctl.set_distance(std::max(side, 1.0) * 1.3);
        ctl.set_sse_threshold(init_sse);
    }

    void set_view_preset(int preset) {
        ctl.set_view_preset(preset);
        if (preset == 0) return;                    // Keep the current 3D orbit framing.

        // Fit the plane actually visible in this orthographic view. Reusing the
        // plan-view distance makes a shallow cloud almost disappear in Front/Side:
        // its small Z extent is framed inside the much larger XY dataset extent.
        const int horizontal_axis = preset == 3 ? 1 : 0; // Top/Front: X; Side: Y
        const int vertical_axis = preset == 1 ? 1 : 2;   // Top: Y; Front/Side: Z
        const double projected_width = std::max(env.extent(horizontal_axis), 1e-6);
        const double projected_height = std::max(env.extent(vertical_axis), 1e-6);
        const double aspect = std::max(double(w) / double(std::max(h, 1u)), 1e-6);
        const double half_height = 0.55 * std::max(projected_height, projected_width / aspect);
        constexpr double tan_half_fov = 0.4142135623730950488; // tan(45 degrees / 2)

        ctl.set_target(env.mid(0), env.mid(1), env.mid(2));
        double distance = half_height / tan_half_fov;
        if (preset == 2 || preset == 3) {
            // Perspective Front/Side: put the eye beyond the near face of the
            // cloud, then leave a small depth margin. These are real 3D cameras,
            // differing only in their viewing direction.
            const int depth_axis = preset == 2 ? 1 : 0;
            distance += 0.55 * std::max(env.extent(depth_axis), 1e-6);
        }
        ctl.set_distance(std::max(distance, 1.0));
    }
};

}  // namespace

extern "C" {

__declspec(dllexport) int lnt_convert_las(const char* las_path, const char* lnt_path,
                                          char* err, int err_cap, uint64_t* points_out,
                                          uint64_t* nodes_out) {
    LNTBuildParams params;
    LNTBuildStats st;
    std::string e;
    if (!convert_las_to_lnt(las_path, lnt_path, params, &st, &e)) {
        set_err(err, err_cap, e);
        return 0;
    }
    if (points_out) *points_out = st.input_points;
    if (nodes_out) *nodes_out = st.node_count;
    return 1;
}

// Reports WHERE an access violation happens (module + offset) before the host's
// own handler (ctypes) turns it into a bare OSError. Diagnostic only: it always
// continues the search, so it never changes behaviour.
static LONG WINAPI report_fault(EXCEPTION_POINTERS* ep) {
    if (ep->ExceptionRecord->ExceptionCode == EXCEPTION_ACCESS_VIOLATION) {
        void* addr = ep->ExceptionRecord->ExceptionAddress;
        HMODULE mod = nullptr;
        char path[MAX_PATH] = "?";
        GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                               GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                           static_cast<LPCSTR>(addr), &mod);
        if (mod) GetModuleFileNameA(mod, path, MAX_PATH);
        std::fprintf(stderr, "[LNT-FAULT] access violation at %p (%s + 0x%llx) %s addr %p\n", addr,
                     path, (unsigned long long)(reinterpret_cast<uintptr_t>(addr) -
                                                reinterpret_cast<uintptr_t>(mod)),
                     ep->ExceptionRecord->ExceptionInformation[0] ? "writing" : "reading",
                     reinterpret_cast<void*>(ep->ExceptionRecord->ExceptionInformation[1]));
        // A null call leaves the return address at the top of the stack.
        uintptr_t ret = 0;
        {
            const uintptr_t* sp = reinterpret_cast<const uintptr_t*>(ep->ContextRecord->Rsp);
            if (!IsBadReadPtr(sp, sizeof(uintptr_t))) ret = *sp;
        }
        HMODULE rmod = nullptr;
        char rpath[MAX_PATH] = "?";
        GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                               GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                           reinterpret_cast<LPCSTR>(ret), &rmod);
        if (rmod) GetModuleFileNameA(rmod, rpath, MAX_PATH);
        std::fprintf(stderr, "[LNT-FAULT] return address %p (%s + 0x%llx)\n",
                     reinterpret_cast<void*>(ret), rpath,
                     (unsigned long long)(ret - reinterpret_cast<uintptr_t>(rmod)));
        std::fflush(stderr);
    }
    return EXCEPTION_CONTINUE_SEARCH;
}

static void* open_impl(void* parent_hwnd, uint64_t attach_handle, void* on_window, int width, int height,
                       const char* lnt_path, const char* vulkan_dll, uint32_t budget_mb,
                       double sse, char* err, int err_cap) {
    static bool fault_hook = (AddVectoredExceptionHandler(1, report_fault), true);
    (void)fault_hook;
    std::string e;
    std::unique_ptr<View> v(new View(uint64_t(budget_mb ? budget_mb : 512) * 1024ull * 1024ull));
    v->init_sse = sse;
    v->w = uint32_t(std::max(width, 1));
    v->h = uint32_t(std::max(height, 1));
    v->lnt_path = lnt_path;
    if (!v->rd.open(lnt_path)) { set_err(err, err_cap, "cannot open .lnt: " + v->rd.error()); return nullptr; }
    v->env = v->rd.get_bounds();
    v->side = v->env.extent(0);
    for (int i = 0; i < 3; ++i) v->origin[i] = v->rd.origin(i);

    std::string dll = (vulkan_dll && vulkan_dll[0]) ? vulkan_dll : VulkanEngine::find_dll();
    if (dll.empty()) { set_err(err, err_cap, "naksha_vulkan.dll not found"); return nullptr; }
    if (!v->eng.load(dll, e)) { set_err(err, err_cap, "engine load failed: " + e); return nullptr; }
    if (!v->eng.is_available()) { set_err(err, err_cap, "no Vulkan device"); return nullptr; }
    if (on_window != nullptr) {
        // Dedicated LNT renderer on the host's own viewport window.
        if (!v->eng.open_on_window(on_window, v->w, v->h, e)) {
            set_err(err, err_cap, "renderer creation failed: " + e);
            return nullptr;
        }
    } else if (attach_handle != 0) {
        // The host application already owns THE renderer (one per process).
        if (!v->eng.attach_existing(attach_handle)) {
            set_err(err, err_cap, "cannot attach to the existing renderer");
            return nullptr;
        }
    } else {
        if (!v->eng.open(v->w, v->h, false, false, e, parent_hwnd)) {
            set_err(err, err_cap, "renderer creation failed: " + e);
            return nullptr;
        }
        if (HWND hw = v->eng.hwnd())
            SetWindowLongPtrA(hw, GWLP_WNDPROC, reinterpret_cast<LONG_PTR>(transparent_proc));
    }

    if (!v->scene.plan(v->rd, e) || !v->scene.reserve(v->eng, e)) {
        set_err(err, err_cap, "GPU arena reservation failed: " + e);
        return nullptr;
    }
    v->eng.set_render_origin(v->origin[0], v->origin[1], v->origin[2]);
    v->eng.set_clear_color(0.0f, 0.0f, 0.0f, 1.0f);
    v->eng.set_display_mode(kEngineClassification);   // the file's own classes, by palette
    v->eng.set_point_size_params(0.12f, 1.5f, 8.0f, 0.0f);
    if (!v->cache.open(lnt_path, e, 3) || !v->streamer.attach(v->rd, &v->cache, e)) {
        set_err(err, err_cap, "streamer failed: " + e);
        return nullptr;
    }
    // ---- interactive streaming policy -------------------------------------------
    // Draw only what the view can use: a point budget ties the amount of data to the
    // zoom level, a coarse base layer guarantees every cell has a stand-in (no black
    // holes while detail loads), and the GPU upload is time-sliced so a camera move
    // never stalls a frame on tile uploads. NAKSHA_LNT_POINT_BUDGET overrides the budget.
    {
        LntLodParams lp = v->streamer.params();
        lp.max_nodes = 20000;
        lp.point_budget = 6000000;
        lp.gate_split_on_residency = true;     // parent retention: a parent stays until its children land
        if (const char* env = std::getenv("NAKSHA_LNT_POINT_BUDGET"))
            if (unsigned long long b = std::strtoull(env, nullptr, 10)) lp.point_budget = b;
        v->streamer.set_params(lp);
        v->streamer.set_base_levels(2);
        v->streamer.set_upload_budget_ms(5.0);
        v->streamer.set_prefetch_ahead(4.0);
        v->streamer.set_merge_draw_ranges(true);
    }
    v->reset();
    v->loop.attach(&v->ctl, &v->cache, v->origin);
    v->loop.set_viewport(v->w, v->h);
    return v.release();
}

extern "C" {

__declspec(dllexport) void* lnt_view_open(void* parent_hwnd, int width, int height,
                                          const char* lnt_path, const char* vulkan_dll,
                                          uint32_t budget_mb, double sse, char* err,
                                          int err_cap) {
    return open_impl(parent_hwnd, 0, nullptr, width, height, lnt_path, vulkan_dll, budget_mb, sse,
                     err, err_cap);
}

// Attach to the renderer the host already created (nkv handle), instead of
// creating a window/renderer. The host owns the window, swapchain and resizing.
// Create a DEDICATED renderer on a window the host owns (the host must have shut
// its own renderer down first: the engine allows one renderer per process).
__declspec(dllexport) void* lnt_view_open_on_window(void* hwnd, int width, int height,
                                                    const char* lnt_path, const char* vulkan_dll,
                                                    uint32_t budget_mb, double sse, char* err,
                                                    int err_cap) {
    return open_impl(nullptr, 0, hwnd, width, height, lnt_path, vulkan_dll, budget_mb, sse, err,
                     err_cap);
}

__declspec(dllexport) void* lnt_view_attach(uint64_t renderer_handle, int width, int height,
                                            const char* lnt_path, const char* vulkan_dll,
                                            uint32_t budget_mb, double sse, char* err,
                                            int err_cap) {
    return open_impl(nullptr, renderer_handle, nullptr, width, height, lnt_path, vulkan_dll, budget_mb,
                     sse, err, err_cap);
}

}  // extern "C"


__declspec(dllexport) void lnt_view_close(void* h) {
    View* v = static_cast<View*>(h);
    if (!v) return;
    if (v->lod) v->lod->shutdown();                      // joins workers, commits built tiles
    if (v->eng.ok()) v->eng.clear_point_draw_ranges();   // leave the shared renderer empty
    v->eng.close();
    v->cache.close();
    delete v;
}

__declspec(dllexport) int lnt_view_resize(void* h, int w, int hh) {
    View* v = static_cast<View*>(h);
    if (!v || w < 1 || hh < 1) return 0;
    v->w = uint32_t(w);
    v->h = uint32_t(hh);
    if (HWND hw = v->eng.hwnd()) SetWindowPos(hw, nullptr, 0, 0, w, hh, SWP_NOZORDER | SWP_NOACTIVATE);
    v->eng.resize(v->w, v->h);
    v->loop.set_viewport(v->w, v->h);
    return 1;
}

__declspec(dllexport) void lnt_view_orbit(void* h, double dx_px, double dy_px) {
    if (View* v = static_cast<View*>(h))
        v->ctl.orbit(float(-dx_px * 0.005), float(dy_px * 0.005));
}
__declspec(dllexport) void lnt_view_pan(void* h, double dx_px, double dy_px) {
    if (View* v = static_cast<View*>(h)) {
        const float k = float(v->ctl.distance() * 0.0012);
        v->ctl.pan(float(-dx_px) * k, float(dy_px) * k);
    }
}
__declspec(dllexport) void lnt_view_zoom(void* h, double wheel_steps) {
    if (View* v = static_cast<View*>(h))
        v->ctl.zoom(float(-v->ctl.goal_distance() * 0.12 * wheel_steps));
}
__declspec(dllexport) void lnt_view_set_sse(void* h, double px) {
    if (View* v = static_cast<View*>(h)) v->ctl.set_sse_threshold(px);
}
// Three 256 x RGB uint8 tables (class palette, elevation ramp, intensity ramp); any may be null.
__declspec(dllexport) int lnt_view_set_luts(void* h, const uint8_t* class_rgb,
                                            const uint8_t* elev_rgb, const uint8_t* int_rgb) {
    View* v = static_cast<View*>(h);
    if (!v || !v->eng.set_point_luts(class_rgb, elev_rgb, int_rgb)) return 0;
    if (class_rgb) for (int i = 0; i < 256; ++i) {
        v->class_lut[i].r = class_rgb[i * 3];
        v->class_lut[i].g = class_rgb[i * 3 + 1];
        v->class_lut[i].b = class_rgb[i * 3 + 2];
    }
    if(class_rgb) v->class_lut_uploaded=false;
    ++v->surface_gen;
    v->node_cache_gen = v->surface_gen;
    v->surface_ids.clear();
    return 1;
}
__declspec(dllexport) int lnt_view_set_classification_lut(void* h, const uint8_t* rgba1024) {
    View* v = static_cast<View*>(h);
    if(!v || !rgba1024) return 0;
    if(v->class_lut_uploaded && v->surface_error.empty() &&
       std::memcmp(v->class_lut,rgba1024,sizeof(v->class_lut))==0) return 1;
    if (!v->eng.set_classification_lut(
        reinterpret_cast<const lnt::PackedClassColor*>(rgba1024))) return 0;
    std::memcpy(v->class_lut, rgba1024, sizeof(v->class_lut));
    v->class_lut_uploaded=true;
    if (v->surface_elev) {
        v->surface_gpu_elev = v->eng.has_surface_elevation_color() && v->visible_root_classes() >= 2;
        v->eng.set_surface_elevation_color(v->surface_gpu_elev, v->surface_z0, v->surface_z1, v->surface_ramp);
    }
    ++v->surface_gen; // discard any job built with the previous palette/visibility
    v->node_cache_gen = v->surface_gen; // samples and cached class sets are palette independent
    v->surface_ids.clear();
    return 1;
}
__declspec(dllexport) void lnt_view_set_mode(void* h, int mode) {
    if (View* v = static_cast<View*>(h)) {
        // Point presentation modes (including native view-space Depth) must
        // take ownership back from a previously selected surface/shading mesh.
        v->surface_requested = false;
        v->surface_urgent = false;
        v->eng.show_surface(false);
        v->eng.set_display_mode(mode);
    }
}
// Back to a top-down plan view (X right, Y up) keeping the current target and
// distance: what the 2D mode uses when the user switches back from 3D orbit.
__declspec(dllexport) void lnt_view_plan_view(void* h) {
    if (View* v = static_cast<View*>(h)) v->ctl.set_angles(0.0, 0.0);
}
// 1 = 2D plan (orthographic, orbit locked; the default), 0 = 3D perspective orbit.
__declspec(dllexport) void lnt_view_set_2d(void* h, int on) {
    if (View* v = static_cast<View*>(h)) v->ctl.set_ortho2d(on != 0);
}
// Native camera presets: 0 perspective 3D, 1 top, 2 front, 3 side.
__declspec(dllexport) void lnt_view_set_view(void* h, int preset) {
    if (View* v = static_cast<View*>(h)) v->set_view_preset(preset);
}
__declspec(dllexport) void lnt_view_reset(void* h) {
    if (View* v = static_cast<View*>(h)) v->reset();
}


// Presentation mode: RGB / Intensity / Elevation / Classification / NormalShading. A uniform
// update only - the resident tiles are not touched. Returns 0 when the engine DLL has no
// nkv_set_shading_mode (rebuild it) or the values are out of range.
__declspec(dllexport) int lnt_view_set_shading(void* h, const ShadingUniforms* config) {
    View* v = static_cast<View*>(h);
    if (!v || !config || !v->eng.set_shading_mode(*config)) return 0;
    const bool request_surface = config->display_mode == uint32_t(LNTDisplayMode::NormalShading);
    const bool needs_new_surface = request_surface &&
        (!v->has_surface_light || (v->surface_elev && !v->surface_gpu_elev) ||
         std::memcmp(v->last_surface_light,config->light_direction,sizeof(v->last_surface_light))!=0);
    v->shading = *config;
    v->surface_requested = request_surface;
    if (!request_surface) {
        v->eng.show_surface(false);
        v->surface_urgent=false;
    } else {
        // Stop presenting square point splats immediately, even before the
        // first full-detail surface has been built and uploaded.
        if (!v->eng.show_surface(true)) return 0;
        v->surface_elev = false;                   // Shading mode: class colours
        v->eng.set_surface_elevation_color(false, 0.0, 1.0, 0);
        std::memcpy(v->last_surface_light,config->light_direction,sizeof(v->last_surface_light));
        v->has_surface_light=true;
        if (needs_new_surface) {
            if (v->surface_job_running) v->surface_cancel->store(true);
            ++v->surface_gen;
            v->node_cache_gen=v->surface_gen; // source records remain cacheable
            v->surface_ids.clear();
        }
        if (!v->surface_ids.empty() && v->surface_key()==v->active_surface_key &&
            v->surface_cut()==v->surface_ids &&
            v->eng.show_surface(true)) v->surface_urgent=false;
        else v->surface_urgent=true;
    }
    return 1;
}
// Surface feature: call AFTER lnt_view_set_shading(NormalShading). Same lit, streamed mesh,
// coloured by world Z through `ramp` over [z0, z1] instead of by class. Colour only:
// the old surface stays on screen until the recoloured mesh has been built.
__declspec(dllexport) int lnt_view_set_surface_elevation(void* h, int on, uint32_t ramp,
                                                         double z0, double z1) {
    View* v = static_cast<View*>(h);
    if (!v || !v->surface_requested || ramp > 3) return 0;
    v->surface_elev = on != 0;
    v->surface_ramp = ramp;
    v->surface_z0 = z0;
    v->surface_z1 = z1 > z0 ? z1 : z0 + 1.0;
    // The single-class Shading path bakes its flat hillshade into cell colours;
    // retain that same path for Surface too. Multi-class uses the shared tile arena.
    v->surface_gpu_elev = v->eng.has_surface_elevation_color() && v->visible_root_classes() >= 2;
    v->eng.set_surface_elevation_color(v->surface_elev && v->surface_gpu_elev, z0, v->surface_z1, ramp);
    if (v->surface_gpu_elev) return 1;
    if (v->surface_job_running) v->surface_cancel->store(true);
    ++v->surface_gen;
    v->node_cache_gen=v->surface_gen; // elevation ramp does not change source records
    v->surface_ids.clear();
    return 1;
}
// Crisp shaded-classification lighting: the parameters _configure_microstation_color_blend_lighting()
// derives on the Python side (azimuth, raw 0..999 Sharpness, Ambient). Geometry and class
// membership do not change, so with a resident crisp surface this is a push-constant update
// only. `light_elevation_deg` <= 0 keeps the configured sun elevation.
__declspec(dllexport) int lnt_view_set_surface_shading(void* h, double azimuth_deg, double sharpness,
                                                       double ambient, double light_elevation_deg) {
    View* v = static_cast<View*>(h);
    if (!v) return 0;
    v->surface_shading.azimuth_deg = float(azimuth_deg);
    v->surface_shading.sharpness = float(sharpness);
    v->surface_shading.ambient = float(ambient);
    if (light_elevation_deg > 0.0) v->surface_shading.light_elevation_deg = float(light_elevation_deg);
    v->has_surface_shading = true;
    if (v->surface_shading_upload) {          // engine-lit mesh: relight in place
        v->push_crisp_shading_params();
        return 1;
    }
    // Legacy CPU-baked path: the shade is baked into the vertex colours, so re-light by
    // rebuilding. Same TIN, so the node source cache is still valid.
    v->invalidate_surface();
    return 1;
}
__declspec(dllexport) int lnt_view_has_shading(void* h) {
    View* v = static_cast<View*>(h);
    return (v && v->eng.has_shading_mode()) ? 1 : 0;
}
__declspec(dllexport) uint64_t lnt_view_surface_triangles(void* h) {
    View* v = static_cast<View*>(h);
    return v ? v->eng.surface_triangles() : 0;
}
// Value ranges for default ramps, from the ROOT node's LOD sample (no file scan):
// out[0..1] = z min/max, out[2..3] = z 1st/99th percentile, out[4..5] = intensity min/max
// (0..255). Returns 0 when the root chunk cannot be read.
__declspec(dllexport) int lnt_view_data_stats(void* h, double* out6) {
    View* v = static_cast<View*>(h);
    if (!v || !out6) return 0;
    const std::vector<LNTNodeEntry>& ns = v->rd.nodes();
    if (ns.empty()) return 0;
    std::vector<RawPoint> pts;
    if (!v->rd.read_node_points(ns[0], pts) || pts.empty()) return 0;
    std::vector<double> z(pts.size());
    double imin = 255.0, imax = 0.0;
    for (std::size_t i = 0; i < pts.size(); ++i) {
        z[i] = pts[i].z;
        imin = std::min<double>(imin, pts[i].intensity);
        imax = std::max<double>(imax, pts[i].intensity);
    }
    std::sort(z.begin(), z.end());
    auto pct = [&](double p) { return z[std::size_t(p * double(z.size() - 1))]; };
    out6[0] = z.front(); out6[1] = z.back();
    out6[2] = pct(0.01); out6[3] = pct(0.99);
    out6[4] = imin; out6[5] = imax;
    return 1;
}

// Upper bound on the points the cut may hold (0 = unlimited). Lower = faster, coarser.
__declspec(dllexport) void lnt_view_set_point_budget(void* h, uint64_t points) {
    if (View* v = static_cast<View*>(h)) {
        LntLodParams lp = v->streamer.params();
        lp.point_budget = points;
        v->streamer.set_params(lp);
    }
}

// ---- section tools: point query, classification write-back, screen <-> world ---------
// Collects every leaf point inside the closed world box [mn, mx]; returns the count, -1 on
// error, -2 when it exceeds max_points (0 = unlimited). Fetch them with lnt_query_fetch.
__declspec(dllexport) int64_t lnt_query_aabb(void* h, const double* mn, const double* mx,
                                             uint64_t max_points, char* err, int err_cap) {
    View* v = static_cast<View*>(h);
    if (!v || !mn || !mx) { set_err(err, err_cap, "bad arguments"); return -1; }
    std::string e;
    const int64_t n = section_query_aabb(v->rd, mn, mx, v->hits, e, max_points);
    if (n < 0) set_err(err, err_cap, e);
    return n;
}
// Collects every leaf point a plan-view brush stroke covers (path: n XY pairs; round or
// square brush of radius r swept along it). Same return values as lnt_query_aabb.
__declspec(dllexport) int64_t lnt_query_stroke(void* h, const double* path, uint64_t n, double radius,
                                               int square, uint64_t max_points, char* err, int err_cap) {
    View* v = static_cast<View*>(h);
    if (!v || (n && !path)) { set_err(err, err_cap, "bad arguments"); return -1; }
    std::string e;
    const int64_t c = section_query_stroke(v->rd, path, n, radius, square != 0, v->hits, e, max_points);
    if (c < 0) set_err(err, err_cap, e);
    return c;
}
__declspec(dllexport) int lnt_query_fetch(void* h, double* xyz, uint8_t* cls, uint8_t* inten,
                                          uint8_t* rgb, uint32_t* node_idx, uint32_t* local_idx) {
    View* v = static_cast<View*>(h);
    return (v && section_fetch(v->rd, v->hits, xyz, cls, inten, rgb, node_idx, local_idx)) ? 1 : 0;
}
__declspec(dllexport) void lnt_query_release(void* h) {
    if (View* v = static_cast<View*>(h)) std::vector<SectionHit>().swap(v->hits);
}

// Sets the classification of leaf points IN THE FILE (and their LOD copies), then makes
// the open viewer show it: the cache payloads are patched and the GPU tiles re-uploaded.
__declspec(dllexport) int lnt_set_classes(void* h, const uint32_t* node_idx,
                                          const uint32_t* local_idx, const uint8_t* new_class,
                                          uint64_t n, char* err, int err_cap) {
    View* v = static_cast<View*>(h);
    if (!v || (n && (!node_idx || !local_idx || !new_class))) { set_err(err, err_cap, "bad arguments"); return 0; }
    std::vector<uint32_t> touched;
    std::vector<SectionRecordEdit> records;
    std::string e;
    try {
        const auto write_start = std::chrono::steady_clock::now();
        if (!section_set_classes(v->rd, v->sec_idx, &v->cache.read_mutex(), node_idx, local_idx,
                                 new_class, n, touched, e, nullptr, &records)) {
            set_err(err, err_cap, e);
            return 0;
        }
        const auto written = std::chrono::steady_clock::now();
        struct Trace {
            std::chrono::steady_clock::time_point a, b;
            size_t n;
            ~Trace() {
                if (std::getenv("NAKSHA_TRACE_BRUSH"))
                    std::fprintf(stderr, "[BRUSH] write %.3f ms, show %.3f ms, %zu records\n",
                        std::chrono::duration<double, std::milli>(b-a).count(),
                        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now()-b).count(), n);
            }
        } trace{write_start, written, records.size()};
        // Every rewritten record (leaves and LOD copies) recolours its resident surface
        // vertices and point tiles in place; nothing is re-meshed unless a class's
        // visibility changed. Already previewed leaf edits are a no-op here.
        std::vector<surface_lod::ClassEdit> edits;
        edits.reserve(records.size());
        for (const auto& r : records) edits.push_back({r.node, r.local, r.x, r.y, r.z, r.cls, r.old_cls});
        v->show_class_edits(edits, false);
        if (v->lod) v->lod->end_edit();
        // The cut-based surface (one visible class, elevation colour) caches meshes with
        // baked colours: those are rebuilt.
        v->node_cache = std::make_shared<View::NodeMeshCache>();
        if (!v->lod_mode_last) {
            ++v->surface_gen;
            v->node_cache_gen = v->surface_gen;
            v->surface_ids.clear();
        }
    } catch (const std::exception& ex) {
        set_err(err, err_cap, ex.what());
        return 0;
    }
    return 1;
}

// Shows class edits immediately, before they are written (lnt_set_classes writes them
// later): resident surface tiles are recoloured in place and point tiles re-uploaded.
// xyz are the records' positions as lnt_query_fetch returned them; old_class is the
// class currently shown. out_stats (optional, 4 values): tiles patched, vertices,
// coverage points, tiles rebuilt.
__declspec(dllexport) int lnt_preview_classes(void* h, const uint32_t* node_idx, const uint32_t* local_idx,
                                              const double* xyz, const uint8_t* old_class,
                                              const uint8_t* new_class, uint64_t n, uint32_t* out_stats) {
    View* v = static_cast<View*>(h);
    if (!v || (n && (!node_idx || !local_idx || !xyz || !old_class || !new_class))) return 0;
    try {
        const auto preview_start = std::chrono::steady_clock::now();
        const std::vector<LNTNodeEntry>& ns = v->rd.nodes();
        std::vector<surface_lod::ClassEdit> edits;
        // A coarse surface uses ancestor copies, whose quantized coordinates differ
        // from leaf XYZ. Resolve the same verified record identities as write-back,
        // without opening the file for writing, so coarse tiles also change now.
        std::vector<uint32_t> touched;
        std::vector<SectionRecordEdit> records;
        std::string error;
        if (!section_set_classes(v->rd, v->sec_idx, &v->cache.read_mutex(), node_idx, local_idx,
                new_class, n, touched, error, nullptr, &records, true, old_class)) return 0;
        edits.reserve(records.size());
        for (const auto& e : records)
            edits.push_back({e.node, e.local, e.x, e.y, e.z, e.cls, e.old_cls});
        const auto resolve_end = std::chrono::steady_clock::now();
        const auto r = v->show_class_edits(edits, true);
        if (std::getenv("NAKSHA_TRACE_BRUSH")) {
            const auto end = std::chrono::steady_clock::now();
            std::fprintf(stderr, "[BRUSH] resolve %.3f ms, patch %.3f ms, %zu records\n",
                std::chrono::duration<double, std::milli>(resolve_end-preview_start).count(),
                std::chrono::duration<double, std::milli>(end-resolve_end).count(), records.size());
        }
        if (out_stats) { out_stats[0] = r.tiles; out_stats[1] = r.vertices; out_stats[2] = r.coverage; out_stats[3] = r.rebuilt; }
    } catch (const std::exception& ex) {
        std::fprintf(stderr, "[LNT] class preview failed: %s\n", ex.what());
        return 0;
    }
    return 1;
}

// 2D plan view only: window pixel (device pixels, y down) <-> world XY on the plane
// z = camera target z. Returns 0 when the view is in 3D orbit mode.
__declspec(dllexport) int lnt_view_screen_to_world(void* h, double px, double py, double* out3) {
    View* v = static_cast<View*>(h);
    if (!v || !out3 || !v->ctl.ortho2d()) return 0;
    const double tn = std::tan(v->ctl.fov_y_deg() * 3.14159265358979323846 / 360.0);
    const double half_h = std::max(v->ctl.distance() * tn, 1e-6);
    const double per_px = 2.0 * half_h / double(std::max<uint32_t>(v->h, 1));
    const double* t = v->ctl.target();
    out3[0] = t[0] + (px - 0.5 * double(v->w)) * per_px;
    out3[1] = t[1] - (py - 0.5 * double(v->h)) * per_px;
    out3[2] = t[2];
    return 1;
}
__declspec(dllexport) int lnt_view_world_to_screen(void* h, double x, double y, double* out2) {
    View* v = static_cast<View*>(h);
    if (!v || !out2 || !v->ctl.ortho2d()) return 0;
    const double tn = std::tan(v->ctl.fov_y_deg() * 3.14159265358979323846 / 360.0);
    const double half_h = std::max(v->ctl.distance() * tn, 1e-6);
    const double px_per_m = double(std::max<uint32_t>(v->h, 1)) / (2.0 * half_h);
    const double* t = v->ctl.target();
    out2[0] = 0.5 * double(v->w) + (x - t[0]) * px_per_m;
    out2[1] = 0.5 * double(v->h) - (y - t[1]) * px_per_m;
    return 1;
}
// Centre the 2D view on (x, y) and show `width_m` metres across the window width.
__declspec(dllexport) void lnt_view_frame_region(void* h, double cx, double cy, double width_m) {
    View* v = static_cast<View*>(h);
    if (!v) return;
    const double* t = v->ctl.target();
    v->ctl.set_target(cx, cy, t[2]);
    const double tn = std::tan(v->ctl.fov_y_deg() * 3.14159265358979323846 / 360.0);
    const double aspect = double(std::max<uint32_t>(v->w, 1)) / double(std::max<uint32_t>(v->h, 1));
    v->ctl.set_distance(std::max(width_m / (2.0 * tn * aspect), 1.0));
}

// ---- cross-section: GPU clip volume + streamer corridor -------------------------
// planes_abcd: plane_count (<= 4) float64 A,B,C,D quadruples in WORLD coordinates,
// normal pointing into the KEPT side. 0 planes switches clipping off. Returns 0 when
// the engine DLL predates nkv_set_clip_volume.
__declspec(dllexport) int lnt_view_set_clip_planes(void* h, const double* planes_abcd,
                                                   uint32_t plane_count) {
    View* v = static_cast<View*>(h);
    return (v && v->eng.set_clip_planes_world(planes_abcd, plane_count)) ? 1 : 0;
}
__declspec(dllexport) int lnt_view_has_clip_volume(void* h) {
    View* v = static_cast<View*>(h);
    return (v && v->eng.has_clip_volume()) ? 1 : 0;
}
// Restrict streaming to a world AABB corridor (min3 / max3): nodes outside it are never
// selected, requested or uploaded. lnt_view_clear_corridor lifts it.
__declspec(dllexport) void lnt_view_set_corridor(void* h, const double* min3, const double* max3) {
    View* v = static_cast<View*>(h);
    if (!v || !min3 || !max3) return;
    v->streamer.set_corridor(BoundingBox::from_min_max(min3[0], min3[1], min3[2],
                                                       max3[0], max3[1], max3[2]));
}
__declspec(dllexport) void lnt_view_clear_corridor(void* h) {
    if (View* v = static_cast<View*>(h)) v->streamer.clear_corridor();
}
// How many directory nodes meet the corridor (a directory scan, no payload read).
__declspec(dllexport) uint64_t lnt_view_corridor_node_count(void* h, const double* min3,
                                                            const double* max3) {
    View* v = static_cast<View*>(h);
    if (!v || !min3 || !max3) return 0;
    return v->streamer.query_nodes_in_corridor(BoundingBox::from_min_max(
        min3[0], min3[1], min3[2], max3[0], max3[1], max3[2])).size();
}

// Light elevation used when exactly one class is visible (app.last_shade_angle). That case
// is CPU-baked, so a change rebuilds the colours; a crisp (multi-class) mesh ignores it.
__declspec(dllexport) int lnt_view_set_surface_single_elevation(void* h, double elevation_deg) {
    View* v = static_cast<View*>(h);
    if (!v) return 0;
    if (v->surface_single_elevation == elevation_deg) return 1;
    v->surface_single_elevation = elevation_deg;
    if (!v->surface_shading_upload) v->invalidate_surface();
    return 1;
}
// Offscreen capture of the engine's current frame (rgba8, rows top-to-bottom). Returns 1 and
// the true extent; `out` may be null to query the extent only. Used by the parity tests.
__declspec(dllexport) int lnt_view_capture(void* h, uint8_t* out, uint64_t capacity,
                                           uint32_t* width, uint32_t* height) {
    View* v = static_cast<View*>(h);
    if (!v) return 0;
    EngineFrame f;
    if (!v->eng.capture_frame(f)) return 0;
    if (width) *width = f.width;
    if (height) *height = f.height;
    if (!out) return 1;
    if (capacity < f.rgba.size()) return 0;
    std::memcpy(out, f.rgba.data(), f.rgba.size());
    return 1;
}
// The old method remains active while a replacement cut is loading/building.
__declspec(dllexport) int lnt_view_surface_method(void* h) {
    return h ? static_cast<View*>(h)->surface_method : 0;
}
__declspec(dllexport) int lnt_view_surface_building(void* h) {
    View* v = static_cast<View*>(h);
    if (!v) return 0;
    if (v->surface_requested && v->lod_mode_last && v->lod) return v->lod->busy() ? 1 : 0;
    return v->surface_job_running ? 1 : 0;
}
// Viewport-LOD statistics (LNTSurfaceLod.h LodStat order). Returns the number written,
// 0 when the LOD path has not been used.
__declspec(dllexport) int lnt_view_surface_lod_stats(void* h, uint64_t* out, int capacity) {
    View* v = static_cast<View*>(h);
    if (!v || !v->lod || !out || capacity <= 0) return 0;
    const int n = std::min<int>(capacity, int(surface_lod::kStatCount));
    std::memcpy(out, v->lod->stats(), sizeof(uint64_t) * size_t(n));
    return n;
}
// 1 while the current frame is drawn by the viewport-LOD path.
__declspec(dllexport) int lnt_view_surface_lod_active(void* h) {
    View* v = static_cast<View*>(h);
    return v && v->surface_requested && v->lod_mode_last ? 1 : 0;
}
// Refinement threshold: a tile is refined while its representative spacing projects to
// more than `pixels` on screen (default 2; NAKSHA_LNT_SURFACE_LOD_PX).
__declspec(dllexport) int lnt_view_set_surface_lod_px(void* h, double pixels) {
    View* v = static_cast<View*>(h);
    if (!v || !(pixels >= 0.25 && pixels <= 64.0)) return 0;
    if (v->lod) v->lod->config.tau_px = pixels;
    else { char buf[32]; std::snprintf(buf, sizeof(buf), "%g", pixels); SetEnvironmentVariableA("NAKSHA_LNT_SURFACE_LOD_PX", buf); }
    v->frame_sig_valid = false;
    return 1;
}
// Blocks until built tiles are committed to the archive (tests / explicit save).
__declspec(dllexport) int lnt_view_surface_flush(void* h) {
    View* v = static_cast<View*>(h);
    if (!v || !v->lod) return 0;
    v->lod->flush();
    return 1;
}
// Latest archive persistence problem (empty when none); valid until the next call.
__declspec(dllexport) const char* lnt_view_surface_persist_message(void* h) {
    static thread_local std::string message;
    View* v = static_cast<View*>(h);
    message = v && v->lod ? v->lod->take_persist_message() : std::string();
    return message.c_str();
}
__declspec(dllexport) uint64_t lnt_view_surface_revision(void* h) {
    return h ? static_cast<View*>(h)->surface_revision : 0;
}
// Queried on the same GUI thread as frame(); the pointer lasts until the next view call.
__declspec(dllexport) const char* lnt_view_surface_error(void* h) {
    return h ? static_cast<View*>(h)->surface_error.c_str() : "";
}
// Active full-detail source records, visible records, native point-pass records.
__declspec(dllexport) int lnt_view_surface_point_counts(void* h, uint64_t* out3) {
    if(!h||!out3) return 0;
    std::memcpy(out3,static_cast<View*>(h)->surface_point_counts,sizeof(uint64_t)*3);
    return 1;
}

__declspec(dllexport) int lnt_view_frame(void* h, float dt, LntViewStats* out) {
    View* v = static_cast<View*>(h);
    if (!v) return 0;
    try {
        const bool skip = v->view_is_idle();
        // Activate a finished upload before drawing, so the first completed
        // surface appears in this frame rather than one frame later.
        if (v->surface_requested && !v->lod_mode_last) v->poll_surface();
        if (!skip) {
            if (!v->loop.render_frame(v->eng, v->scene, v->streamer, dt, !v->surface_requested)) {
                std::fprintf(stderr, "[LNT] frame failed: %s\n", v->loop.error().c_str());
                return 0;
            }
            if(v->loop.presented())v->last_render_at = std::chrono::steady_clock::now();
            else v->frame_sig_valid=false; // retry latest camera, without the idle debounce
            if (v->pending_surface_method >= 0 && v->loop.presented()) {
                v->surface_method = v->pending_surface_method;
                std::memcpy(v->surface_point_counts,v->pending_surface_point_counts,sizeof(v->surface_point_counts));
                v->pending_surface_method = -1;
            }
        }
        const bool lod_mode = v->surface_requested && v->lod_eligible() && v->ensure_lod();
        if (lod_mode) {
            if (!v->lod_mode_last) {
                // Entering the tiled path: stop the cut builder; its mesh is released by
                // the tile arena (one surface on the GPU at a time).
                if (v->surface_job_running) v->surface_cancel->store(true);
                v->eng.show_surface(true);
            }
            v->lod_mode_last = true;
            v->lod_sync();
            std::string err;
            const LntLodParams& lp = v->streamer.params();
            v->lod_changed = v->lod->update(v->loop.camera(), v->eng, lp.use_corridor ? &lp.corridor : nullptr, &err);
            if (!err.empty()) {
                std::fprintf(stderr, "[LNT surface] %s; using the cut-based surface\n", err.c_str());
                v->lod->deactivate(v->eng);
                v->lod_failed = true;
                v->lod_mode_last = false;
                v->surface_ids.clear();
                v->surface_urgent = true;
            } else {
                if (v->lod_changed) v->frame_sig_valid = false;
                const uint64_t* st = v->lod->stats();
                v->surface_point_counts[0] = st[surface_lod::kStatSourceCovered];
                v->surface_point_counts[1] = st[surface_lod::kStatRepresentedDrawn];
                v->surface_point_counts[2] = st[surface_lod::kStatCoverageDrawn];
                if (st[surface_lod::kStatDrawnTiles]) v->surface_method = 1;
                v->surface_error = v->lod->failure().empty() ? std::string()
                    : v->lod->failure() + " (a coarser tile is shown there; original records unchanged)";
            }
        } else if (v->lod_mode_last && v->surface_requested) {
            // Another surface path takes over (one visible class, elevation colour): free
            // the arena for it. Point modes keep the tiles resident for a quick return.
            if (v->lod) v->lod->deactivate(v->eng);
            v->lod_mode_last = false;
            v->surface_ids.clear();
            v->surface_urgent = true;
        }
        if (!skip && v->surface_requested && !v->lod_mode_last) {
            const auto& ids = v->surface_cut();
            const auto surface_key=v->surface_key();
            if (ids != v->pending_surface_ids || surface_key!=v->pending_surface_key) {
                v->pending_surface_key=surface_key;
                v->pending_surface_ids = ids;
                v->surface_cut_changed = std::chrono::steady_clock::now();
                // The previous surface stays visible while the new one is rebuilt.
            }
            // No cancel-and-restart while the camera moves: a build always finishes (its
            // reads stay cached) and the next one starts at most every 250 ms, so a pan
            // is never competing with a stream of restarted builds.
            v->poll_surface();
            const auto now = std::chrono::steady_clock::now();
            if (!ids.empty() && (ids != v->surface_ids || surface_key!=v->active_surface_key) && !v->surface_job_running &&
                (v->surface_urgent ||
                 (std::chrono::duration<double, std::milli>(now - v->surface_last_start).count() >= 250.0 &&
                  std::chrono::duration<double, std::milli>(now - v->surface_cut_changed).count() >= 300.0))) {
                if (v->rebuild_surface(ids)) { v->surface_last_start = now; v->surface_urgent=false; }
            }
        }
    } catch (const std::exception& ex) {
        std::fprintf(stderr, "[LNT] frame threw: %s\n", ex.what());
        return 0;
    }
    if (out) {
        const LNTStreamerStats& s = v->loop.stats();
        out->active_nodes = s.active_nodes;
        out->points_rendered = s.total_points_rendered;
        out->points_wanted = s.wanted_points;
        out->pending_loads = s.pending_async_loads;
        out->evictions = s.total_evictions;
        out->_pad = 0;
        out->cache_used_bytes = s.cache_bytes_used;
        out->cache_capacity_bytes = s.cache_capacity_bytes;
        out->frame_ms = s.frame_time_ms;
        out->update_ms = s.streamer_update_ms;
        out->sse = s.sse_threshold;
        out->distance = v->ctl.distance();
    }
    return 1;
}

__declspec(dllexport) uint64_t lnt_view_renderer_handle(void* h) {
    auto* v=static_cast<View*>(h);return v?v->eng.raw_handle():0;
}

__declspec(dllexport) void lnt_view_info(void* h, uint64_t* points, uint64_t* nodes,
                                         double* extent3) {
    View* v = static_cast<View*>(h);
    if (!v) return;
    if (points) *points = v->rd.get_total_points();
    if (nodes) *nodes = v->rd.get_total_nodes();
    if (extent3) for (int i = 0; i < 3; ++i) extent3[i] = v->env.extent(i);
}

}  // extern "C"
