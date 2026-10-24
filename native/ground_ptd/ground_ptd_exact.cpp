// Optional Ground PTD 2-D simplex lookup for a SciPy 1.15.3 Delaunay TIN.
// This is a port of the project's already-tested ordered Numba walker.
// No source point, classification, or triangulation is modified here.
// Compilers MUST NOT enable unsafe FP reassociation or contraction.
// The caller must audit returned IDs against COMPLETE SciPy queries before
// approving a platform/compiler combination for production.
#include <cmath>
#include <cstdint>
#include <limits>
#include <new>
#include <algorithm>
#include <array>
#include <chrono>
#include <vector>

#if defined(_WIN32)
#define NAKSHA_EXPORT extern "C" __declspec(dllexport)
#else
#define NAKSHA_EXPORT extern "C" __attribute__((visibility("default")))
#endif

namespace {
constexpr double EPS = 100.0 * std::numeric_limits<double>::epsilon();

inline double c(const double* transform, int32_t id, int row, int col) {
    return transform[static_cast<int64_t>(id)*6 + row*2 + col];
}
inline int32_t other(const int32_t* neighbors, int32_t id, int col) {
    return neighbors[static_cast<int64_t>(id)*3 + col];
}
inline double eq(const double* equations, int32_t id, int col) {
    return equations[static_cast<int64_t>(id)*4 + col];
}

// Match _brute_2d's sequential simplex order and degenerate-neighbor rules.
int32_t brute(double x, double y, int32_t count,
              const int32_t* neighbors, const double* transform) {
    const double broad = std::sqrt(EPS);
    double bar[3];
    for (int32_t sid = 0; sid < count; ++sid) {
        if (!std::isnan(c(transform,sid,0,0))) {
            bar[2] = 1.0;
            bool inside = true;
            for (int k=0; k<2; ++k) {
                bar[k]=0.0;
                bar[k] += c(transform,sid,k,0)*(x-c(transform,sid,2,0));
                bar[k] += c(transform,sid,k,1)*(y-c(transform,sid,2,1));
                bar[2] -= bar[k];
                if (!(bar[k]>=-EPS && bar[k]<=1.0+EPS)) {
                    inside=false;
                    break;
                }
            }
            if (inside && bar[2]>=-EPS && bar[2]<=1.0+EPS) return sid;
        } else {
            for (int k=0; k<3; ++k) {
                int32_t neighbor=other(neighbors,sid,k);
                if (neighbor==-1 || std::isnan(c(transform,neighbor,0,0))) continue;
                bar[2]=1.0;
                for (int m=0; m<2; ++m) {
                    bar[m]=0.0;
                    bar[m] += c(transform,neighbor,m,0)*(x-c(transform,neighbor,2,0));
                    bar[m] += c(transform,neighbor,m,1)*(y-c(transform,neighbor,2,1));
                    bar[2]-=bar[m];
                }
                bool inside=true;
                for (int m=0; m<3; ++m) {
                    const double low=(other(neighbors,neighbor,m)==sid) ? -broad : -EPS;
                    if (!(bar[m]>=low && bar[m]<=1.0+EPS)) {
                        inside=false;
                        break;
                    }
                }
                if (inside) return neighbor;
            }
        }
    }
    return -1;
}
}

// Optional triangle-AABB binning. The grid is only a *proposal* mechanism:
// a simplex ID can bypass the ordered walker only if the query is STRICTLY
// interior to one regular triangle. For edges, vertices, nearly singular TINs,
// outside-hull points, skipped giant triangles and every other doubtful case,
// use the existing SciPy-compatible ordered walker, retaining its start state.
// A certified interior point must choose the same simplex regardless of the
// walk's start state in a conforming Qhull 2-D triangulation.
namespace {
constexpr double CERT_INTERIOR = 1e-7;
constexpr int64_t MAX_BIN_ENTRIES = 8000000;
constexpr int MAX_CELLS_PER_TRIANGLE = 64;
constexpr int64_t MAX_FINE_ENTRIES = 8000000;
constexpr int MIN_REFINE_OCCUPANCY = 48;
constexpr int FINE_SIDE = 4;

struct GridIndex {
    int nx=0, ny=0;
    double minx=0, miny=0, scale_x=0, scale_y=0;
    std::vector<int64_t> offsets;
    std::vector<int32_t> simplices;
    // Stage 4 is optional and does not alter the Stage 3 grid code path.
    // All AABB filters can only force an ordered-walker fallback; they never
    // certify a triangle which Stage 3 would reject.
    struct Bounds { double x0, x1, y0, y1; };
    std::vector<Bounds> bounds;
    std::vector<int32_t> fine_slot;
    std::vector<std::array<int64_t, 17>> fine_offsets;
    std::vector<int32_t> fine_simplices;
    bool adaptive=false;

    int fine_coord(double value, double minimum, double scale, int coarse) const noexcept {
        const double z = (value-minimum)*scale*double(FINE_SIDE);
        if (!std::isfinite(z)) return 0;
        const double local=z-double(FINE_SIDE)*coarse;
        if (local<=0) return 0;
        if (local>=FINE_SIDE) return FINE_SIDE-1;
        return static_cast<int>(local);
    }

    void build_fine_bins() {
        const size_t ncells=size_t(nx)*size_t(ny);
        fine_slot.assign(ncells, -1);
        int64_t global_entries=0;
        for (size_t bin=0;bin<ncells;++bin) {
            const int64_t count=offsets[bin+1]-offsets[bin];
            if (count<MIN_REFINE_OCCUPANCY) continue;
            const int cx=static_cast<int>(bin%size_t(nx));
            const int cy=static_cast<int>(bin/size_t(nx));
            std::array<std::vector<int32_t>,16> lists;
            for (int64_t k=offsets[bin];k<offsets[bin+1];++k) {
                const int32_t sid=simplices[size_t(k)];
                const Bounds& bb=bounds[size_t(sid)];
                const int x0=fine_coord(bb.x0,minx,scale_x,cx);
                const int x1=fine_coord(bb.x1,minx,scale_x,cx);
                const int y0=fine_coord(bb.y0,miny,scale_y,cy);
                const int y1=fine_coord(bb.y1,miny,scale_y,cy);
                for (int fy=y0;fy<=y1;++fy)
                    for (int fx=x0;fx<=x1;++fx)
                        lists[size_t(fy*FINE_SIDE+fx)].push_back(sid);
            }
            int64_t local_entries=0;
            for (const auto& list:lists) local_entries+=int64_t(list.size());
            // Retain the original Stage 3 coarse cell when refinement is too
            // expensive. This changes no returned simplex IDs.
            if (global_entries+local_entries>MAX_FINE_ENTRIES) continue;
            fine_slot[bin]=static_cast<int32_t>(fine_offsets.size());
            std::array<int64_t,17> entry{};
            for (int k=0;k<16;++k) {
                entry[size_t(k)]=static_cast<int64_t>(fine_simplices.size());
                fine_simplices.insert(fine_simplices.end(),lists[size_t(k)].begin(),
                                      lists[size_t(k)].end());
            }
            entry[16]=static_cast<int64_t>(fine_simplices.size());
            fine_offsets.push_back(entry);
            global_entries+=local_entries;
        }
    }

    int cell_x(double v) const noexcept {
        double z=(v-minx)*scale_x;
        if (!std::isfinite(z) || z<=0) return 0;
        if (z>=nx) return nx-1;
        return static_cast<int>(z);
    }
    int cell_y(double v) const noexcept {
        double z=(v-miny)*scale_y;
        if (!std::isfinite(z) || z<=0) return 0;
        if (z>=ny) return ny-1;
        return static_cast<int>(z);
    }

    struct Cells {
        int x0=0, y0=0, x1=-1, y1=-1;
        bool eligible=false;
    };

    bool build(const int32_t* faces, int64_t ntri,
               const double* points, int64_t npoints,
               double low_x, double low_y,
               double high_x, double high_y, bool use_adaptive=false) {
        if (!faces || !points || ntri<=0 || npoints<=0 ||
            !(high_x>low_x && high_y>low_y) ||
            !std::isfinite(low_x) || !std::isfinite(low_y) ||
            !std::isfinite(high_x) || !std::isfinite(high_y)) return false;
        adaptive=use_adaptive;
        const double span_x=high_x-low_x, span_y=high_y-low_y;
        const double aspect=std::max(0.125,std::min(8.0,span_x/span_y));
        const double target=std::max(1024.0,std::min(262144.0,double(ntri)/4.0));
        nx=std::max(16,std::min(512,static_cast<int>(std::sqrt(target*aspect))));
        ny=std::max(16,std::min(512,static_cast<int>(std::sqrt(target/aspect))));
        minx=low_x; miny=low_y;
        scale_x=nx/span_x; scale_y=ny/span_y;
        if (!(std::isfinite(scale_x) && std::isfinite(scale_y) &&
              scale_x>0 && scale_y>0)) return false;
        const size_t ncells=size_t(nx)*size_t(ny);
        std::vector<int64_t> counts(ncells,0);
        std::vector<Cells> spans(static_cast<size_t>(ntri));
        if (adaptive) bounds.resize(static_cast<size_t>(ntri));
        int64_t total=0;
        for (int64_t sid=0; sid<ntri; ++sid) {
            double xs[3], ys[3];
            for (int k=0;k<3;++k) {
                const int32_t v=faces[sid*3+k];
                if (v<0 || v>=npoints) return false;
                xs[k]=points[int64_t(v)*2];
                ys[k]=points[int64_t(v)*2+1];
                if (!std::isfinite(xs[k]) || !std::isfinite(ys[k])) return false;
            }
            const double ax=std::min(xs[0],std::min(xs[1],xs[2]));
            const double bx=std::max(xs[0],std::max(xs[1],xs[2]));
            const double ay=std::min(ys[0],std::min(ys[1],ys[2]));
            const double by=std::max(ys[0],std::max(ys[1],ys[2]));
            if (adaptive) bounds[size_t(sid)]={ax,bx,ay,by};
            Cells range;
            range.x0=cell_x(ax); range.x1=cell_x(bx);
            range.y0=cell_y(ay); range.y1=cell_y(by);
            int64_t count=int64_t(range.x1-range.x0+1)*
                          int64_t(range.y1-range.y0+1);
            // Omit exceptionally long TIN triangles: never approximate them.
            // The ordered walker still resolves points not certified elsewhere.
            if (count<=0 || count>MAX_CELLS_PER_TRIANGLE ||
                total+count>MAX_BIN_ENTRIES) continue;
            range.eligible=true;
            spans[size_t(sid)]=range;
            total+=count;
            for (int cy=range.y0;cy<=range.y1;++cy)
                for (int cx=range.x0;cx<=range.x1;++cx)
                    ++counts[size_t(cy)*nx+cx];
        }
        offsets.resize(ncells+1,0);
        for (size_t i=0;i<ncells;++i) offsets[i+1]=offsets[i]+counts[i];
        simplices.resize(size_t(total));
        // Write original IDs in simplex order, never spatial-sort the queries.
        auto cursor=offsets;
        for (int64_t sid=0;sid<ntri;++sid) {
            const Cells& range=spans[size_t(sid)];
            if (!range.eligible) continue;
            for (int cy=range.y0;cy<=range.y1;++cy)
                for (int cx=range.x0;cx<=range.x1;++cx) {
                    const size_t bin=size_t(cy)*nx+cx;
                    simplices[size_t(cursor[bin]++)]=static_cast<int32_t>(sid);
                }
        }
        if (adaptive) build_fine_bins();
        return true;
    }

    int32_t certify(double x, double y, const double* transform,
                    int64_t* metrics=nullptr) const noexcept {
        if (!std::isfinite(x) || !std::isfinite(y)) return -1;
        const int cx=cell_x(x), cy=cell_y(y);
        const size_t bin=size_t(cy)*nx+cx;
        const std::vector<int32_t>* list=&simplices;
        int64_t begin=offsets[bin], end=offsets[bin+1];
        if (adaptive && fine_slot[bin]>=0) {
            const int slot=fine_slot[bin];
            const int fx=fine_coord(x,minx,scale_x,cx);
            const int fy=fine_coord(y,miny,scale_y,cy);
            const auto& fine=fine_offsets[size_t(slot)];
            const int sub=fy*FINE_SIDE+fx;
            list=&fine_simplices;
            begin=fine[size_t(sub)]; end=fine[size_t(sub+1)];
            if (metrics) ++metrics[3];
        }
        int32_t certified=-1;
        for (int64_t p=begin;p<end;++p) {
            const int32_t sid=(*list)[size_t(p)];
            if (metrics) ++metrics[1];
            if (adaptive) {
                const Bounds& bb=bounds[size_t(sid)];
                if (x<bb.x0 || x>bb.x1 || y<bb.y0 || y>bb.y1) {
                    if (metrics) ++metrics[2];
                    continue;
                }
            }
            if (std::isnan(c(transform,sid,0,0))) continue;
            // Evaluate in the same component order as the original walker.
            double b0=0.0, b1=0.0, b2=1.0;
            b0+=c(transform,sid,0,0)*(x-c(transform,sid,2,0));
            b0+=c(transform,sid,0,1)*(y-c(transform,sid,2,1));
            b2-=b0;
            if (!(b0>CERT_INTERIOR && b0<1.0-CERT_INTERIOR)) continue;
            b1+=c(transform,sid,1,0)*(x-c(transform,sid,2,0));
            b1+=c(transform,sid,1,1)*(y-c(transform,sid,2,1));
            b2-=b1;
            if (b1>CERT_INTERIOR && b1<1.0-CERT_INTERIOR &&
                b2>CERT_INTERIOR && b2<1.0-CERT_INTERIOR) {
                if (!adaptive) return sid;  // unchanged Stage 3 behavior
                if (certified>=0 && certified!=sid) {
                    // Numerical overlap/ambiguous Qhull facets are possible
                    // on near-degenerate survey geometry. Never guess which
                    // simplex SciPy would choose: run the ordered walker.
                    if (metrics) ++metrics[6];
                    return -1;
                }
                certified=sid;
            }
        }
        return certified;
    }
};
}  // namespace

// Return 0 for success, 1 for incomplete walk (caller must run SciPy on
// the COMPLETE original query), negative for invalid input. Always serial:
// a single last-simplex search state must be shared by every ordered query.
static int find_simplex_impl(
    const double* xy, int64_t nquery,
    const int32_t* neighbors, const double* equations, const double* transform,
    int64_t ntri, double low_x, double low_y, double high_x, double high_y,
    double paraboloid_scale, double paraboloid_shift, int32_t* out,
    const GridIndex* grid, int64_t* fast_hits,
    int64_t* metrics=nullptr) noexcept {
    if (!xy || !neighbors || !equations || !transform || !out ||
        nquery<0 || ntri<=0 || ntri>std::numeric_limits<int32_t>::max() ||
        !std::isfinite(paraboloid_scale) || !std::isfinite(paraboloid_shift)) {
        return -1;
    }
    const int32_t count=static_cast<int32_t>(ntri);
    if (fast_hits) *fast_hits=0;
    int32_t start=0;
    for (int64_t row=0; row<nquery; ++row) {
        double x=xy[row*2], y=xy[row*2+1];
        out[row]=-1;
        if (!std::isfinite(x) || !std::isfinite(y)) return -1;
        if (x<low_x-EPS || x>high_x+EPS ||
            y<low_y-EPS || y>high_y+EPS) continue;
        if (grid && x>=low_x && x<=high_x && y>=low_y && y<=high_y) {
            const int32_t certified=grid->certify(x,y,transform,metrics);
            if (certified>=0) {
                out[row]=certified;
                start=certified;  // Same walker start for the next ordered query.
                if (fast_hits) ++*fast_hits;
                continue;
            }
        }
        int32_t sid=start;
        if (sid<0 || sid>=count) sid=0;
        double z=0.0;
        z+=x*x;
        z+=y*y;
        z*=paraboloid_scale;
        z+=paraboloid_shift;
        double best=eq(equations,sid,3);
        best+=eq(equations,sid,0)*x;
        best+=eq(equations,sid,1)*y;
        best+=eq(equations,sid,2)*z;
        bool changed=true;
        int64_t steps=0;
        while (changed) {
            if (best>0.0) break;
            changed=false;
            for (int k=0;k<3;++k) {
                int32_t n=other(neighbors,sid,k);
                if (n==-1) continue;
                double dist=eq(equations,n,3);
                dist+=eq(equations,n,0)*x;
                dist+=eq(equations,n,1)*y;
                dist+=eq(equations,n,2)*z;
                if (dist>best+EPS*(1.0+std::fabs(best))) {
                    sid=n;
                    best=dist;
                    changed=true;
                }
            }
            ++steps;
            if (steps>ntri+1) return 1;
        }
        start=sid;
        bool finished=false;
        for (int64_t cycle=0; cycle<1+ntri/4; ++cycle) {
            int inside=1;
            double bar[3];
            for (int k=0;k<3;++k) {
                if (k==2) {
                    bar[2]=1.0;
                    bar[2]-=bar[0];
                    bar[2]-=bar[1];
                } else {
                    bar[k]=0.0;
                    bar[k]+=c(transform,sid,k,0)*(x-c(transform,sid,2,0));
                    bar[k]+=c(transform,sid,k,1)*(y-c(transform,sid,2,1));
                }
                if (bar[k]<-EPS) {
                    int32_t n=other(neighbors,sid,k);
                    if (n==-1) {
                        start=sid;
                        sid=-1;
                        inside=2;
                        break;
                    }
                    sid=n;
                    inside=-1;
                    break;
                } else if (bar[k]<=1.0+EPS) {
                    // accepted coordinate; continue
                } else {
                    inside=0;
                }
            }
            if (inside==-1) continue;
            if (inside==2) { finished=true; break; }
            if (inside==1) { start=sid; finished=true; break; }
            sid=brute(x,y,count,neighbors,transform);
            start=sid;
            finished=true;
            break;
        }
        if (!finished) {
            sid=brute(x,y,count,neighbors,transform);
            start=sid;
        }
        out[row]=sid;
    }
    return 0;
}

// Preserve the Stage 1 ABI and implementation when the grid flag is off.
NAKSHA_EXPORT int naksha_ground_find_simplex_2d(
    const double* xy, int64_t nquery,
    const int32_t* neighbors, const double* equations, const double* transform,
    int64_t ntri, double low_x, double low_y, double high_x, double high_y,
    double paraboloid_scale, double paraboloid_shift, int32_t* out) noexcept {
    return find_simplex_impl(xy,nquery,neighbors,equations,transform,ntri,
                             low_x,low_y,high_x,high_y,
                             paraboloid_scale,paraboloid_shift,out,nullptr,nullptr);
}

// Optional Stage 3 grid proposal, with exact ordered fallback per uncertain
// point. -1 invalid, 1 incomplete (Python falls back to full SciPy query).
NAKSHA_EXPORT int naksha_ground_find_simplex_2d_grid(
    const double* xy, int64_t nquery,
    const int32_t* neighbors, const double* equations, const double* transform,
    int64_t ntri, double low_x, double low_y, double high_x, double high_y,
    double paraboloid_scale, double paraboloid_shift,
    const int32_t* faces, const double* points, int64_t npoints,
    int32_t* out, int64_t* fast_hits) noexcept {
    if (fast_hits) *fast_hits=0;
    if (!xy || !neighbors || !equations || !transform || !out ||
        !faces || !points || !fast_hits || nquery<0 || ntri<=0 || npoints<=0 ||
        ntri>std::numeric_limits<int32_t>::max()) return -1;
    try {
        GridIndex grid;
        if (!grid.build(faces,ntri,points,npoints,low_x,low_y,high_x,high_y)) {
            return find_simplex_impl(xy,nquery,neighbors,equations,transform,ntri,
                                     low_x,low_y,high_x,high_y,
                                     paraboloid_scale,paraboloid_shift,out,nullptr,fast_hits);
        }
        return find_simplex_impl(xy,nquery,neighbors,equations,transform,ntri,
                                 low_x,low_y,high_x,high_y,
                                 paraboloid_scale,paraboloid_shift,out,&grid,fast_hits);
    } catch (...) {
        return find_simplex_impl(xy,nquery,neighbors,equations,transform,ntri,
                                 low_x,low_y,high_x,high_y,
                                 paraboloid_scale,paraboloid_shift,out,nullptr,fast_hits);
    }
}

NAKSHA_EXPORT int naksha_ground_kernel_api_version() noexcept { return 1; }

// Stage 4 adds a strictly optional bounding-box and adaptive fine-bin
// rejection filter. Like Stage 3 it never guesses a simplex ID: if the
// triangle is not certified strictly interior, the Stage 1 ordered walker
// handles the ORIGINAL query in its ORIGINAL sequence. No anchor geometry,
// Qhull options, candidate order, PTD thresholds or classifications change.
// metrics[0]=certified, [1]=triangle probes, [2]=AABB prunes,
// [3]=fine-bin queries, [4]=refined parent bins, [5]=fine-index entries,
// [6]=ambiguous overlaps sent to the original ordered walker.
// timings[0]=index build seconds, timings[1]=query seconds.
NAKSHA_EXPORT int naksha_ground_find_simplex_2d_grid_stage4(
    const double* xy, int64_t nquery,
    const int32_t* neighbors, const double* equations, const double* transform,
    int64_t ntri, double low_x, double low_y, double high_x, double high_y,
    double paraboloid_scale, double paraboloid_shift,
    const int32_t* faces, const double* points, int64_t npoints,
    int32_t* out, int64_t* metrics, double* timings) noexcept {
    if (!metrics || !timings) return -1;
    for (int k=0;k<7;++k) metrics[k]=0;
    timings[0]=timings[1]=0.0;
    if (!xy || !neighbors || !equations || !transform || !out ||
        !faces || !points || nquery<0 || ntri<=0 || npoints<=0 ||
        ntri>std::numeric_limits<int32_t>::max()) return -1;
    using Clock=std::chrono::steady_clock;
    try {
        const auto build_start=Clock::now();
        GridIndex grid;
        if (!grid.build(faces,ntri,points,npoints,low_x,low_y,high_x,high_y,true)) {
            const auto lookup_start=Clock::now();
            const int status=find_simplex_impl(xy,nquery,neighbors,equations,transform,ntri,
                              low_x,low_y,high_x,high_y,paraboloid_scale,
                              paraboloid_shift,out,nullptr,nullptr);
            timings[1]=std::chrono::duration<double>(Clock::now()-lookup_start).count();
            return status;
        }
        timings[0]=std::chrono::duration<double>(Clock::now()-build_start).count();
        metrics[4]=static_cast<int64_t>(grid.fine_offsets.size());
        metrics[5]=static_cast<int64_t>(grid.fine_simplices.size());
        const auto lookup_start=Clock::now();
        const int status=find_simplex_impl(xy,nquery,neighbors,equations,transform,ntri,
                           low_x,low_y,high_x,high_y,paraboloid_scale,
                           paraboloid_shift,out,&grid,metrics,metrics);
        timings[1]=std::chrono::duration<double>(Clock::now()-lookup_start).count();
        return status;
    } catch (...) {
        // Complete conservative fallback; never publish a partial proposal.
        for (int k=0;k<7;++k) metrics[k]=0;
        const auto lookup_start=Clock::now();
        const int status=find_simplex_impl(xy,nquery,neighbors,equations,transform,ntri,
                              low_x,low_y,high_x,high_y,paraboloid_scale,
                              paraboloid_shift,out,nullptr,nullptr);
        timings[1]=std::chrono::duration<double>(Clock::now()-lookup_start).count();
        return status;
    }
}
