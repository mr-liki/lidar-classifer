#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <unordered_map>
#include <vector>
#include <utility>

namespace naksha {
// Native counterpart of _compute_boundary_flags. ONE grid spans all resident
// tiles, so tile seams are not scan edges. Camera/style changes use cached flags.
class StructuralBoundary {
    struct Chunk {
        std::vector<float> xyz;
        std::vector<int16_t> classes; // -1 = classification not resident
        uint64_t version = 0;
        std::size_t size() const { return classes.size(); }
    };
    struct Key {
        int64_t x, y;
        bool operator==(const Key& b) const { return x == b.x && y == b.y; }
    };
    struct Hash {
        std::size_t operator()(const Key& k) const {
            uint64_t x = static_cast<uint64_t>(k.x) * 0x9e3779b97f4a7c15ULL;
            uint64_t y = static_cast<uint64_t>(k.y) * 0xc2b2ae3d27d4eb4fULL;
            return static_cast<std::size_t>(x ^ (y + (x << 6) + (x >> 2)));
        }
    };
    struct Cell {
        double z = 0;
        double topZ = -std::numeric_limits<double>::infinity();
        std::size_t count = 0;
        int cls = -1, neighbourClass = -1;
        bool edge = false;
    };
    std::map<std::size_t, std::shared_ptr<Chunk>> chunks_;
    uint64_t nextVersion_ = 0;
    bool dirty_ = true;
    uint64_t revision_ = 0;
public:
    struct Range { std::size_t first, count; uint64_t version; };
    std::vector<Range> ranges() const {
        std::vector<Range> out;
        for (const auto& [first,c] : chunks_) out.push_back({first,c->size(),c->version});
        return out;
    }
    bool matches(const Range& r) const {
        auto it = chunks_.upper_bound(r.first);
        if (it == chunks_.begin()) return false;
        --it;
        return r.first >= it->first && r.first+r.count <= it->first+it->second->size() && it->second->version == r.version;
    }
    uint64_t revision() const { return revision_; }
    void invalidate() { dirty_ = true; ++revision_; }
    // Resident parents and fine descendants may overlap in XY. Only the
    // selected draw cut belongs in a boundary grid; hidden LODs corrupt cell
    // heights/classes and flag flat building interiors as structural edges.
    void restrictToRanges(const std::vector<std::pair<uint32_t,uint32_t>>& ranges) {
        std::map<std::size_t,std::shared_ptr<Chunk>> selected;
        for (const auto& [first,count] : ranges) {
            const std::size_t end = std::size_t(first)+count;
            auto it = chunks_.upper_bound(first);
            if (it != chunks_.begin()) --it;
            for (; it != chunks_.end() && it->first < end; ++it) {
                const auto lo = std::max(std::size_t(first),it->first);
                const auto hi = std::min(end,it->first+it->second->size());
                if (lo >= hi) continue;
                if (lo==it->first && hi==it->first+it->second->size()) selected.emplace(lo,it->second);
                else {
                    auto c = std::make_shared<Chunk>();
                    c->version = it->second->version;
                    c->xyz.assign(it->second->xyz.begin()+(lo-it->first)*3,it->second->xyz.begin()+(hi-it->first)*3);
                    c->classes.assign(it->second->classes.begin()+lo-it->first,it->second->classes.begin()+hi-it->first);
                    selected.emplace(lo,std::move(c));
                }
            }
        }
        chunks_ = std::move(selected);
    }
    bool dirty() const { return dirty_; }
    void clear() { chunks_.clear(); invalidate(); }
    void positions(std::size_t first, const float* xyz, std::size_t count) {
        // Arena slots may be reused or partially replaced. Retain only the
        // unaffected portions of an old tile, never stale overlapping points.
        const auto end = first + count;
        std::map<std::size_t, std::shared_ptr<Chunk>> retained;
        for (auto it = chunks_.begin(); it != chunks_.end();) {
            const auto lo = it->first, hi = lo + it->second->size();
            if (lo >= end || hi <= first) { ++it; continue; }
            auto keep = [&](std::size_t a, std::size_t b) {
                Chunk c;
                c.xyz.assign(it->second->xyz.begin() + (a-lo)*3, it->second->xyz.begin() + (b-lo)*3);
                c.classes.assign(it->second->classes.begin() + a-lo, it->second->classes.begin() + b-lo);
                c.version = ++nextVersion_;
                retained.emplace(a, std::make_shared<Chunk>(std::move(c)));
            };
            if (lo < first) keep(lo, first);
            if (hi > end) keep(end, hi);
            it = chunks_.erase(it);
        }
        chunks_.merge(retained);
        Chunk c;
        c.xyz.assign(xyz, xyz + count*3);
        c.classes.assign(count, -1);
        c.version = ++nextVersion_;
        chunks_[first] = std::make_shared<Chunk>(std::move(c));
        invalidate();
    }
    void classification(std::size_t first, const uint8_t* cls, std::size_t count) {
        const auto end = first + count;
        auto it = chunks_.upper_bound(first);
        if (it != chunks_.begin()) --it;
        for (; it != chunks_.end() && it->first < end; ++it) {
            const auto lo = std::max(first, it->first);
            const auto hi = std::min(end, it->first + it->second->size());
            if (lo >= hi) continue;
            // Workers retain immutable geometry; edits copy only the affected tile.
            if (it->second.use_count() > 1) it->second = std::make_shared<Chunk>(*it->second);
            it->second->version = ++nextVersion_;
            for (auto i = lo; i < hi; ++i) it->second->classes[i-it->first] = cls[i-first];
        }
        invalidate();
    }
    std::vector<uint8_t> compute(std::size_t pointCount, bool streamedCut = false) const {
        std::vector<uint8_t> flags(pointCount, 0);
        double xmin = std::numeric_limits<double>::infinity(), ymin = xmin, zmin = xmin;
        double xmax = -xmin, ymax = -xmin, zmax = -xmin;
        std::size_t n = 0;
        auto valid = [](const float* p) { return std::isfinite(p[0]) && std::isfinite(p[1]) && std::isfinite(p[2]); };
        for (const auto& [first, chunk] : chunks_) {
            const auto& c = *chunk;
            for (std::size_t i = 0; i < c.size() && first+i < pointCount; ++i) {
                const float* p = c.xyz.data() + i*3;
                if (!valid(p)) continue;
                xmin = std::min(xmin, double(p[0])); xmax = std::max(xmax, double(p[0]));
                ymin = std::min(ymin, double(p[1])); ymax = std::max(ymax, double(p[1]));
                zmin = std::min(zmin, double(p[2])); zmax = std::max(zmax, double(p[2]));
                ++n;
            }
        }
        if (!n) return flags;
        const double res = std::max(0.05, std::sqrt(std::max(xmax-xmin, 1.0) * std::max(ymax-ymin, 1.0) / n) * 1.5);
        const double threshold = std::max(0.5, std::max(zmax-zmin, 1.0) * 0.03);
        auto key = [&](const float* p) { return Key{int64_t(std::floor((p[0]-xmin)/res)), int64_t(std::floor((p[1]-ymin)/res))}; };
        // Sparse cells keep long/narrow sections and widely separated tiles
        // bounded by resident point count instead of XY bounding-box area.
        std::unordered_map<Key, Cell, Hash> grid;
        grid.reserve(n / 2 + 1);
        for (const auto& [first, chunk] : chunks_) {
            const auto& c = *chunk;
            for (std::size_t i = 0; i < c.size() && first+i < pointCount; ++i) {
                const float* p = c.xyz.data() + i*3;
                if (!valid(p)) continue;
                auto& cell = grid[key(p)];
                cell.z += p[2]; ++cell.count;
                if (streamedCut) {
                    // Use the visible upper return for the plan-view grid.
                    // Ground/facade returns beneath a roof must not randomly
                    // replace its class or pull its height down in each cell.
                    if (p[2] > cell.topZ) { cell.topZ=p[2]; cell.cls=c.classes[i]; }
                } else if (c.classes[i] >= 0) cell.cls = c.classes[i];
            }
        }
        for (auto& [k, c] : grid) c.z = streamedCut ? c.topZ : c.z/c.count;
        if (streamedCut) {
            // A thinned roof may have an isolated cell containing only lower
            // ground/facade returns. Close small lower-class sampling holes
            // when at least five neighbours agree on an upper surface. Keep
            // true class boundaries and same-class roof steps intact.
            struct Fill { Key key; double z; int cls; };
            for (int pass=0;pass<2;++pass) {
                std::vector<Fill> fills;
                for (const auto& [k,c] : grid) {
                    int ids[8]={},counts[8]={},groups=0;
                    double heights[8]={};
                    for (int dx=-1;dx<=1;++dx) for(int dy=-1;dy<=1;++dy) {
                        if (!dx && !dy) continue;
                        auto it=grid.find({k.x+dx,k.y+dy});
                        if (it==grid.end()) continue;
                        const auto& nb=it->second;
                        if (nb.cls<0 || nb.cls==c.cls || nb.z-c.z<=threshold) continue;
                        int g=0;for(;g<groups && ids[g]!=nb.cls;++g) {}
                        if(g==groups)ids[groups++]=nb.cls;
                        ++counts[g];heights[g]+=nb.z;
                    }
                    for(int g=0;g<groups;++g) if(counts[g]>=5) {
                        fills.push_back({k,heights[g]/counts[g],ids[g]});break;
                    }
                }
                if(fills.empty())break;
                for(const auto& fill:fills) {auto& c=grid.at(fill.key);c.z=fill.z;c.cls=fill.cls;}
            }
        }
        const auto maxKey = key(std::array<float,3>{float(xmax),float(ymax),float(zmax)}.data());
        for (auto& [k, c] : grid) {
            int occupied = 0, fitCount = 0;
            double xx=0,xy=0,yy=0,bx=0,by=0;
            bool heightEdge = false;
            for (int dx = -1; dx <= 1; ++dx) for (int dy = -1; dy <= 1; ++dy) {
                if (!dx && !dy) continue;
                auto it = grid.find({k.x+dx, k.y+dy});
                if (it == grid.end()) continue;
                ++occupied;
                const auto& nb = it->second;
                const double dz = nb.z-c.z;
                heightEdge |= std::abs(dz) > threshold;
                if (streamedCut && (c.cls < 0 || nb.cls == c.cls)) {
                    ++fitCount; xx+=dx*dx; xy+=dx*dy; yy+=dy*dy;
                    bx+=dx*dz; by+=dy*dz;
                }
                if (nb.cls >= 0) {
                    if (c.neighbourClass == -1) c.neighbourClass = nb.cls;
                    else if (c.neighbourClass != nb.cls) c.neighbourClass = -2;
                }
            }
            const double determinant = xx*yy-xy*xy;
            if (streamedCut && fitCount>=3 && determinant>1e-6) {
                // Coarse LOD sampling creates larger height increments along
                // an ordinary pitched roof. Test departure from its local
                // plane, rather than marking every point on the slope as an
                // edge. Half the step threshold preserves sharp roof steps.
                const double sx=(bx*yy-by*xy)/determinant;
                const double sy=(by*xx-bx*xy)/determinant;
                heightEdge=false;
                for (int dx=-1;dx<=1;++dx) for(int dy=-1;dy<=1;++dy) {
                    if (!dx && !dy) continue;
                    auto it=grid.find({k.x+dx,k.y+dy});
                    if (it!=grid.end())
                        heightEdge |= std::abs(it->second.z-c.z-sx*dx-sy*dy)>threshold*.5;
                }
            }
            // Interior empty LOD cells are sampling gaps, not scan boundaries.
            // Real object edges still come from class/height discontinuities.
            // Keep exterior limits; full clouds/sections retain VTK's exact
            // eight-neighbour scan-edge criterion.
            const bool exterior=k.x==0 || k.y==0 || k.x==maxKey.x || k.y==maxKey.y;
            c.edge=heightEdge || (occupied<4 && (!streamedCut || exterior));
        }
        for (const auto& [first, chunk] : chunks_) {
            const auto& c = *chunk;
            for (std::size_t i = 0; i < c.size() && first+i < pointCount; ++i) {
                const float* p = c.xyz.data() + i*3;
                if (!valid(p)) continue;
                const auto& cell = grid.at(key(p));
                const bool classEdge = c.classes[i] >= 0 && cell.neighbourClass != -1 && cell.neighbourClass != c.classes[i];
                flags[first+i] = (cell.edge || classEdge) ? 255 : 0;
            }
        }
        return flags;
    }
    void markClean() { dirty_ = false; }
};
} // namespace naksha
