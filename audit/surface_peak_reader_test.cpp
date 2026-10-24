#ifdef READER_BASELINE
#include "../patch_backups/surface_peak_fix/LNTReader.h"
#else
#include "../native/lnt/include/LNTReader.h"
#endif
#ifdef MESH_BASELINE
#include "../patch_backups/surface_peak_fix/LNTSurfaceMesh.h"
#else
#include "../native/lnt/include/LNTSurfaceMesh.h"
#endif
#include <cstdlib>
#include <cstdio>
#include <memory>

using namespace lnt;
void require(bool condition,int line) {
    if(!condition) { std::fprintf(stderr,"Regression check failed at line %d\n",line); std::exit(1); }
}
#define CHECK(expression) require(bool(expression),__LINE__)
void peak_test(bool hidden, bool split, bool reverse) {
    std::vector<SurfaceTile> tiles(split?2:1);
    for(auto& t:tiles) {
        t.cmin[0]=t.cmin[1]=-2; t.cmin[2]=-1;
        t.cmax[0]=t.cmax[1]=2; t.cmax[2]=11;
    }
    tiles[0].pts={{-1,-1,0,2,0,0,0,0},{1,-1,0,2,0,0,0,0},
                  {1,1,0,2,0,0,0,0},{-1,1,0,2,0,0,0,0},
                  {0,0,0,2,0,0,0,0}};
    tiles[split?1:0].pts.push_back({0,0,10,5,0,0,0,0});
    if(reverse) std::reverse(tiles[0].pts.begin(),tiles[0].pts.end());
    SurfaceParams params; params.max_edge_abs=-1;
    params.class_visible[5]=!hidden;
    const auto mesh=build_surface_mesh(tiles,params,2);
    CHECK(mesh.triangles()==4);
    int tips=0;
    for(auto vertex:mesh.faces) {
        const auto i=static_cast<size_t>(vertex)*3;
        if(mesh.positions[i]==0 && mesh.positions[i+1]==0) {
            CHECK(mesh.positions[i+2]==(hidden?0:10)); ++tips;
        }
    }
    CHECK(tips==4);
}
int main(int argc,char** argv) {
    if(argc==1) {
        for(bool hidden:{false,true}) for(bool split:{false,true}) for(bool reverse:{false,true})
            peak_test(hidden,split,reverse);
        std::puts("PASS peak triangles use highest visible return, independent of input order and tile ownership");
        return 0;
    }
    // Reproduce descriptor pressure without requiring hundreds of files.
    CHECK(_setmaxstdio(96)>=96);
    std::vector<std::unique_ptr<Reader>> readers;
    for(int i=0;i<40;++i) {
        auto reader=std::make_unique<Reader>();
        if(!reader->open(argv[1])) {
            std::fprintf(stderr,"reader %d: %s\n",i,reader->error().c_str());
            return 2;
        }
        readers.push_back(std::move(reader));
    }
    for(auto& reader:readers) for(const auto& node:reader->nodes()) {
        std::vector<RawPoint> points;
        if(!reader->read_node_points(node,points)) return 3;
        CHECK(points.size()==node.point_count);
    }
    std::puts("PASS 40 concurrent collection readers access every source under a 96-stream limit");
}
