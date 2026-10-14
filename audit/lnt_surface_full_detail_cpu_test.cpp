#include "../native/lnt/include/LNTSurfaceMesh.h"
#include <cassert>
#include <cstdio>
#include <set>
#include <tuple>
#include <random>

using namespace lnt;
int main() {
    // Exact topology parity against the comparison-sort sweep, including
    // quantization ties and duplicate input records.
    std::mt19937_64 rng(932);
    for(int fixture=0;fixture<3;++fixture) {
        std::vector<double> xy;
        for(int i=0;i<50000;++i) {
            double x=double(rng()%1000000)/1000,y=double(rng()%1000000)/1000;
            if(fixture==1) {x=double(i%223);y=double(i/223);}
            if(fixture==2) {x=std::round(x);y=std::round(y);}
            xy.push_back(x);xy.push_back(y);
        }
        for(int i=0;i<200;++i) {xy.push_back(xy[2*i]);xy.push_back(xy[2*i+1]);}
        Delaunay2D fast(xy,true),reference(xy,false);
        assert(fast.triangles()==reference.triangles());
        assert(fast.halfedges()==reference.halfedges());
    }
    std::puts("PASS radix sweep has identical topology on random, grid, quantized and duplicate XY inputs");
    std::vector<SurfaceTile> tiles(8);
    for(int column=0;column<4;++column) for(int layer=0;layer<2;++layer) {
        auto& t=tiles[2*column+layer];
        int cx=column%2,cy=column/2;
        t.cmin[0]=cx*16;t.cmax[0]=(cx+1)*16;
        t.cmin[1]=cy*16;t.cmax[1]=(cy+1)*16;
        for(int y=0;y<16;++y) for(int x=0;x<16;++x) {
            double px=cx*16+x+.2+layer*.35,py=cy*16+y+.25+layer*.2;
            t.pts.push_back({px,py,.1*px+.2*py,uint8_t(layer?5:2),20,80,160,128});
        }
    }
    SurfaceParams p;p.compact_crisp=true;p.bake_colors=false;p.halo_fraction=.02;
    std::vector<const SurfaceTile*> refs;
    for(const auto& t:tiles) refs.push_back(&t);
    SurfacePlan plan;plan_surface(refs,p,2,plan);
    auto mesh=build_surface_mesh(tiles,p,3);
    assert(mesh.visible_points==2048);
    assert(mesh.mesh_referenced_points+mesh.coverage_positions.size()/3==2048);
    std::set<std::tuple<double,double,double>> represented;
    for(const auto* stream:{&mesh.positions,&mesh.coverage_positions})
        for(size_t i=0;i<stream->size();i+=3)
            represented.emplace((*stream)[i],(*stream)[i+1],(*stream)[i+2]);
    for(const auto& t:tiles) for(const auto& q:t.pts)
        assert(represented.count({q.x,q.y,q.z})==1);
    std::vector<SurfaceTile> planar;
    partition_surface_plan(refs,p,plan,planar,128);
    std::set<const SurfacePoint*> assigned;
    for(const auto& t:planar) for(const auto* q:t.point_refs) assert(assigned.insert(q).second);
    assert(assigned.size()==2048);
    for(const auto& t:tiles) for(const auto& q:t.pts) assert(assigned.count(&q)==1);
    auto planar_mesh=build_surface_mesh(planar,p,3);
    auto reference_params=p;reference_params.bake_colors=true;
    auto reference_mesh=build_surface_mesh(planar,reference_params,3);
    assert(planar_mesh.faces==reference_mesh.faces && planar_mesh.positions==reference_mesh.positions);
    assert(planar_mesh.visible_points==2048 &&
           planar_mesh.mesh_referenced_points+planar_mesh.coverage_positions.size()/3==2048);
    represented.clear();
    for(const auto* stream:{&planar_mesh.positions,&planar_mesh.coverage_positions})
        for(size_t i=0;i<stream->size();i+=3)
            represented.emplace((*stream)[i],(*stream)[i+1],(*stream)[i+2]);
    for(const auto& t:tiles) for(const auto& q:t.pts) assert(represented.count({q.x,q.y,q.z})==1);
    // Test the plan partition's seams too, using the same interior samples.
    mesh=std::move(planar_mesh);
    // Interior samples along both XY column seams must meet exactly one face.
    std::set<double> x_seams,y_seams;
    for(const auto& t:planar) {
        if(t.cmax[0]<plan.xy_max[0]-1e-9) x_seams.insert(t.cmax[0]);
        if(t.cmax[1]<plan.xy_max[1]-1e-9) y_seams.insert(t.cmax[1]);
    }
    std::vector<double> xs(x_seams.begin(),x_seams.end()),ys(y_seams.begin(),y_seams.end());
    for(int s=0;s<10000;++s) {
        const bool axis=s%2;
        const double along=1.000037137+double(rng()%290000)/10000;
        double px=axis?along:xs[s%xs.size()]+.00123;
        double py=axis?ys[s%ys.size()]+.00123:along;
        int hits=0;
        for(size_t f=0;f<mesh.faces.size();f+=3) {
            auto vertex=[&](int k,int a){return mesh.positions[3*mesh.faces[f+k]+a];};
            auto cross=[&](int a,int b){return (vertex(b,0)-vertex(a,0))*(py-vertex(a,1))-
                                              (vertex(b,1)-vertex(a,1))*(px-vertex(a,0));};
            double a=cross(0,1),b=cross(1,2),c=cross(2,0);
            hits+=(a>1e-10&&b>1e-10&&c>1e-10)||(a< -1e-10&&b< -1e-10&&c< -1e-10);
        }
        if(hits!=1) std::fprintf(stderr,"seam sample %d: %.12f %.12f hits=%d\n",s,px,py,hits);
        assert(hits==1);
    }
    // A hidden first layer must not steal ownership from the visible layer.
    p.class_visible[2]=0;
    auto hidden=build_surface_mesh(tiles,p,3);
    assert(hidden.visible_points==1024);
    assert(hidden.mesh_referenced_points+hidden.coverage_positions.size()/3==1024);
    for(auto cls:hidden.vertex_class) assert(cls==5);
    std::puts("PASS all 2048 records represented; 10000 plan-seam samples covered once; hidden classes handled");
}
