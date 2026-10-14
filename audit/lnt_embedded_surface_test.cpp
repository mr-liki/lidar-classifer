#define NOMINMAX
#include "../native/lnt/include/LNTEmbeddedSurface.h"
#include <cassert>
#include <filesystem>
#include <iostream>
#include <chrono>
int main(){
    using namespace lnt;namespace es=embedded_surface;
    const std::string path="audit/embedded_surface_test.lnt";
    std::filesystem::copy_file("native/lnt/test_shading.lnt",path,std::filesystem::copy_options::overwrite_existing);
    Reader rd;assert(rd.open(path));auto base=es::base_end(rd);
    std::filesystem::resize_file(path,base);
    auto source=es::source_digest(rd);SurfaceParams p;std::vector<uint64_t> ids{1,2};auto settings=es::settings_digest(p,ids);
    SurfaceMesh m;m.positions={0,0,0,1,0,0,0,1,0};m.faces={0,1,2};m.vertex_class={1,2,1};m.blocks={3};m.visible_points=4;m.mesh_referenced_points=3;m.coverage_positions={2,2,0};m.coverage_colors={255,0,0};m.input_points=rd.directory().total_points;
    assert(es::save(path,base,source,settings,m,2,SurfaceMethod::Barycentric,true));
    Reader reopened;assert(reopened.open(path));assert(es::source_digest(reopened)==source);
    SurfaceMesh out;unsigned classes=0;SurfaceMethod method;bool lit=false;
    auto load=[&](const es::Digest& s){return es::load(reopened,s,settings,out,classes,method,lit);};
    assert(load(source));assert(out.positions==m.positions&&out.faces==m.faces&&out.vertex_class==m.vertex_class&&out.coverage_positions==m.coverage_positions&&out.coverage_colors==m.coverage_colors);assert(out.mesh_referenced_points+out.coverage_positions.size()/3==out.visible_points);assert(classes==2&&lit&&method==SurfaceMethod::Barycentric);
    auto wrong=source;wrong[0]^=1;assert(!load(wrong));p.class_visible[1]=0;auto changed=es::settings_digest(p,ids);assert(!es::load(reopened,source,changed,out,classes,method,lit));
    // Existing v1 caches remain readable and are marked for background upgrade.
    assert(es::save(path,base,source,settings,m,2,SurfaceMethod::Barycentric,true,false));
    {std::fstream f(path,std::ios::in|std::ios::out|std::ios::binary);f.seekg(-std::streamoff(sizeof(es::Footer)),std::ios::end);es::Footer footer;f.read(reinterpret_cast<char*>(&footer),sizeof(footer));std::vector<char> body(size_t(footer.bytes));f.seekg(base);f.read(body.data(),body.size());es::Hash hash;hash.add(body.data(),body.size());footer.body_hash=hash.result();footer.magic[6]='1';es::Hash fh;fh.add(&footer,offsetof(es::Footer,footer_hash));footer.footer_hash=fh.result();f.seekp(base+footer.bytes);f.write(reinterpret_cast<char*>(&footer),sizeof(footer));}
    bool legacy=false;assert(es::load(reopened,source,settings,out,classes,method,lit,&legacy)&&legacy);
    assert(es::save(path,base,source,settings,m,2,SurfaceMethod::Barycentric,true));legacy=true;assert(es::load(reopened,source,settings,out,classes,method,lit,&legacy)&&!legacy);
    {std::fstream f(path,std::ios::in|std::ios::out|std::ios::binary);f.seekp(base);char bad=42;f.write(&bad,1);}assert(!load(source));
    assert(es::save(path,base,source,settings,m,2,SurfaceMethod::Barycentric,true));assert(load(source));
    auto size=std::filesystem::file_size(path);assert(es::save(path,base,source,settings,m,2,SurfaceMethod::Barycentric,true));assert(std::filesystem::file_size(path)==size);
    std::filesystem::resize_file(path,size-7);assert(!load(source));assert(reopened.open(path));assert(es::source_digest(reopened)==source);
    // Exercise multiple independent frames per array and parallel decoding,
    // including exact double positions and duplicated halo vertices.
    SurfaceMesh large;const size_t n=1200000;
    large.positions.resize(n*3);large.faces.resize(n*6);large.vertex_class.resize(n);
    large.input_points=rd.directory().total_points;large.visible_points=large.mesh_referenced_points=large.input_points;
    for(size_t i=0;i<n;++i){size_t record=i%size_t(large.input_points);large.positions[i*3]=600000.0+double(record%1000)*.013;large.positions[i*3+1]=676000.0+double(record/1000)*.017;large.positions[i*3+2]=double(record%51)*.001;large.vertex_class[i]=uint8_t(record%8);for(size_t a=0;a<6;++a)large.faces[i*6+a]=int32_t((i+a)%n);}
    es::Digest detected;assert(es::save(path,base,source,settings,large,8,SurfaceMethod::Barycentric,true,false));assert(es::cached_source(reopened,settings,detected)&&detected==source);
    auto before=std::chrono::steady_clock::now();assert(load(source));auto raw_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-before).count();auto raw_bytes=std::filesystem::file_size(path)-base;
    assert(out.positions==large.positions&&out.faces==large.faces&&out.vertex_class==large.vertex_class);
    assert(es::save(path,base,source,settings,large,8,SurfaceMethod::Barycentric,true));
    before=std::chrono::steady_clock::now();assert(load(source));auto compressed_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-before).count();
    assert(out.positions==large.positions&&out.faces==large.faces&&out.vertex_class==large.vertex_class);assert(out.visible_points==large.visible_points);
    {std::fstream f(path,std::ios::in|std::ios::out|std::ios::binary);f.seekg(base+sizeof(es::MeshChunkHeader)+13);char b;f.read(&b,1);b^=1;f.seekp(base+sizeof(es::MeshChunkHeader)+13);f.write(&b,1);}assert(!load(source));
    std::cout<<"CACHE CHECK: raw "<<raw_bytes/1e6<<" MB, framed "<<(std::filesystem::file_size(path)-base)/1e6<<" MB; warm load raw "<<raw_ms<<" ms, framed "<<compressed_ms<<" ms\n";
    {std::fstream f(path,std::ios::in|std::ios::out|std::ios::binary);f.seekg(64);char b;f.read(&b,1);b^=1;f.seekp(64);f.write(&b,1);}assert(es::source_digest(reopened)!=source);
    reopened.close();rd.close();std::filesystem::remove(path);
    std::cout<<"PASS embedded roundtrip, exact point accounting, unchanged original bytes, settings/source invalidation, corruption, interruption and bounded replacement\n";
}
