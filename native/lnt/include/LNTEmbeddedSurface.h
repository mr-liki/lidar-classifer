#pragma once
// Derived mesh trailer. The v2 header, point chunks and directory stay untouched.
// A missing, interrupted or stale trailer is a cache miss, never a partial mesh.
#include "LNTReader.h"
#include "LNTSurfaceMesh.h"
#include "LNTMeshChunks.h"
#include <windows.h>
#include <bcrypt.h>
#include <array>
#include <fstream>
#include <limits>
#include <stdexcept>

namespace lnt { namespace embedded_surface {
using Digest = std::array<unsigned char,32>;
// CRC32C is a hardware-assisted integrity check for the large derived arrays.
// Original data and settings retain SHA-256 identity checks.
class BodyHash {
    uint32_t crc=0xffffffff;
    __attribute__((target("sse4.2"))) static uint32_t hardware(uint32_t c,const unsigned char* p,size_t n){
        while(n>=8){uint64_t v;std::memcpy(&v,p,8);c=uint32_t(__builtin_ia32_crc32di(c,v));p+=8;n-=8;}
        while(n--)c=__builtin_ia32_crc32qi(c,*p++);return c;
    }
public:
    void add(const void* data,size_t bytes){
        auto p=static_cast<const unsigned char*>(data);
        if(__builtin_cpu_supports("sse4.2")){crc=hardware(crc,p,bytes);return;}
        static const auto table=[](){std::array<uint32_t,256> t{};for(unsigned i=0;i<256;++i){uint32_t c=i;for(int j=0;j<8;++j)c=(c>>1)^((c&1)?0x82f63b78:0);t[i]=c;}return t;}();
        while(bytes--)crc=table[(crc^*p++)&255]^(crc>>8);
    }
    Digest result(){Digest d{};uint32_t c=~crc;std::memcpy(d.data(),&c,4);return d;}
};
class Hash {
    HMODULE dll=nullptr; BCRYPT_ALG_HANDLE alg=nullptr; BCRYPT_HASH_HANDLE hash=nullptr;
    std::vector<unsigned char> object;
    decltype(&BCryptOpenAlgorithmProvider) open;
    decltype(&BCryptGetProperty) property;
    decltype(&BCryptCreateHash) create;
    decltype(&BCryptHashData) update;
    decltype(&BCryptFinishHash) finish;
    decltype(&BCryptDestroyHash) destroy;
    decltype(&BCryptCloseAlgorithmProvider) close;
public:
    Hash() {
        dll=LoadLibraryW(L"bcrypt.dll"); if(!dll) throw std::runtime_error("SHA256 unavailable");
#define LNT_HASH_PROC(field,name) field=reinterpret_cast<decltype(field)>(GetProcAddress(dll,name)); if(!field) throw std::runtime_error("SHA256 unavailable")
        LNT_HASH_PROC(open,"BCryptOpenAlgorithmProvider"); LNT_HASH_PROC(property,"BCryptGetProperty");
        LNT_HASH_PROC(create,"BCryptCreateHash"); LNT_HASH_PROC(update,"BCryptHashData");
        LNT_HASH_PROC(finish,"BCryptFinishHash"); LNT_HASH_PROC(destroy,"BCryptDestroyHash");
        LNT_HASH_PROC(close,"BCryptCloseAlgorithmProvider");
#undef LNT_HASH_PROC
        ULONG size=0,got=0;
        if(open(&alg,BCRYPT_SHA256_ALGORITHM,nullptr,0)<0 || property(alg,BCRYPT_OBJECT_LENGTH,reinterpret_cast<PUCHAR>(&size),sizeof(size),&got,0)<0) throw std::runtime_error("SHA256 initialization failed");
        object.resize(size);
        if(create(alg,&hash,object.data(),size,nullptr,0,0)<0) throw std::runtime_error("SHA256 initialization failed");
    }
    ~Hash(){if(hash)destroy(hash);if(alg)close(alg,0);if(dll)FreeLibrary(dll);}
    void add(const void* data,size_t bytes) {
        auto p=static_cast<const unsigned char*>(data);
        while(bytes){ULONG n=ULONG(std::min<size_t>(bytes,16*1024*1024));if(update(hash,const_cast<PUCHAR>(p),n,0)<0)throw std::runtime_error("SHA256 failed");p+=n;bytes-=n;}
    }
    template<class T> void value(const T& v){add(&v,sizeof(v));}
    Digest result(){Digest d{};if(finish(hash,d.data(),ULONG(d.size()),0)<0)throw std::runtime_error("SHA256 failed");return d;}
};
inline uint64_t base_end(const Reader& rd){return rd.get_index_offset()+16+uint64_t(rd.nodes().size())*52;}
inline Digest source_digest(const Reader& rd){
    std::ifstream in(rd.path(),std::ios::binary);Hash h;std::vector<char> buffer(8*1024*1024);
    uint64_t left=base_end(rd);
    while(left){size_t n=size_t(std::min<uint64_t>(left,buffer.size()));if(!in.read(buffer.data(),n))throw std::runtime_error("cannot fingerprint .lnt source");h.add(buffer.data(),n);left-=n;}
    return h.result();
}
inline Digest settings_digest(const SurfaceParams& p,const std::vector<uint64_t>& ids){
    Hash h;uint32_t algorithm=2;h.value(algorithm); // upper-envelope duplicate XY
    h.value(p.method);h.value(p.azimuth_deg);h.value(p.elevation_deg);h.value(p.ambient);h.value(p.max_edge_factor);
    h.value(p.color_source);h.value(p.ramp);h.value(p.z_min);h.value(p.z_max);h.value(p.i_min);h.value(p.i_max);
    h.add(p.class_rgb,sizeof(p.class_rgb));h.add(p.class_visible,sizeof(p.class_visible));
    h.value(p.single_elevation_deg);h.value(p.compact_crisp);h.value(p.halo_fraction);
    // Baked colours depend on the light model itself: Sharpness changes them even though
    // azimuth/elevation/ambient did not, so the crisp flag and its parameters must key the
    // cached mesh or a Sharpness change would reuse the previous bake.
    h.value(p.crisp_shade);h.value(p.surface.azimuth_deg);h.value(p.surface.sharpness);
    h.value(p.surface.ambient);h.value(p.surface.light_elevation_deg);
    h.value(p.surface.ambient_floor);h.value(p.surface.directional_energy);
    const char* halo=std::getenv("NAKSHA_LNT_HALO_FRACTION");if(halo)h.add(halo,std::strlen(halo));
    h.add(ids.data(),ids.size()*sizeof(uint64_t));return h.result();
}
struct Footer {
    char magic[8]={'L','N','T','S','H','D','3','\0'};
    uint64_t offset=0,bytes=0,source=0,visible=0,referenced=0,dropped=0;
    uint64_t sizes[9]{};
    uint32_t classes=0,method=0,engine_lit=0,reserved=0;
    Digest source_hash{},settings_hash{},body_hash{},footer_hash{};
};
static_assert(sizeof(Footer)==272&&offsetof(Footer,footer_hash)==240,"embedded mesh footer layout changed");
inline bool cached_source(const Reader& rd,const Digest& settings,Digest& source){
    try{
        std::ifstream in(rd.path(),std::ios::binary|std::ios::ate);auto end=in.tellg();
        if(end<std::streamoff(base_end(rd)+sizeof(Footer)))return false;
        Footer f;in.seekg(end-std::streamoff(sizeof(f)));if(!in.read(reinterpret_cast<char*>(&f),sizeof(f)))return false;
        if(std::memcmp(f.magic,"LNTSHD",6)||f.magic[6]<'1'||f.magic[6]>'3'||f.magic[7]||f.reserved||f.offset!=base_end(rd)||f.bytes!=uint64_t(end)-sizeof(f)-f.offset||f.settings_hash!=settings)return false;
        Hash h;h.add(&f,offsetof(Footer,footer_hash));if(h.result()!=f.footer_hash)return false;
        source=f.source_hash;return true;
    }catch(...){return false;}
}
template<class F> inline void arrays(SurfaceMesh& m,F f){
    f(m.positions,0);f(m.faces,1);f(m.coverage_positions,2);f(m.coverage_colors,3);
    f(m.vertex_colors,4);f(m.colors,5);f(m.blocks,6);f(m.vertex_class,7);f(m.mixed_faces,8);
}
inline bool load(const Reader& rd,const Digest& source,const Digest& settings,SurfaceMesh& mesh, unsigned& classes,SurfaceMethod& method,bool& engine_lit,bool* legacy=nullptr){
    try {
        std::ifstream in(rd.path(),std::ios::binary|std::ios::ate);auto end=in.tellg();
        if(end<std::streamoff(base_end(rd)+sizeof(Footer)))return false;
        Footer f;in.seekg(end-std::streamoff(sizeof(f)));if(!in.read(reinterpret_cast<char*>(&f),sizeof(f)))return false;
        Hash footer_check;footer_check.add(&f,offsetof(Footer,footer_hash));if(footer_check.result()!=f.footer_hash)return false;
        const bool old=std::memcmp(f.magic,"LNTSHD1\0",8)==0;
        const bool chunks=std::memcmp(f.magic,"LNTSHD3\0",8)==0;
        if((!old&&!chunks&&std::memcmp(f.magic,"LNTSHD2\0",8))||f.reserved||f.offset!=base_end(rd)||f.bytes!=uint64_t(end)-sizeof(f)-f.offset||f.source_hash!=source||f.settings_hash!=settings||f.classes>256||f.method>1||f.engine_lit>1)return false;
        SurfaceMesh m;uint64_t total=0;
        arrays(m,[&](auto& v,int i){using T=typename std::decay_t<decltype(v)>::value_type;if(f.sizes[i]>uint64_t(std::numeric_limits<size_t>::max())/sizeof(T))throw std::runtime_error("invalid cache size");uint64_t n=f.sizes[i]*sizeof(T);if(n>std::numeric_limits<uint64_t>::max()-total)throw std::runtime_error("invalid cache size");total+=n;});
        if(!chunks&&total!=f.bytes)return false;
        MEMORYSTATUSEX mem{};mem.dwLength=sizeof(mem);if(GlobalMemoryStatusEx(&mem)&&total>mem.ullAvailPhys*3/4)return false;
        if(chunks){
            MeshMapping mapped(rd.path(),f.offset,uint64_t(end));MeshCodec codec;Hash metadata;
            struct Task {MeshChunkHeader header;const char* src;char* dst;};
            std::vector<Task> tasks;uint64_t cursor=f.offset,limit=f.offset+f.bytes;
            // Validate all frame bounds and the authenticated directory before allocating.
            arrays(m,[&](auto& v,int i){using T=typename std::decay_t<decltype(v)>::value_type;uint64_t left=f.sizes[i]*sizeof(T);while(left){
                if(limit-cursor<sizeof(MeshChunkHeader))throw std::runtime_error("incomplete mesh frame");
                MeshChunkHeader h;std::memcpy(&h,mapped.at(cursor),sizeof(h));cursor+=sizeof(h);metadata.value(h);
                if(h.raw!=std::min<uint64_t>(left,kMeshChunkBytes)||!h.stored||h.stored>kMeshChunkBytes||h.stored>limit-cursor||h.codec>3||(h.codec==0&&h.raw!=h.stored)||(h.codec&&!codec.available())||(h.codec==2&&sizeof(T)!=8)||(h.codec==3&&sizeof(T)!=4))throw std::runtime_error("invalid mesh frame");
                tasks.push_back({h,mapped.at(cursor),nullptr});cursor+=h.stored;left-=h.raw;
            }});
            if(cursor!=limit||metadata.result()!=f.body_hash)return false;
            size_t task=0;
            arrays(m,[&](auto& v,int i){v.resize(size_t(f.sizes[i]));size_t left=v.size()*sizeof(v[0]);auto dst=reinterpret_cast<char*>(v.data());while(left){auto& t=tasks[task++];t.dst=dst;dst+=t.header.raw;left-=size_t(t.header.raw);}});
            std::atomic<size_t> next{0};std::atomic<bool> failed{false};std::vector<std::thread> workers;
            auto work=[&](){try{std::vector<char> scratch;while(!failed.load()){size_t i=next.fetch_add(1);if(i>=tasks.size())break;const auto& t=tasks[i];if(t.header.codec>=2){scratch.resize(size_t(t.header.raw));codec.decode(scratch.data(),scratch.size(),t.src,size_t(t.header.stored));if(t.header.codec==2)shuffle_words<8,false>(t.dst,scratch.data(),scratch.size());else shuffle_words<4,false>(t.dst,scratch.data(),scratch.size());}else if(t.header.codec)codec.decode(t.dst,size_t(t.header.raw),t.src,size_t(t.header.stored));else std::memcpy(t.dst,t.src,size_t(t.header.raw));BodyHash checksum;checksum.add(t.dst,size_t(t.header.raw));auto d=checksum.result();uint32_t crc;std::memcpy(&crc,d.data(),4);if(crc!=t.header.crc)throw std::runtime_error("mesh checksum mismatch");}}catch(...){failed.store(true);}};
            const size_t count=total<32*1024*1024?1:std::min<size_t>(4,tasks.size());
            if(count<=1)work();
            else try{for(size_t i=0;i<count;++i)workers.emplace_back(work);}catch(...){failed.store(true);}
            for(auto& worker:workers)worker.join();if(failed.load())return false;
        } else {
        std::unique_ptr<Hash> hash;if(old)hash=std::make_unique<Hash>();BodyHash fast;in.seekg(f.offset);
        arrays(m,[&](auto& v,int i){v.resize(size_t(f.sizes[i]));auto p=reinterpret_cast<char*>(v.data());size_t left=v.size()*sizeof(v[0]);while(left){size_t n=std::min<size_t>(left,8*1024*1024);if(!in.read(p,n))throw std::runtime_error("incomplete cache");if(old)hash->add(p,n);else fast.add(p,n);p+=n;left-=n;}});
        if((old?hash->result():fast.result())!=f.body_hash)return false;
        }
        if(m.positions.size()%3||m.faces.size()%3||m.coverage_positions.size()%3||m.coverage_colors.size()!=m.coverage_positions.size()||f.visible>f.source||f.referenced>f.visible||m.coverage_positions.size()/3!=f.visible-f.referenced||m.vertex_class.size()!=m.vertices())return false;
        for(auto index:m.faces)if(index<0||uint64_t(index)>=m.vertices())return false;
        for(auto index:m.mixed_faces)if(index<0||uint64_t(index)>=m.triangles())return false;
        if(!m.colors.empty()&&m.colors.size()!=m.faces.size())return false;
        if(!m.vertex_colors.empty()&&m.vertex_colors.size()!=m.positions.size())return false;
        m.input_points=f.source;m.visible_points=f.visible;m.mesh_referenced_points=f.referenced;m.dropped_long_edges=f.dropped;
        mesh=std::move(m);classes=f.classes;method=SurfaceMethod(f.method);engine_lit=f.engine_lit!=0;if(legacy)*legacy=!chunks;return true;
    }catch(...){return false;}
}
// "what: <system message> (code N)" for the calling thread's last Win32 error.
inline std::string system_error_text(const char* what,DWORD code=GetLastError()){
    char buf[512]={};DWORD n=FormatMessageA(FORMAT_MESSAGE_FROM_SYSTEM|FORMAT_MESSAGE_IGNORE_INSERTS,nullptr,code,MAKELANGID(LANG_NEUTRAL,SUBLANG_DEFAULT),buf,sizeof(buf),nullptr);
    while(n&&(buf[n-1]=='\r'||buf[n-1]=='\n'||buf[n-1]==' '||buf[n-1]=='.'))buf[--n]=0;
    return std::string(what)+": "+(n?buf:"unknown error")+" (code "+std::to_string(code)+")";
}
// Only a full-file cut is persisted: camera cuts do not grow the source file.
// Rewrite derived bytes only; interruption cannot damage any original point.
// Network (UNC) paths are written like local ones; `error` receives the real reason
// when nothing was stored. A tiled surface archive (LNTSurfaceArchive.h) in the tail is
// never replaced by a whole-file mesh.
inline bool save(const std::string& path,uint64_t offset,const Digest& source,const Digest& settings,SurfaceMesh& mesh,unsigned classes,SurfaceMethod method,bool engine_lit,bool chunks=true,std::string* error=nullptr){
    auto fail=[&](const std::string& why){if(error)*error=why;return false;};
    HANDLE file=CreateFileA(path.c_str(),GENERIC_READ|GENERIC_WRITE,FILE_SHARE_READ|FILE_SHARE_WRITE,nullptr,OPEN_EXISTING,FILE_ATTRIBUTE_NORMAL,nullptr);
    if(file==INVALID_HANDLE_VALUE)return fail(system_error_text("cannot open .lnt for writing"));
    OVERLAPPED lock{};lock.Offset=DWORD(offset);lock.OffsetHigh=DWORD(offset>>32);
    if(!LockFileEx(file,LOCKFILE_EXCLUSIVE_LOCK|LOCKFILE_FAIL_IMMEDIATELY,0,MAXDWORD,0x7fffffff,&lock)){const std::string why=system_error_text("derived tail is locked by another reader or writer");CloseHandle(file);return fail(why);}
    {
        LARGE_INTEGER size{};char magic[8]={};DWORD got=0;OVERLAPPED at{};
        if(GetFileSizeEx(file,&size)&&uint64_t(size.QuadPart)>=offset+448){
            const uint64_t pos=uint64_t(size.QuadPart)-448;at.Offset=DWORD(pos);at.OffsetHigh=DWORD(pos>>32);
            if(ReadFile(file,magic,8,&got,&at)&&got==8&&std::memcmp(magic,"LNTSLA1\0",8)==0){
                UnlockFileEx(file,0,MAXDWORD,0x7fffffff,&lock);CloseHandle(file);
                return fail("a tiled surface archive occupies the derived tail");
            }
        }
    }
    bool ok=false;
    try{
        Footer f;f.offset=offset;f.source=mesh.input_points;f.visible=mesh.visible_points;f.referenced=mesh.mesh_referenced_points;f.dropped=mesh.dropped_long_edges;f.classes=classes;f.method=uint32_t(method);f.engine_lit=engine_lit;f.source_hash=source;f.settings_hash=settings;
        LARGE_INTEGER pos{};pos.QuadPart=offset;if(!SetFilePointerEx(file,pos,nullptr,FILE_BEGIN))throw std::runtime_error("cache seek failed");
        BodyHash hash;Hash metadata;MeshCodec codec;std::vector<char> compressed,candidate,shuffled;
        auto write=[&](const void* data,size_t bytes){auto p=static_cast<const char*>(data);while(bytes){DWORD n=DWORD(std::min<size_t>(bytes,8*1024*1024)),written=0;if(!WriteFile(file,p,n,&written,nullptr)||written!=n)throw std::runtime_error("cache write failed");p+=n;bytes-=n;}};
        arrays(mesh,[&](auto& v,int i){
            f.sizes[i]=v.size();size_t left=v.size()*sizeof(v[0]);auto src=reinterpret_cast<const char*>(v.data());
            if(!chunks){hash.add(src,left);write(src,left);f.bytes+=left;return;}
            while(left){
                size_t n=std::min(left,kMeshChunkBytes);MeshChunkHeader h;h.raw=n;
                BodyHash crc;crc.add(src,n);auto d=crc.result();std::memcpy(&h.crc,d.data(),4);
                h.codec=codec.encode_frame(src,n,sizeof(v[0]),compressed,candidate,shuffled);
                h.stored=h.codec?compressed.size():n;metadata.value(h);
                write(&h,sizeof(h));write(h.codec?compressed.data():src,size_t(h.stored));
                f.bytes+=sizeof(h)+h.stored;left-=n;src+=n;
            }
        });
        if(!chunks)f.magic[6]='2';
        f.body_hash=chunks?metadata.result():hash.result();Hash footer_check;footer_check.add(&f,offsetof(Footer,footer_hash));f.footer_hash=footer_check.result();write(&f,sizeof(f));if(!SetEndOfFile(file)||!FlushFileBuffers(file))throw std::runtime_error("cache commit failed");ok=true;
        if(chunks){uint64_t raw=0;arrays(mesh,[&](auto& v,int){raw+=v.size()*sizeof(v[0]);});std::fprintf(stderr,"[LNT surface] embedded mesh storage: %.1f MB raw -> %.1f MB stored, lossless%s\n",double(raw)/1e6,double(f.bytes)/1e6,codec.available()?" zstd":" raw frames");}
    }catch(const std::exception& e){if(error)*error=system_error_text(e.what());}catch(...){if(error)*error="unknown write failure";}
    UnlockFileEx(file,0,MAXDWORD,0x7fffffff,&lock);CloseHandle(file);return ok;
}
} }

