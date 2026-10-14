#pragma once
#include <windows.h>
#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace lnt { namespace embedded_surface {
// Independent lossless frames: no quantization, point reduction or float conversion.
constexpr size_t kMeshChunkBytes=8*1024*1024;
struct MeshChunkHeader { uint64_t raw=0,stored=0;uint32_t crc=0,codec=0; };
static_assert(sizeof(MeshChunkHeader)==24,"mesh chunk layout changed");
// Reorder bytes only, improving repeated exponent/high-index byte compression.
// Reverse transformation restores the original doubles and indices bit for bit.
template<size_t Width,bool Encode> inline void shuffle_words(char* dst,const char* src,size_t bytes){
    const size_t words=bytes/Width;
    for(size_t i=0;i<words;++i)for(size_t b=0;b<Width;++b){
        if constexpr(Encode)dst[b*words+i]=src[i*Width+b];
        else dst[i*Width+b]=src[b*words+i];
    }
}
class MeshCodec {
    HMODULE dll=nullptr;
    using Bound=size_t(*)(size_t);
    using Compress=size_t(*)(void*,size_t,const void*,size_t,int);
    using Decompress=size_t(*)(void*,size_t,const void*,size_t);
    using Error=unsigned(*)(size_t);
    Bound bound=nullptr;Compress compress=nullptr;Decompress decompress=nullptr;Error error=nullptr;
    static void anchor(){}
public:
    MeshCodec(){
        HMODULE owner=nullptr;char name[32768]{};
        if(GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS|GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,reinterpret_cast<LPCSTR>(&anchor),&owner)&&GetModuleFileNameA(owner,name,sizeof(name))){
            std::string path(name);auto slash=path.find_last_of("\\/");if(slash!=std::string::npos){path.resize(slash);path+="/../libzstd.dll";dll=LoadLibraryA(path.c_str());}
        }
        if(!dll)dll=LoadLibraryA("native/libzstd.dll");
        if(!dll)dll=LoadLibraryA("libzstd.dll");
        if(!dll)return;
        bound=reinterpret_cast<Bound>(GetProcAddress(dll,"ZSTD_compressBound"));
        compress=reinterpret_cast<Compress>(GetProcAddress(dll,"ZSTD_compress"));
        decompress=reinterpret_cast<Decompress>(GetProcAddress(dll,"ZSTD_decompress"));
        error=reinterpret_cast<Error>(GetProcAddress(dll,"ZSTD_isError"));
    }
    ~MeshCodec(){if(dll)FreeLibrary(dll);}
    bool available()const{return bound&&compress&&decompress&&error;}
    bool encode(const char* src,size_t n,std::vector<char>& dst,int level=1)const {
        if(!available())return false;
        dst.resize(bound(n));auto size=compress(dst.data(),dst.size(),src,n,level);
        if(error(size)||size>=n)return false;dst.resize(size);return true;
    }
    uint32_t encode_frame(const char* src,size_t n,size_t width,std::vector<char>& output,
                          std::vector<char>& candidate,std::vector<char>& shuffled)const {
        const bool plain=encode(src,n,output);
        if(available()&&n>=4096&&(width==4||width==8)){
            shuffled.resize(n);
            if(width==8)shuffle_words<8,true>(shuffled.data(),src,n);
            else shuffle_words<4,true>(shuffled.data(),src,n);
            // Prefer direct decode unless byte reordering makes the frame substantially smaller.
            if(encode(shuffled.data(),n,candidate)&&(!plain||candidate.size()<output.size()*0.85)){
                output.swap(candidate);return width==8?2:3;
            }
        }
        return plain?1:0;
    }
    void decode(char* dst,size_t n,const char* src,size_t stored)const {
        if(!available())throw std::runtime_error("embedded mesh requires libzstd.dll");
        auto size=decompress(dst,n,src,stored);if(error(size)||size!=n)throw std::runtime_error("invalid compressed mesh frame");
    }
};
class MeshMapping {
    HANDLE file=INVALID_HANDLE_VALUE,mapping=nullptr;const char* data=nullptr;
    OVERLAPPED lock{};bool locked=false;
public:
    explicit MeshMapping(const std::string& path,uint64_t start,uint64_t expected_size){
        file=CreateFileA(path.c_str(),GENERIC_READ,FILE_SHARE_READ|FILE_SHARE_WRITE|FILE_SHARE_DELETE,nullptr,OPEN_EXISTING,FILE_ATTRIBUTE_NORMAL,nullptr);
        if(file==INVALID_HANDLE_VALUE)throw std::runtime_error("cannot open embedded mesh");
        lock.Offset=DWORD(start);lock.OffsetHigh=DWORD(start>>32);
        locked=LockFileEx(file,LOCKFILE_FAIL_IMMEDIATELY,0,MAXDWORD,0x7fffffff,&lock)!=0;
        LARGE_INTEGER size{};
        if(!locked||!GetFileSizeEx(file,&size)||uint64_t(size.QuadPart)!=expected_size){if(locked)UnlockFileEx(file,0,MAXDWORD,0x7fffffff,&lock);CloseHandle(file);throw std::runtime_error("embedded mesh is changing");}
        mapping=CreateFileMappingA(file,nullptr,PAGE_READONLY,0,0,nullptr);
        if(mapping)data=static_cast<const char*>(MapViewOfFile(mapping,FILE_MAP_READ,0,0,0));
        if(!data){if(mapping)CloseHandle(mapping);UnlockFileEx(file,0,MAXDWORD,0x7fffffff,&lock);CloseHandle(file);throw std::runtime_error("cannot map embedded mesh");}
    }
    ~MeshMapping(){if(data)UnmapViewOfFile(data);if(mapping)CloseHandle(mapping);if(locked)UnlockFileEx(file,0,MAXDWORD,0x7fffffff,&lock);if(file!=INVALID_HANDLE_VALUE)CloseHandle(file);}
    const char* at(uint64_t offset)const{return data+offset;}
};
} }
