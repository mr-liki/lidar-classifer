// Naksha Vulkan-first Ground PTD: GPU certified-interior simplex proposals.
// A conservative offscreen compute context; DOES NOT implement GPU Delaunay.
// The application always owns CPU/VTK compatibility and LNT commits.
#include <vulkan/vulkan.h>
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(_WIN32)
#define NK_API extern "C" __declspec(dllexport)
#else
#define NK_API extern "C" __attribute__((visibility("default")))
#endif

namespace {
constexpr std::uint32_t MAX_TRI = 500000;
constexpr std::uint32_t MAX_ENTRIES = 2500000;
constexpr std::uint32_t GRID_OFFSETS = 4097;
constexpr std::uint32_t BATCH = 32768;
constexpr std::uint32_t WG = 128;
constexpr std::array<VkDeviceSize, 5> SIZE{
    VkDeviceSize(MAX_TRI)*6u*8u,
    VkDeviceSize(GRID_OFFSETS)*4u,
    VkDeviceSize(MAX_ENTRIES)*4u,
    VkDeviceSize(BATCH)*2u*8u,
    VkDeviceSize(BATCH)*4u,
};

struct Push {
    std::uint32_t count;
    std::uint32_t reserved;
    double x, y, invx, invy;
};
static_assert(sizeof(Push) == 40, "Vulkan push-constant ABI mismatch");

void message(char* dst, int cap, const std::string& s) noexcept {
    if (!dst || cap <= 0) return;
    const auto n = std::min(std::size_t(cap-1), s.size());
    std::memcpy(dst, s.data(), n);
    dst[n] = '\0';
}
void check(VkResult rc, const char* name) {
    if (rc != VK_SUCCESS) throw std::runtime_error(std::string(name) + " Vulkan code=" + std::to_string(rc));
}
struct Buffer {
    VkBuffer handle = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    void* mapped = nullptr;
};

class VkLocator {
public:
    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice physical = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VkQueue queue = VK_NULL_HANDLE;
    std::uint32_t family = 0;
    VkShaderModule shader = VK_NULL_HANDLE;
    VkDescriptorSetLayout layout = VK_NULL_HANDLE;
    VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
    VkDescriptorSet descriptorSet = VK_NULL_HANDLE;
    VkCommandPool cmdPool = VK_NULL_HANDLE;
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;
    Buffer buffers[5]{};   // DEVICE_LOCAL shader-visible SSBOs
    Buffer staging[5]{};   // host-coherent transfer in/out
    std::mutex mutex;
    bool ready = false;
    std::string deviceName;

    explicit VkLocator(const std::string& path) {
        try { initialize(path); }
        catch (...) { cleanup(); throw; }
    }
    ~VkLocator() { cleanup(); }
    VkLocator(const VkLocator&) = delete;
    VkLocator& operator=(const VkLocator&) = delete;

    void destroy(Buffer& b) noexcept {
        if (!device) return;
        if (b.mapped) vkUnmapMemory(device, b.memory);
        if (b.handle) vkDestroyBuffer(device, b.handle, nullptr);
        if (b.memory) vkFreeMemory(device, b.memory, nullptr);
        b = {};
    }
    void cleanup() noexcept {
        if (device) vkDeviceWaitIdle(device);
        if (device && fence) vkDestroyFence(device, fence, nullptr);
        if (device && cmdPool) vkDestroyCommandPool(device, cmdPool, nullptr);
        if (device && descriptorPool) vkDestroyDescriptorPool(device, descriptorPool, nullptr);
        if (device && pipeline) vkDestroyPipeline(device, pipeline, nullptr);
        if (device && pipelineLayout) vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
        if (device && layout) vkDestroyDescriptorSetLayout(device, layout, nullptr);
        if (device && shader) vkDestroyShaderModule(device, shader, nullptr);
        for (auto& b : buffers) destroy(b);
        for (auto& b : staging) destroy(b);
        if (device) vkDestroyDevice(device, nullptr);
        if (instance) vkDestroyInstance(instance, nullptr);
        device = VK_NULL_HANDLE;
        instance = VK_NULL_HANDLE;
        ready = false;
    }
    void allocate(Buffer& b, VkDeviceSize bytes, VkBufferUsageFlags usage,
                  VkMemoryPropertyFlags required) {
        VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        bi.size = bytes;
        bi.usage = usage;
        bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        check(vkCreateBuffer(device, &bi, nullptr, &b.handle), "vkCreateBuffer");
        VkMemoryRequirements rq{};
        vkGetBufferMemoryRequirements(device, b.handle, &rq);
        VkPhysicalDeviceMemoryProperties mem{};
        vkGetPhysicalDeviceMemoryProperties(physical, &mem);
        std::uint32_t chosen = UINT32_MAX;
        for (std::uint32_t i=0; i<mem.memoryTypeCount; ++i) {
            if ((rq.memoryTypeBits & (1u<<i)) && (mem.memoryTypes[i].propertyFlags & required) == required) {
                chosen = i;
                break;
            }
        }
        if (chosen == UINT32_MAX) throw std::runtime_error("Required device-local or host-coherent Vulkan memory unavailable");
        VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        ai.allocationSize = rq.size;
        ai.memoryTypeIndex = chosen;
        check(vkAllocateMemory(device, &ai, nullptr, &b.memory), "vkAllocateMemory");
        check(vkBindBufferMemory(device, b.handle, b.memory, 0), "vkBindBufferMemory");
        if (required & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)
            check(vkMapMemory(device, b.memory, 0, bytes, 0, &b.mapped), "vkMapMemory");
    }
    void initialize(const std::string& shaderPath) {
        std::ifstream in(shaderPath, std::ios::binary | std::ios::ate);
        if (!in) throw std::runtime_error("Vulkan locator SPIR-V missing");
        const std::streamoff len = in.tellg();
        if (len <= 0 || len > 1024*1024 || (len%4)!=0)
            throw std::runtime_error("Invalid SPIR-V shader size");
        std::vector<std::uint32_t> spirv(std::size_t(len)/4);
        in.seekg(0);
        if (!in.read(reinterpret_cast<char*>(spirv.data()), std::streamsize(len)) || spirv[0] != 0x07230203)
            throw std::runtime_error("Invalid SPIR-V contents");
        VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
        app.pApplicationName = "Naksha Ground Vulkan Indexed Locator";
        app.apiVersion = VK_API_VERSION_1_1;
        VkInstanceCreateInfo ci{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
        ci.pApplicationInfo = &app;
        check(vkCreateInstance(&ci,nullptr,&instance), "vkCreateInstance");
        std::uint32_t num=0;
        check(vkEnumeratePhysicalDevices(instance, &num, nullptr), "vkEnumeratePhysicalDevices");
        if (!num) throw std::runtime_error("No Vulkan GPU available");
        std::vector<VkPhysicalDevice> devices(num);
        check(vkEnumeratePhysicalDevices(instance,&num,devices.data()), "vkEnumeratePhysicalDevices list");
        int top=-1;
        for (auto d : devices) {
            VkPhysicalDeviceFeatures f{};
            VkPhysicalDeviceProperties props{};
            vkGetPhysicalDeviceFeatures(d,&f);
            vkGetPhysicalDeviceProperties(d,&props);
            if (!f.shaderFloat64 || props.limits.maxPushConstantsSize < sizeof(Push) ||
                props.limits.maxComputeWorkGroupCount[0] < BATCH/WG ||
                props.limits.maxStorageBufferRange < SIZE[0] ||
                props.limits.maxStorageBufferRange < SIZE[2]) continue;
            std::uint32_t nfamilies=0;
            vkGetPhysicalDeviceQueueFamilyProperties(d,&nfamilies,nullptr);
            std::vector<VkQueueFamilyProperties> families(nfamilies);
            vkGetPhysicalDeviceQueueFamilyProperties(d,&nfamilies,families.data());
            for (std::uint32_t i=0; i<nfamilies; ++i) {
                if (!families[i].queueCount || !(families[i].queueFlags & VK_QUEUE_COMPUTE_BIT)) continue;
                int score = (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU ? 100 : 0)
                    + (props.vendorID == 0x10de ? 20 : 0)
                    + (!(families[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) ? 1 : 0);
                if (score > top) {
                    top=score; physical=d; family=i; deviceName=props.deviceName;
                }
            }
        }
        if (!physical) throw std::runtime_error("No FP64 Vulkan GPU with required buffer range/compute");
        const float priority=1.0f;
        VkDeviceQueueCreateInfo qi{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
        qi.queueFamilyIndex=family;
        qi.queueCount=1;
        qi.pQueuePriorities=&priority;
        VkPhysicalDeviceFeatures enabled{};
        enabled.shaderFloat64=VK_TRUE;
        VkDeviceCreateInfo di{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
        di.queueCreateInfoCount=1;
        di.pQueueCreateInfos=&qi;
        di.pEnabledFeatures=&enabled;
        check(vkCreateDevice(physical,&di,nullptr,&device), "vkCreateDevice");
        vkGetDeviceQueue(device,family,0,&queue);
        VkShaderModuleCreateInfo sm{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        sm.codeSize=std::size_t(len);
        sm.pCode=spirv.data();
        check(vkCreateShaderModule(device,&sm,nullptr,&shader), "vkCreateShaderModule");
        std::array<VkDescriptorSetLayoutBinding,5> bindings{};
        for (std::uint32_t i=0;i<bindings.size();++i) {
            bindings[i].binding=i;
            bindings[i].descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            bindings[i].descriptorCount=1;
            bindings[i].stageFlags=VK_SHADER_STAGE_COMPUTE_BIT;
        }
        VkDescriptorSetLayoutCreateInfo dsl{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        dsl.bindingCount=std::uint32_t(bindings.size());
        dsl.pBindings=bindings.data();
        check(vkCreateDescriptorSetLayout(device,&dsl,nullptr,&layout), "vkCreateDescriptorSetLayout");
        VkPushConstantRange range{};
        range.stageFlags=VK_SHADER_STAGE_COMPUTE_BIT;
        range.size=sizeof(Push);
        VkPipelineLayoutCreateInfo pl{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        pl.setLayoutCount=1;
        pl.pSetLayouts=&layout;
        pl.pushConstantRangeCount=1;
        pl.pPushConstantRanges=&range;
        check(vkCreatePipelineLayout(device,&pl,nullptr,&pipelineLayout), "vkCreatePipelineLayout");
        VkPipelineShaderStageCreateInfo stage{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
        stage.stage=VK_SHADER_STAGE_COMPUTE_BIT;
        stage.module=shader;
        stage.pName="main";
        VkComputePipelineCreateInfo info{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        info.stage=stage;
        info.layout=pipelineLayout;
        check(vkCreateComputePipelines(device,VK_NULL_HANDLE,1,&info,nullptr,&pipeline), "vkCreateComputePipelines");
        // Discrete NVIDIA T400 has scarce CPU-visible VRAM; placing large
        // random-access triangle buffers in HOST_VISIBLE memory would force
        // shader PCIe reads. Keep SSBOs in fast DEVICE_LOCAL memory and use
        // host-coherent staging buffers only for PCIe upload/readback.
        for (std::uint32_t i=0;i<buffers_count();++i) {
            const VkBufferUsageFlags deviceUsage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                (i < 4 ? VK_BUFFER_USAGE_TRANSFER_DST_BIT : VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
            allocate(buffers[i],SIZE[i],deviceUsage,VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
            const VkBufferUsageFlags stagingUsage = i < 4 ? VK_BUFFER_USAGE_TRANSFER_SRC_BIT : VK_BUFFER_USAGE_TRANSFER_DST_BIT;
            allocate(staging[i],SIZE[i],stagingUsage,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        }
        VkDescriptorPoolSize poolSize{};
        poolSize.type=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        poolSize.descriptorCount=5;
        VkDescriptorPoolCreateInfo pi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        pi.maxSets=1;
        pi.poolSizeCount=1;
        pi.pPoolSizes=&poolSize;
        check(vkCreateDescriptorPool(device,&pi,nullptr,&descriptorPool), "vkCreateDescriptorPool");
        VkDescriptorSetAllocateInfo alloc{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        alloc.descriptorPool=descriptorPool;
        alloc.descriptorSetCount=1;
        alloc.pSetLayouts=&layout;
        check(vkAllocateDescriptorSets(device,&alloc,&descriptorSet), "vkAllocateDescriptorSets");
        std::array<VkDescriptorBufferInfo,5> dbi{};
        std::array<VkWriteDescriptorSet,5> writes{};
        for (std::uint32_t i=0;i<dbi.size();++i) {
            dbi[i].buffer=buffers[i].handle;
            dbi[i].range=SIZE[i];
            writes[i].sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[i].dstSet=descriptorSet;
            writes[i].dstBinding=i;
            writes[i].descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[i].descriptorCount=1;
            writes[i].pBufferInfo=&dbi[i];
        }
        vkUpdateDescriptorSets(device,std::uint32_t(writes.size()),writes.data(),0,nullptr);
        VkCommandPoolCreateInfo cpi{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
        cpi.queueFamilyIndex=family;
        cpi.flags=VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        check(vkCreateCommandPool(device,&cpi,nullptr,&cmdPool), "vkCreateCommandPool");
        VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        cai.commandPool=cmdPool;
        cai.level=VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cai.commandBufferCount=1;
        check(vkAllocateCommandBuffers(device,&cai,&cmd), "vkAllocateCommandBuffers");
        VkFenceCreateInfo fi{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        check(vkCreateFence(device,&fi,nullptr,&fence), "vkCreateFence");
        ready=true;
    }
    static constexpr std::size_t buffers_count() { return 5; }

    void run(const double* transform, std::uint32_t ntri,
             const std::uint32_t* offsets, std::uint32_t n_off,
             const std::uint32_t* entries, std::uint32_t n_entries,
             const double* xy, std::uint32_t nquery,
             double lo_x,double lo_y,double inv_x,double inv_y,
             std::int32_t* out) {
        std::lock_guard<std::mutex> lock(mutex);
        if (!ready || !transform || !offsets || !entries || !xy || !out ||
            !ntri || ntri>MAX_TRI || n_off!=GRID_OFFSETS || !n_entries || n_entries>MAX_ENTRIES ||
            !nquery || nquery>10000000 || offsets[0]!=0 || offsets[GRID_OFFSETS-1]!=n_entries ||
            !(inv_x>0.0) || !(inv_y>0.0))
            throw std::runtime_error("Vulkan locator input invalid or exceeds VRAM/buffer limit");
        for (std::uint32_t i=0;i<GRID_OFFSETS-1;i++) {
            if (offsets[i]>offsets[i+1] || offsets[i]>n_entries)
                throw std::runtime_error("Vulkan locator corrupt CSR grid offsets");
        }
        for (std::uint32_t i=0;i<n_entries;i++) {
            if (entries[i]>=ntri) throw std::runtime_error("Vulkan locator corrupt simplex index");
        }
        std::memcpy(staging[0].mapped,transform,std::size_t(ntri)*6u*8u);
        std::memcpy(staging[1].mapped,offsets,std::size_t(GRID_OFFSETS)*4u);
        std::memcpy(staging[2].mapped,entries,std::size_t(n_entries)*4u);
        for (std::uint32_t begin=0;begin<nquery;) {
            std::uint32_t count=std::min(BATCH,nquery-begin);
            std::memcpy(staging[3].mapped,xy+std::size_t(begin)*2u,std::size_t(count)*2u*8u);
            check(vkResetCommandBuffer(cmd,0),"vkResetCommandBuffer");
            VkCommandBufferBeginInfo cb{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
            cb.flags=VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
            check(vkBeginCommandBuffer(cmd,&cb),"vkBeginCommandBuffer");
            // HOST writes -> transfer reads from coherent staging memory.
            VkMemoryBarrier hostToTransfer{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
            hostToTransfer.srcAccessMask=VK_ACCESS_HOST_WRITE_BIT;
            hostToTransfer.dstAccessMask=VK_ACCESS_TRANSFER_READ_BIT;
            vkCmdPipelineBarrier(cmd,VK_PIPELINE_STAGE_HOST_BIT,VK_PIPELINE_STAGE_TRANSFER_BIT,
                                 0,1,&hostToTransfer,0,nullptr,0,nullptr);
            if (begin == 0) {
                for (std::uint32_t i=0;i<3;i++) {
                    VkBufferCopy region{};
                    region.size=(i==0) ? std::size_t(ntri)*6u*8u :
                                (i==1) ? std::size_t(GRID_OFFSETS)*4u : std::size_t(n_entries)*4u;
                    vkCmdCopyBuffer(cmd,staging[i].handle,buffers[i].handle,1,&region);
                }
            }
            VkBufferCopy queryRegion{};
            queryRegion.size=std::size_t(count)*2u*8u;
            vkCmdCopyBuffer(cmd,staging[3].handle,buffers[3].handle,1,&queryRegion);
            VkMemoryBarrier toShader{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
            toShader.srcAccessMask=VK_ACCESS_TRANSFER_WRITE_BIT;
            toShader.dstAccessMask=VK_ACCESS_SHADER_READ_BIT;
            vkCmdPipelineBarrier(cmd,VK_PIPELINE_STAGE_TRANSFER_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                 0,1,&toShader,0,nullptr,0,nullptr);
            vkCmdBindPipeline(cmd,VK_PIPELINE_BIND_POINT_COMPUTE,pipeline);
            vkCmdBindDescriptorSets(cmd,VK_PIPELINE_BIND_POINT_COMPUTE,pipelineLayout,
                                    0,1,&descriptorSet,0,nullptr);
            Push pc{count,0u,lo_x,lo_y,inv_x,inv_y};
            vkCmdPushConstants(cmd,pipelineLayout,VK_SHADER_STAGE_COMPUTE_BIT,0,sizeof(pc),&pc);
            vkCmdDispatch(cmd,(count+WG-1u)/WG,1,1);
            VkMemoryBarrier toCopy{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
            toCopy.srcAccessMask=VK_ACCESS_SHADER_WRITE_BIT;
            toCopy.dstAccessMask=VK_ACCESS_TRANSFER_READ_BIT;
            vkCmdPipelineBarrier(cmd,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_PIPELINE_STAGE_TRANSFER_BIT,
                                 0,1,&toCopy,0,nullptr,0,nullptr);
            VkBufferCopy resultRegion{};
            resultRegion.size=std::size_t(count)*4u;
            vkCmdCopyBuffer(cmd,buffers[4].handle,staging[4].handle,1,&resultRegion);
            VkMemoryBarrier toHost{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
            toHost.srcAccessMask=VK_ACCESS_TRANSFER_WRITE_BIT;
            toHost.dstAccessMask=VK_ACCESS_HOST_READ_BIT;
            vkCmdPipelineBarrier(cmd,VK_PIPELINE_STAGE_TRANSFER_BIT,VK_PIPELINE_STAGE_HOST_BIT,
                                 0,1,&toHost,0,nullptr,0,nullptr);
            check(vkEndCommandBuffer(cmd),"vkEndCommandBuffer");
            check(vkResetFences(device,1,&fence),"vkResetFences");
            VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
            si.commandBufferCount=1;
            si.pCommandBuffers=&cmd;
            check(vkQueueSubmit(queue,1,&si,fence),"vkQueueSubmit");
            // Fenced and bounded. If GPU cannot finish within 20s, CPU fallback.
            check(vkWaitForFences(device,1,&fence,VK_TRUE,20000000000ull),"vkWaitForFences");
            std::memcpy(out+begin,staging[4].mapped,std::size_t(count)*4u);
            begin+=count;
        }
    }
};
}

NK_API int nk_ground_vk_locator_create(const char* shader_path, void** out,
                                       char* err, int err_cap) noexcept {
    if (out) *out=nullptr;
    try {
        if (!shader_path || !out) throw std::runtime_error("Vulkan locator shader path/context is null");
        auto ctx=std::make_unique<VkLocator>(shader_path);
        message(err,err_cap,"FP64 Vulkan indexed simplex ready: "+ctx->deviceName);
        *out=ctx.release();
        return 0;
    } catch (const std::exception& ex) {
        message(err,err_cap,ex.what());
        return 1;
    } catch (...) {
        message(err,err_cap,"Unknown Vulkan locator init error");
        return 2;
    }
}

NK_API int nk_ground_vk_locator_locate(void* pointer,
                     const double* transforms, std::uint32_t ntri,
                     const std::uint32_t* offsets, std::uint32_t noff,
                     const std::uint32_t* entries, std::uint32_t nentries,
                     const double* query_xy, std::uint32_t nquery,
                     double minx,double miny,double invx,double invy,
                     std::int32_t* out,char* err,int err_cap) noexcept {
    try {
        if (!pointer) throw std::runtime_error("No Vulkan locator context");
        static_cast<VkLocator*>(pointer)->run(transforms,ntri,offsets,noff,
                                               entries,nentries,query_xy,nquery,
                                               minx,miny,invx,invy,out);
        message(err,err_cap,"Vulkan parallel indexed lookup complete");
        return 0;
    } catch (const std::exception& ex) {
        message(err,err_cap,ex.what());
        return 3;
    } catch (...) {
        message(err,err_cap,"Unknown Vulkan locator compute error");
        return 4;
    }
}
NK_API void nk_ground_vk_locator_release(void* ctx) noexcept {
    try { delete static_cast<VkLocator*>(ctx); } catch (...) { }
}
