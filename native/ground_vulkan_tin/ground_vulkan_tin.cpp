// Naksha Vulkan TIN facet builder (experimental, isolated from renderer).
// Delaunay TOPOLOGY is still determined by SciPy/Qhull on the CPU. This Vulkan
// shader constructs the normal/edge geometry for each triangle in parallel.
// Python verifies bit-exact outputs against the original reference geometry,
// substitutes reference facets on any mismatches, and commits nothing to LNT.
// This is NOT a GPU replacement for Delaunay connectivity construction.
#include <vulkan/vulkan.h>
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>
#include <cstdio>
#include <cmath>

#if defined(_WIN32)
#define NKVT_API extern "C" __declspec(dllexport)
#else
#define NKVT_API extern "C" __attribute__((visibility("default")))
#endif

namespace {
constexpr std::uint32_t kCapacity = 16384;
constexpr std::uint32_t kLocalSize = 128;
constexpr VkDeviceSize kInputBytes = VkDeviceSize(kCapacity) * 9 * sizeof(double);
constexpr VkDeviceSize kOutputBytes = VkDeviceSize(kCapacity) * 5 * sizeof(double);

struct PushConstants {
    std::uint32_t count;
};
static_assert(sizeof(PushConstants) == 4, "GLSL push constant layout mismatch");

void put_message(char* out, int capacity, const std::string& msg) noexcept {
    if (!out || capacity <= 0) return;
    const auto n = std::min<std::size_t>(std::size_t(capacity - 1), msg.size());
    std::memcpy(out, msg.data(), n);
    out[n] = '\0';
}
void vk_check(VkResult result, const char* operation) {
    if (result != VK_SUCCESS) {
        throw std::runtime_error(std::string(operation) + " failed Vulkan code " + std::to_string(result));
    }
}

struct Buffer {
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    void* mapped = nullptr;
};

struct Compute {
    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice physical = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VkQueue queue = VK_NULL_HANDLE;
    std::uint32_t family = 0;
    VkShaderModule shader = VK_NULL_HANDLE;
    VkDescriptorSetLayout descriptorLayout = VK_NULL_HANDLE;
    VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
    VkDescriptorSet descriptorSet = VK_NULL_HANDLE;
    VkCommandPool commandPool = VK_NULL_HANDLE;
    VkCommandBuffer commandBuffer = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;
    Buffer input, output;
    std::string deviceName;
    std::mutex mutex;
    bool ready = false;

    explicit Compute(const std::string& shaderPath) { try { initialize(shaderPath); } catch (...) { cleanup(); throw; } }
    ~Compute() { cleanup(); }
    Compute(const Compute&) = delete;
    Compute& operator=(const Compute&) = delete;

    void cleanup() noexcept {
        if (device) vkDeviceWaitIdle(device);
        if (device && fence) vkDestroyFence(device, fence, nullptr);
        if (device && commandPool) vkDestroyCommandPool(device, commandPool, nullptr);
        if (device && descriptorPool) vkDestroyDescriptorPool(device, descriptorPool, nullptr);
        if (device && pipeline) vkDestroyPipeline(device, pipeline, nullptr);
        if (device && pipelineLayout) vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
        if (device && descriptorLayout) vkDestroyDescriptorSetLayout(device, descriptorLayout, nullptr);
        if (device && shader) vkDestroyShaderModule(device, shader, nullptr);
        destroy_buffer(input);
        destroy_buffer(output);
        if (device) vkDestroyDevice(device, nullptr);
        if (instance) vkDestroyInstance(instance, nullptr);
        device = VK_NULL_HANDLE;
        instance = VK_NULL_HANDLE;
        ready = false;
    }
    void destroy_buffer(Buffer& b) noexcept {
        if (!device) return;
        if (b.mapped) vkUnmapMemory(device, b.memory);
        if (b.buffer) vkDestroyBuffer(device, b.buffer, nullptr);
        if (b.memory) vkFreeMemory(device, b.memory, nullptr);
        b = {};
    }
    void init_buffer(Buffer& b, VkDeviceSize bytes) {
        VkBufferCreateInfo ci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        ci.size = bytes;
        ci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
        ci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        vk_check(vkCreateBuffer(device, &ci, nullptr, &b.buffer), "vkCreateBuffer");
        VkMemoryRequirements requirements{};
        vkGetBufferMemoryRequirements(device, b.buffer, &requirements);
        VkPhysicalDeviceMemoryProperties mem{};
        vkGetPhysicalDeviceMemoryProperties(physical, &mem);
        std::uint32_t selected = UINT32_MAX;
        for (std::uint32_t i = 0; i < mem.memoryTypeCount; ++i) {
            VkMemoryPropertyFlags requested = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
            if ((requirements.memoryTypeBits & (1u << i)) &&
                (mem.memoryTypes[i].propertyFlags & requested) == requested) {
                selected = i;
                break;
            }
        }
        if (selected == UINT32_MAX)
            throw std::runtime_error("No HOST_VISIBLE+HOST_COHERENT Vulkan memory; CPU fallback required");
        VkMemoryAllocateInfo alloc{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        alloc.allocationSize = requirements.size;
        alloc.memoryTypeIndex = selected;
        vk_check(vkAllocateMemory(device, &alloc, nullptr, &b.memory), "vkAllocateMemory");
        vk_check(vkBindBufferMemory(device, b.buffer, b.memory, 0), "vkBindBufferMemory");
        vk_check(vkMapMemory(device, b.memory, 0, bytes, 0, &b.mapped), "vkMapMemory");
    }
    void initialize(const std::string& shaderPath) {
        std::ifstream stream(shaderPath, std::ios::binary | std::ios::ate);
        if (!stream) throw std::runtime_error("Compiled Vulkan TIN SPIR-V shader not found");
        const std::streamoff size = stream.tellg();
        if (size <= 0 || size > 1024 * 1024 || (size % 4) != 0)
            throw std::runtime_error("Invalid Vulkan TIN shader size");
        std::vector<std::uint32_t> spirv(std::size_t(size) / 4);
        stream.seekg(0);
        if (!stream.read(reinterpret_cast<char*>(spirv.data()), std::streamsize(size)))
            throw std::runtime_error("Cannot read Vulkan TIN shader");
        if (spirv[0] != 0x07230203u) throw std::runtime_error("Invalid SPIR-V magic");

        VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
        app.pApplicationName = "Naksha Vulkan TIN Facets";
        app.apiVersion = VK_API_VERSION_1_1;
        VkInstanceCreateInfo inst{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
        inst.pApplicationInfo = &app;
        vk_check(vkCreateInstance(&inst, nullptr, &instance), "vkCreateInstance");
        std::uint32_t count = 0;
        vk_check(vkEnumeratePhysicalDevices(instance, &count, nullptr), "vkEnumeratePhysicalDevices");
        if (!count) throw std::runtime_error("No Vulkan compute physical device");
        std::vector<VkPhysicalDevice> devices(count);
        vk_check(vkEnumeratePhysicalDevices(instance, &count, devices.data()), "vkEnumeratePhysicalDevices list");
        int score = -1;
        for (auto dev : devices) {
            VkPhysicalDeviceFeatures features{};
            VkPhysicalDeviceProperties props{};
            VkQueueFamilyProperties qp{};
            std::uint32_t familyCount = 0;
            vkGetPhysicalDeviceFeatures(dev, &features);
            vkGetPhysicalDeviceProperties(dev, &props);
            vkGetPhysicalDeviceQueueFamilyProperties(dev, &familyCount, nullptr);
            if (!features.shaderFloat64 || props.limits.maxStorageBufferRange < kInputBytes ||
                props.limits.maxStorageBufferRange < kOutputBytes ||
                props.limits.maxPushConstantsSize < sizeof(PushConstants) ||
                props.limits.maxComputeWorkGroupCount[0] < (kCapacity + kLocalSize - 1) / kLocalSize)
                continue;
            std::vector<VkQueueFamilyProperties> families(familyCount);
            vkGetPhysicalDeviceQueueFamilyProperties(dev, &familyCount, families.data());
            for (std::uint32_t i = 0; i < familyCount; ++i) {
                qp = families[i];
                if (!(qp.queueFlags & VK_QUEUE_COMPUTE_BIT) || qp.queueCount == 0) continue;
                const int candidate = ((props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) ? 100 : 0)
                                    + ((props.vendorID == 0x10DEu) ? 20 : 0)
                                    + ((qp.queueFlags & VK_QUEUE_GRAPHICS_BIT) ? 0 : 1);
                if (candidate > score) {
                    score = candidate;
                    physical = dev;
                    family = i;
                    deviceName = props.deviceName;
                }
            }
        }
        if (!physical) throw std::runtime_error("No Vulkan GPU supporting shaderFloat64 and compute; CPU TIN fallback required");
        const float priority = 1.0f;
        VkDeviceQueueCreateInfo queueInfo{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
        queueInfo.queueFamilyIndex = family;
        queueInfo.queueCount = 1;
        queueInfo.pQueuePriorities = &priority;
        VkPhysicalDeviceFeatures requested{};
        requested.shaderFloat64 = VK_TRUE;
        VkDeviceCreateInfo devInfo{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
        devInfo.queueCreateInfoCount = 1;
        devInfo.pQueueCreateInfos = &queueInfo;
        devInfo.pEnabledFeatures = &requested;
        vk_check(vkCreateDevice(physical, &devInfo, nullptr, &device), "vkCreateDevice");
        vkGetDeviceQueue(device, family, 0, &queue);

        VkShaderModuleCreateInfo moduleInfo{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        moduleInfo.codeSize = std::size_t(size);
        moduleInfo.pCode = spirv.data();
        vk_check(vkCreateShaderModule(device, &moduleInfo, nullptr, &shader), "vkCreateShaderModule");
        VkDescriptorSetLayoutBinding bindings[2]{};
        for (std::uint32_t i = 0; i < 2; ++i) {
            bindings[i].binding = i;
            bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            bindings[i].descriptorCount = 1;
            bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        }
        VkDescriptorSetLayoutCreateInfo dl{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        dl.bindingCount = 2;
        dl.pBindings = bindings;
        vk_check(vkCreateDescriptorSetLayout(device, &dl, nullptr, &descriptorLayout), "vkCreateDescriptorSetLayout");
        VkPushConstantRange range{};
        range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        range.size = sizeof(PushConstants);
        VkPipelineLayoutCreateInfo pl{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        pl.setLayoutCount = 1;
        pl.pSetLayouts = &descriptorLayout;
        pl.pushConstantRangeCount = 1;
        pl.pPushConstantRanges = &range;
        vk_check(vkCreatePipelineLayout(device, &pl, nullptr, &pipelineLayout), "vkCreatePipelineLayout");
        VkPipelineShaderStageCreateInfo stage{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
        stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        stage.module = shader;
        stage.pName = "main";
        VkComputePipelineCreateInfo cp{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        cp.stage = stage;
        cp.layout = pipelineLayout;
        vk_check(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &cp, nullptr, &pipeline), "vkCreateComputePipelines");

        init_buffer(input, kInputBytes);
        init_buffer(output, kOutputBytes);
        VkDescriptorPoolSize poolSize{};
        poolSize.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        poolSize.descriptorCount = 2;
        VkDescriptorPoolCreateInfo poolInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        poolInfo.maxSets = 1;
        poolInfo.poolSizeCount = 1;
        poolInfo.pPoolSizes = &poolSize;
        vk_check(vkCreateDescriptorPool(device, &poolInfo, nullptr, &descriptorPool), "vkCreateDescriptorPool");
        VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        ai.descriptorPool = descriptorPool;
        ai.descriptorSetCount = 1;
        ai.pSetLayouts = &descriptorLayout;
        vk_check(vkAllocateDescriptorSets(device, &ai, &descriptorSet), "vkAllocateDescriptorSets");
        VkDescriptorBufferInfo infos[2]{};
        infos[0] = {input.buffer, 0, kInputBytes};
        infos[1] = {output.buffer, 0, kOutputBytes};
        VkWriteDescriptorSet writes[2]{};
        for (std::uint32_t i = 0; i < 2; ++i) {
            writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[i].dstSet = descriptorSet;
            writes[i].dstBinding = i;
            writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[i].descriptorCount = 1;
            writes[i].pBufferInfo = &infos[i];
        }
        vkUpdateDescriptorSets(device, 2, writes, 0, nullptr);

        VkCommandPoolCreateInfo pool{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
        pool.queueFamilyIndex = family;
        pool.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        vk_check(vkCreateCommandPool(device, &pool, nullptr, &commandPool), "vkCreateCommandPool");
        VkCommandBufferAllocateInfo commandAlloc{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        commandAlloc.commandPool = commandPool;
        commandAlloc.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        commandAlloc.commandBufferCount = 1;
        vk_check(vkAllocateCommandBuffers(device, &commandAlloc, &commandBuffer), "vkAllocateCommandBuffers");
        VkFenceCreateInfo fi{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        vk_check(vkCreateFence(device, &fi, nullptr, &fence), "vkCreateFence");
        ready = true;
    }
    void build_facets(const double* vertices, std::uint32_t count, double* facets) {
        std::lock_guard<std::mutex> lock(mutex);
        if (!ready || !vertices || !facets || !count || count > kCapacity)
            throw std::runtime_error("Invalid Vulkan TIN compute input");
        std::memcpy(input.mapped, vertices, std::size_t(count) * 9 * sizeof(double));
        vk_check(vkResetCommandBuffer(commandBuffer, 0), "vkResetCommandBuffer");
        VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vk_check(vkBeginCommandBuffer(commandBuffer, &begin), "vkBeginCommandBuffer");
        VkBufferMemoryBarrier hostToCompute{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
        hostToCompute.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        hostToCompute.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        hostToCompute.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT;
        hostToCompute.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        hostToCompute.buffer = input.buffer;
        hostToCompute.offset = 0;
        hostToCompute.size = VkDeviceSize(count) * 9 * sizeof(double);
        vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_HOST_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 1, &hostToCompute, 0, nullptr);
        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                                pipelineLayout, 0, 1, &descriptorSet, 0, nullptr);
        PushConstants params{count};
        vkCmdPushConstants(commandBuffer, pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT,
                           0, sizeof(params), &params);
        vkCmdDispatch(commandBuffer, (count + kLocalSize - 1u) / kLocalSize, 1, 1);
        VkBufferMemoryBarrier computeToHost{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
        computeToHost.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        computeToHost.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        computeToHost.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        computeToHost.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
        computeToHost.buffer = output.buffer;
        computeToHost.offset = 0;
        computeToHost.size = VkDeviceSize(count) * 5 * sizeof(double);
        vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_HOST_BIT, 0, 0, nullptr, 1, &computeToHost, 0, nullptr);
        vk_check(vkEndCommandBuffer(commandBuffer), "vkEndCommandBuffer");
        vk_check(vkResetFences(device, 1, &fence), "vkResetFences");
        VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        submit.commandBufferCount = 1;
        submit.pCommandBuffers = &commandBuffer;
        vk_check(vkQueueSubmit(queue, 1, &submit, fence), "vkQueueSubmit");
        vk_check(vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX), "vkWaitForFences");
        std::memcpy(facets, output.mapped, std::size_t(count) * 5 * sizeof(double));
    }
};
}

NKVT_API int nk_ground_vk_tin_create(const char* shader_path, void** handle, char* message, int capacity) noexcept {
    if (handle) *handle = nullptr;
    try {
        if (!shader_path || !handle) throw std::runtime_error("Missing Vulkan TIN shader path/handle");
        auto context = std::make_unique<Compute>(shader_path);
        put_message(message, capacity, std::string("Vulkan TIN facet shader ready: ") + context->deviceName);
        *handle = context.release();
        return 0;
    } catch (const std::exception& ex) {
        put_message(message, capacity, ex.what());
        return -1;
    } catch (...) {
        put_message(message, capacity, "Unknown Vulkan TIN initialization failure");
        return -2;
    }
}
NKVT_API int nk_ground_vk_tin_build(void* handle, const double* packed_vertices, std::uint32_t count,
                                    double* output_facets, char* message, int capacity) noexcept {
    try {
        if (!handle) throw std::runtime_error("Vulkan TIN context unavailable");
        static_cast<Compute*>(handle)->build_facets(packed_vertices, count, output_facets);
        put_message(message, capacity, "OK");
        return 0;
    } catch (const std::exception& ex) {
        put_message(message, capacity, ex.what());
        return -1;
    } catch (...) {
        put_message(message, capacity, "Unknown Vulkan TIN dispatch failure");
        return -2;
    }
}
NKVT_API void nk_ground_vk_tin_release(void* handle) noexcept {
    delete static_cast<Compute*>(handle);
}
NKVT_API std::uint32_t nk_ground_vk_tin_max_batch() noexcept { return kCapacity; }
