#include "naksha/core/vulkan/VulkanInstance.h"
#include <map>
#include <mutex>

namespace naksha {
namespace vulkan {
namespace {
struct SharedInstance {
    VkInstance instance = VK_NULL_HANDLE;
    VkDebugUtilsMessengerEXT messenger = VK_NULL_HANDLE;
    unsigned references = 0;
};
std::mutex instanceMutex;
std::map<std::string, SharedInstance> sharedInstances;
std::string InstanceKey(const VulkanInstanceConfig& config) {
    std::string key = config.enableValidation ? "validation" : "production";
    for (const auto& extension : config.requiredExtensions) key += "|ext:" + extension;
    for (const auto& layer : config.requiredLayers) key += "|layer:" + layer;
    return key;
}
}

VulkanInstance::~VulkanInstance() { Shutdown(); }

bool VulkanInstance::Initialize(const VulkanInstanceConfig& config) {
    std::lock_guard<std::mutex> lock(instanceMutex);
    if (instance_) return true;
    const auto key = InstanceKey(config);
    auto found = sharedInstances.find(key);
    if (found != sharedInstances.end()) {
        instance_ = found->second.instance;
        debugMessenger_ = found->second.messenger;
        validationEnabled_ = config.enableValidation;
        ++found->second.references;
        volkLoadInstance(instance_);
        return true;
    }
    if (volkInitialize() != VK_SUCCESS) {
        return false;
    }

    validationEnabled_ = config.enableValidation;

    VkApplicationInfo appInfo{};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = config.appName.c_str();
    appInfo.applicationVersion = config.appVersion;
    appInfo.pEngineName = "NakshaPointEngine";
    appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.apiVersion = VK_API_VERSION_1_3;

    std::vector<const char*> extensions;
    for (auto& ext : config.requiredExtensions) extensions.push_back(ext.c_str());
    std::vector<const char*> layers;
    for (auto& layer : config.requiredLayers) layers.push_back(layer.c_str());

    if (validationEnabled_) {
        extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
        layers.push_back("VK_LAYER_KHRONOS_validation");
    }

    VkInstanceCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    createInfo.pApplicationInfo = &appInfo;
    createInfo.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
    createInfo.ppEnabledExtensionNames = extensions.data();
    createInfo.enabledLayerCount = static_cast<uint32_t>(layers.size());
    createInfo.ppEnabledLayerNames = layers.data();

    if (vkCreateInstance(&createInfo, nullptr, &instance_) != VK_SUCCESS) {
        return false;
    }
    volkLoadInstance(instance_);

    if (validationEnabled_) {
        auto createDebugFunc = (PFN_vkCreateDebugUtilsMessengerEXT)
            vkGetInstanceProcAddr(instance_, "vkCreateDebugUtilsMessengerEXT");
        if (createDebugFunc) {
            VkDebugUtilsMessengerCreateInfoEXT debugInfo{};
            debugInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
            debugInfo.messageSeverity =
                VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
                VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
            debugInfo.messageType =
                VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
            debugInfo.pfnUserCallback = DebugCallback;
            createDebugFunc(instance_, &debugInfo, nullptr, &debugMessenger_);
        }
    }
    sharedInstances.emplace(key, SharedInstance{instance_, debugMessenger_, 1});
    return true;
}

void VulkanInstance::Shutdown() {
    std::lock_guard<std::mutex> lock(instanceMutex);
    if (!instance_) return;
    for (auto it = sharedInstances.begin(); it != sharedInstances.end(); ++it) {
        if (it->second.instance != instance_) continue;
        if (--it->second.references != 0) {
            instance_ = VK_NULL_HANDLE;
            debugMessenger_ = VK_NULL_HANDLE;
            return;
        }
        sharedInstances.erase(it);
        break;
    }
    if (debugMessenger_) {
        auto func = (PFN_vkDestroyDebugUtilsMessengerEXT)
            vkGetInstanceProcAddr(instance_, "vkDestroyDebugUtilsMessengerEXT");
        if (func) func(instance_, debugMessenger_, nullptr);
        debugMessenger_ = VK_NULL_HANDLE;
    }
    if (instance_) {
        vkDestroyInstance(instance_, nullptr);
        instance_ = VK_NULL_HANDLE;
    }
}

VKAPI_ATTR VkBool32 VKAPI_CALL VulkanInstance::DebugCallback(
    VkDebugUtilsMessageSeverityFlagBitsEXT severity,
    VkDebugUtilsMessageTypeFlagsEXT type,
    const VkDebugUtilsMessengerCallbackDataEXT* callbackData,
    void* userData) {
    (void)type; (void)userData;
    if (severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT)
        s_errorCount_++;
    else if (severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT)
        s_warningCount_++;
    if (severity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) {
        // Tag the severity: the layer does NOT put "Validation Error: ..." in
        // pMessage for every VUID, so a log parser cannot otherwise tell an
        // error from a warning. Tests count "[Vulkan Validation][ERROR] lines".
        const char* tag = (severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT)
                              ? "[ERROR]"
                              : "[WARN]";
        fprintf(stderr, "[Vulkan Validation]%s %s\n", tag, callbackData->pMessage);
    }
    return VK_FALSE;
}

int VulkanInstance::GetValidationErrorCount() { return s_errorCount_.load(); }
int VulkanInstance::GetValidationWarningCount() { return s_warningCount_.load(); }
void VulkanInstance::ResetValidationCounters() { s_errorCount_ = 0; s_warningCount_ = 0; }

} // namespace vulkan
} // namespace naksha
