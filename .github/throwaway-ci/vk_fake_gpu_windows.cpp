#include <windows.h>

#include <cstdint>
#include <cstring>

#define VKAPI_CALL __stdcall

using VkHandle = void *;
using VkGetInstanceProcAddr = void * (VKAPI_CALL *)(VkHandle, const char *);

static HMODULE real_loader() {
    static HMODULE module = LoadLibraryW(L"vulkan-real.dll");
    if (!module) {
        TerminateProcess(GetCurrentProcess(), 126);
    }
    return module;
}

template<typename T>
static T real_symbol(const char * name) {
    T fn = reinterpret_cast<T>(GetProcAddress(real_loader(), name));
    if (!fn) {
        TerminateProcess(GetCurrentProcess(), 127);
    }
    return fn;
}

static void mark_cpu_as_integrated(void * properties, size_t device_type_offset) {
    auto * device_type = reinterpret_cast<std::uint32_t *>(
        static_cast<unsigned char *>(properties) + device_type_offset);
    if (*device_type == 4) {
        *device_type = 1;
    }
}

extern "C" __declspec(dllexport) void VKAPI_CALL vkGetPhysicalDeviceProperties(
        VkHandle physical_device,
        void * properties) {
    using Function = void (VKAPI_CALL *)(VkHandle, void *);
    real_symbol<Function>("vkGetPhysicalDeviceProperties")(physical_device, properties);
    mark_cpu_as_integrated(properties, 16);
}

extern "C" __declspec(dllexport) void VKAPI_CALL vkGetPhysicalDeviceProperties2(
        VkHandle physical_device,
        void * properties) {
    using Function = void (VKAPI_CALL *)(VkHandle, void *);
    real_symbol<Function>("vkGetPhysicalDeviceProperties2")(physical_device, properties);
    mark_cpu_as_integrated(properties, 32);
}

extern "C" __declspec(dllexport) void VKAPI_CALL vkGetPhysicalDeviceProperties2KHR(
        VkHandle physical_device,
        void * properties) {
    using Function = void (VKAPI_CALL *)(VkHandle, void *);
    real_symbol<Function>("vkGetPhysicalDeviceProperties2KHR")(physical_device, properties);
    mark_cpu_as_integrated(properties, 32);
}

extern "C" __declspec(dllexport) void * VKAPI_CALL vkGetInstanceProcAddr(
        VkHandle instance,
        const char * name) {
    if (std::strcmp(name, "vkGetPhysicalDeviceProperties") == 0) {
        return reinterpret_cast<void *>(vkGetPhysicalDeviceProperties);
    }
    if (std::strcmp(name, "vkGetPhysicalDeviceProperties2") == 0) {
        return reinterpret_cast<void *>(vkGetPhysicalDeviceProperties2);
    }
    if (std::strcmp(name, "vkGetPhysicalDeviceProperties2KHR") == 0) {
        return reinterpret_cast<void *>(vkGetPhysicalDeviceProperties2KHR);
    }
    return real_symbol<VkGetInstanceProcAddr>("vkGetInstanceProcAddr")(instance, name);
}

extern "C" __declspec(dllexport) void VKAPI_CALL vkCmdCopyBuffer(
        VkHandle command_buffer,
        VkHandle source,
        VkHandle destination,
        std::uint32_t region_count,
        const void * regions) {
    using Function = void (VKAPI_CALL *)(
        VkHandle, VkHandle, VkHandle, std::uint32_t, const void *);
    real_symbol<Function>("vkCmdCopyBuffer")(
        command_buffer, source, destination, region_count, regions);
}

extern "C" __declspec(dllexport) void VKAPI_CALL vkGetPhysicalDeviceFeatures2(
        VkHandle physical_device,
        void * features) {
    using Function = void (VKAPI_CALL *)(VkHandle, void *);
    real_symbol<Function>("vkGetPhysicalDeviceFeatures2")(physical_device, features);
}
