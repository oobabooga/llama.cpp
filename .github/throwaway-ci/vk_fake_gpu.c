#define _GNU_SOURCE

#include <dlfcn.h>
#include <string.h>
#include <vulkan/vulkan.h>

static void * vulkan_loader(void) {
    static void * handle;
    if (!handle) {
        handle = dlopen("libvulkan.so.1", RTLD_NOW | RTLD_LOCAL);
    }
    return handle;
}

static PFN_vkGetInstanceProcAddr next_gipa(void) {
    static PFN_vkGetInstanceProcAddr fn;
    if (!fn) {
        fn = (PFN_vkGetInstanceProcAddr) dlsym(vulkan_loader(), "vkGetInstanceProcAddr");
    }
    return fn;
}

static void mark_cpu_as_integrated(VkPhysicalDeviceProperties * properties) {
    if (properties->deviceType == VK_PHYSICAL_DEVICE_TYPE_CPU) {
        properties->deviceType = VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU;
    }
}

VKAPI_ATTR void VKAPI_CALL vkGetPhysicalDeviceProperties(
        VkPhysicalDevice physical_device,
        VkPhysicalDeviceProperties * properties) {
    PFN_vkGetPhysicalDeviceProperties fn = (PFN_vkGetPhysicalDeviceProperties)
        dlsym(vulkan_loader(), "vkGetPhysicalDeviceProperties");
    fn(physical_device, properties);
    mark_cpu_as_integrated(properties);
}

VKAPI_ATTR void VKAPI_CALL vkGetPhysicalDeviceProperties2(
        VkPhysicalDevice physical_device,
        VkPhysicalDeviceProperties2 * properties) {
    PFN_vkGetPhysicalDeviceProperties2 fn = (PFN_vkGetPhysicalDeviceProperties2)
        dlsym(vulkan_loader(), "vkGetPhysicalDeviceProperties2");
    fn(physical_device, properties);
    mark_cpu_as_integrated(&properties->properties);
}

VKAPI_ATTR void VKAPI_CALL vkGetPhysicalDeviceProperties2KHR(
        VkPhysicalDevice physical_device,
        VkPhysicalDeviceProperties2 * properties) {
    PFN_vkGetPhysicalDeviceProperties2KHR fn = (PFN_vkGetPhysicalDeviceProperties2KHR)
        dlsym(vulkan_loader(), "vkGetPhysicalDeviceProperties2KHR");
    fn(physical_device, properties);
    mark_cpu_as_integrated(&properties->properties);
}

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL vkGetInstanceProcAddr(
        VkInstance instance,
        const char * name) {
    if (strcmp(name, "vkGetPhysicalDeviceProperties") == 0) {
        return (PFN_vkVoidFunction) vkGetPhysicalDeviceProperties;
    }
    if (strcmp(name, "vkGetPhysicalDeviceProperties2") == 0) {
        return (PFN_vkVoidFunction) vkGetPhysicalDeviceProperties2;
    }
    if (strcmp(name, "vkGetPhysicalDeviceProperties2KHR") == 0) {
        return (PFN_vkVoidFunction) vkGetPhysicalDeviceProperties2KHR;
    }
    return next_gipa()(instance, name);
}
