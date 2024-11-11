/*
 * Based on https://github.com/tgfrerer/island/tree/wip/apps/examples/bitonic_merge_sort_example
 */

#include <android/asset_manager.h>
#include <android/log.h>
#include <android/native_window.h>
#include <android/native_window_jni.h>
#include <assert.h>
#include <vulkan/vulkan.h>

#include <array>
#include <fstream>
#include <map>
#include <optional>
#include <set>
#include <sstream>
#include <string>
#include <vector>
#include <random>

namespace vkt {
#define LOG_TAG "hellovkjni"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)
#define VK_CHECK(x)                           \
  do {                                        \
    VkResult err = x;                         \
    if (err) {                                \
      LOGE("Detected Vulkan error: %d", err); \
      abort();                                \
    }                                         \
  } while (0)

struct QueueFamilyIndices {
    std::optional<uint32_t> computeFamily;
    bool isComplete() {
        return computeFamily.has_value();
    }
};

struct Parameters {
    enum eAlgorithmVariant : uint32_t {
        eLocalBitonicMergeSort = 0,
        eLocalDisperse         = 1,
        eBigFlip               = 2,
        eBigDisperse           = 3,
    };
    uint32_t          h;
    eAlgorithmVariant algorithm;
};

struct ANativeWindowDeleter {
    void operator()(ANativeWindow *window) { ANativeWindow_release(window); }
};

std::vector<uint8_t> LoadBinaryFileToVector(const char *file_path, AAssetManager *assetManager) {
    std::vector<uint8_t> file_content;
    assert(assetManager);
    AAsset *file = AAssetManager_open(assetManager, file_path, AASSET_MODE_BUFFER);
    size_t file_length = AAsset_getLength(file);

    file_content.resize(file_length);

    AAsset_read(file, file_content.data(), file_length);
    AAsset_close(file);
    return file_content;
}

const char *toStringMessageSeverity(VkDebugUtilsMessageSeverityFlagBitsEXT s) {
    switch (s) {
        case VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT:
            return "VERBOSE";
        case VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT:
            return "ERROR";
        case VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT:
            return "WARNING";
        case VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT:
            return "INFO";
        default:
            return "UNKNOWN";
    }
}

const char *toStringMessageType(VkDebugUtilsMessageTypeFlagsEXT s) {
    if (s == (VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
              VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
              VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT))
        return "General | Validation | Performance";
    if (s == (VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
              VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT))
        return "Validation | Performance";
    if (s == (VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
              VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT))
        return "General | Performance";
    if (s == (VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT))
        return "Performance";
    if (s == (VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
              VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT))
        return "General | Validation";
    if (s == VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT) return "Validation";
    if (s == VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT) return "General";
    return "Unknown";
}

static VKAPI_ATTR VkBool32 VKAPI_CALL
debugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
              VkDebugUtilsMessageTypeFlagsEXT messageType,
              const VkDebugUtilsMessengerCallbackDataEXT *pCallbackData,
              void * /* pUserData */) {
    auto ms = toStringMessageSeverity(messageSeverity);
    auto mt = toStringMessageType(messageType);
    printf("[%s: %s]\n%s\n", ms, mt, pCallbackData->pMessage);
    return VK_FALSE;
}

static void populateDebugMessengerCreateInfo(VkDebugUtilsMessengerCreateInfoEXT &createInfo) {
    createInfo = {
            VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT,
            nullptr,
            0,
            VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT |
            VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
            VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT,
            VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
            VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
            VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT,
            debugCallback
    };
}

static VkResult CreateDebugUtilsMessengerEXT(
        VkInstance instance, const VkDebugUtilsMessengerCreateInfoEXT *pCreateInfo,
        const VkAllocationCallbacks *pAllocator,
        VkDebugUtilsMessengerEXT *pDebugMessenger) {
    auto func = (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(
            instance, "vkCreateDebugUtilsMessengerEXT");
    if (func != nullptr) {
        return func(instance, pCreateInfo, pAllocator, pDebugMessenger);
    } else {
        return VK_ERROR_EXTENSION_NOT_PRESENT;
    }
}

static void DestroyDebugUtilsMessengerEXT(
        VkInstance instance, VkDebugUtilsMessengerEXT debugMessenger,
        const VkAllocationCallbacks *pAllocator) {
    auto func = (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(
            instance, "vkDestroyDebugUtilsMessengerEXT");
    if (func != nullptr) {
        func(instance, debugMessenger, pAllocator);
    }
}

#define BUFFER_LENGTH (128 * 1024)
#define BUFFER_SIZE (BUFFER_LENGTH * sizeof(int32_t))

class HelloVK {
public:
    void initVulkanCompute();
    void initSortData();
    void checkDataSorted();
    void compute();
    void cleanupCompute();
    void reset(ANativeWindow *newWindow, AAssetManager *newManager);
    bool initialized = false;

private:
    void createInstance();
    void setupDebugMessenger();
    void pickPhysicalDevice();
    void createLogicalDeviceAndQueue();
    void createDescriptorSetLayout();
    void createCommandPool();
    void createCommandBuffer();
    void createSyncObjects();
    static QueueFamilyIndices findQueueFamilies(VkPhysicalDevice device);
    bool checkDeviceExtensionSupport(VkPhysicalDevice device);
    bool isDeviceSuitable(VkPhysicalDevice device);
    bool checkValidationLayerSupport();
    static std::vector<const char *> getRequiredExtensions(bool enableValidation);
    VkShaderModule createShaderModule(const std::vector<uint8_t> &code);
    void updateUniformBuffer(Parameters::eAlgorithmVariant algorithm, uint32_t h);
    uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties, VkDeviceSize memorySize);
    void createBuffer(VkDeviceSize size, VkBufferUsageFlags usage,
                      VkMemoryPropertyFlags properties, VkBuffer &buffer,
                      VkDeviceMemory &bufferMemory);
    void copyBuffer(VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size);
    void createStorageBuffers();
    void createUniformBuffers();
    void createComputePipeline();
    void createDescriptorPool();
    void createDescriptorSets();

    void dispatch(uint32_t workgroupCount);
    void localBitonicMergeSort(uint32_t h, uint32_t workgroupCount);
    void bigFlip(uint32_t h, uint32_t workgroupCount);
    void localDisperse(uint32_t h, uint32_t workgroupCount);
    void bigDisperse(uint32_t h, uint32_t workgroupCount);

    /*
     * In order to enable validation layer toggle this to true and
     * follow the README.md instructions concerning the validation
     * layers. You will be required to add separate vulkan validation
     * '*.so' files in order to enable this.
     *
     * The validation layers are not shipped with the APK as they are sizeable.
     */
    bool enableValidationLayers = false;

    const std::vector<const char *> validationLayers = {"VK_LAYER_KHRONOS_validation"};
    const std::vector<const char *> deviceExtensions = {};
    std::unique_ptr<ANativeWindow, ANativeWindowDeleter> window;
    AAssetManager *assetManager;
    VkInstance instance;
    VkDebugUtilsMessengerEXT debugMessenger;

    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    VkDevice device;
    VkCommandPool commandPool;
    VkCommandBuffer commandBuffer;
    std::vector<VkCommandBuffer> commandBuffers;

    uint32_t queueFamilyIndex;
    VkQueue computeQueue;

    VkDescriptorSetLayout descriptorSetLayout;
    VkPipelineLayout pipelineLayout;
    VkPipeline computePipeline;

    std::vector<VkBuffer> storageBuffers;
    std::vector<VkDeviceMemory> storageBuffersMemory;
    std::vector<VkBuffer> uniformBuffers;
    std::vector<VkDeviceMemory> uniformBuffersMemory;
    std::vector<void*> uniformBuffersMapped;

    VkDescriptorPool descriptorPool;
    std::vector<VkDescriptorSet> descriptorSets;
    VkDescriptorSet descriptorSet;
    VkFence computeFence;

    int32_t *sortData;
    int32_t sortDataOriginal[BUFFER_LENGTH];
    uint32_t workGroupSize;
};


void HelloVK::initVulkanCompute() {
    createInstance();
    pickPhysicalDevice();
    createLogicalDeviceAndQueue();
    setupDebugMessenger();
    createDescriptorSetLayout();
    createStorageBuffers();
    createUniformBuffers();
    createDescriptorPool();
    createDescriptorSets();
    createComputePipeline();
    createCommandPool();
    createCommandBuffer();
    createSyncObjects();
    initialized = true;
}

/*
 *	Create a buffer with specified usage and memory properties
 *  Upon creation, these buffers will list memory requirements which need to be
 *  satisfied by the device in use in order to be created.
 */
void HelloVK::createBuffer(VkDeviceSize size, VkBufferUsageFlags usage,
                           VkMemoryPropertyFlags properties, VkBuffer &buffer,
                           VkDeviceMemory &bufferMemory) {
    const VkBufferCreateInfo bufferCreateInfo = {
            VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            nullptr,
            0,
            size,
            usage,
            VK_SHARING_MODE_EXCLUSIVE,
            1,
            &queueFamilyIndex
    };

    VK_CHECK(vkCreateBuffer(device, &bufferCreateInfo, nullptr, &buffer));

    VkMemoryRequirements memRequirements;
    vkGetBufferMemoryRequirements(device, buffer, &memRequirements);
    uint32_t memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties, size);

    const VkMemoryAllocateInfo memoryAllocateInfo = {
            VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            nullptr,
            memRequirements.size,
            memoryTypeIndex
    };

    VK_CHECK(vkAllocateMemory(device, &memoryAllocateInfo, nullptr, &bufferMemory));

    vkBindBufferMemory(device, buffer, bufferMemory, 0);
}

/*
 * Finds the index of the memory heap which matches a particular buffer's memory
 * requirements. Vulkan manages these requirements as a bitset, in this case
 * expressed through a uint32_t.
 */
uint32_t HelloVK::findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties, VkDeviceSize memorySize) {
    VkPhysicalDeviceMemoryProperties memoryProperties;
    vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memoryProperties);

    // set memoryTypeIndex to an invalid entry in the properties.memoryTypes array
    uint32_t memoryTypeIndex = VK_MAX_MEMORY_TYPES;

    for (uint32_t i = 0; i < memoryProperties.memoryTypeCount; i++) {
        if ((typeFilter & (1 << i)) &&
            (memoryProperties.memoryTypes[i].propertyFlags & properties) == properties &&
            (memorySize < memoryProperties.memoryHeaps[memoryProperties.memoryTypes[i].heapIndex].size)) {
            return i;
        }
    }

    return memoryTypeIndex;
}

void HelloVK::copyBuffer(VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size) {
    VkCommandBufferBeginInfo beginInfo {
            VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            nullptr,
            VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT
    };

    vkBeginCommandBuffer(commandBuffer, &beginInfo);
    VkBufferCopy copyRegion {
        0,
        0,
        size
    };
    vkCmdCopyBuffer(commandBuffer, srcBuffer, dstBuffer, 1, &copyRegion);
    vkEndCommandBuffer(commandBuffer);

    VkSubmitInfo submitInfo {
            VK_STRUCTURE_TYPE_SUBMIT_INFO,
            nullptr,
            0,
            nullptr,
            nullptr,
            1,
            &commandBuffer
    };

    vkQueueSubmit(computeQueue, 1, &submitInfo, VK_NULL_HANDLE);
    vkQueueWaitIdle(computeQueue);

    vkResetCommandBuffer(commandBuffer, 0);
}

void HelloVK::createStorageBuffers() {
    storageBuffers.resize(1);
    storageBuffersMemory.resize(1);

    createBuffer(BUFFER_SIZE, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                 VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                 storageBuffers[0], storageBuffersMemory[0]);
}

void HelloVK::createUniformBuffers() {
    VkDeviceSize bufferSize = sizeof(Parameters);
    uniformBuffers.resize(1);
    uniformBuffersMemory.resize(1);
    uniformBuffersMapped.resize(1);

    createBuffer(bufferSize, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                 VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                 uniformBuffers[0], uniformBuffersMemory[0]);
    vkMapMemory(device, uniformBuffersMemory[0], 0, bufferSize, 0, &uniformBuffersMapped[0]);
}

void HelloVK::createDescriptorSetLayout() {
    VkDescriptorSetLayoutBinding descriptorSetLayoutBindings[2] = {
            {
                    0,
                    VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    1,
                    VK_SHADER_STAGE_COMPUTE_BIT,
                    nullptr
            },
            {
                    1,
                    VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                    1,
                    VK_SHADER_STAGE_COMPUTE_BIT,
                    nullptr
            }
    };

    VkDescriptorSetLayoutCreateInfo descriptorSetLayoutCreateInfo = {
            VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            nullptr,
            0,
            2,
            descriptorSetLayoutBindings
    };
    VK_CHECK(vkCreateDescriptorSetLayout(device, &descriptorSetLayoutCreateInfo, nullptr, &descriptorSetLayout));
}

void HelloVK::reset(ANativeWindow *newWindow, AAssetManager *newManager) {
  window.reset(newWindow);
  assetManager = newManager;
}

void HelloVK::initSortData() {
    // Create a staging buffer used to upload data to the gpu
    VkBuffer stagingBuffer;
    VkDeviceMemory stagingBufferMemory;
    createBuffer(BUFFER_SIZE, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);

    int32_t* data;
    vkMapMemory(device, stagingBufferMemory, 0, BUFFER_SIZE, 0, reinterpret_cast<void **>(&data));
    std::uniform_int_distribution<size_t> uniform_dist (INT32_MIN, INT32_MAX);
    std::default_random_engine engine(std::chrono::system_clock::now().time_since_epoch().count());
    for (uint32_t k = 0; k < BUFFER_LENGTH; k++) {
        data[k] = static_cast<int32_t>(uniform_dist(engine));
        sortDataOriginal[k] = data[k];
    }
    vkUnmapMemory(device, stagingBufferMemory);

    copyBuffer(stagingBuffer, storageBuffers[0], BUFFER_SIZE);

    vkDestroyBuffer(device, stagingBuffer, nullptr);
    vkFreeMemory(device, stagingBufferMemory, nullptr);
}

void HelloVK::checkDataSorted() {
    // Create a staging buffer used to download data from the gpu
    VkBuffer stagingBuffer;
    VkDeviceMemory stagingBufferMemory;
    createBuffer(BUFFER_SIZE, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);

    int32_t* data;
    vkMapMemory(device, stagingBufferMemory, 0, BUFFER_SIZE, 0, reinterpret_cast<void **>(&data));

    copyBuffer(storageBuffers[0], stagingBuffer, BUFFER_SIZE);
    bool error = false;
    for (uint32_t k = 0; k < BUFFER_LENGTH - 1; k++) {
        if (data[k + 1] > data[k]) {
            LOGE("Error occurred in compute shader");
            error = true;
            break;
        }
    }
    if (!error) {
        LOGI("Sort successful");
    }
    vkUnmapMemory(device, stagingBufferMemory);
    vkDestroyBuffer(device, stagingBuffer, nullptr);
    vkFreeMemory(device, stagingBufferMemory, nullptr);
}

void HelloVK::dispatch(uint32_t workgroupCount) {
    vkResetFences(device, 1, &computeFence);

    VkCommandBufferBeginInfo commandBufferBeginInfo = {
            VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            nullptr,
            VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
            nullptr
    };
    VK_CHECK(vkBeginCommandBuffer(commandBuffer, &commandBufferBeginInfo));
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, computePipeline);
    vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                            pipelineLayout, 0, 1, &descriptorSet, 0, nullptr);
    vkCmdDispatch(commandBuffer, workgroupCount, 1, 1);
    VkBufferMemoryBarrier memoryBufferBarrier {
            VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
            nullptr,
            0,
            VK_ACCESS_SHADER_READ_BIT,
            queueFamilyIndex,
            queueFamilyIndex,
            storageBuffers[0],
            0,
            VK_WHOLE_SIZE
    };
    vkCmdPipelineBarrier(
            commandBuffer,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0,
            0, nullptr,
            1, &memoryBufferBarrier,
            0, nullptr);
    VK_CHECK(vkEndCommandBuffer(commandBuffer));

    VkSubmitInfo submitInfo = {
            VK_STRUCTURE_TYPE_SUBMIT_INFO,
            nullptr,
            0,
            nullptr,
            nullptr,
            1,
            &commandBuffer,
            0,
            nullptr
    };
    VK_CHECK(vkQueueSubmit(computeQueue, 1, &submitInfo, computeFence));
    vkWaitForFences(device, 1, &computeFence, VK_TRUE, UINT64_MAX);

    vkResetCommandBuffer(commandBuffer, 0);
};

void HelloVK::localBitonicMergeSort(uint32_t h, uint32_t workgroupCount) {
    updateUniformBuffer(Parameters::eAlgorithmVariant::eLocalBitonicMergeSort, h);
    dispatch(workgroupCount);
};

void HelloVK::bigFlip(uint32_t h, uint32_t workgroupCount) {
    updateUniformBuffer(Parameters::eAlgorithmVariant::eBigFlip, h);
    dispatch(workgroupCount);
};

void HelloVK::localDisperse(uint32_t h, uint32_t workgroupCount) {
    updateUniformBuffer(Parameters::eAlgorithmVariant::eLocalDisperse, h);
    dispatch(workgroupCount);
};

void HelloVK::bigDisperse(uint32_t h, uint32_t workgroupCount) {
    updateUniformBuffer(Parameters::eAlgorithmVariant::eBigDisperse, h);
    dispatch(workgroupCount);
};

// layout(local_size_x_id = 1) in;
#define LOCAL_SIZE_X_CONST_ID     1

void HelloVK::compute() {
    initSortData();
    size_t n = BUFFER_LENGTH;

    uint32_t max_workgroup_size = workGroupSize;
    uint32_t workgroup_size_x = 1;

// Adjust workgroup_size_x to get as close to max_workgroup_size as possible.
    if ( n < max_workgroup_size * 2 ) {
        workgroup_size_x = n / 2;
    } else {
        workgroup_size_x = max_workgroup_size;
    }
    const uint32_t workgroupCount = n / (workgroup_size_x * 2 );

    uint32_t h = workgroup_size_x * 2;
    assert( h <= n );
    assert( h % 2 == 0 );

    localBitonicMergeSort(h, workgroupCount);
    // we must now double h, as this happens before every flip
    h *= 2;

    for ( ; h <= n; h *= 2 ) {
        bigFlip(h, workgroupCount);

        for (uint32_t hh = h / 2; hh > 1; hh /= 2) {

            if (hh <= workgroup_size_x * 2) {
                // We can fit all elements for a disperse operation into continuous shader
                // workgroup local memory, which means we can complete the rest of the
                // cascade using a single shader invocation.
                localDisperse(hh, workgroupCount);
                break;
            } else {
                bigDisperse(hh, workgroupCount );
            }
        }
    }
    checkDataSorted();
}

void HelloVK::createDescriptorPool() {
    std::vector<VkDescriptorPoolSize> poolSizes {
            {
                    VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    1
            },
            {
                    VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                    1
            }
    };

    VkDescriptorPoolCreateInfo descriptorPoolCreateInfo = {
            VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            nullptr,
            0,
            1,
            static_cast<uint32_t>(poolSizes.size()),
            poolSizes.data()
    };

    VK_CHECK(vkCreateDescriptorPool(device, &descriptorPoolCreateInfo, nullptr, &descriptorPool));
}

void HelloVK::createDescriptorSets() {
    VkDescriptorSetAllocateInfo descriptorSetAllocateInfo = {
            VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            nullptr,
            descriptorPool,
            1,
            &descriptorSetLayout
    };

    VK_CHECK(vkAllocateDescriptorSets(device, &descriptorSetAllocateInfo, &descriptorSet));

    VkDescriptorBufferInfo storageBufferInfo = {
            storageBuffers[0],
            0,
            VK_WHOLE_SIZE
    };

    VkDescriptorBufferInfo uniformBufferInfo{
            uniformBuffers[0],
            0,
            sizeof(Parameters)
    };

    VkWriteDescriptorSet writeDescriptorSet[2] = {
            {
                    VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                    nullptr,
                    descriptorSet,
                    0,
                    0,
                    1,
                    VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    nullptr,
                    &storageBufferInfo,
                    nullptr
            },
            {
                    VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                    nullptr,
                    descriptorSet,
                    1,
                    0,
                    1,
                    VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                    nullptr,
                    &uniformBufferInfo,
                    nullptr
            }
    };
    vkUpdateDescriptorSets(device, 2, writeDescriptorSet, 0, nullptr);
}

void HelloVK::cleanupCompute() {
    vkDeviceWaitIdle(device);

    vkUnmapMemory(device, uniformBuffersMemory[0]);

    vkDestroyBuffer(device, uniformBuffers[0], nullptr);
    vkFreeMemory(device, uniformBuffersMemory[0], nullptr);

    vkDestroyDescriptorPool(device, descriptorPool, nullptr);
    vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);

    vkDestroyBuffer(device, storageBuffers[0], nullptr);
    vkFreeMemory(device, storageBuffersMemory[0], nullptr);

    vkDestroyFence(device, computeFence, nullptr);

    vkDestroyCommandPool(device, commandPool, nullptr);
    vkDestroyPipeline(device, computePipeline, nullptr);
    vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
    vkDestroyDevice(device, nullptr);
    if (enableValidationLayers) {
        DestroyDebugUtilsMessengerEXT(instance, debugMessenger, nullptr);
    }
    vkDestroyInstance(instance, nullptr);
    initialized = false;
    LOGI("Cleanup completed successfully");
}

void HelloVK::setupDebugMessenger() {
    if (!enableValidationLayers) {
        return;
    }

    VkDebugUtilsMessengerCreateInfoEXT createInfo{};
    populateDebugMessengerCreateInfo(createInfo);

    VK_CHECK(CreateDebugUtilsMessengerEXT(instance, &createInfo, nullptr,
                                          &debugMessenger));
}

bool HelloVK::checkValidationLayerSupport() {
    uint32_t layerCount;
    vkEnumerateInstanceLayerProperties(&layerCount, nullptr);

    std::vector<VkLayerProperties> availableLayers(layerCount);
    vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());

    for (const char *layerName : validationLayers) {
        bool layerFound = false;
        for (const auto &layerProperties : availableLayers) {
            if (strcmp(layerName, layerProperties.layerName) == 0) {
                layerFound = true;
                break;
            }
        }

        if (!layerFound) {
            return false;
        }
    }
    return true;
}

std::vector<const char *> HelloVK::getRequiredExtensions(bool enableValidationLayers) {
    std::vector<const char *> extensions;
    if (enableValidationLayers) {
        extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    }
    return extensions;
}

void HelloVK::createInstance() {
    assert(!enableValidationLayers || checkValidationLayerSupport());
    auto requiredExtensions = getRequiredExtensions(enableValidationLayers);
    const VkApplicationInfo applicationInfo = {
            VK_STRUCTURE_TYPE_APPLICATION_INFO,
            nullptr,
            "VKComputeSample",
            0,
            "",
            0,
            VK_MAKE_VERSION(1, 0, 9)
    };

    VkInstanceCreateInfo createInfo = {
            VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
            nullptr,
            0,
            &applicationInfo,
            0,
            nullptr,
            0,
            nullptr
    };

    VK_CHECK(vkCreateInstance(&createInfo, nullptr, &instance));
    if (enableValidationLayers) {
        VkDebugUtilsMessengerCreateInfoEXT debugCreateInfo{};
        createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
        createInfo.ppEnabledLayerNames = validationLayers.data();
        populateDebugMessengerCreateInfo(debugCreateInfo);
        createInfo.pNext = (VkDebugUtilsMessengerCreateInfoEXT *)&debugCreateInfo;
    } else {
        createInfo.enabledLayerCount = 0;
        createInfo.pNext = nullptr;
    }
    VK_CHECK(vkCreateInstance(&createInfo, nullptr, &instance));

    uint32_t extensionCount = 0;
    vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, nullptr);
    std::vector<VkExtensionProperties> extensions(extensionCount);
    vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount,extensions.data());
    LOGI("available extensions");
    for (const auto &extension : extensions) {
        LOGI("\t %s", extension.extensionName);
    }
}

// BEGIN DEVICE SUITABILITY
// Functions to find a suitable physical device to execute Vulkan commands.

QueueFamilyIndices HelloVK::findQueueFamilies(VkPhysicalDevice device) {
    QueueFamilyIndices indices;

    uint32_t queueFamilyCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, nullptr);

    std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
    vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount,
                                             queueFamilies.data());

    int i = 0;
    for (const auto &queueFamily : queueFamilies) {
        if (queueFamily.queueFlags & VK_QUEUE_COMPUTE_BIT) {
            indices.computeFamily = i;
        }

        if (indices.isComplete()) {
            break;
        }

        i++;
    }
    return indices;
}

bool HelloVK::checkDeviceExtensionSupport(VkPhysicalDevice device) {
    uint32_t extensionCount;
    vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount,
                                         nullptr);

    std::vector<VkExtensionProperties> availableExtensions(extensionCount);
    vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount,
                                         availableExtensions.data());

    std::set<std::string> requiredExtensions(deviceExtensions.begin(),
                                             deviceExtensions.end());

    for (const auto &extension : availableExtensions) {
        requiredExtensions.erase(extension.extensionName);
    }

    return requiredExtensions.empty();
}

bool HelloVK::isDeviceSuitable(VkPhysicalDevice device) {
    QueueFamilyIndices indices = findQueueFamilies(device);
    bool extensionsSupported = checkDeviceExtensionSupport(device);
    return indices.isComplete() && extensionsSupported;
}

void HelloVK::pickPhysicalDevice() {
    uint32_t deviceCount = 0;
    vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);

    assert(deviceCount > 0);  // failed to find GPUs with Vulkan support!

    std::vector<VkPhysicalDevice> devices(deviceCount);
    vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());

    for (const auto &device : devices) {
        if (isDeviceSuitable(device)) {
            physicalDevice = device;
            break;
        }
    }

    assert(physicalDevice != VK_NULL_HANDLE);  // failed to find a suitable GPU!
}
// END DEVICE SUITABILITY

void HelloVK::createLogicalDeviceAndQueue() {
    QueueFamilyIndices indices = findQueueFamilies(physicalDevice);
    const float queuePrioritory = 1.0f;
    const VkDeviceQueueCreateInfo deviceQueueCreateInfo = {
            VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
            nullptr,
            0,
            indices.computeFamily.value(),
            1,
            &queuePrioritory
    };

    const VkDeviceCreateInfo deviceCreateInfo = {
            VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
            nullptr,
            0,
            1,
            &deviceQueueCreateInfo,
            0,
            nullptr,
            0,
            nullptr,
            nullptr
    };

    VK_CHECK(vkCreateDevice(physicalDevice, &deviceCreateInfo, nullptr, &device));

    queueFamilyIndex = indices.computeFamily.value();
    vkGetDeviceQueue(device, queueFamilyIndex, 0, &computeQueue);
}

void HelloVK::createComputePipeline() {
    VkPhysicalDeviceProperties properties;
    vkGetPhysicalDeviceProperties(physicalDevice, &properties);

    workGroupSize = std::min(properties.limits.maxComputeWorkGroupSize[0], properties.limits.maxComputeSharedMemorySize / 2);

    VkSpecializationMapEntry specializationMapEntry {
            LOCAL_SIZE_X_CONST_ID,
            0,
            sizeof(workGroupSize)
    };

    VkSpecializationInfo specializationInfo {
        1,
        &specializationMapEntry,
        sizeof(workGroupSize),
        &workGroupSize
    };

    auto compShaderCode = LoadBinaryFileToVector("shaders/bitonic.comp.spv", assetManager);
    VkShaderModule compShaderModule = createShaderModule(compShaderCode);

    VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo = {
            VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            nullptr,
            0,
            1,
            &descriptorSetLayout,
            0,
            nullptr
    };

    VK_CHECK(vkCreatePipelineLayout(device, &pipelineLayoutCreateInfo, nullptr, &pipelineLayout));

    VkComputePipelineCreateInfo computePipelineCreateInfo = {
            VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
            nullptr,
            0,
            {
                    VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                    nullptr,
                    0,
                    VK_SHADER_STAGE_COMPUTE_BIT,
                    compShaderModule,
                    "main",
                    &specializationInfo
            },
            pipelineLayout,
            0,
            0
    };

    VK_CHECK(vkCreateComputePipelines(device, 0, 1, &computePipelineCreateInfo, nullptr, &computePipeline));
}

void HelloVK::updateUniformBuffer(Parameters::eAlgorithmVariant algorithm, uint32_t h) {
    Parameters parameters {
            h,
            algorithm
    };
    memcpy(uniformBuffersMapped[0], &parameters, sizeof(parameters));
}

VkShaderModule HelloVK::createShaderModule(const std::vector<uint8_t> &code) {
    // Satisfies alignment requirements since the allocator in vector ensures worst case requirements
    VkShaderModuleCreateInfo createInfo = {
            VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
            nullptr,
            0,
            code.size(),
            reinterpret_cast<const uint32_t *>(code.data())
    };

    VkShaderModule shaderModule;
    VK_CHECK(vkCreateShaderModule(device, &createInfo, nullptr, &shaderModule));
    return shaderModule;
}

void HelloVK::createCommandPool() {
    VkCommandPoolCreateInfo commandPoolCreateInfo = {
            VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
            nullptr,
            VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
            queueFamilyIndex
    };
    VK_CHECK(vkCreateCommandPool(device, &commandPoolCreateInfo, nullptr, &commandPool));
}

void HelloVK::createCommandBuffer() {
    VkCommandBufferAllocateInfo commandBufferAllocateInfo = {
            VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            nullptr,
            commandPool,
            VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            1
    };

    VK_CHECK(vkAllocateCommandBuffers(device, &commandBufferAllocateInfo, &commandBuffer));
}

void HelloVK::createSyncObjects() {
    VkFenceCreateInfo fenceInfo{
            VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
            nullptr,
            VK_FENCE_CREATE_SIGNALED_BIT
    };
    vkCreateFence(device, &fenceInfo, nullptr, &computeFence);
}

}  // namespace vkt