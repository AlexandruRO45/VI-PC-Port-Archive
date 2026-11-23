/* Standalone Vulkan reproducer for an AMD Windows driver miscompile (Leonida #5533, TEST-1296). No Leonida code:
 * it creates a device, runs one SPIR-V compute module on one workgroup of 64 threads and checks what each thread
 * wrote. See README.md in this folder.
 *
 *   vk_repro <module.spv> [device-name substring] [expected word, hex]
 *
 * The module gets one storage buffer (set 0, binding 0) of 64 uints, zero-filled; thread i writes word i.
 * The expected word defaults to 40004000. Output: the device and driver, then every distinct word with its
 * thread count. Exit 0 when all 64 words are the expected one, 1 when any differs, 2 on a Vulkan or usage error.
 *
 * Build (MSVC):  cl /nologo /O2 vk_repro.c /I %VULKAN_SDK%\Include /link /LIBPATH:%VULKAN_SDK%\Lib vulkan-1.lib
 * Build (gcc):   gcc -O1 vk_repro.c -o vk_repro -lvulkan
 */
#include <vulkan/vulkan.h>

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define THREADS 64

#define CK(x)                                                                    \
    do {                                                                         \
        VkResult r_ = (x);                                                       \
        if (r_ != VK_SUCCESS) {                                                  \
            fprintf(stderr, "%s failed: VkResult %d\n", #x, (int)r_);            \
            exit(2);                                                             \
        }                                                                        \
    } while (0)

static int contains_nocase(const char *hay, const char *needle) {
    size_t n = strlen(needle);
    for (; *hay; ++hay) {
        size_t i = 0;
        while (i < n && hay[i] && tolower((unsigned char)hay[i]) == tolower((unsigned char)needle[i])) ++i;
        if (i == n) return 1;
    }
    return n == 0;
}

static unsigned char *read_file(const char *path, size_t *size) {
    FILE *f = fopen(path, "rb");
    unsigned char *data;
    long n;
    if (!f) {
        fprintf(stderr, "cannot open %s\n", path);
        exit(2);
    }
    fseek(f, 0, SEEK_END);
    n = ftell(f);
    fseek(f, 0, SEEK_SET);
    data = (unsigned char *)malloc((size_t)n);
    if (!data || fread(data, 1, (size_t)n, f) != (size_t)n) {
        fprintf(stderr, "cannot read %s\n", path);
        exit(2);
    }
    fclose(f);
    *size = (size_t)n;
    return data;
}

int main(int argc, char **argv) {
    const char *want_dev = argc > 2 ? argv[2] : "";
    const unsigned expected = argc > 3 ? (unsigned)strtoul(argv[3], NULL, 16) : 0x40004000u;
    VkApplicationInfo app = {VK_STRUCTURE_TYPE_APPLICATION_INFO};
    VkInstanceCreateInfo ici = {VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    VkInstance instance;
    VkPhysicalDevice phys[16], pd = VK_NULL_HANDLE;
    VkPhysicalDeviceProperties props = {0};
    VkQueueFamilyProperties qprops[16];
    uint32_t count = 16, qcount = 16, qfam = UINT32_MAX, i, mt = UINT32_MAX;
    float prio = 1.0f;
    VkDeviceQueueCreateInfo qci = {VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    VkDeviceCreateInfo dci = {VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    VkDevice dev;
    VkQueue queue;
    VkBufferCreateInfo bci = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    VkBuffer buf;
    VkMemoryRequirements req;
    VkPhysicalDeviceMemoryProperties mp;
    VkMemoryAllocateInfo mai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    VkDeviceMemory mem;
    uint32_t *words;
    size_t code_size;
    unsigned char *code;
    VkShaderModuleCreateInfo smci = {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    VkShaderModule sm;
    VkDescriptorSetLayoutBinding binding = {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL};
    VkDescriptorSetLayoutCreateInfo dslci = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    VkDescriptorSetLayout dsl;
    VkPipelineLayoutCreateInfo plci = {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    VkPipelineLayout pl;
    VkComputePipelineCreateInfo cpci = {VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    VkPipeline pipe;
    VkDescriptorPoolSize ps = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1};
    VkDescriptorPoolCreateInfo dpci = {VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    VkDescriptorPool dp;
    VkDescriptorSetAllocateInfo dsai = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    VkDescriptorSet ds;
    VkDescriptorBufferInfo dbi;
    VkWriteDescriptorSet wds = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    VkCommandPoolCreateInfo cpi = {VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    VkCommandPool pool;
    VkCommandBufferAllocateInfo cbai = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    VkCommandBuffer cb;
    VkCommandBufferBeginInfo cbbi = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    VkMemoryBarrier mb = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    VkSubmitInfo si = {VK_STRUCTURE_TYPE_SUBMIT_INFO};
    unsigned seen[THREADS];
    int seen_count[THREADS], distinct = 0, bad = 0;

    if (argc < 2) {
        fprintf(stderr, "usage: vk_repro <module.spv> [device-name substring] [expected word, hex]\n");
        return 2;
    }

    app.apiVersion = VK_API_VERSION_1_2; /* SPIR-V 1.5 modules */
    ici.pApplicationInfo = &app;
    CK(vkCreateInstance(&ici, NULL, &instance));
    {
        VkResult r = vkEnumeratePhysicalDevices(instance, &count, phys);
        if (r != VK_SUCCESS && r != VK_INCOMPLETE) CK(r);
    }
    for (i = 0; i < count; ++i) {
        vkGetPhysicalDeviceProperties(phys[i], &props);
        if (contains_nocase(props.deviceName, want_dev)) {
            pd = phys[i];
            break;
        }
    }
    if (pd == VK_NULL_HANDLE) {
        fprintf(stderr, "no Vulkan device name contains \"%s\"\n", want_dev);
        return 2;
    }
    printf("device: %s (vendor 0x%04x, device 0x%04x), driverVersion 0x%08x, Vulkan %u.%u.%u\n", props.deviceName,
           props.vendorID, props.deviceID, props.driverVersion, VK_API_VERSION_MAJOR(props.apiVersion),
           VK_API_VERSION_MINOR(props.apiVersion), VK_API_VERSION_PATCH(props.apiVersion));

    vkGetPhysicalDeviceQueueFamilyProperties(pd, &qcount, qprops);
    for (i = 0; i < qcount; ++i)
        if (qprops[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
            qfam = i;
            break;
        }
    qci.queueFamilyIndex = qfam;
    qci.queueCount = 1;
    qci.pQueuePriorities = &prio;
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos = &qci;
    CK(vkCreateDevice(pd, &dci, NULL, &dev));
    vkGetDeviceQueue(dev, qfam, 0, &queue);

    bci.size = THREADS * sizeof(uint32_t);
    bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    CK(vkCreateBuffer(dev, &bci, NULL, &buf));
    vkGetBufferMemoryRequirements(dev, buf, &req);
    vkGetPhysicalDeviceMemoryProperties(pd, &mp);
    for (i = 0; i < mp.memoryTypeCount; ++i) {
        const VkMemoryPropertyFlags need = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        if ((req.memoryTypeBits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & need) == need) {
            mt = i;
            break;
        }
    }
    mai.allocationSize = req.size;
    mai.memoryTypeIndex = mt;
    CK(vkAllocateMemory(dev, &mai, NULL, &mem));
    CK(vkBindBufferMemory(dev, buf, mem, 0));
    CK(vkMapMemory(dev, mem, 0, bci.size, 0, (void **)&words));
    memset(words, 0, (size_t)bci.size);

    code = read_file(argv[1], &code_size);
    smci.codeSize = code_size;
    smci.pCode = (const uint32_t *)code;
    CK(vkCreateShaderModule(dev, &smci, NULL, &sm));
    dslci.bindingCount = 1;
    dslci.pBindings = &binding;
    CK(vkCreateDescriptorSetLayout(dev, &dslci, NULL, &dsl));
    plci.setLayoutCount = 1;
    plci.pSetLayouts = &dsl;
    CK(vkCreatePipelineLayout(dev, &plci, NULL, &pl));
    cpci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    cpci.stage.module = sm;
    cpci.stage.pName = "main";
    cpci.layout = pl;
    CK(vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &cpci, NULL, &pipe));

    dpci.maxSets = 1;
    dpci.poolSizeCount = 1;
    dpci.pPoolSizes = &ps;
    CK(vkCreateDescriptorPool(dev, &dpci, NULL, &dp));
    dsai.descriptorPool = dp;
    dsai.descriptorSetCount = 1;
    dsai.pSetLayouts = &dsl;
    CK(vkAllocateDescriptorSets(dev, &dsai, &ds));
    dbi.buffer = buf;
    dbi.offset = 0;
    dbi.range = bci.size;
    wds.dstSet = ds;
    wds.descriptorCount = 1;
    wds.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    wds.pBufferInfo = &dbi;
    vkUpdateDescriptorSets(dev, 1, &wds, 0, NULL);

    cpi.queueFamilyIndex = qfam;
    CK(vkCreateCommandPool(dev, &cpi, NULL, &pool));
    cbai.commandPool = pool;
    cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cbai.commandBufferCount = 1;
    CK(vkAllocateCommandBuffers(dev, &cbai, &cb));
    CK(vkBeginCommandBuffer(cb, &cbbi));
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pl, 0, 1, &ds, 0, NULL);
    vkCmdDispatch(cb, 1, 1, 1);
    mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &mb, 0, NULL, 0,
                         NULL);
    CK(vkEndCommandBuffer(cb));
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cb;
    CK(vkQueueSubmit(queue, 1, &si, VK_NULL_HANDLE));
    CK(vkQueueWaitIdle(queue));

    for (i = 0; i < THREADS; ++i) {
        int k;
        for (k = 0; k < distinct && seen[k] != words[i]; ++k) {
        }
        if (k == distinct) {
            seen[distinct] = words[i];
            seen_count[distinct++] = 0;
        }
        ++seen_count[k];
        bad += words[i] != expected;
    }
    for (i = 0; i < (uint32_t)distinct; ++i)
        printf("0x%08x in %d of %d threads%s\n", seen[i], seen_count[i], THREADS,
               seen[i] == expected ? " (expected)" : "");
    printf("%s: %d of %d threads wrote the expected 0x%08x\n", bad ? "MISMATCH" : "PASS", THREADS - bad, THREADS,
           expected);

    vkDestroyCommandPool(dev, pool, NULL);
    vkDestroyDescriptorPool(dev, dp, NULL);
    vkDestroyPipeline(dev, pipe, NULL);
    vkDestroyPipelineLayout(dev, pl, NULL);
    vkDestroyDescriptorSetLayout(dev, dsl, NULL);
    vkDestroyShaderModule(dev, sm, NULL);
    vkUnmapMemory(dev, mem);
    vkDestroyBuffer(dev, buf, NULL);
    vkFreeMemory(dev, mem, NULL);
    vkDestroyDevice(dev, NULL);
    vkDestroyInstance(instance, NULL);
    free(code);
    return bad ? 1 : 0;
}
