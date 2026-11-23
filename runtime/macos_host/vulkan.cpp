// vulkan-1.dll for Windows guests, on MoltenVK. Vulkan structs have the same layout on Win64 and SysV x86-64, so a
// guest call only needs its calling convention changed and gs switched to the host while MoltenVK runs. Every
// function pointer handed to the guest is a small stub into one generic converter; the few commands with float
// parameters (whose registers differ between the two ABIs) get typed thunks. The Win32 surface is a CAMetalLayer.
// ponytail: no VkAllocationCallbacks (passed through, must be null), no debug-messenger callbacks into the guest,
// float-command thunks keep the pointer of the last device that asked for them.
#define VK_NO_PROTOTYPES
#include <vulkan/vulkan.h>
#include <vulkan/vulkan_metal.h>

#include "host.h"

#include <dlfcn.h>
#include <sys/mman.h>

#include <cstring>
#include <map>
#include <mutex>

extern "C" void host_ms_to_sysv();  // rax = SysV target; MS ABI integer/pointer arguments, up to 12

// Spill the MS argument registers, switch gs to the host, move the arguments to SysV registers and stack slots
// (args 7..12 are copied unconditionally: reading past a shorter list only reads the caller's frame), call, switch
// back. rdi, rsi and xmm6-15 are callee-saved in the MS ABI but not in SysV, so they are kept here.
__asm__(
    ".text\n.globl _host_ms_to_sysv\n_host_ms_to_sysv:\n"
    "  pushq %rbp\n  movq %rsp, %rbp\n  pushq %rbx\n  pushq %rdi\n  pushq %rsi\n  pushq %r12\n"
    "  subq $0x100, %rsp\n"
    "  movq %rcx, 0x30(%rsp)\n  movq %rdx, 0x38(%rsp)\n  movq %r8, 0x40(%rsp)\n  movq %r9, 0x48(%rsp)\n"
    "  movdqu %xmm6, 0x50(%rsp)\n  movdqu %xmm7, 0x60(%rsp)\n  movdqu %xmm8, 0x70(%rsp)\n  movdqu %xmm9, 0x80(%rsp)\n"
    "  movdqu %xmm10, 0x90(%rsp)\n  movdqu %xmm11, 0xa0(%rsp)\n  movdqu %xmm12, 0xb0(%rsp)\n"
    "  movdqu %xmm13, 0xc0(%rsp)\n  movdqu %xmm14, 0xd0(%rsp)\n  movdqu %xmm15, 0xe0(%rsp)\n"
    "  movq %rax, %r12\n  movq %gs:0x30, %rbx\n  movq 0x1f00(%rbx), %rdi\n  callq __thread_set_tsd_base\n"
    "  movq 0x40(%rbp), %rax\n  movq %rax, 0x00(%rsp)\n  movq 0x48(%rbp), %rax\n  movq %rax, 0x08(%rsp)\n"
    "  movq 0x50(%rbp), %rax\n  movq %rax, 0x10(%rsp)\n  movq 0x58(%rbp), %rax\n  movq %rax, 0x18(%rsp)\n"
    "  movq 0x60(%rbp), %rax\n  movq %rax, 0x20(%rsp)\n  movq 0x68(%rbp), %rax\n  movq %rax, 0x28(%rsp)\n"
    "  movq 0x30(%rsp), %rdi\n  movq 0x38(%rsp), %rsi\n  movq 0x40(%rsp), %rdx\n  movq 0x48(%rsp), %rcx\n"
    "  movq 0x30(%rbp), %r8\n  movq 0x38(%rbp), %r9\n"
    "  callq *%r12\n"
    "  movq %rax, %r12\n  movq %rbx, %rdi\n  callq __thread_set_tsd_base\n  movq %r12, %rax\n"
    "  movdqu 0x50(%rsp), %xmm6\n  movdqu 0x60(%rsp), %xmm7\n  movdqu 0x70(%rsp), %xmm8\n  movdqu 0x80(%rsp), %xmm9\n"
    "  movdqu 0x90(%rsp), %xmm10\n  movdqu 0xa0(%rsp), %xmm11\n  movdqu 0xb0(%rsp), %xmm12\n"
    "  movdqu 0xc0(%rsp), %xmm13\n  movdqu 0xd0(%rsp), %xmm14\n  movdqu 0xe0(%rsp), %xmm15\n"
    "  leaq -32(%rbp), %rsp\n  popq %r12\n  popq %rsi\n  popq %rdi\n  popq %rbx\n  popq %rbp\n  retq\n");

namespace host {

void* metal_layer_for_window(void* hwnd);  // window.mm

namespace {

PFN_vkGetInstanceProcAddr g_gipa = nullptr;  // MoltenVK's
PFN_vkGetDeviceProcAddr g_gdpa = nullptr;    // exported directly: vkGetInstanceProcAddr(NULL, ...) does not give it

void load_moltenvk() {
    if (g_gipa) return;
    setenv("MVK_CONFIG_FAST_MATH_ENABLED", "0", 0);  // as Leonida's macOS tools do (docs/design.md)
    const char* path = std::getenv("LEONIDA_MOLTENVK");
    void* lib = dlopen(path && *path ? path : "@rpath/libMoltenVK.dylib", RTLD_NOW | RTLD_LOCAL);
    if (!lib) fail(std::string("cannot load MoltenVK (set LEONIDA_MOLTENVK): ") + dlerror());
    g_gipa = reinterpret_cast<PFN_vkGetInstanceProcAddr>(dlsym(lib, "vkGetInstanceProcAddr"));
    g_gdpa = reinterpret_cast<PFN_vkGetDeviceProcAddr>(dlsym(lib, "vkGetDeviceProcAddr"));
    if (!g_gipa || !g_gdpa) fail("MoltenVK lacks vkGetInstanceProcAddr or vkGetDeviceProcAddr");
}

// One 24-byte stub per host function: movabs rax, target; jmp [rip+0]; .quad host_ms_to_sysv.
std::mutex g_stub_lock;
std::map<void*, void*> g_stubs;
uint8_t* g_stub_page = nullptr;
size_t g_stub_used = 0;
constexpr size_t kStubPage = 1 << 16, kStubSize = 24;

void* stub_for(void* target) {
    if (!target) return nullptr;
    std::lock_guard<std::mutex> lock(g_stub_lock);
    if (auto it = g_stubs.find(target); it != g_stubs.end()) return it->second;
    if (!g_stub_page || g_stub_used + kStubSize > kStubPage) {
        void* page = mmap(nullptr, kStubPage, PROT_READ | PROT_WRITE | PROT_EXEC, MAP_PRIVATE | MAP_ANON, -1, 0);
        if (page == MAP_FAILED) fail("cannot map Vulkan stubs");
        g_stub_page = static_cast<uint8_t*>(page);
        g_stub_used = 0;
    }
    uint8_t* s = g_stub_page + g_stub_used;
    g_stub_used += kStubSize;
    uint64_t t = reinterpret_cast<uint64_t>(target), conv = reinterpret_cast<uint64_t>(&host_ms_to_sysv);
    s[0] = 0x48; s[1] = 0xB8; std::memcpy(s + 2, &t, 8);
    s[10] = 0xFF; s[11] = 0x25; std::memset(s + 12, 0, 4); std::memcpy(s + 16, &conv, 8);
    g_stubs[target] = s;
    return s;
}

// ---- commands whose float parameters sit in different registers on each side ----
PFN_vkCmdSetLineWidth g_line_width;
PFN_vkCmdSetDepthBias g_depth_bias;
PFN_vkCmdSetDepthBounds g_depth_bounds;
PFN_vkSetDeviceMemoryPriorityEXT g_memory_priority;
PFN_vkCmdSetExtraPrimitiveOverestimationSizeEXT g_overestimation;
void cmd_set_line_width(VkCommandBuffer c, float w) { g_line_width(c, w); }
void cmd_set_depth_bias(VkCommandBuffer c, float a, float b, float s) { g_depth_bias(c, a, b, s); }
void cmd_set_depth_bounds(VkCommandBuffer c, float lo, float hi) { g_depth_bounds(c, lo, hi); }
void set_memory_priority(VkDevice d, VkDeviceMemory m, float p) { g_memory_priority(d, m, p); }
void cmd_set_overestimation(VkCommandBuffer c, float s) { g_overestimation(c, s); }

template <typename Pfn, typename F> void* typed(Pfn& slot, PFN_vkVoidFunction real, F& thunk) {
    slot = reinterpret_cast<Pfn>(real);
    return real ? reinterpret_cast<void*>(thunk) : nullptr;
}
#define FLOAT_COMMAND(name, slot, fn) \
    if (!std::strcmp(n, name)) return typed(slot, real, Thunk<decltype(fn), &fn>::call);

void* float_command(const char* n, PFN_vkVoidFunction real) {
    FLOAT_COMMAND("vkCmdSetLineWidth", g_line_width, cmd_set_line_width)
    FLOAT_COMMAND("vkCmdSetDepthBias", g_depth_bias, cmd_set_depth_bias)
    FLOAT_COMMAND("vkCmdSetDepthBounds", g_depth_bounds, cmd_set_depth_bounds)
    FLOAT_COMMAND("vkSetDeviceMemoryPriorityEXT", g_memory_priority, set_memory_priority)
    FLOAT_COMMAND("vkCmdSetExtraPrimitiveOverestimationSizeEXT", g_overestimation, cmd_set_overestimation)
    return reinterpret_cast<void*>(-1);
}

// ---- the Win32 surface, as Metal ----
const char* const kWin32Surface = "VK_KHR_win32_surface";
const char* const kMetalSurface = "VK_EXT_metal_surface";

VkResult create_instance(const VkInstanceCreateInfo* info, const VkAllocationCallbacks* alloc, VkInstance* out) {
    std::vector<const char*> ext(info->ppEnabledExtensionNames, info->ppEnabledExtensionNames + info->enabledExtensionCount);
    for (auto& e : ext)
        if (!std::strcmp(e, kWin32Surface)) e = kMetalSurface;
    VkInstanceCreateInfo copy = *info;
    copy.enabledExtensionCount = static_cast<uint32_t>(ext.size());
    copy.ppEnabledExtensionNames = ext.data();
    auto real = reinterpret_cast<PFN_vkCreateInstance>(g_gipa(nullptr, "vkCreateInstance"));
    return real(&copy, alloc, out);
}

VkResult enumerate_instance_extensions(const char* layer, uint32_t* count, VkExtensionProperties* props) {
    auto real = reinterpret_cast<PFN_vkEnumerateInstanceExtensionProperties>(
        g_gipa(nullptr, "vkEnumerateInstanceExtensionProperties"));
    VkResult r = real(layer, count, props);
    if (props)
        for (uint32_t i = 0; i < *count; ++i)
            if (!std::strcmp(props[i].extensionName, kMetalSurface))
                std::strncpy(props[i].extensionName, kWin32Surface, VK_MAX_EXTENSION_NAME_SIZE);
    return r;
}

struct Win32SurfaceInfo {  // VkWin32SurfaceCreateInfoKHR, without windows.h
    VkStructureType sType;
    const void* pNext;
    VkFlags flags;
    void* hinstance;
    void* hwnd;
};
VkResult create_win32_surface(VkInstance instance, const Win32SurfaceInfo* info, const VkAllocationCallbacks* alloc,
                              VkSurfaceKHR* out) {
    VkMetalSurfaceCreateInfoEXT metal{};
    metal.sType = VK_STRUCTURE_TYPE_METAL_SURFACE_CREATE_INFO_EXT;
    metal.pLayer = metal_layer_for_window(info->hwnd);
    if (!metal.pLayer) return VK_ERROR_INITIALIZATION_FAILED;
    auto real = reinterpret_cast<PFN_vkCreateMetalSurfaceEXT>(g_gipa(instance, "vkCreateMetalSurfaceEXT"));
    return real ? real(instance, &metal, alloc, out) : VK_ERROR_EXTENSION_NOT_PRESENT;
}
VkBool32 win32_presentation_support(VkPhysicalDevice, uint32_t) { return VK_TRUE; }

void* get_instance_proc_addr(VkInstance instance, const char* name);
void* get_device_proc_addr(VkDevice device, const char* name) {
    PFN_vkVoidFunction real = g_gdpa(device, name);
    void* f = float_command(name, real);
    return f != reinterpret_cast<void*>(-1) ? f : stub_for(reinterpret_cast<void*>(real));
}

#define SPECIAL(sym, fn) \
    if (!std::strcmp(name, sym)) return reinterpret_cast<void*>(&Thunk<decltype(fn), &fn>::call);

void* get_instance_proc_addr(VkInstance instance, const char* name) {
    load_moltenvk();
    SPECIAL("vkGetInstanceProcAddr", get_instance_proc_addr)
    SPECIAL("vkGetDeviceProcAddr", get_device_proc_addr)
    SPECIAL("vkCreateInstance", create_instance)
    SPECIAL("vkEnumerateInstanceExtensionProperties", enumerate_instance_extensions)
    SPECIAL("vkCreateWin32SurfaceKHR", create_win32_surface)
    SPECIAL("vkGetPhysicalDeviceWin32PresentationSupportKHR", win32_presentation_support)
    PFN_vkVoidFunction real = g_gipa(instance, name);
    void* f = float_command(name, real);
    return f != reinterpret_cast<void*>(-1) ? f : stub_for(reinterpret_cast<void*>(real));
}

}  // namespace

Exports vulkan_exports() {
    return {EXPORT_AS("vulkan-1.dll", "vkGetInstanceProcAddr", get_instance_proc_addr)};
}

}  // namespace host
