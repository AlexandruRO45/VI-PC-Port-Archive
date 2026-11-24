#include "ffx_provider_loop.h"
#include "ffx_loop_loader.cpp"
#include <cstdlib>
#include <cstdio>
#include <memory>

namespace
{
struct Request {
    RecurrentConfig config;
    LoopContext* context;
    void** observed;
};
uint32_t inner_create(void** output, const void* descriptor, const Allocator*)
{
    const auto& request = *static_cast<const Request*>(descriptor);
    const auto result = create_loop_context(*request.context, output, request.config);
    *request.observed = *output;
    return result;
}
uint32_t release_borrowed(void** output, const Allocator*)
{
    *output = nullptr;
    return 0;
}
} // namespace

uint32_t check_loop_loader(LoopContext& ctx, void** inner, const RecurrentConfig& config)
{
    static const Provider provider{inner_create, release_borrowed};
    const Host host{&provider, std::calloc};
    void* observed = &ctx;
    Request request{config, &ctx, &observed};
    void* wrapper = &ctx;
    const auto result = create_context(&wrapper, &request, nullptr, &host);
    if (result != 0) {
        if (wrapper || observed)
            throw std::runtime_error("Published context on provider failure");
        *inner = nullptr;
        std::puts("loader_wrapper=null");
        return result;
    }
    std::unique_ptr<void, decltype(&std::free)> storage(wrapper, std::free);
    if (!wrapper || wrapper == &ctx.state || load<const Provider*>(reinterpret_cast<word>(wrapper)) != &provider)
        throw std::runtime_error("Invalid loader wrapper or callback table");
    *inner = load<void*>(reinterpret_cast<word>(wrapper) + 8);
    if (*inner != observed)
        throw std::runtime_error("Loader replaced the provider handle");
    std::puts("loader_wrapper=published callback_table=matched");
    return result;
}
