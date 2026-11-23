// Lanterns, stage 25: the scale frame (lanterns_scale.h, lanterns_scale_gpu.h) run for many frames.
//
//   lanterns.exe --stage 25 --shaders <dir> --out <dir> [--frames N] [--seconds S] [--draws D] [--threads T]
//                [--ref-every K] [--sample-us U] [--timeout-ms MS] [--max-dumps M] [--jitter-us J]
//
// Every frame draws the same thing, so every readback (main target, resolved aux target, top target, both particle
// sets) must be byte for byte the first one. A frame that differs is a finding: its readback, the first frame's, and
// the queue-level log of the frames around it are written to <out> (stage25-bad-<frame>.*). A frame that does not
// finish within the timeout is a hang: the fences' completed values and the log are written, and the run stops.
// The first frame's main target is checked against the CPU reference at every pixel (in bands), and every K-th frame at
// a quarter-density lattice whose offset moves each time; both particle sets are compared with the CPU's simulation.
// Each D3D12 call is timed by class (lanterns_scale_prof.h); with --sample-us a sampling profiler runs beside the
// frames and writes stage25-samples.txt, which tools turn into the layer's hot functions.
#include "lanterns_scale.h"
#include "lanterns_scale_gpu.h"
#include "lanterns_scale_prof.h"

#include "lanterns_post.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

namespace lanterns::scale {
namespace {

struct Options {
    // The default is a short run for ctest; run_lanterns.py --soak asks for 10,000 frames with a check every 1,000.
    std::uint64_t frames = 60;
    double seconds = 0;
    unsigned draws = 5000, threads = 4, ref_every = 20, sample_us = 0, timeout_ms = 20000, max_dumps = 3, jitter_us = 0;
};

bool write_file(const std::filesystem::path& path, const void* data, std::size_t size) {
    std::ofstream out(path, std::ios::binary);
    out.write(static_cast<const char*>(data), std::streamsize(size));
    return bool(out);
}

std::uint64_t hash64(const std::uint8_t* data, std::size_t bytes) {
    std::uint64_t h[4] = {0x9e3779b97f4a7c15ull, 0xc2b2ae3d27d4eb4full, 0x165667b19e3779f9ull, 0x27d4eb2f165667c5ull};
    std::size_t i = 0;
    for (; i + 32 <= bytes; i += 32)
        for (int k = 0; k < 4; ++k) {
            std::uint64_t w;
            std::memcpy(&w, data + i + 8 * k, 8);
            h[k] = (h[k] ^ w) * 0x100000001b3ull;
            h[k] ^= h[k] >> 29;
        }
    std::uint64_t r = h[0] ^ (h[1] << 1) ^ (h[2] << 2) ^ (h[3] << 3);
    for (; i < bytes; ++i) r = (r ^ data[i]) * 0x100000001b3ull;
    return r;
}

Color main_pixel(const std::uint8_t* data, unsigned x, unsigned y) {
    Color c;
    for (unsigned k = 0; k < 4; ++k) {
        std::uint16_t h;
        std::memcpy(&h, data + kOffMain + y * kMainPitch + (x * 4 + k) * 2, 2);
        c[k] = half_value(h);
    }
    return c;
}

// Where frame data differs from the first frame's: per region, the differing pixels (or records) and their box.
struct Region {
    const char* name;
    std::uint64_t offset, pitch;
    unsigned width, height, bytes_per_pixel;
};
const Region kRegions[5] = {{"main", kOffMain, kMainPitch, kW, kH, 8},
                            {"aux", kOffAux, kAuxPitch, kAuxW, kAuxH, 8},
                            {"top", kOffTop, kTopPitch, kTopW, kTopH, 4},
                            {"particles A", kOffPartA, kParticles * sizeof(Instance), kParticles, 1, sizeof(Instance)},
                            {"particles B", kOffPartB, kParticles * sizeof(Instance), kParticles, 1, sizeof(Instance)}};

std::string describe_difference(const std::uint8_t* a, const std::uint8_t* b) {
    std::string out;
    for (const Region& r : kRegions) {
        unsigned long long n = 0;
        unsigned x0 = ~0u, y0 = ~0u, x1 = 0, y1 = 0;
        for (unsigned y = 0; y < r.height; ++y) {
            const std::uint64_t row = r.offset + y * r.pitch;
            if (std::memcmp(a + row, b + row, std::size_t(r.width) * r.bytes_per_pixel) == 0) continue;
            for (unsigned x = 0; x < r.width; ++x)
                if (std::memcmp(a + row + x * r.bytes_per_pixel, b + row + x * r.bytes_per_pixel, r.bytes_per_pixel)) {
                    ++n;
                    x0 = std::min(x0, x), y0 = std::min(y0, y), x1 = std::max(x1, x), y1 = std::max(y1, y);
                }
        }
        if (!n) continue;
        char line[200];
        std::snprintf(line, sizeof(line), "%s%s: %llu differ in (%u,%u)-(%u,%u)", out.empty() ? "" : "; ", r.name, n, x0,
                      y0, x1, y1);
        out += line;
    }
    return out;
}

void print_timers(const std::filesystem::path& out_dir, std::uint64_t frames, double ns_per_tick) {
    const ClassCounters t = timers().total();
    std::vector<unsigned> order;
    for (unsigned k = 0; k < kClassCount; ++k)
        if (t.count[k]) order.push_back(k);
    std::sort(order.begin(), order.end(), [&](unsigned a, unsigned b) { return t.ticks[a] > t.ticks[b]; });
    double all = 0;
    for (unsigned k : order)
        if (k != kFenceWaitCpu) all += double(t.ticks[k]) * ns_per_tick;
    std::ofstream tsv(out_dir / "stage25-calls.tsv");
    tsv << "class\tcalls\tcalls_per_frame\tns_per_call\tus_per_frame\tshare\n";
    std::printf("lanterns: CPU time in D3D12 calls over %llu frames (TSC %.4f ns/tick), all recording threads:\n",
                (unsigned long long)frames, ns_per_tick);
    for (unsigned k : order) {
        const double ns = double(t.ticks[k]) * ns_per_tick;
        const double share = k == kFenceWaitCpu ? 0 : ns / all;
        std::printf("lanterns:   %-48s %10.1f calls/frame %9.1f ns/call %9.1f us/frame %5.1f%%\n", call_class_name(k),
                    double(t.count[k]) / double(frames), ns / double(t.count[k]), ns / 1000.0 / double(frames),
                    100.0 * share);
        tsv << call_class_name(k) << '\t' << t.count[k] << '\t' << double(t.count[k]) / double(frames) << '\t'
            << ns / double(t.count[k]) << '\t' << ns / 1000.0 / double(frames) << '\t' << share << '\n';
    }
}

}  // namespace

int stage25(int argc, char** argv, const std::filesystem::path& shader_dir, const std::filesystem::path& out_dir) {
    Options o;
    for (int i = 1; i + 1 < argc; i += 2) {
        const std::string key = argv[i];
        const char* v = argv[i + 1];
        if (key == "--frames") o.frames = std::strtoull(v, nullptr, 10);
        else if (key == "--seconds") o.seconds = std::atof(v);
        else if (key == "--draws") o.draws = unsigned(std::atoi(v));
        else if (key == "--threads") o.threads = unsigned(std::atoi(v));
        else if (key == "--ref-every") o.ref_every = unsigned(std::atoi(v));
        else if (key == "--sample-us") o.sample_us = unsigned(std::atoi(v));
        else if (key == "--timeout-ms") o.timeout_ms = unsigned(std::atoi(v));
        else if (key == "--max-dumps") o.max_dumps = unsigned(std::atoi(v));
        else if (key == "--jitter-us") o.jitter_us = unsigned(std::atoi(v));
    }
    if (o.draws < 8 || o.threads < 1 || o.threads > 16 || o.frames < 2) {
        std::fprintf(stderr, "lanterns: stage 25 needs --draws >= 8, --threads 1..16, --frames >= 2\n");
        return 2;
    }
    Shaders shaders;
    if (!load_shaders(shader_dir, 3, shaders)) return 1;
    std::vector<Particle> cpu_a = initial_particles(), cpu_b = initial_particles();
    for (unsigned k = 0; k < kStepsA; ++k) step_particles(cpu_a);
    for (unsigned k = 0; k < kStepsB; ++k) step_particles(cpu_b);
    const auto want_a = particle_instances(cpu_a), want_b = particle_instances(cpu_b);
    const World world = make_world(o.draws, cpu_a);

    const auto t_setup = std::chrono::steady_clock::now();
    ScaleGpu gpu;
    if (!gpu.create(world, shaders, o.threads)) return 1;
    gpu.set_jitter(o.jitter_us);
    const double setup_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_setup).count();
    std::printf("lanterns: stage 25 set up in %.1f s: %u pipelines, %u root signatures, %zu draws in the main pass "
                "(%u instanced, %u objects), %zu aux (4x MSAA), %zu top (RGBA8); %u recording threads, jitter up to %u us\n", setup_s,
                gpu.pipelines(), gpu.root_signatures(), world.scene.draws.size(), world.instanced_draws, world.objects,
                world.aux_draws.size() + 1, world.top_draws.size(), o.threads, o.jitter_us);

    std::vector<std::uint8_t> golden;
    std::uint64_t golden_hash = 0, mismatches = 0, first_bad = 0, dumps = 0, ref_checks = 0, ref_failed = 0;
    unsigned long long ref_pixels = 0, ref_exact = 0, ref_uncertain = 0;
    bool particles_ok = true, full_ok = false;
    Sampler sampler;
    bool sampling = false;
    constexpr std::uint64_t kWarmup = 10;
    std::vector<double> frame_ms;
    std::ofstream frames_csv(out_dir / "stage25-frames.csv");
    frames_csv << "frame,frame_to_frame_ms,submit_call_ms,consume_ms,same\n";
    std::vector<double> submit_ms;
    const auto started = std::chrono::steady_clock::now();
    auto last_submit = started;
    std::string hang;

    const auto consume = [&](std::uint64_t g) -> bool {
        const auto t0 = std::chrono::steady_clock::now();
        const auto waited = gpu.wait(g, o.timeout_ms);
        if (waited != ScaleGpu::Outcome::ok) {
            hang = "frame " + std::to_string(g) + " did not finish within " + std::to_string(o.timeout_ms) + " ms; " +
                   gpu.fence_status();
            return false;
        }
        const std::uint8_t* data = gpu.map_readback(g);
        if (!data) {
            hang = "Map of the readback failed at frame " + std::to_string(g);
            return false;
        }
        bool same = true;
        if (g == 1) {
            golden.assign(data, data + kReadbackBytes);
            golden_hash = hash64(data, kReadbackBytes);
            particles_ok = std::memcmp(data + kOffPartA, want_a.data(), kParticles * sizeof(Instance)) == 0 &&
                           std::memcmp(data + kOffPartB, want_b.data(), kParticles * sizeof(Instance)) == 0;
            write_file(out_dir / "stage25-frame1.bin", data, kReadbackBytes);
            const auto t_ref = std::chrono::steady_clock::now();
            const SparseCheck full = check_sparse(world, Lattice{}, [&](unsigned x, unsigned y) { return main_pixel(data, x, y); });
            full_ok = full.pass();
            write_file(out_dir / "stage25-frame1.marks.u8", full.marks.data(), full.marks.size());
            std::printf("lanterns: frame 1 against the CPU reference at every pixel (%.1f s): %llu of %llu certain pixels "
                        "exact, %llu of %llu uncertain pixels one of their candidates; %u faces culled, %u of uncertain "
                        "orientation, %u edge-on; particle sets A and B %s\n",
                        std::chrono::duration<double>(std::chrono::steady_clock::now() - t_ref).count(), full.exact,
                        full.certain, full.plausible, full.uncertain, full.culled_faces, full.unsure_orientation,
                        full.degenerate_faces, particles_ok ? "exact" : "DIFFERENT from the CPU's simulation");
            if (!full.pass()) {
                std::printf("lanterns: frame 1: wrong pixels by the expected draw's pipeline variant (cull, flavour):");
                for (unsigned v = 0; v <= kVariants; ++v)
                    if (full.wrong_by_variant[v])
                        std::printf(" [%s%s] %llu", v == kVariants ? "clear colour" : v % 3 == 0 ? "none, " : v % 3 == 1 ? "back, " : "front, ",
                                    v == kVariants ? "" : flavour_name(variant_flavour(v)), full.wrong_by_variant[v]);
                std::printf("\n");
            }
            if (!full.pass())
                std::printf("lanterns: frame 1: first bad pixel (%u, %u): gpu (%g, %g, %g, %g), reference (%g, %g, %g, %g)\n",
                            full.first_bad_x, full.first_bad_y, full.first_bad_gpu[0], full.first_bad_gpu[1],
                            full.first_bad_gpu[2], full.first_bad_gpu[3], full.first_bad_ref[0], full.first_bad_ref[1],
                            full.first_bad_ref[2], full.first_bad_ref[3]);
            ref_pixels += full.certain + full.uncertain, ref_exact += full.exact, ref_uncertain += full.uncertain;
        } else if (std::memcmp(data, golden.data(), kReadbackBytes) != 0) {
            same = false;
            ++mismatches;
            if (!first_bad) first_bad = g;
            const std::string what = describe_difference(golden.data(), data);
            std::printf("lanterns: FINDING: frame %llu differs from frame 1 (hash %016llx, frame 1 %016llx): %s\n",
                        (unsigned long long)g, (unsigned long long)hash64(data, kReadbackBytes),
                        (unsigned long long)golden_hash, what.c_str());
            if (dumps < o.max_dumps) {
                ++dumps;
                const std::string base = "stage25-bad-" + std::to_string(g);
                write_file(out_dir / (base + ".bin"), data, kReadbackBytes);
                const std::string log = what + "\n" + gpu.fence_status() + "\n" + gpu.frame_log(g);
                write_file(out_dir / (base + ".log"), log.data(), log.size());
            }
        }
        if (o.ref_every && g > 1 && g % o.ref_every == 0) {
            const unsigned k = unsigned(g / o.ref_every);
            const SparseCheck c = check_sparse(world, Lattice{4, 4, k % 4, (k / 4) % 4},
                                               [&](unsigned x, unsigned y) { return main_pixel(data, x, y); });
            ++ref_checks;
            ref_failed += !c.pass();
            ref_pixels += c.certain + c.uncertain, ref_exact += c.exact, ref_uncertain += c.uncertain;
            if (!c.pass())
                std::printf("lanterns: FINDING: frame %llu against the reference (lattice offset %u,%u): %llu of %llu "
                            "certain exact, first bad (%u, %u)\n", (unsigned long long)g, k % 4, (k / 4) % 4, c.exact,
                            c.certain, c.first_bad_x, c.first_bad_y);
        }
        gpu.unmap_readback(g);
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        frames_csv << g << ',' << (frame_ms.size() >= g ? frame_ms[g - 1] : 0.0) << ','
                   << (submit_ms.size() >= g ? submit_ms[g - 1] : 0.0) << ',' << ms << ',' << (same ? 1 : 0)
                   << '\n';
        return true;
    };

    std::uint64_t f = 1, measured_from = 0;
    for (;; ++f) {
        if (f == kWarmup + 1) {   // measure from here: the first frame's reference and the warm-up are left out
            timers().reset();
            timers().start_clock();
            measured_from = f;
            if (o.sample_us) sampling = sampler.start(o.sample_us, 20u << 20);
        }
        const auto t_submit = std::chrono::steady_clock::now();
        const auto outcome = gpu.submit(f, o.timeout_ms);
        const auto now = std::chrono::steady_clock::now();
        submit_ms.push_back(std::chrono::duration<double, std::milli>(now - t_submit).count());
        frame_ms.push_back(std::chrono::duration<double, std::milli>(now - last_submit).count());
        last_submit = now;
        if (outcome != ScaleGpu::Outcome::ok) {
            hang = std::string(outcome == ScaleGpu::Outcome::hang ? "HANG" : "ERROR") + " submitting frame " +
                   std::to_string(f) + "; " + gpu.fence_status();
            break;
        }
        if (f == 1) {
            const FrameStats& s = gpu.stats();
            std::printf("lanterns: one frame: %u draws, %u dispatches, %u command lists in %u ExecuteCommandLists, %u "
                        "barriers, %u descriptor copies, %u PSO changes, %u root signature changes, %u copies, "
                        "3 queues\n", s.draws, s.dispatches, s.lists, s.executes, s.barriers, s.descriptor_copies,
                        s.pso_changes, s.rs_changes, s.copies);
        }
        if (f >= 2 && !consume(f - 1)) break;
        const double elapsed = std::chrono::duration<double>(now - started).count();
        if (f % 500 == 0)
            std::printf("lanterns: %llu frames, %llu differ, %.1f s, %.2f frames/s\n", (unsigned long long)f,
                        (unsigned long long)mismatches, elapsed, double(f) / elapsed);
        if ((o.seconds > 0 && elapsed >= o.seconds) || (o.seconds <= 0 && f >= o.frames)) {
            if (!consume(f)) break;
            break;
        }
    }
    if (sampling) sampler.stop();
    const double ns_per_tick = timers().ns_per_tick();
    const std::uint64_t measured = measured_from && f >= measured_from ? f - measured_from + 1 : 0;
    if (!hang.empty()) {
        std::printf("lanterns: FINDING: %s\n", hang.c_str());
        const std::string log = hang + "\n" + gpu.frame_log(f);
        write_file(out_dir / "stage25-hang.log", log.data(), log.size());
        std::fflush(stdout);
        std::_Exit(4);   // the device may be wedged: do not wait for it in the destructors
    }
    if (!gpu.idle(o.timeout_ms)) {
        std::printf("lanterns: FINDING: the queues did not go idle; %s\n", gpu.fence_status().c_str());
        std::_Exit(4);
    }
    const double total_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    if (measured) print_timers(out_dir, measured, ns_per_tick);
    if (sampling) {
        sampler.write(out_dir / "stage25-samples.txt");
        std::printf("lanterns: %zu profiler samples in stage25-samples.txt\n", sampler.samples());
    }
    std::vector<double> sorted(frame_ms.begin() + std::min<std::size_t>(frame_ms.size(), kWarmup), frame_ms.end());
    std::sort(sorted.begin(), sorted.end());
    const auto pct = [&](double q) { return sorted.empty() ? 0.0 : sorted[std::min(sorted.size() - 1, std::size_t(q * sorted.size()))]; };
    std::printf("lanterns: %llu frames in %.1f s (%.2f frames/s); frame-to-frame time median %.2f ms, p99 %.2f ms, max %.2f "
                "ms; %llu frames differ from frame 1 (first: %llu); %llu reference checks, %llu failed; %llu pixels "
                "compared with the reference, %llu exact, %llu uncertain\n", (unsigned long long)f, total_s,
                double(f) / total_s, pct(0.5), pct(0.99), sorted.empty() ? 0.0 : sorted.back(),
                (unsigned long long)mismatches, (unsigned long long)first_bad, (unsigned long long)ref_checks,
                (unsigned long long)ref_failed, ref_pixels, ref_exact, ref_uncertain);
    if (mismatches || !particles_ok || !full_ok || ref_failed) {
        std::fprintf(stderr, "lanterns: stage 25 FAILED\n");
        return 1;
    }
    const FrameStats& s = gpu.stats();
    std::printf("Lanterns stage 25: PASS, %llu frames at %ux%u identical to the first (hash %016llx), %u draws and %u "
                "dispatches per frame on 3 queues, frame 1 exact against the CPU reference at every certain pixel, %llu "
                "lattice checks exact\n", (unsigned long long)f, kW, kH, (unsigned long long)golden_hash, s.draws,
                s.dispatches, (unsigned long long)ref_checks);
    return 0;
}

}  // namespace lanterns::scale