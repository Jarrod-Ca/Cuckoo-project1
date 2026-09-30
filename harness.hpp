// harness.hpp — the two experiments, writing CSVs for plot_results.py.
//
//   Threshold experiment: fill a fixed-size table (rehash disabled) until the
//     first failed insert, many seeds, d=2/b=1 and d=2/b=4, at several eviction
//     budgets. Also records the eviction-chain length of every insert.
//   Latency experiment: per-lookup latency percentiles (hits and misses) for the
//     table (d=2/b=1, d=2/b=4) and the baseline across a sweep of loads, plus
//     probe counts.
//
// Rules the code follows (from experiment-plan.md):
//   - every key is generated before any timing window opens;
//   - fixed seeds everywhere, recorded in the output;
//   - no allocation or formatting inside a timed loop: latencies go into a
//     pre-allocated buffer and percentiles are computed afterwards;
//   - every lookup result is consumed, so the compiler cannot delete it;
//   - warm-up pass before every timed pass;
//   - timer overhead is measured and reported, not silently subtracted.

#pragma once

#include "cuckoo_table.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#if defined(_MSC_VER)
#include <intrin.h>
#elif defined(__x86_64__) || defined(__i386__)
#include <cpuid.h>
#include <x86intrin.h>
#else
#error "The latency timer uses the x86 time-stamp counter (rdtsc)."
#endif

#ifndef RESULTS_DIR
#define RESULTS_DIR "results"
#endif

namespace harness {

// ---------------------------------------------------------------------------
// Settings. Everything the report needs to quote is here.
// ---------------------------------------------------------------------------
struct Settings {
    // Threshold experiment
    std::size_t threshold_slots = std::size_t{1} << 20;
    std::uint64_t threshold_seeds = 20;              // seeds 1..N
    // Latency experiment
    std::size_t latency_slots = std::size_t{1} << 20;
    std::size_t lookups_per_trial = std::size_t{1} << 19;
    std::size_t trials = 5;
};

inline Settings quick_settings() {   // smoke test: same code, tiny sizes
    Settings s;
    s.threshold_slots = std::size_t{1} << 14;
    s.threshold_seeds = 3;
    s.latency_slots = std::size_t{1} << 14;
    s.lookups_per_trial = std::size_t{1} << 13;
    s.trials = 2;
    return s;
}

// ---------------------------------------------------------------------------
// Timing: the CPU time-stamp counter.
//
// On Windows std::chrono::steady_clock ticks every 100 ns, longer than a
// lookup, so it cannot time one operation. rdtsc counts at a fixed rate (the
// CPU's base frequency) regardless of turbo. lfence/rdtscp stop the CPU from
// running the lookup before the first read or after the second, and the
// signal fences stop the compiler moving code across the reads.
// ---------------------------------------------------------------------------
inline std::uint64_t tsc_begin() noexcept {
    std::atomic_signal_fence(std::memory_order_seq_cst);
    _mm_lfence();
    const std::uint64_t t = __rdtsc();
    _mm_lfence();
    std::atomic_signal_fence(std::memory_order_seq_cst);
    return t;
}

inline std::uint64_t tsc_end() noexcept {
    std::atomic_signal_fence(std::memory_order_seq_cst);
    unsigned aux = 0;
    const std::uint64_t t = __rdtscp(&aux);
    _mm_lfence();
    std::atomic_signal_fence(std::memory_order_seq_cst);
    return t;
}

// Writing each lookup result here is what stops the compiler from deleting
// the lookup: a volatile store cannot be optimised away.
inline volatile std::uint64_t g_sink = 0;

// Ticks per nanosecond, measured against steady_clock over ~250 ms.
inline double calibrate_tsc() {
    using clock = std::chrono::steady_clock;
    const auto c0 = clock::now();
    const std::uint64_t t0 = tsc_begin();
    while (clock::now() - c0 < std::chrono::milliseconds(250)) {}
    const auto c1 = clock::now();
    const std::uint64_t t1 = tsc_end();
    const double ns = static_cast<double>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(c1 - c0).count());
    return static_cast<double>(t1 - t0) / ns;
}

// ---------------------------------------------------------------------------
// Environment details for run_info.txt.
// ---------------------------------------------------------------------------
inline std::string cpu_brand() {
    unsigned regs[12] = {};
#if defined(_MSC_VER)
    int r[4];
    for (int i = 0; i < 3; ++i) {
        __cpuid(r, static_cast<int>(0x80000002u + i));
        std::memcpy(regs + 4 * i, r, sizeof r);
    }
#else
    for (unsigned i = 0; i < 3; ++i)
        __get_cpuid(0x80000002u + i, &regs[4 * i], &regs[4 * i + 1], &regs[4 * i + 2], &regs[4 * i + 3]);
#endif
    char buf[49] = {};
    std::memcpy(buf, regs, 48);
    std::string s(buf);
    const auto first = s.find_first_not_of(' ');
    return first == std::string::npos ? s : s.substr(first);
}

inline std::string compiler_string() {
    std::ostringstream o;
#if defined(_MSC_VER) && !defined(__clang__)
    o << "MSVC " << _MSC_FULL_VER;
#elif defined(__clang__)
    o << "Clang " << __clang_major__ << "." << __clang_minor__ << "." << __clang_patchlevel__;
#elif defined(__GNUC__)
    o << "GCC " << __GNUC__ << "." << __GNUC_MINOR__ << "." << __GNUC_PATCHLEVEL__;
#endif
    return o.str();
}

inline bool release_build() {
#ifdef NDEBUG
    return true;
#else
    return false;
#endif
}

inline std::string local_timestamp() {
    const std::time_t now = std::time(nullptr);
    std::tm tm{};
#if defined(_MSC_VER)
    localtime_s(&tm, &now);
#else
    localtime_r(&now, &tm);
#endif
    std::ostringstream o;
    o << std::put_time(&tm, "%Y-%m-%d %H:%M:%S");
    return o.str();
}

// ---------------------------------------------------------------------------
// Data generation. Distinct random 64-bit keys, never the reserved kEmpty.
// ---------------------------------------------------------------------------
inline std::vector<std::uint64_t> make_keys(std::size_t n, std::uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::vector<std::uint64_t> keys;
    keys.reserve(n);
    while (keys.size() < n) {
        const std::uint64_t k = rng();
        if (k != cuckoo::kEmpty) keys.push_back(k);
    }
    // Duplicates among 2^20 random 64-bit values have probability ~3e-8;
    // removing them keeps every key distinct without changing order much.
    std::vector<std::uint64_t> sorted = keys;
    std::sort(sorted.begin(), sorted.end());
    if (std::adjacent_find(sorted.begin(), sorted.end()) != sorted.end()) {
        std::unordered_map<std::uint64_t, bool> seen;
        std::vector<std::uint64_t> unique;
        for (auto k : keys) if (seen.emplace(k, true).second) unique.push_back(k);
        keys.swap(unique);
    }
    return keys;
}

// Percentile of a sorted sample, nearest-rank method.
template <class T>
inline T percentile(const std::vector<T>& sorted, double p) {
    if (sorted.empty()) return T{};
    std::size_t rank = static_cast<std::size_t>(std::ceil(p * static_cast<double>(sorted.size())));
    if (rank == 0) rank = 1;
    return sorted[rank - 1];
}

// ===========================================================================
// THRESHOLD EXPERIMENT
// ===========================================================================
//
// For each (d, b, budget, seed): a table of `threshold_slots` slots with
// rehash disabled; insert fresh keys until the first Failed. The load factor
// just before that insert is the failure point. The eviction count of every
// insert is kept in a pre-allocated vector and summarised per 0.01-wide load
// bin afterwards.
//
// Budgets are multiples of the table's default (16 * log2(slots)): the b = 4
// result depends on the budget, so the budget is a column, not a constant.

inline void run_threshold(const Settings& st, std::ostream& log) {
    namespace fs = std::filesystem;
    std::ofstream fails(fs::path(RESULTS_DIR) / "threshold_failures.csv");
    std::ofstream chains(fs::path(RESULTS_DIR) / "threshold_chains.csv");
    fails << "d,b,slots,budget,budget_mult,seed,inserted,failure_load\n";
    chains << "d,b,slots,budget,budget_mult,seed,load_bin,count,mean,p50,p99,p999,max\n";
    fails << std::setprecision(9);
    chains << std::setprecision(9);

    struct Case { std::size_t d, b; std::vector<std::size_t> mults; };
    const std::vector<Case> cases = {
        {2, 1, {1, 4}},
        {2, 4, {1, 4, 16}},
    };

    const std::size_t S = st.threshold_slots;
    std::vector<std::uint32_t> chain(S);   // pre-allocated: one entry per insert

    for (const Case& c : cases) {
        std::size_t default_budget = 0;
        {
            cuckoo::Config probe;
            probe.d = c.d; probe.b = c.b; probe.num_buckets = S / c.b;
            default_budget = cuckoo::CuckooTable(probe).max_evictions();
        }
        for (std::size_t mult : c.mults) {
            const std::size_t budget = default_budget * mult;
            double sum = 0;
            for (std::uint64_t seed = 1; seed <= st.threshold_seeds; ++seed) {
                cuckoo::Config cfg;
                cfg.d = c.d;
                cfg.b = c.b;
                cfg.num_buckets = S / c.b;
                cfg.seed = seed;
                cfg.max_evictions = budget;
                cfg.rehash_on_failure = false;        // observe failure, don't repair it
                cuckoo::CuckooTable table(cfg);

                const auto keys = make_keys(S, 0x7E57'0000ULL + seed);   // before the loop

                std::size_t inserted = 0;
                for (std::uint64_t k : keys) {
                    const auto r = table.insert(k, k);
                    if (r == cuckoo::InsertResult::Failed) break;
                    chain[inserted] = static_cast<std::uint32_t>(table.last_eviction_count());
                    ++inserted;
                }
                const double failure_load = static_cast<double>(inserted) / static_cast<double>(S);
                sum += failure_load;
                fails << c.d << ',' << c.b << ',' << S << ',' << budget << ',' << mult << ','
                      << seed << ',' << inserted << ',' << failure_load << '\n';

                // Chain lengths per 0.01 load bin (load *before* each insert).
                const std::size_t bins = 100;
                std::vector<std::vector<std::uint32_t>> per_bin(bins);
                for (std::size_t i = 0; i < inserted; ++i) {
                    const std::size_t bin = std::min(bins - 1, i * bins / S);
                    per_bin[bin].push_back(chain[i]);
                }
                for (std::size_t bin = 0; bin < bins; ++bin) {
                    auto& v = per_bin[bin];
                    if (v.empty()) continue;
                    std::sort(v.begin(), v.end());
                    double mean = 0;
                    for (auto x : v) mean += x;
                    mean /= static_cast<double>(v.size());
                    chains << c.d << ',' << c.b << ',' << S << ',' << budget << ',' << mult << ','
                           << seed << ',' << static_cast<double>(bin) / bins << ',' << v.size() << ','
                           << mean << ',' << percentile(v, 0.50) << ',' << percentile(v, 0.99) << ','
                           << percentile(v, 0.999) << ',' << v.back() << '\n';
                }
            }
            log << "  threshold d=" << c.d << " b=" << c.b << " budget=" << budget
                << " (" << mult << "x): mean failure load "
                << std::setprecision(6) << sum / static_cast<double>(st.threshold_seeds) << "\n";
        }
    }
}

// ===========================================================================
// LATENCY EXPERIMENT
// ===========================================================================
//
// Same slot count S for every structure. At load a, n = a*S keys are inserted.
//   Table:    S slots, rehash disabled, large eviction budget (only lookups
//             are measured, so the budget does not matter beyond letting
//             the build succeed).
//   Baseline: rehash(S) before inserting, so it has S buckets and its own
//             load factor is also a. Pre-sizing this way keeps the bucket
//             count fixed across the sweep and avoids any rehash mid-build;
//             the actual bucket_count is written to the CSV because
//             implementations round it (MSVC to a power of two).
//
// Per trial: a fresh random query order (hits drawn from the inserted keys,
// misses from keys never inserted), one untimed warm-up pass, one timed pass
// with a timestamp pair around every lookup, and one batch pass timed as a
// whole (throughput, no fences) as a cross-check.

struct LatencyRow {
    std::string structure;
    std::size_t d = 0, b = 0;
};

template <class Lookup>
inline void time_lookups(const std::vector<std::uint64_t>& queries, Lookup&& lookup,
                         std::vector<std::uint32_t>& ticks, double& batch_ns_per_op) {
    // Warm-up: same queries, untimed.
    std::uint64_t acc = 0;
    for (std::uint64_t q : queries) acc += lookup(q);
    g_sink = acc;

    // Timed, one lookup at a time.
    for (std::size_t i = 0; i < queries.size(); ++i) {
        const std::uint64_t t0 = tsc_begin();
        g_sink = lookup(queries[i]);
        const std::uint64_t t1 = tsc_end();
        ticks[i] = static_cast<std::uint32_t>(std::min<std::uint64_t>(t1 - t0, 0xFFFFFFFFu));
    }

    // Batch: whole loop timed once; lookups may overlap in the CPU.
    const auto c0 = std::chrono::steady_clock::now();
    acc = 0;
    for (std::uint64_t q : queries) acc += lookup(q);
    g_sink = acc;
    const auto c1 = std::chrono::steady_clock::now();
    batch_ns_per_op = static_cast<double>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(c1 - c0).count()) /
        static_cast<double>(queries.size());
}

inline void run_latency(const Settings& st, double ticks_per_ns, std::ostream& log) {
    namespace fs = std::filesystem;
    std::ofstream out(fs::path(RESULTS_DIR) / "latency.csv");
    out << "structure,d,b,load,n,buckets,trial,kind,p50_ns,p90_ns,p99_ns,p999_ns,max_ns,"
           "mean_ns,batch_mean_ns,mean_probes,max_probes\n";
    out << std::setprecision(9);

    const std::size_t S = st.latency_slots;
    const std::size_t M = st.lookups_per_trial;
    const std::vector<double> loads = {0.10, 0.20, 0.30, 0.40, 0.45, 0.50,
                                       0.60, 0.70, 0.80, 0.90, 0.95};

    struct Structure { std::string name; std::size_t d, b; double max_load; };
    const std::vector<Structure> structures = {
        {"cuckoo_d2b1", 2, 1, 0.45},   // threshold ~0.5: stay below it
        {"cuckoo_d2b4", 2, 4, 0.95},
        {"baseline", 0, 0, 0.95},
    };

    std::vector<std::uint32_t> ticks(M);
    std::vector<std::uint32_t> sorted(M);
    std::vector<std::uint64_t> queries(M);

    for (const Structure& s : structures) {
        for (double load : loads) {
            if (load > s.max_load + 1e-9) continue;
            const std::size_t n = static_cast<std::size_t>(std::llround(load * static_cast<double>(S)));
            const auto keys = make_keys(n, 0xA11CE000ULL + static_cast<std::uint64_t>(load * 1000));
            const auto misses = make_keys(M, 0xB0B00000ULL + static_cast<std::uint64_t>(load * 1000));

            // Build (untimed).
            std::unique_ptr<cuckoo::CuckooTable> table;
            std::unordered_map<std::uint64_t, std::uint64_t> base;
            std::size_t buckets = 0;
            if (s.name == "baseline") {
                base.rehash(S);
                for (auto k : keys) base.emplace(k, k);
                buckets = base.bucket_count();
            } else {
                cuckoo::Config cfg;
                cfg.d = s.d; cfg.b = s.b; cfg.num_buckets = S / s.b;
                cfg.seed = 0xC0C0ULL;
                cfg.max_evictions = 100000;
                cfg.rehash_on_failure = false;
                table = std::make_unique<cuckoo::CuckooTable>(cfg);
                bool ok = true;
                for (auto k : keys)
                    if (table->insert(k, k) == cuckoo::InsertResult::Failed) { ok = false; break; }
                if (!ok) {
                    log << "  latency " << s.name << " load " << load << ": build failed, skipped\n";
                    continue;
                }
                buckets = table->num_buckets();
            }

            // Lookup functions: return something derived from the result so
            // it has to be computed.
            auto look_table = [&](std::uint64_t k) -> std::uint64_t {
                const std::uint64_t* v = table->find(k);
                return v ? *v : 1;
            };
            auto look_base = [&](std::uint64_t k) -> std::uint64_t {
                auto it = base.find(k);
                return it != base.end() ? it->second : 1;
            };
            auto probes = [&](std::uint64_t k) -> std::size_t {
                if (table) return table->probe_count(k);
                std::size_t p = 0;
                const auto bk = base.bucket(k);
                for (auto it = base.begin(bk); it != base.end(bk); ++it) {
                    ++p;
                    if (it->first == k) break;
                }
                return p == 0 ? 1 : p;   // an empty bucket still costs one probe
            };

            for (std::size_t trial = 0; trial < st.trials; ++trial) {
                std::mt19937_64 rng(0x5EED0000ULL + trial);
                for (int kind = 0; kind < 2; ++kind) {   // 0 = hit, 1 = miss
                    for (std::size_t i = 0; i < M; ++i)
                        queries[i] = kind == 0 ? keys[rng() % n] : misses[(i + trial * 7919) % M];

                    double batch_ns = 0;
                    if (table) time_lookups(queries, look_table, ticks, batch_ns);
                    else       time_lookups(queries, look_base, ticks, batch_ns);

                    // Probe counts: separate untimed pass.
                    double probe_sum = 0;
                    std::size_t probe_max = 0;
                    for (std::uint64_t q : queries) {
                        const std::size_t p = probes(q);
                        probe_sum += static_cast<double>(p);
                        probe_max = std::max(probe_max, p);
                    }

                    std::copy(ticks.begin(), ticks.end(), sorted.begin());
                    std::sort(sorted.begin(), sorted.end());
                    double mean = 0;
                    for (auto t : sorted) mean += t;
                    mean /= static_cast<double>(M);
                    auto ns = [&](double t) { return t / ticks_per_ns; };

                    out << s.name << ',' << s.d << ',' << s.b << ',' << load << ',' << n << ','
                        << buckets << ',' << trial << ',' << (kind == 0 ? "hit" : "miss") << ','
                        << ns(percentile(sorted, 0.50)) << ',' << ns(percentile(sorted, 0.90)) << ','
                        << ns(percentile(sorted, 0.99)) << ',' << ns(percentile(sorted, 0.999)) << ','
                        << ns(sorted.back()) << ',' << ns(mean) << ',' << batch_ns << ','
                        << probe_sum / static_cast<double>(M) << ',' << probe_max << '\n';
                }
            }
            log << "  latency " << s.name << " load " << load << " done\n";
        }
    }
}

// ---------------------------------------------------------------------------
// Timer overhead: the same timestamp pair around an empty body.
// ---------------------------------------------------------------------------
inline void timer_overhead(double ticks_per_ns, std::size_t samples, double& p50_ns, double& p99_ns) {
    std::vector<std::uint32_t> t(samples);
    for (std::size_t i = 0; i < samples; ++i) {
        const std::uint64_t a = tsc_begin();
        g_sink = i;
        const std::uint64_t b = tsc_end();
        t[i] = static_cast<std::uint32_t>(b - a);
    }
    std::sort(t.begin(), t.end());
    p50_ns = percentile(t, 0.50) / ticks_per_ns;
    p99_ns = percentile(t, 0.99) / ticks_per_ns;
}

inline int run_all(const Settings& st, bool quick, std::ostream& log) {
    namespace fs = std::filesystem;
    fs::create_directories(RESULTS_DIR);

    if (!release_build() && !quick) {
        log << "Refusing to run the experiments in a Debug build: the timings would be\n"
               "meaningless. Switch the configuration to x64 Release.\n";
        return 2;
    }

    const double tpn = calibrate_tsc();
    double ovh50 = 0, ovh99 = 0;
    timer_overhead(tpn, 1 << 20, ovh50, ovh99);

    {
        std::ofstream info(fs::path(RESULTS_DIR) / "run_info.txt");
        info << std::setprecision(9)
             << "timestamp: " << local_timestamp() << "\n"
             << "cpu: " << cpu_brand() << "\n"
             << "compiler: " << compiler_string() << "\n"
             << "build: " << (release_build() ? "Release (NDEBUG)" : "Debug") << "\n"
             << "mode: " << (quick ? "QUICK SMOKE TEST - not for the report" : "full") << "\n"
             << "tsc_ticks_per_ns: " << tpn << "\n"
             << "timer_overhead_p50_ns: " << ovh50 << "\n"
             << "timer_overhead_p99_ns: " << ovh99 << "\n"
             << "threshold_slots: " << st.threshold_slots << "\n"
             << "threshold_seeds: 1.." << st.threshold_seeds << "\n"
             << "latency_slots: " << st.latency_slots << "\n"
             << "lookups_per_trial: " << st.lookups_per_trial << "\n"
             << "trials: " << st.trials << "\n"
             << "power_plan: FILL IN BY HAND (e.g. High performance, on mains power)\n"
             << "other_load: FILL IN BY HAND (what else was running)\n";
    }
    log << "TSC " << tpn << " ticks/ns; timer overhead p50 " << ovh50 << " ns, p99 " << ovh99 << " ns\n";

    log << "Threshold experiment:\n";
    run_threshold(st, log);
    log << "Latency experiment:\n";
    run_latency(st, tpn, log);
    log << "Results written to " << fs::absolute(RESULTS_DIR).string() << "\n";
    return 0;
}

}  // namespace harness
