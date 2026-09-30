// correctness_check.hpp
// Randomised tests against std::unordered_map, run at the start of main().
// Uses explicit checks instead of assert() because Release builds define NDEBUG.

#pragma once

#include "cuckoo_table.hpp"

#include <cstddef>
#include <cstdint>
#include <ostream>
#include <random>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace cuckoo::test {

namespace detail {

inline std::vector<std::uint64_t> random_keys(std::size_t n, std::uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::vector<std::uint64_t> keys;
    keys.reserve(n);
    std::unordered_set<std::uint64_t> seen;
    while (keys.size() < n) {
        const std::uint64_t k = rng();
        if (k != kEmpty && seen.insert(k).second) keys.push_back(k);
    }
    return keys;
}

// Random inserts, lookups and erases, compared with std::unordered_map after
// every op. Starts from a tiny table with rehash on, so rebuilds and growth
// happen often. Keys come from a small pool so repeats are common.
inline bool differential(const Config& cfg, std::size_t ops, std::uint64_t seed, std::ostream& log) {
    CuckooTable table(cfg);
    std::unordered_map<std::uint64_t, std::uint64_t> ref;
    const auto universe = random_keys(2000, seed ^ 0xABCDEF);
    std::mt19937_64 rng(seed);

    auto fail = [&](const char* what, std::size_t op, std::uint64_t key) {
        log << "  FAIL [differential d=" << cfg.d << " b=" << cfg.b << "] " << what
            << " at op " << op << ", key " << key << "\n";
        return false;
    };

    for (std::size_t op = 0; op < ops; ++op) {
        const std::uint64_t key = universe[rng() % universe.size()];
        const unsigned kind = static_cast<unsigned>(rng() % 10);

        if (kind < 5) {                                   // 50% insert/update
            const std::uint64_t value = rng();
            const bool existed = ref.count(key) != 0;
            const InsertResult r = table.insert(key, value);
            ref[key] = value;
            if (r == InsertResult::Failed) return fail("insert failed with rehash enabled", op, key);
            if (existed != (r == InsertResult::Updated)) return fail("Inserted/Updated mismatch", op, key);
        } else if (kind < 8) {                            // 30% lookup
            const std::uint64_t* v = table.find(key);
            const auto it = ref.find(key);
            if ((v != nullptr) != (it != ref.end())) return fail("presence mismatch", op, key);
            if (v && *v != it->second) return fail("value mismatch", op, key);
        } else {                                          // 20% erase
            if (table.erase(key) != (ref.erase(key) == 1)) return fail("erase mismatch", op, key);
        }

        if (table.size() != ref.size()) return fail("size mismatch", op, key);

        if (op % 1000 == 999) {
            if (!table.check_invariant()) return fail("invariant violated", op, key);
            for (const auto& [k, v] : ref) {
                const std::uint64_t* got = table.find(k);
                if (!got || *got != v) return fail("full sweep mismatch", op, k);
            }
        }
    }
    log << "  ok  differential d=" << cfg.d << " b=" << cfg.b << " (" << ops << " ops, "
        << table.rehash_count() << " rehashes, final buckets " << table.num_buckets() << ")\n";
    return true;
}

// Rehash off: fill a fixed table until an insert fails. Checks the failed
// insert left the table unchanged and that the table still works afterwards.
inline bool failure_path(std::size_t d, std::size_t b, std::uint64_t seed, std::ostream& log) {
    Config cfg;
    cfg.d = d;
    cfg.b = b;
    cfg.num_buckets = 4096 / b;          // 4096 slots for any b
    cfg.seed = seed;
    cfg.rehash_on_failure = false;
    CuckooTable table(cfg);

    const auto keys = random_keys(table.capacity() + 1, seed ^ 0x1234);
    std::size_t inserted = 0;
    std::uint64_t failed_key = kEmpty;
    for (const std::uint64_t k : keys) {
        if (table.insert(k, k * 3 + 1) == InsertResult::Failed) { failed_key = k; break; }
        ++inserted;
    }

    auto fail = [&](const char* what) {
        log << "  FAIL [failure path d=" << d << " b=" << b << "] " << what << "\n";
        return false;
    };

    if (failed_key == kEmpty) return fail("table accepted more keys than it has slots");
    if (table.size() != inserted) return fail("size changed on failed insert");
    if (table.contains(failed_key)) return fail("failed key is present");
    if (!table.check_invariant()) return fail("invariant violated after failure");
    for (std::size_t i = 0; i < inserted; ++i) {
        const std::uint64_t* v = table.find(keys[i]);
        if (!v || *v != keys[i] * 3 + 1) return fail("an existing key was lost or corrupted");
    }

    // Free some space and retry the key that failed.
    for (std::size_t i = 0; i < inserted / 10; ++i) table.erase(keys[i]);
    if (table.insert(failed_key, 7) != InsertResult::Inserted) return fail("retry after erase failed");
    if (!table.check_invariant()) return fail("invariant violated after retry");

    log << "  ok  failure path d=" << d << " b=" << b << " (first failure at load "
        << static_cast<double>(inserted) / static_cast<double>(table.capacity())
        << ", " << table.max_evictions() << "-eviction budget)\n";
    return true;
}

// The reserved key kEmpty must be rejected.
inline bool sentinel(std::ostream& log) {
    CuckooTable table(Config{});
    table.insert(1, 1);
    bool threw = false;
    try { table.insert(kEmpty, 0); } catch (const std::invalid_argument&) { threw = true; }
    if (!threw || table.contains(kEmpty) || table.erase(kEmpty) || table.size() != 1) {
        log << "  FAIL [sentinel] reserved key not handled\n";
        return false;
    }
    log << "  ok  sentinel key rejected\n";
    return true;
}

}  // namespace detail

inline bool run_correctness_checks(std::ostream& log) {
    log << "Correctness checks:\n";
    bool ok = true;

    const std::size_t geometries[][2] = {{2, 1}, {2, 4}, {3, 1}, {4, 2}};
    std::uint64_t seed = 1;
    for (const auto& g : geometries) {
        Config cfg;
        cfg.d = g[0];
        cfg.b = g[1];
        cfg.num_buckets = 4;             // tiny, so rebuilds happen often
        cfg.seed = seed;
        ok = ok && detail::differential(cfg, 200000, seed++, log);
    }
    for (const auto& g : geometries) ok = ok && detail::failure_path(g[0], g[1], seed++, log);
    ok = ok && detail::sentinel(log);

    return ok;
}

}  // namespace cuckoo::test
