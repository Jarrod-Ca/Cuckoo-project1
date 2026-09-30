// cuckoo_table.hpp
// Cuckoo hash table with d hash functions and b slots per bucket.
// Header-only, C++17.
//
// Slots are stored in one flat array split into num_buckets buckets of b slots.
// Bucket i is slots [i*b, i*b + b). A slot is 16 bytes (key, value) and the
// array is 64-byte aligned, so with b = 4 each bucket is one cache line.
// All d hash functions map into the whole table.
//
// Invariant:
//   (I1) every stored key is in one of its own d candidate buckets
//   (I2) no key is stored twice
// insert() maintains this. find() and erase() rely on it and only check the
// d candidate buckets.
//
// Empty slots have key == kEmpty (UINT64_MAX), so that key can't be stored.

#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <new>
#include <random>
#include <stdexcept>
#include <unordered_set>
#include <utility>
#include <vector>

#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_ARM64))
#include <intrin.h>
#endif

namespace cuckoo {

inline constexpr std::uint64_t kEmpty = std::numeric_limits<std::uint64_t>::max();
inline constexpr std::size_t kMaxD = 8;
inline constexpr std::size_t kNone = std::numeric_limits<std::size_t>::max();

// Simple tabulation hashing. Each of the 8 key bytes indexes its own table of
// 256 random words and the results are XORed. Different seeds give
// independent functions.
class TabulationHash {
public:
    explicit TabulationHash(std::uint64_t seed) {
        std::mt19937_64 rng(seed);
        for (auto& table : tables_)
            for (auto& word : table) word = rng();
    }

    std::uint64_t operator()(std::uint64_t key) const noexcept {
        std::uint64_t h = 0;
        for (std::size_t i = 0; i < 8; ++i) {
            h ^= tables_[i][key & 0xFF];
            key >>= 8;
        }
        return h;
    }

private:
    std::array<std::array<std::uint64_t, 256>, 8> tables_{};
};

// High 64 bits of a * b. Maps a hash into [0, n) as floor(h * n / 2^64),
// which avoids a division on every probe.
inline std::uint64_t mul_high(std::uint64_t a, std::uint64_t b) noexcept {
#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_ARM64))
    return __umulh(a, b);
#elif defined(__SIZEOF_INT128__)
    __extension__ typedef unsigned __int128 u128;   // GCC/Clang only
    return static_cast<std::uint64_t>((static_cast<u128>(a) * b) >> 64);
#else
    const std::uint64_t a_lo = a & 0xFFFFFFFFu, a_hi = a >> 32;
    const std::uint64_t b_lo = b & 0xFFFFFFFFu, b_hi = b >> 32;
    const std::uint64_t p0 = a_lo * b_lo, p1 = a_lo * b_hi;
    const std::uint64_t p2 = a_hi * b_lo, p3 = a_hi * b_hi;
    const std::uint64_t mid = (p0 >> 32) + (p1 & 0xFFFFFFFFu) + (p2 & 0xFFFFFFFFu);
    return p3 + (p1 >> 32) + (p2 >> 32) + (mid >> 32);
#endif
}

struct Slot {
    std::uint64_t key;
    std::uint64_t value;
};

// Fixed-size slot array with 64-byte alignment (std::vector only guarantees 16).
class SlotArray {
public:
    SlotArray() = default;
    explicit SlotArray(std::size_t n)
        : n_(n),
          p_(static_cast<Slot*>(::operator new[](n * sizeof(Slot), std::align_val_t{64}))) {
        for (std::size_t i = 0; i < n_; ++i) p_[i] = Slot{kEmpty, 0};
    }
    Slot& operator[](std::size_t i) noexcept { return p_[i]; }
    const Slot& operator[](std::size_t i) const noexcept { return p_[i]; }
    std::size_t size() const noexcept { return n_; }

private:
    struct Free {
        void operator()(Slot* p) const noexcept { ::operator delete[](p, std::align_val_t{64}); }
    };
    std::size_t n_ = 0;
    std::unique_ptr<Slot[], Free> p_;
};

struct Config {
    std::size_t d = 2;                // hash functions (2..kMaxD)
    std::size_t b = 1;                // slots per bucket (>= 1)
    std::size_t num_buckets = 1024;   // initial bucket count (>= 1)
    std::uint64_t seed = 0x5EEDC0FFEEULL;
    std::size_t max_evictions = 0;    // 0 = default, see set_eviction_limit()
    bool rehash_on_failure = true;    // false: a failed insert returns Failed
};

enum class InsertResult { Inserted, Updated, Failed };

class CuckooTable {
public:
    explicit CuckooTable(const Config& cfg)
        : cfg_(cfg), num_buckets_(cfg.num_buckets), seed_rng_(cfg.seed) {
        if (cfg_.d < 2 || cfg_.d > kMaxD) throw std::invalid_argument("d must be in [2, 8]");
        if (cfg_.b < 1) throw std::invalid_argument("b must be >= 1");
        if (num_buckets_ < 1) throw std::invalid_argument("num_buckets must be >= 1");
        walk_state_ = seed_rng_();
        reseed();
        slots_ = SlotArray(num_buckets_ * cfg_.b);
        set_eviction_limit();
    }

    // Inserts the key, or updates its value if it's already present.
    // Returns Failed only when rehash_on_failure is false, and in that case
    // the table is left unchanged.
    InsertResult insert(std::uint64_t key, std::uint64_t value) {
        if (key == kEmpty) throw std::invalid_argument("key UINT64_MAX is reserved");

        // Check for an existing copy first so the key is never stored twice (I2).
        if (Slot* s = find_slot(key)) {
            s->value = value;
            last_evictions_ = 0;
            return InsertResult::Updated;
        }

        if (try_place(key, value)) {
            ++size_;
            return InsertResult::Inserted;
        }

        // try_place rolled back, so the table is unchanged.
        if (!cfg_.rehash_on_failure) return InsertResult::Failed;

        const std::size_t failed_walk = last_evictions_;
        rebuild_including(key, value);
        last_evictions_ = failed_walk;   // report the failed walk, not the rebuild
        ++size_;
        return InsertResult::Inserted;
    }

    // Returns a pointer to the value, or nullptr. Checks at most d*b slots.
    const std::uint64_t* find(std::uint64_t key) const noexcept {
        const Slot* s = const_cast<CuckooTable*>(this)->find_slot(key);
        return s ? &s->value : nullptr;
    }

    bool contains(std::uint64_t key) const noexcept { return find(key) != nullptr; }

    // No tombstones needed: emptying a slot can't break (I1) for other keys.
    bool erase(std::uint64_t key) noexcept {
        Slot* s = find_slot(key);
        if (!s) return false;
        s->key = kEmpty;
        s->value = 0;
        --size_;
        return true;
    }

    std::size_t size() const noexcept { return size_; }
    std::size_t capacity() const noexcept { return slots_.size(); }   // slots, not buckets
    std::size_t num_buckets() const noexcept { return num_buckets_; }
    double load_factor() const noexcept {
        return static_cast<double>(size_) / static_cast<double>(slots_.size());
    }
    std::size_t max_evictions() const noexcept { return max_evictions_; }
    std::size_t rehash_count() const noexcept { return rehashes_; }

    // Evictions done by the last insert: 0 for an update, max_evictions() for
    // a failed insert, and the failed walk's length if the insert caused a rebuild.
    std::size_t last_eviction_count() const noexcept { return last_evictions_; }

    // Checks (I1), (I2) and the size counter. O(capacity), for tests.
    bool check_invariant() const {
        std::unordered_set<std::uint64_t> seen;
        std::size_t count = 0;
        for (std::size_t idx = 0; idx < slots_.size(); ++idx) {
            const std::uint64_t k = slots_[idx].key;
            if (k == kEmpty) continue;
            ++count;
            const std::size_t here = idx / cfg_.b;
            bool in_candidate = false;
            for (std::size_t i = 0; i < cfg_.d; ++i)
                if (bucket_of(i, k) == here) in_candidate = true;
            if (!in_candidate) return false;              // (I1)
            if (!seen.insert(k).second) return false;     // (I2)
        }
        return count == size_;
    }

private:
    std::size_t bucket_of(std::size_t i, std::uint64_t key) const noexcept {
        return static_cast<std::size_t>(mul_high(hashes_[i](key), num_buckets_));
    }

    Slot* find_slot(std::uint64_t key) noexcept {
        // kEmpty would match any empty slot, so reject it here.
        if (key == kEmpty) return nullptr;
        for (std::size_t i = 0; i < cfg_.d; ++i) {
            Slot* bucket = &slots_[bucket_of(i, key) * cfg_.b];
            for (std::size_t s = 0; s < cfg_.b; ++s)
                if (bucket[s].key == key) return &bucket[s];
        }
        return nullptr;
    }

    // Eviction walk. Places a key that isn't already in the table.
    // Returns false on failure, with the table restored to how it was.
    //
    // We always hold one item, `carried` (initially the new key). If one of
    // its candidate buckets has a free slot, it goes there and we're done.
    // Otherwise it's swapped into an occupied slot in one of its candidate
    // buckets and the displaced item becomes `carried`. Items only ever move
    // into their own candidate buckets, so (I1) holds throughout.
    //
    // For d = 2, b = 1 this is the standard algorithm. For larger d or b the
    // victim is picked at random (random-walk insertion).
    bool try_place(std::uint64_t key, std::uint64_t value) {
        Slot carried{key, value};
        std::size_t came_from = kNone;   // bucket `carried` was just evicted from
        path_.clear();                   // slots swapped through, for rollback

        for (;;) {
            // Candidate buckets, skipping the one we just came from. It's full
            // (the previous item was just put there), and choosing it would
            // just swap the two items back and forth. For d = 2 this means the
            // key always moves to its other bucket.
            std::size_t cand[kMaxD];
            std::size_t nc = 0;
            for (std::size_t i = 0; i < cfg_.d; ++i) {
                const std::size_t bk = bucket_of(i, carried.key);
                if (bk == came_from) continue;
                bool dup = false;   // two hash functions can give the same bucket
                for (std::size_t j = 0; j < nc; ++j) dup |= (cand[j] == bk);
                if (!dup) cand[nc++] = bk;
            }

            // Use a free slot if there is one.
            for (std::size_t j = 0; j < nc; ++j) {
                Slot* bucket = &slots_[cand[j] * cfg_.b];
                for (std::size_t s = 0; s < cfg_.b; ++s) {
                    if (bucket[s].key == kEmpty) {
                        bucket[s] = carried;
                        last_evictions_ = path_.size();
                        return true;
                    }
                }
            }

            // Out of budget. The budget also stops cycles from looping forever.
            // Undo the swaps in reverse order: this restores every slot and
            // puts the original key back in `carried`.
           /* if (path_.size() == max_evictions_) {
                for (auto it = path_.rbegin(); it != path_.rend(); ++it)
                    std::swap(carried, slots_[*it]);
                last_evictions_ = path_.size();
                return false;
            }*/

            // If every hash of `carried` pointed at came_from, allow it anyway.
            if (nc == 0) cand[nc++] = came_from;

            const std::size_t victim_bucket = (nc == 1) ? cand[0] : cand[next_rand() % nc];
            const std::size_t victim_slot =
                victim_bucket * cfg_.b + (cfg_.b == 1 ? 0 : next_rand() % cfg_.b);

            // `carried` goes into its own candidate bucket; the displaced item
            // is now carried.
            std::swap(carried, slots_[victim_slot]);
            path_.push_back(victim_slot);   // reserved to max_evictions_, no allocation
            came_from = victim_bucket;
        }
    }

    // Rebuild with every item plus the one that failed. First try the same
    // size with new hash functions, then double the bucket count on each
    // later attempt. Growth only happens on failure.
    void rebuild_including(std::uint64_t key, std::uint64_t value) {
        std::vector<Slot> items;
        items.reserve(size_ + 1);
        for (std::size_t i = 0; i < slots_.size(); ++i)
            if (slots_[i].key != kEmpty) items.push_back(slots_[i]);
        items.push_back(Slot{key, value});

        for (std::size_t attempt = 0;; ++attempt) {
            ++rehashes_;
            if (attempt > 0) num_buckets_ *= 2;
            reseed();
            slots_ = SlotArray(num_buckets_ * cfg_.b);
            set_eviction_limit();

            bool ok = true;
            for (const Slot& it : items) {
                if (!try_place(it.key, it.value)) { ok = false; break; }
            }
            if (ok) return;
        }
    }

    void reseed() {
        hashes_.clear();
        hashes_.reserve(cfg_.d);
        for (std::size_t i = 0; i < cfg_.d; ++i) hashes_.emplace_back(seed_rng_());
    }

    // Default budget: max(64, 16 * ceil(log2(slots))).
    void set_eviction_limit() {
        if (cfg_.max_evictions != 0) {
            max_evictions_ = cfg_.max_evictions;
        } else {
            std::size_t lg = 0;
            while ((std::size_t{1} << lg) < slots_.size()) ++lg;
            max_evictions_ = std::max<std::size_t>(64, 16 * lg);
        }
        path_.reserve(max_evictions_);
    }

    // splitmix64, used to pick victims.
    std::uint64_t next_rand() noexcept {
        std::uint64_t z = (walk_state_ += 0x9E3779B97F4A7C15ULL);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        return z ^ (z >> 31);
    }

    Config cfg_;
    std::size_t num_buckets_;
    std::mt19937_64 seed_rng_;        // all hash seeds come from here, so runs are reproducible
    std::uint64_t walk_state_ = 0;
    std::vector<TabulationHash> hashes_;
    SlotArray slots_;
    std::size_t size_ = 0;
    std::size_t max_evictions_ = 0;
    std::vector<std::size_t> path_;
    std::size_t last_evictions_ = 0;
    std::size_t rehashes_ = 0;
};

}  // namespace cuckoo
