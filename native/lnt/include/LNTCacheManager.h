// LNTCacheManager.h - the asynchronous, budgeted, LRU payload cache.
//
// WHAT IS BEING CACHED, AND WHY THAT IS THE INTERESTING PART. A .lnt file is a
// directory of nodes, each owning one contiguous chunk of decoded points. The
// render path wants the payloads of a small subset - the nodes the camera can
// actually see - but decoding is a disk read, and a disk read on the render
// thread is a frame hitch. This class is the boundary: the render thread asks
// for a node and never blocks on the disk; a worker thread does the reading; and
// the resident set is held under a fixed byte budget with least-recently-used
// eviction, so a camera flight over a dataset larger than RAM stays inside a
// known footprint instead of growing until the machine swaps.
//
// THE THREE THINGS A NAIVE VERSION OF THIS GETS WRONG, all of which are handled
// here on purpose.
//
// 1. THE READER IS NOT SHARED. lnt::Reader owns a single std::ifstream and
//    implements a node read as seek-then-read on that one stream, and its
//    last_error_ is a single mutable string. Driving that same Reader from a
//    worker while the main thread also reads is a data race on the file
//    position: two reads interleave and decode points from the WRONG chunk -
//    silently, non-deterministically, and only under load. So the loader opens
//    its OWN Reader on the same path. Two independent streams on one file cost
//    nothing (the directory is 52 bytes per node, loaded once) and remove the
//    race by construction rather than by locking every call. The mutex below
//    protects this class's own state, not the stream.
//
// 2. THE BUDGET IS AN INVARIANT, NOT AN ASPIRATION. Eviction happens BEFORE a
//    payload is admitted, and a payload that cannot be admitted is never
//    inserted - it stays queued. So resident_bytes() <= budget holds at every
//    instant, including inside the loader, and not merely once per frame after
//    a cleanup pass. The alternative (insert, then evict) requires a transient
//    overshoot and makes "never exceeds the limit" unprovable.
//
//    The awkward case is real and is not papered over: while every resident
//    entry is PINNED (the current frame is about to draw it, so evicting it
//    would be a cache miss in the same frame it was fetched), nothing can be
//    freed. The budget wins - the load is deferred and retried when the pin set
//    changes, counted in stats().deferred. A guarantee that holds only when it
//    is convenient is not a guarantee.
//
// 3. A PAYLOAD BIGGER THAN THE BUDGET MUST NOT SPIN. admit() would defer it
//    forever: it can never fit, so the worker would re-queue it on every pass and
//    never make progress. It is rejected once, permanently, as
//    stats().too_large, and reported.
//
// LOCKING. One mutex guards the entry table, the LRU list, the queue and the
// counters. The worker holds it only for bookkeeping - never across the disk
// read - so a load in flight does not stall the render thread's residency
// queries. peek() returns a pointer into an entry's vector, so it is documented
// to be used while the caller holds the pin on that node; eviction is the only
// thing that can invalidate it, and a pinned entry is never evicted.

#ifndef LNT_CACHE_MANAGER_H
#define LNT_CACHE_MANAGER_H

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <list>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "LNTFormat.h"
#include "LNTReader.h"
#include "LNTTypes.h"

namespace lnt {

// 2 GiB of decoded payload. One RawPoint is 32 bytes (3x float64 + 4 uint8), so this
// is ~67 million points resident at once. It was 512 MiB (16.7 M points), which a
// zoom-and-pan over a multi-billion-point file overran: every tile the cut needed
// evicted a tile the next frame needed, so tiles were missing (drawn as holes or at
// LOD density) while the loader thrashed. The budget stays an INVARIANT, not a target:
// resident bytes never exceed it (see admit_locked). Hosts with less RAM, or a
// multi-billion-point file on a workstation with more, pass their own figure
// (kLargeCloudCacheBudgetBytes is the 4 GiB tier).
static constexpr uint64_t kDefaultCacheBudgetBytes = 2048ull * 1024ull * 1024ull;
static constexpr uint64_t kLargeCloudCacheBudgetBytes = 4096ull * 1024ull * 1024ull;

enum class LntResidency : uint8_t {
    Absent = 0,    // not resident, no load in flight
    Queued = 1,    // enqueued, the worker has not picked it up
    Loading = 2,   // the worker is reading it right now
    Resident = 3,  // decoded payload available
    Failed = 4     // unreadable, or too large for the budget. Terminal.
};

inline const char* residency_name(LntResidency r) {
    switch (r) {
        case LntResidency::Absent:   return "absent";
        case LntResidency::Queued:   return "queued";
        case LntResidency::Loading:  return "loading";
        case LntResidency::Resident: return "resident";
        case LntResidency::Failed:   return "failed";
    }
    return "?";
}

struct LntCacheStats {
    uint64_t budget_bytes = 0;
    uint64_t resident_bytes = 0;
    // Peak resident bytes, maintained by the class itself under the lock rather
    // than sampled by the caller. A sampled maximum can miss a transient
    // overshoot; this one cannot.
    uint64_t peak_bytes = 0;
    uint64_t resident_nodes = 0;
    uint64_t loads = 0;            // payloads successfully decoded
    uint64_t load_failures = 0;    // unreadable chunks
    uint64_t evictions = 0;
    uint64_t deferred = 0;         // admissions refused because the budget was all pinned
    uint64_t too_large = 0;        // payloads larger than the entire budget
    uint64_t hits = 0;
    uint64_t misses = 0;
    // Longest a MAIN-THREAD call ever spent waiting for the worker. This is the
    // evidence for "does not block the render thread": it is measured here, at
    // the only place a block could happen, rather than inferred from frame times.
    uint64_t max_main_wait_us = 0;
    uint64_t queued = 0;
    uint64_t loading = 0;

    bool within_budget() const { return peak_bytes <= budget_bytes; }
    std::string describe() const;
};

// ========================================================================
// LntCacheManager
// ========================================================================
class LntCacheManager {
public:
    // The budget is a constructor argument rather than a setter so that the
    // invariant "resident <= budget" cannot be violated by a caller that
    // shrinks the budget under a full cache without an eviction pass. (A caller
    // that genuinely needs to change it can close() and reopen.)
    explicit LntCacheManager(uint64_t budget_bytes = kDefaultCacheBudgetBytes)
        : budget_(budget_bytes) {}

    ~LntCacheManager() { close(); }

    LntCacheManager(const LntCacheManager&) = delete;
    LntCacheManager& operator=(const LntCacheManager&) = delete;

    // Opens the loader's OWN Reader on `lnt_path` and starts the worker thread.
    // Returns false with a reason when the file cannot be opened - in which case
    // no thread is started and every request is reported Failed.
    //
    // `workers` loader threads run in parallel, each with its OWN Reader (note 1 above:
    // a Reader is one seek-then-read stream and cannot be shared), so several tiles are
    // in flight at once - the disk and the decode both overlap. The default of 1 is the
    // original single loader; the interactive viewer asks for more.
    inline bool open(const std::string& lnt_path, std::string& err, unsigned workers = 1) {
        std::lock_guard<std::mutex> lock(mu_);
        if (open_) { err = "cache is already open"; return false; }
        if (budget_ == 0) { err = "cache budget is zero"; return false; }
        if (workers < 1) workers = 1;
        if (workers > 16) workers = 16;
        loaders_.clear();
        for (unsigned i = 0; i < workers; ++i) {
            std::unique_ptr<Reader> r(new Reader());
            if (!r->open(lnt_path)) {
                err = "cache loader could not open '" + lnt_path + "': " + r->error();
                loaders_.clear();
                return false;
            }
            loaders_.push_back(std::move(r));
        }
        path_ = lnt_path;
        stop_ = false;
        open_ = true;
        for (unsigned i = 0; i < workers; ++i)
            workers_.emplace_back([this, i] { worker_main(i); });
        return true;
    }
    unsigned worker_count() const { return unsigned(loaders_.size()); }

    // Stops and joins the worker, and drops every payload. Safe to call twice.
    inline void close() {
        {
            std::lock_guard<std::mutex> lock(mu_);
            if (!open_ && workers_.empty()) { entries_.clear(); lru_.clear(); return; }
            stop_ = true;
            open_ = false;
        }
        work_cv_.notify_all();
        for (std::thread& t : workers_) if (t.joinable()) t.join();
        workers_.clear();
        std::lock_guard<std::mutex> lock(mu_);
        entries_.clear();
        lru_.clear();
        queue_.clear();
        queued_.clear();
        pinned_.clear();
        bytes_ = 0;
        loaders_.clear();
    }

    bool is_open() const {
        std::lock_guard<std::mutex> lock(mu_);
        return open_;
    }

    // ---- main-thread API: none of these touch the disk --------------------

    inline LntResidency residency(uint64_t node_id) const {
        std::lock_guard<std::mutex> lock(mu_);
        const Entry* e = find(node_id);
        return e == nullptr ? LntResidency::Absent : LntResidency(e->state);
    }

    // Enqueues `node_id` for loading unless it is already resident, already
    // queued/loading, or permanently Failed. Never blocks.
    inline void request(uint64_t node_id) {
        std::lock_guard<std::mutex> lock(mu_);
        request_locked(node_id);
    }

    inline void request_all(const std::vector<uint64_t>& ids) {
        std::lock_guard<std::mutex> lock(mu_);
        for (uint64_t id : ids) request_locked(id);
    }

    // The decoded payload, or nullptr when it is not resident. Marks the entry
    // most-recently-used. The pointer is valid until that node is evicted, so a
    // caller that intends to use it must have pinned it (set_pinned) - which is
    // exactly what the streamer does with the current frame's selection.
    inline const std::vector<RawPoint>* peek(uint64_t node_id) {
        std::lock_guard<std::mutex> lock(mu_);
        Entry* e = find(node_id);
        if (e == nullptr || e->state != int(LntResidency::Resident)) return nullptr;
        touch_locked(node_id, e);
        return &e->points;
    }

    // ---- request scheduling -------------------------------------------------
    // Drops every request that is still WAITING in the queue, is in `eligible` (the ids
    // the caller itself requested - it must not cancel anyone else's) and is not in
    // `wanted`.
    // A camera that moved (or zoomed) since a request was made leaves its tiles
    // queued behind the ones the new view needs; loading them would spend the disk on
    // data nobody will draw. A read already in flight is left alone.
    inline uint64_t cancel_unwanted(const std::unordered_set<uint64_t>& wanted,
                                    const std::unordered_set<uint64_t>& eligible) {
        std::lock_guard<std::mutex> lock(mu_);
        std::deque<uint64_t> keep;
        uint64_t dropped = 0;
        for (uint64_t id : queue_) {
            if (wanted.count(id) || !eligible.count(id)) { keep.push_back(id); continue; }
            queued_.erase(id);
            Entry* e = find(id);
            if (e != nullptr && e->state == int(LntResidency::Queued))
                e->state = int(LntResidency::Absent);
            ++dropped;
        }
        queue_.swap(keep);
        return dropped;
    }
    // Reorders the waiting queue so `ids` (in the order given: coarse tiles first) are
    // loaded before everything else. Ids that are not queued are ignored.
    inline void prioritize(const std::vector<uint64_t>& ids) {
        std::lock_guard<std::mutex> lock(mu_);
        if (queue_.size() < 2 || ids.empty()) return;
        std::unordered_map<uint64_t, std::size_t> rank;
        rank.reserve(ids.size() * 2);
        for (std::size_t i = 0; i < ids.size(); ++i) rank.emplace(ids[i], i);
        std::vector<uint64_t> v(queue_.begin(), queue_.end());
        std::stable_sort(v.begin(), v.end(), [&rank](uint64_t a, uint64_t b) {
            const auto ia = rank.find(a), ib = rank.find(b);
            const std::size_t ra = ia == rank.end() ? ~std::size_t(0) : ia->second;
            const std::size_t rb = ib == rank.end() ? ~std::size_t(0) : ib->second;
            return ra < rb;
        });
        queue_.assign(v.begin(), v.end());
    }

    // ---- in-place edits ----------------------------------------------------
    // The file's bytes for a node changed (a classification edit). A resident
    // payload is patched from `fresh` (same point count, same order); a payload
    // that is mid-read may hold the OLD bytes, so it is marked stale and the
    // worker re-reads it instead of admitting it. Queued / absent entries need
    // nothing: they will read the new bytes.
    inline void refresh_node(uint64_t id, const std::vector<RawPoint>& fresh) {
        std::lock_guard<std::mutex> lock(mu_);
        Entry* e = find(id);
        if (e == nullptr) return;
        if (e->state == int(LntResidency::Resident)) {
            if (e->points.size() == fresh.size())
                for (std::size_t i = 0; i < fresh.size(); ++i)
                    e->points[i].classification = fresh[i].classification;
        } else if (e->state == int(LntResidency::Loading)) {
            stale_.insert(id);
        }
    }

    // Class edits on one node's records: applied in place when it is resident (returns
    // true: its tile must be re-uploaded); a node being loaded is re-read afterwards.
    // `span` (optional) receives the classes of records [*span_first, +span.size()): the
    // smallest run covering every record whose class actually changed (empty: none did),
    // so the caller can rewrite just those bytes of the uploaded tile.
    inline bool patch_classes(uint64_t id, const std::vector<std::pair<uint32_t, uint8_t>>& edits,
                              std::vector<uint8_t>* span = nullptr, uint32_t* span_first = nullptr) {
        std::lock_guard<std::mutex> lock(mu_);
        Entry* e = find(id);
        if (e == nullptr) return false;
        if (e->state == int(LntResidency::Resident)) {
            uint32_t lo = UINT32_MAX, hi = 0;
            for (const auto& ed : edits) {
                if (ed.first >= e->points.size()) continue;
                uint8_t& c = e->points[ed.first].classification;
                if (c == ed.second) continue;
                c = ed.second;
                lo = std::min(lo, ed.first);
                hi = std::max(hi, ed.first);
            }
            if (span) {
                span->clear();
                if (lo <= hi) {
                    span->resize(size_t(hi - lo) + 1);
                    for (uint32_t i = lo; i <= hi; ++i) (*span)[i - lo] = e->points[i].classification;
                }
                if (span_first) *span_first = lo <= hi ? lo : 0;
            }
            return true;
        }
        if (e->state == int(LntResidency::Loading)) stale_.insert(id);
        return false;
    }

    // Held (shared) by every loader across a file read. An editor takes it EXCLUSIVELY
    // while it rewrites a chunk so a read can never observe half of the write.
    inline std::shared_mutex& read_mutex() { return read_mu_; }

    // ---- pinning ---------------------------------------------------------

    // Pins exactly this set: entries in `ids` are not evictable, everything else
    // becomes evictable again. Called once per frame with the selection, before
    // it is drawn. Also wakes the worker, because a pin change is what unblocks
    // a deferred admission.
    inline void set_pinned(const std::vector<uint64_t>& ids) {
        {
            std::lock_guard<std::mutex> lock(mu_);
            pinned_.clear();
            for (uint64_t id : ids) pinned_.insert(id);
        }
        work_cv_.notify_all();
    }

    inline void clear_pinned() {
        {
            std::lock_guard<std::mutex> lock(mu_);
            pinned_.clear();
        }
        work_cv_.notify_all();
    }

    // ---- back-pressure --------------------------------------------------

    // While held, the worker does not start new loads. This is a real capability
    // (suspend streaming across a modal operation so the disk is not fighting
    // for something the user is waiting on) and it is also what makes the
    // non-blocking test deterministic: with the loader held, a payload that was
    // requested cannot possibly have arrived, so "the render call returned and
    // did not wait" is a statement about this code and not a race the test won.
    inline void hold(bool suspended) {
        {
            std::lock_guard<std::mutex> lock(mu_);
            held_ = suspended;
        }
        work_cv_.notify_all();
    }
    inline bool held() const {
        std::lock_guard<std::mutex> lock(mu_);
        return held_;
    }

    // Blocks until nothing is Queued or Loading, or the timeout expires.
    // Teardown and tests only - the render path must never call this, and the
    // harness asserts that the flight never needed to.
    inline bool drain(uint64_t timeout_ms = 10000) {
        const auto deadline = std::chrono::steady_clock::now() +
                              std::chrono::milliseconds(timeout_ms);
        std::unique_lock<std::mutex> lock(mu_);
        while (pending_locked() != 0) {
            if (done_cv_.wait_until(lock, deadline) == std::cv_status::timeout)
                return pending_locked() == 0;
        }
        return true;
    }

    // ---- accounting ------------------------------------------------------

    inline LntCacheStats stats() const {
        std::lock_guard<std::mutex> lock(mu_);
        LntCacheStats s;
        s.budget_bytes = budget_;
        s.resident_bytes = bytes_;
        s.peak_bytes = peak_;
        s.resident_nodes = resident_nodes_;
        s.loads = loads_;
        s.load_failures = load_failures_;
        s.evictions = evictions_;
        s.deferred = deferred_;
        s.too_large = too_large_;
        s.hits = hits_;
        s.misses = misses_;
        s.max_main_wait_us = max_main_wait_us_;
        for (const auto& kv : entries_) {
            if (kv.second.state == int(LntResidency::Queued)) ++s.queued;
            else if (kv.second.state == int(LntResidency::Loading)) ++s.loading;
        }
        return s;
    }

    inline uint64_t resident_bytes() const {
        std::lock_guard<std::mutex> lock(mu_);
        return bytes_;
    }
    // Cumulative payloads that finished decoding. One counter under one lock,
    // unlike stats(), which walks the whole entry table to count queued/loading -
    // so the streamer can poll this every frame to notice that a payload a
    // residency gate was waiting for has landed, without paying a table walk for
    // a question about a single number.
    inline uint64_t loads() const {
        std::lock_guard<std::mutex> lock(mu_);
        return loads_;
    }
    // ---- test hook: simulated disk latency -------------------------------
    // The loader sleeps this long BEFORE starting each read, so "the children are
    // still on their way from disk" is something a test can hold open for exactly
    // as long as it wants instead of racing the machine's actual I/O. Only the
    // WORKER sleeps: residency queries and the render thread are unaffected, which
    // is the whole property this file promises and the reason the delay is here
    // rather than in peek() or request(). 0 (the default) does nothing.
    inline void set_load_delay_ms(uint32_t ms) {
        std::lock_guard<std::mutex> lock(mu_);
        load_delay_ms_ = ms;
    }
    inline uint32_t load_delay_ms() const {
        std::lock_guard<std::mutex> lock(mu_);
        return load_delay_ms_;
    }
    inline uint64_t budget_bytes() const { return budget_; }
    inline uint64_t peak_bytes() const {
        std::lock_guard<std::mutex> lock(mu_);
        return peak_;
    }
    inline uint64_t pending() const {
        std::lock_guard<std::mutex> lock(mu_);
        return pending_locked();
    }
    // Bytes the entry table holds for a node whose payload is 5,000 points -
    // used by the harness to size a budget it knows will force eviction.
    static uint64_t payload_bytes(uint64_t points) {
        return points * sizeof(RawPoint);
    }
    const std::string& path() const { return path_; }

private:
    // `in_lru` is a bool rather than a comparison of lru_it against lru_.end():
    // a default-constructed list iterator is singular, and comparing a singular
    // iterator to end() is undefined behaviour, not a false answer. Every entry
    // in the table is created by operator[] without joining the LRU list, so
    // most entries really do carry a singular iterator at some point.
    struct Entry {
        std::vector<RawPoint> points;
        uint64_t bytes = 0;
        int state = int(LntResidency::Absent);
        bool in_lru = false;
        std::list<uint64_t>::iterator lru_it{};
    };

    inline Entry* find(uint64_t id) {
        auto it = entries_.find(id);
        return it == entries_.end() ? nullptr : &it->second;
    }
    inline const Entry* find(uint64_t id) const {
        auto it = entries_.find(id);
        return it == entries_.end() ? nullptr : &it->second;
    }

    inline uint64_t pending_locked() const {
        return queued_.size() + loading_;
    }

    inline void touch_locked(uint64_t id, Entry* e) {
        if (e->in_lru) {
            lru_.erase(e->lru_it);
            e->in_lru = false;
        }
        lru_.push_front(id);
        e->lru_it = lru_.begin();
        e->in_lru = true;
    }

    inline void unlink_lru_locked(Entry* e) {
        if (!e->in_lru) return;
        lru_.erase(e->lru_it);
        e->in_lru = false;
    }

    inline void request_locked(uint64_t id) {
        Entry& e = entries_[id];
        if (e.state == int(LntResidency::Resident)) { ++hits_; return; }
        if (e.state == int(LntResidency::Failed)) return;   // terminal: do not retry
        if (e.state == int(LntResidency::Loading)) return;
        if (e.state == int(LntResidency::Queued) && queued_.count(id)) return;
        ++misses_;
        e.state = int(LntResidency::Queued);
        queue_.push_back(id);
        queued_.insert(id);
        work_cv_.notify_all();
    }

    inline bool is_pinned_locked(uint64_t id) const {
        return pinned_.count(id) != 0;
    }

    // Evicts least-recently-used unpinned entries until `needed` more bytes fit.
    // Returns false when it cannot - the payload must then NOT be inserted.
    //
    // TWO PASSES, deliberately. A single reverse walk that erased as it went
    // would invalidate the very iterator it was about to step, so the victims
    // are chosen first and released afterwards. The choice pass is a pure read
    // of the LRU list and of each entry's pinned flag, which is why it can
    // afford to be a separate loop.
    inline bool admit_locked(uint64_t needed, uint64_t& freed) {
        freed = 0;
        if (needed > budget_) return false;
        std::vector<uint64_t> victims;
        uint64_t reclaimable = 0;
        const uint64_t target = (needed > budget_) ? 0 : budget_ - needed;
        for (auto it = lru_.rbegin(); it != lru_.rend(); ++it) {
            if (bytes_ - reclaimable <= target) break;
            const uint64_t id = *it;
            const Entry* e = find(id);
            if (e == nullptr || e->state != int(LntResidency::Resident)) continue;
            if (is_pinned_locked(id)) continue;
            victims.push_back(id);
            reclaimable += e->bytes;
        }
        for (uint64_t id : victims) {
            Entry* e = find(id);
            if (e == nullptr) continue;
            freed += e->bytes;
            bytes_ -= e->bytes;
            --resident_nodes_;
            std::vector<RawPoint>().swap(e->points);   // release, do not just resize
            e->bytes = 0;
            e->state = int(LntResidency::Absent);
            unlink_lru_locked(e);
            ++evictions_;
        }
        return bytes_ + needed <= budget_;
    }

    inline void worker_main(unsigned idx) {
        Reader& loader = *loaders_[idx];
        for (;;) {
            uint64_t id = 0;
            uint32_t delay_ms = 0;              // test hook, read under mu_ below
            {
                std::unique_lock<std::mutex> lock(mu_);
                work_cv_.wait(lock, [this] {
                    return stop_ || (!held_ && !queue_.empty());
                });
                if (stop_) return;
                if (held_ || queue_.empty()) continue;
                id = queue_.front();
                queue_.pop_front();
                queued_.erase(id);
                Entry* e = find(id);
                if (e == nullptr) continue;
                e->state = int(LntResidency::Loading);
                ++loading_;
                delay_ms = load_delay_ms_;      // read under mu_ ...
                // The disk read happens with the lock RELEASED: the loader has
                // its own Reader, and holding this mutex across a read would
                // stall every residency query on the render thread.
            }   // ... and the sleep below also happens outside it, for the same reason

            // Simulated disk latency. Deliberately outside the lock: a sleep that
            // held mu_ would block every residency() the render thread makes for
            // the whole delay, which is exactly the render-thread wait on I/O that
            // this file promises never to cause.
            if (delay_ms > 0)
                std::this_thread::sleep_for(std::chrono::milliseconds(delay_ms));

            std::vector<RawPoint> pts;
            bool ok = false;
            std::string read_err;
            {
                std::shared_lock<std::shared_mutex> read_guard(read_mu_);
                ok = loader.read_node_points(id, pts);
                if (!ok) read_err = loader.last_error();
            }
            if (!ok) {
                std::lock_guard<std::mutex> lk(mu_);
                last_read_error_ = read_err;
            }

            {
                std::unique_lock<std::mutex> lock(mu_);
                --loading_;
                Entry& e = entries_[id];
                if (ok && stale_.erase(id) != 0) {
                    // The chunk was edited while this read was in flight.
                    e.state = int(LntResidency::Queued);
                    queue_.push_front(id);
                    queued_.insert(id);
                    continue;
                }
                if (!ok) {
                    e.state = int(LntResidency::Failed);
                    ++load_failures_;
                    done_cv_.notify_all();
                    continue;
                }
                const uint64_t needed = uint64_t(pts.size()) * sizeof(RawPoint);
                uint64_t freed = 0;
                if (!admit_locked(needed, freed)) {
                    // Two distinct reasons, and they must not be conflated: a
                    // payload that cannot fit in the whole budget is terminal,
                    // everything else is deferred until the pin set moves.
                    if (needed > budget_) {
                        e.state = int(LntResidency::Failed);
                        ++too_large_;
                        done_cv_.notify_all();
                        continue;
                    }
                    ++deferred_;
                    e.state = int(LntResidency::Queued);
                    queue_.push_back(id);
                    queued_.insert(id);
                    done_cv_.notify_all();
                    // Wait for the pin set to change (or a short timeout, so a
                    // release that races this point is still picked up) rather
                    // than re-queueing and immediately retrying the same
                    // admission in a hot loop. wait_for releases the mutex, so
                    // the render thread is never behind this.
                    work_cv_.wait_for(lock, std::chrono::milliseconds(2));
                    continue;
                }
                e.points = std::move(pts);
                e.bytes = needed;
                bytes_ += needed;
                if (bytes_ > peak_) peak_ = bytes_;
                ++resident_nodes_;
                e.state = int(LntResidency::Resident);
                touch_locked(id, &e);
                ++loads_;
                done_cv_.notify_all();
            }
        }
    }

    mutable std::mutex mu_;
    std::shared_mutex read_mu_;   // loaders hold it shared; a chunk editor holds it exclusive
    std::condition_variable work_cv_;
    std::condition_variable done_cv_;
    std::vector<std::thread> workers_;

    std::unordered_map<uint64_t, Entry> entries_;
    std::list<uint64_t> lru_;          // front = most recently used
    std::deque<uint64_t> queue_;       // FIFO of ids waiting to load
    std::unordered_set<uint64_t> queued_;
    std::unordered_set<uint64_t> pinned_;
    std::unordered_set<uint64_t> stale_;   // edited while Loading: re-read

    std::vector<std::unique_ptr<Reader>> loaders_;   // one OWN reader per worker - see note 1
    std::string last_read_error_;
    std::string path_;
    uint64_t budget_ = kDefaultCacheBudgetBytes;
    uint64_t bytes_ = 0;
    uint64_t peak_ = 0;
    uint64_t resident_nodes_ = 0;
    uint64_t loading_ = 0;
    uint64_t loads_ = 0, load_failures_ = 0, evictions_ = 0, deferred_ = 0,
             too_large_ = 0, hits_ = 0, misses_ = 0, max_main_wait_us_ = 0;
    uint32_t load_delay_ms_ = 0;      // test hook, see set_load_delay_ms()
    bool stop_ = false;
    bool held_ = false;
    bool open_ = false;

public:
    // The loader's last read error, for a failure report.
    std::string last_error() const {
        std::lock_guard<std::mutex> lock(mu_);
        return last_read_error_;
    }
};

inline std::string LntCacheStats::describe() const {
    std::string s;
    s.reserve(256);
    auto mb = [](uint64_t b) { return double(b) / (1024.0 * 1024.0); };
    char buf[256];
    std::snprintf(buf, sizeof(buf),
                  "resident %.1f MiB / %.1f MiB budget (peak %.1f MiB), %llu nodes, "
                  "%llu loads, %llu evictions, %llu deferred, %llu too-large",
                  mb(resident_bytes), mb(budget_bytes), mb(peak_bytes),
                  (unsigned long long)resident_nodes, (unsigned long long)loads,
                  (unsigned long long)evictions, (unsigned long long)deferred,
                  (unsigned long long)too_large);
    s += buf;
    std::snprintf(buf, sizeof(buf), ", hits %llu / misses %llu",
                  (unsigned long long)hits, (unsigned long long)misses);
    s += buf;
    return s;
}

}  // namespace lnt

#endif  // LNT_CACHE_MANAGER_H
