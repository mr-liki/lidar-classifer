#pragma once
// LNTSurfaceArchive.h - the persistent surface LOD archive stored after the .lnt node
// directory (derived data only; the v2 header, point chunks and directory are never
// written by this file).
//
//   [original .lnt bytes .......... base_end)
//   [tile record][tile record]...[index segment][footer]      <- commit 1
//   [tile record]...[index segment][footer]                    <- commit 2 (appended)
//
// * A tile record is an independently readable, CRC32C-checked, optionally zstd
//   compressed surface tile (LNTSurfaceLod.h defines its payload).
// * An index segment lists the records of its commit and chains to the previous
//   segment by offset AND SHA-256, so a reader walks the whole index from the footer.
//   Every 64 commits a snapshot segment (the complete index) restarts the chain.
// * The footer is the last 448 bytes of the file. It carries the SHA-256 of every
//   original byte [0, base_end) (source identity), of the header + directory, of the
//   latest segment, up to eight geometry-settings digests (one per class-visibility
//   configuration), and a commit stamp (file size + last-write time set at commit).
//   An unchanged stamp proves no other writer touched the file since the commit, so a
//   reopen need not re-hash the original points; any mismatch re-hashes them.
//
// Writes only ever append (or truncate back to base_end on a reset), always after
// base_end, under an exclusive byte-range lock. An interrupted commit leaves a torn
// tail after the last good footer: readers find that footer by a bounded backward scan
// and the next writer truncates the torn bytes. The original points stay readable in
// every case, because nothing before base_end is ever opened for writing here.
#include "LNTEmbeddedSurface.h"
#include <windows.h>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <vector>

namespace lnt { namespace surface_archive {
using embedded_surface::Digest;

constexpr uint32_t kMaxSlots = 8;
constexpr uint32_t kSnapshotEvery = 64;
constexpr uint64_t kRecoveryScanBytes = 64ull << 20;

struct BlobHeader {
    char magic[4] = {'L', 'N', 'T', 'T'};
    uint32_t codec = 0;          // 0 raw, 1 zstd
    uint64_t key = 0;
    uint64_t raw = 0, stored = 0;
    uint32_t crc_raw = 0, crc_stored = 0;
    Digest settings{};           // the slot digest this tile was built for
};
static_assert(sizeof(BlobHeader) == 72, "tile record header layout changed");

struct IndexEntry {
    uint64_t key = 0, offset = 0;          // offset of the BlobHeader
    uint64_t raw = 0, stored = 0;
    uint64_t source_points = 0;            // original leaf records inside the tile cell
    uint32_t crc_raw = 0, crc_stored = 0;
    uint32_t vertices = 0, triangles = 0;  // triangles: owned + fringe
    uint32_t coverage = 0, represented = 0;
    uint32_t codec = 0;
    uint16_t slot = 0, flags = 0;          // flags bit0: exact (full resolution)
    double bmin[3]{}, bmax[3]{};
};
static_assert(sizeof(IndexEntry) == 120, "index entry layout changed");

struct SegmentHeader {
    char magic[8] = {'L', 'N', 'T', 'S', 'E', 'G', '1', '\0'};
    uint64_t prev_offset = 0, prev_bytes = 0;
    uint64_t count = 0;
    uint32_t snapshot = 0, reserved = 0;
    Digest prev_hash{};
};
static_assert(sizeof(SegmentHeader) == 72, "index segment header layout changed");

struct Footer {
    char magic[8] = {'L', 'N', 'T', 'S', 'L', 'A', '1', '\0'};
    uint64_t base_end = 0;
    uint64_t segment_offset = 0, segment_bytes = 0;
    uint64_t tile_count = 0;
    uint64_t stamp_size = 0;     // file size right after this commit (== footer end)
    int64_t stamp_mtime = 0;     // FILETIME written by this commit
    uint32_t chain_length = 0, slot_count = 0;
    Digest source_hash{}, dir_hash{}, segment_hash{};
    Digest slots[kMaxSlots]{};
    Digest footer_hash{};
};
static_assert(sizeof(Footer) == 448 && offsetof(Footer, footer_hash) == 416, "archive footer layout changed");

inline std::wstring widen(const std::string& path) {
    for (UINT cp : {UINT(CP_UTF8), UINT(CP_ACP)}) {
        const DWORD flags = cp == CP_UTF8 ? MB_ERR_INVALID_CHARS : 0;
        const int n = MultiByteToWideChar(cp, flags, path.data(), int(path.size()), nullptr, 0);
        if (n <= 0) continue;
        std::wstring w(size_t(n), L'\0');
        MultiByteToWideChar(cp, flags, path.data(), int(path.size()), &w[0], n);
        return w;
    }
    return std::wstring(path.begin(), path.end());
}

// "what: <system message> (code N)" for the calling thread's last Win32 error.
inline std::string win_error(const char* what, DWORD code = GetLastError()) {
    char buf[512] = {};
    DWORD n = FormatMessageA(FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS, nullptr,
                             code, MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT), buf, sizeof(buf), nullptr);
    while (n && (buf[n - 1] == '\r' || buf[n - 1] == '\n' || buf[n - 1] == ' ' || buf[n - 1] == '.')) buf[--n] = 0;
    return std::string(what) + ": " + (n ? buf : "unknown error") + " (code " + std::to_string(code) + ")";
}

inline uint64_t pad8(uint64_t v) { return (v + 7) & ~uint64_t(7); }

inline uint32_t crc32c(const void* data, size_t bytes) {
    embedded_surface::BodyHash h;
    h.add(data, bytes);
    const auto d = h.result();
    uint32_t c;
    std::memcpy(&c, d.data(), 4);
    return c;
}

inline Digest footer_digest(const Footer& f) {
    embedded_surface::Hash h;
    h.add(&f, offsetof(Footer, footer_hash));
    return h.result();
}

class File {
public:
    HANDLE h = INVALID_HANDLE_VALUE;
    File() = default;
    File(const File&) = delete;
    File& operator=(const File&) = delete;
    ~File() { close(); }
    void close() { if (h != INVALID_HANDLE_VALUE) CloseHandle(h); h = INVALID_HANDLE_VALUE; }
    bool open_read(const std::string& path, std::string* err) {
        close();
        h = CreateFileW(widen(path).c_str(), GENERIC_READ,
                        FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr,
                        OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
        if (h == INVALID_HANDLE_VALUE && err) *err = win_error("cannot open .lnt for reading");
        return h != INVALID_HANDLE_VALUE;
    }
    bool open_write(const std::string& path, std::string* err) {
        close();
        h = CreateFileW(widen(path).c_str(), GENERIC_READ | GENERIC_WRITE,
                        FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr, OPEN_EXISTING,
                        FILE_ATTRIBUTE_NORMAL, nullptr);
        if (h == INVALID_HANDLE_VALUE && err) *err = win_error("cannot open .lnt for writing the surface archive");
        return h != INVALID_HANDLE_VALUE;
    }
    uint64_t size() const {
        LARGE_INTEGER s{};
        return GetFileSizeEx(h, &s) ? uint64_t(s.QuadPart) : 0;
    }
    // Positional read: independent of any file pointer, safe on a private handle.
    bool read_at(uint64_t off, void* dst, size_t n) const {
        auto p = static_cast<char*>(dst);
        while (n) {
            OVERLAPPED o{};
            o.Offset = DWORD(off);
            o.OffsetHigh = DWORD(off >> 32);
            DWORD want = DWORD(std::min<size_t>(n, 16u << 20)), got = 0;
            if (!ReadFile(h, p, want, &got, &o) || got != want) return false;
            p += got; off += got; n -= got;
        }
        return true;
    }
    bool write_at(uint64_t off, const void* src, size_t n, std::string* err) {
        auto p = static_cast<const char*>(src);
        while (n) {
            OVERLAPPED o{};
            o.Offset = DWORD(off);
            o.OffsetHigh = DWORD(off >> 32);
            DWORD want = DWORD(std::min<size_t>(n, 8u << 20)), put = 0;
            if (!WriteFile(h, p, want, &put, &o) || put != want) {
                if (err) *err = win_error("surface archive write failed");
                return false;
            }
            p += put; off += put; n -= put;
        }
        return true;
    }
    bool truncate(uint64_t size, std::string* err) {
        LARGE_INTEGER pos{};
        pos.QuadPart = LONGLONG(size);
        if (!SetFilePointerEx(h, pos, nullptr, FILE_BEGIN) || !SetEndOfFile(h)) {
            if (err) *err = win_error("surface archive truncate failed");
            return false;
        }
        return true;
    }
    int64_t mtime() const {
        FILETIME w{};
        if (!GetFileTime(h, nullptr, nullptr, &w)) return 0;
        return int64_t((uint64_t(w.dwHighDateTime) << 32) | w.dwLowDateTime);
    }
};

// SHA-256 of the 64-byte header plus the node directory: a cheap identity check that
// always runs, independent of the commit stamp.
inline bool directory_digest(const File& f, uint64_t index_offset, uint64_t base_end, Digest& out) {
    if (base_end < index_offset || index_offset < 64) return false;
    std::vector<char> b(static_cast<size_t>(64 + (base_end - index_offset)));
    if (!f.read_at(0, b.data(), 64) || !f.read_at(index_offset, b.data() + 64, size_t(base_end - index_offset))) return false;
    embedded_surface::Hash h;
    h.add(b.data(), b.size());
    out = h.result();
    return true;
}

// SHA-256 of every original byte [0, base_end): points, header and directory.
inline bool source_digest(const File& f, uint64_t base_end, Digest& out, const std::atomic<bool>* cancel = nullptr) {
    try {
        embedded_surface::Hash h;
        std::vector<char> buffer(8u << 20);
        for (uint64_t off = 0; off < base_end;) {
            if (cancel && cancel->load()) return false;
            const size_t n = size_t(std::min<uint64_t>(base_end - off, buffer.size()));
            if (!f.read_at(off, buffer.data(), n)) return false;
            h.add(buffer.data(), n);
            off += n;
        }
        out = h.result();
        return true;
    } catch (...) { return false; }
}

struct State {
    bool present = false;        // a valid archive footer was found
    bool recovered = false;      // ...but not at the file end (torn tail after it)
    bool legacy_tail = false;    // an LNTSHD1/2/3 monolithic mesh occupies the tail
    uint64_t file_size = 0;
    uint64_t footer_end = 0;     // offset right after the valid footer
    int64_t mtime = 0;           // file last-write time when read
    Footer footer;
    std::vector<IndexEntry> entries;   // all slots; a later commit's entry wins per (slot,key)
    std::string note;            // why no archive was used
    bool stamp_ok() const {
        return present && !recovered && footer.stamp_size == file_size && footer.stamp_mtime == mtime;
    }
    int slot_of(const Digest& settings) const {
        for (uint32_t i = 0; i < footer.slot_count && i < kMaxSlots; ++i)
            if (footer.slots[i] == settings) return int(i);
        return -1;
    }
};

inline bool valid_footer_at(const File& f, uint64_t pos, uint64_t base_end, Footer& out) {
    if (pos < base_end) return false;
    Footer ft;
    if (!f.read_at(pos, &ft, sizeof(ft))) return false;
    if (std::memcmp(ft.magic, "LNTSLA1\0", 8) != 0 || ft.base_end != base_end) return false;
    if (ft.stamp_size != pos + sizeof(Footer) || ft.slot_count > kMaxSlots) return false;
    if (ft.segment_offset < base_end || ft.segment_offset + ft.segment_bytes > pos) return false;
    if (footer_digest(ft) != ft.footer_hash) return false;
    out = ft;
    return true;
}

// Parses the archive at the end of the file. Never throws, never writes.
inline State read_state(const File& f, uint64_t base_end, const Digest& dir_hash) {
    State s;
    s.file_size = f.size();
    s.mtime = f.mtime();
    if (s.file_size <= base_end) { s.note = "no derived data after the node directory"; return s; }
    const uint64_t tail = s.file_size - base_end;
    Footer ft;
    bool found = tail >= sizeof(Footer) && valid_footer_at(f, s.file_size - sizeof(Footer), base_end, ft);
    if (found) s.footer_end = s.file_size;
    if (!found && tail >= sizeof(embedded_surface::Footer)) {
        char magic[8] = {};
        if (f.read_at(s.file_size - sizeof(embedded_surface::Footer), magic, 8) &&
            std::memcmp(magic, "LNTSHD", 6) == 0) {
            s.legacy_tail = true;
            s.note = "legacy whole-file shading mesh in the tail";
            return s;
        }
    }
    if (!found) {
        // Torn commit: find the last good footer. Every record and segment end is padded
        // to an absolute multiple of 8 bytes, so every footer starts on that grid.
        const uint64_t scan = std::min(tail, kRecoveryScanBytes);
        std::vector<char> buf(static_cast<size_t>(scan));
        const uint64_t start = s.file_size - scan;
        if (scan >= sizeof(Footer) && f.read_at(start, buf.data(), buf.size())) {
            for (uint64_t pos = (s.file_size - sizeof(Footer)) & ~uint64_t(7); pos >= start && pos >= base_end; pos -= 8) {
                if (std::memcmp(buf.data() + (pos - start), "LNTSLA1\0", 8) == 0 &&
                    valid_footer_at(f, pos, base_end, ft)) {
                    found = true;
                    s.recovered = true;
                    s.footer_end = pos + sizeof(Footer);
                    break;
                }
            }
        }
        if (!found) { s.note = "no valid surface archive footer (torn or foreign tail)"; return s; }
    }
    if (ft.dir_hash != dir_hash) { s.note = "surface archive belongs to a different node directory"; return s; }
    // Walk the segment chain from the newest to the snapshot/first segment.
    std::vector<std::vector<IndexEntry>> chain;
    uint64_t off = ft.segment_offset, bytes = ft.segment_bytes;
    Digest expect = ft.segment_hash;
    const uint64_t limit = s.footer_end - sizeof(Footer);
    for (uint32_t guard = 0; guard < 4096; ++guard) {
        if (bytes < sizeof(SegmentHeader) || off < base_end || off + bytes > limit) { s.note = "surface archive index out of range"; return s; }
        std::vector<char> seg(static_cast<size_t>(bytes));
        if (!f.read_at(off, seg.data(), seg.size())) { s.note = "surface archive index unreadable"; return s; }
        embedded_surface::Hash h;
        h.add(seg.data(), seg.size());
        if (h.result() != expect) { s.note = "surface archive index checksum mismatch"; return s; }
        SegmentHeader sh;
        std::memcpy(&sh, seg.data(), sizeof(sh));
        if (std::memcmp(sh.magic, "LNTSEG1\0", 8) != 0 || sh.count != (bytes - sizeof(sh)) / sizeof(IndexEntry) ||
            sizeof(sh) + sh.count * sizeof(IndexEntry) != bytes) { s.note = "surface archive index malformed"; return s; }
        std::vector<IndexEntry> e(size_t(sh.count));
        if (sh.count) std::memcpy(e.data(), seg.data() + sizeof(sh), size_t(sh.count) * sizeof(IndexEntry));
        for (const auto& x : e)
            if (x.offset < base_end || x.offset + sizeof(BlobHeader) + x.stored > limit || x.slot >= ft.slot_count ||
                x.codec > 1 || (x.codec == 0 && x.raw != x.stored)) { s.note = "surface archive record out of range"; return s; }
        chain.push_back(std::move(e));
        if (sh.snapshot || sh.prev_offset == 0) break;
        off = sh.prev_offset; bytes = sh.prev_bytes; expect = sh.prev_hash;
    }
    std::map<std::pair<uint16_t, uint64_t>, IndexEntry> merged;
    for (auto it = chain.rbegin(); it != chain.rend(); ++it)       // oldest first: newer wins
        for (const auto& e : *it) merged[{e.slot, e.key}] = e;
    s.entries.reserve(merged.size());
    for (auto& kv : merged) s.entries.push_back(kv.second);
    s.footer = ft;
    s.present = true;
    return s;
}

// Reads one record and checks it against its index entry (key, sizes, both CRCs).
inline bool read_record(const File& f, const IndexEntry& e, const Digest& settings,
                        std::vector<char>& raw, embedded_surface::MeshCodec& codec, std::string* err) {
    BlobHeader bh;
    if (!f.read_at(e.offset, &bh, sizeof(bh))) { if (err) *err = "record unreadable"; return false; }
    if (std::memcmp(bh.magic, "LNTT", 4) != 0 || bh.key != e.key || bh.raw != e.raw || bh.stored != e.stored ||
        bh.codec != e.codec || bh.crc_raw != e.crc_raw || bh.crc_stored != e.crc_stored || bh.settings != settings) {
        if (err) *err = "record header mismatch";
        return false;
    }
    if (e.raw > (512ull << 20) || e.stored > (512ull << 20)) { if (err) *err = "record too large"; return false; }
    std::vector<char> stored(static_cast<size_t>(e.stored));
    if (!f.read_at(e.offset + sizeof(bh), stored.data(), stored.size())) { if (err) *err = "record unreadable"; return false; }
    if (crc32c(stored.data(), stored.size()) != e.crc_stored) { if (err) *err = "record checksum mismatch"; return false; }
    try {
        if (e.codec == 1) {
            raw.resize(size_t(e.raw));
            codec.decode(raw.data(), raw.size(), stored.data(), stored.size());
        } else raw.swap(stored);
    } catch (const std::exception& ex) { if (err) *err = ex.what(); return false; }
    if (crc32c(raw.data(), raw.size()) != e.crc_raw) { if (err) *err = "decoded record checksum mismatch"; return false; }
    return true;
}

// One encoded tile waiting for a commit.
struct PendingRecord {
    Digest settings{};
    IndexEntry meta;          // everything but offset/slot
    std::vector<char> stored; // record body (compressed or raw)
};

struct CommitResult {
    bool ok = false;
    bool source_changed = false;   // another writer touched the file: re-hash before trusting it
    bool permanent = false;        // read-only / access denied: stop trying this session
    std::string error;
    State state;                   // the archive as it stands after the commit
    int64_t mtime = 0;
};

// Appends `records` (all for settings already resolved to slots inside) as one commit.
// `expected_mtime` is the last-write time at which `source_hash` was established (or
// of this process's previous commit); a different time means some other writer touched
// the file and the commit is refused (source_changed) rather than stamping over it.
// `max_tail_bytes` bounds the derived tail: a commit that would grow it beyond that
// restarts the archive with just these records (0 = no bound).
inline CommitResult commit(const std::string& path, uint64_t base_end, uint64_t index_offset,
                           const Digest& source_hash, int64_t expected_mtime,
                           std::vector<PendingRecord>& records, uint64_t max_tail_bytes = 0) {
    CommitResult r;
    File f;
    if (!f.open_write(path, &r.error)) {
        const DWORD code = GetLastError();
        r.permanent = code == ERROR_ACCESS_DENIED || code == ERROR_WRITE_PROTECT || code == ERROR_FILE_NOT_FOUND ||
                      code == ERROR_PATH_NOT_FOUND || code == ERROR_BAD_NETPATH;
        return r;
    }
    OVERLAPPED lock{};
    lock.Offset = DWORD(base_end);
    lock.OffsetHigh = DWORD(base_end >> 32);
    if (!LockFileEx(f.h, LOCKFILE_EXCLUSIVE_LOCK | LOCKFILE_FAIL_IMMEDIATELY, 0, MAXDWORD, 0x7fffffff, &lock)) {
        r.error = win_error("surface archive is locked by another reader or writer");
        return r;
    }
    struct Unlock { HANDLE h; OVERLAPPED* o; ~Unlock() { UnlockFileEx(h, 0, MAXDWORD, 0x7fffffff, o); } } unlock{f.h, &lock};
    Digest dir{};
    if (!directory_digest(f, index_offset, base_end, dir)) { r.error = "cannot read the .lnt directory"; return r; }
    const int64_t now_mtime = f.mtime();
    if (expected_mtime != 0 && now_mtime != expected_mtime) {
        r.source_changed = true;
        r.error = "the .lnt file was modified by another writer; original data will be re-verified";
        return r;
    }
    State st = read_state(f, base_end, dir);
    bool reset = !st.present || st.footer.source_hash != source_hash;
    Footer ft = reset ? Footer{} : st.footer;
    std::vector<IndexEntry> all = reset ? std::vector<IndexEntry>{} : st.entries;
    // Resolve each record's slot; more than kMaxSlots configurations restarts the archive.
    auto slot_for = [&](const Digest& d) -> int {
        for (uint32_t i = 0; i < ft.slot_count; ++i) if (ft.slots[i] == d) return int(i);
        if (ft.slot_count >= kMaxSlots) return -1;
        ft.slots[ft.slot_count] = d;
        return int(ft.slot_count++);
    };
    for (auto& rec : records) {
        if (slot_for(rec.settings) < 0) { reset = true; break; }
    }
    if (reset) {
        ft = Footer{};
        all.clear();
        for (auto& rec : records) if (slot_for(rec.settings) < 0) { r.error = "too many surface configurations in one commit"; return r; }
    }
    if (!reset && max_tail_bytes) {
        uint64_t add = 4096;
        for (const auto& rec : records) add += sizeof(BlobHeader) + rec.stored.size() + 8 + sizeof(IndexEntry);
        if (st.footer_end - base_end + add > max_tail_bytes) {
            std::fprintf(stderr, "[LNT surface] surface archive reached its %.0f MB bound; restarting it\n", double(max_tail_bytes) / 1e6);
            reset = true;
            ft = Footer{};
            all.clear();
            for (auto& rec : records) if (slot_for(rec.settings) < 0) { r.error = "too many surface configurations in one commit"; return r; }
        }
    }
    uint64_t cursor = reset ? base_end : st.footer_end;
    if (reset || st.recovered || st.file_size != cursor) {
        if (!f.truncate(cursor, &r.error)) return r;
    }
    std::vector<IndexEntry> fresh;
    fresh.reserve(records.size());
    std::vector<char> pad(8, 0);
    for (auto& rec : records) {
        IndexEntry e = rec.meta;
        e.slot = uint16_t(slot_for(rec.settings));
        e.offset = cursor;
        e.stored = rec.stored.size();
        BlobHeader bh;
        bh.codec = e.codec; bh.key = e.key; bh.raw = e.raw; bh.stored = e.stored;
        bh.crc_raw = e.crc_raw; bh.crc_stored = e.crc_stored; bh.settings = rec.settings;
        if (!f.write_at(cursor, &bh, sizeof(bh), &r.error)) return r;
        if (!rec.stored.empty() && !f.write_at(cursor + sizeof(bh), rec.stored.data(), rec.stored.size(), &r.error)) return r;
        const uint64_t end = cursor + sizeof(bh) + rec.stored.size();
        if (pad8(end) != end && !f.write_at(end, pad.data(), size_t(pad8(end) - end), &r.error)) return r;
        cursor = pad8(end);
        fresh.push_back(e);
    }
    // Index: the new records chained to the previous segment, or a full snapshot.
    std::map<std::pair<uint16_t, uint64_t>, IndexEntry> merged;
    for (const auto& e : all) merged[{e.slot, e.key}] = e;
    for (const auto& e : fresh) merged[{e.slot, e.key}] = e;
    const bool snapshot = reset || ft.chain_length + 1 >= kSnapshotEvery;
    SegmentHeader sh;
    std::vector<IndexEntry> seg_entries;
    if (snapshot) {
        for (auto& kv : merged) seg_entries.push_back(kv.second);
        sh.snapshot = 1;
        ft.chain_length = 1;
    } else {
        seg_entries = fresh;
        sh.prev_offset = ft.segment_offset;
        sh.prev_bytes = ft.segment_bytes;
        sh.prev_hash = ft.segment_hash;
        ft.chain_length += 1;
    }
    sh.count = seg_entries.size();
    std::vector<char> seg(sizeof(sh) + seg_entries.size() * sizeof(IndexEntry));
    std::memcpy(seg.data(), &sh, sizeof(sh));
    if (!seg_entries.empty()) std::memcpy(seg.data() + sizeof(sh), seg_entries.data(), seg_entries.size() * sizeof(IndexEntry));
    if (!f.write_at(cursor, seg.data(), seg.size(), &r.error)) return r;
    {
        embedded_surface::Hash h;
        h.add(seg.data(), seg.size());
        ft.segment_hash = h.result();
    }
    ft.segment_offset = cursor;
    ft.segment_bytes = seg.size();
    cursor = pad8(cursor + seg.size());
    if (cursor != ft.segment_offset + seg.size() &&
        !f.write_at(ft.segment_offset + seg.size(), pad.data(), size_t(cursor - ft.segment_offset - seg.size()), &r.error)) return r;
    // Stamp: a whole-second time, so file servers that round timestamps keep it exact.
    FILETIME now{};
    GetSystemTimeAsFileTime(&now);
    uint64_t t = (uint64_t(now.dwHighDateTime) << 32) | now.dwLowDateTime;
    t -= t % 10000000ull;
    std::memcpy(ft.magic, "LNTSLA1\0", 8);
    ft.base_end = base_end;
    ft.tile_count = merged.size();
    ft.stamp_size = cursor + sizeof(Footer);
    ft.stamp_mtime = int64_t(t);
    ft.source_hash = source_hash;
    ft.dir_hash = dir;
    ft.footer_hash = footer_digest(ft);
    if (!f.write_at(cursor, &ft, sizeof(ft), &r.error)) return r;
    if (!f.truncate(ft.stamp_size, &r.error)) return r;
    if (!FlushFileBuffers(f.h)) { r.error = win_error("surface archive flush failed"); return r; }
    FILETIME ws{DWORD(t), DWORD(t >> 32)};
    if (!SetFileTime(f.h, nullptr, nullptr, &ws)) { r.error = win_error("surface archive stamp failed"); return r; }
    r.mtime = f.mtime();
    r.state.present = true;
    r.state.file_size = ft.stamp_size;
    r.state.footer_end = ft.stamp_size;
    r.state.mtime = r.mtime;
    r.state.footer = ft;
    for (auto& kv : merged) r.state.entries.push_back(kv.second);
    r.ok = true;
    return r;
}

}  // namespace surface_archive
}  // namespace lnt
