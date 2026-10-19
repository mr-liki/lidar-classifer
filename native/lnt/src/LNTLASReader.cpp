// LNTLASReader.cpp - see include/LNTLASReader.h.

#include "../include/LNTLASReader.h"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace lnt {

namespace {

bool seek64(std::FILE* file, int64_t offset, int origin) {
#if defined(_WIN32)
    return _fseeki64(file, offset, origin) == 0;
#else
    return fseeko(file, static_cast<off_t>(offset), origin) == 0;
#endif
}

int64_t tell64(std::FILE* file) {
#if defined(_WIN32)
    return _ftelli64(file);
#else
    return static_cast<int64_t>(ftello(file));
#endif
}

// Minimum record length per point format, and where its colour lives.
struct FormatInfo { uint16_t min_len; int rgb_off; };
bool format_info(uint8_t fmt, FormatInfo& fi) {
    static const FormatInfo t[11] = {
        {20, -1}, {28, -1}, {26, 20}, {34, 28}, {57, -1}, {63, 28},
        {30, -1}, {36, 30}, {38, 30}, {59, -1}, {67, 30}};
    if (fmt > 10) return false;
    fi = t[fmt];
    return true;
}

template <class T> T rd(const uint8_t* p) { T v; std::memcpy(&v, p, sizeof(T)); return v; }

}  // namespace

void LASReader::close() {
    if (f_) std::fclose(f_);
    f_ = nullptr;
    count_ = cursor_ = 0;
    raw_.clear();
}

bool LASReader::open(const std::string& path) {
    close();
    error_.clear();
    auto fail = [&](const std::string& m) { close(); error_ = m; return false; };

    f_ = std::fopen(path.c_str(), "rb");
    if (!f_) return fail("cannot open '" + path + "'");
    if (!seek64(f_, 0, SEEK_END)) return fail("cannot seek to end of LAS file");
    const int64_t fsz = tell64(f_);
    if (fsz < 0) return fail("cannot determine LAS file size");
    if (!seek64(f_, 0, SEEK_SET)) return fail("cannot seek to start of LAS file");
    if (fsz < static_cast<int64_t>(kLASHeader12Bytes))
        return fail("file too small for a LAS header");

    std::memset(&hdr_, 0, sizeof(hdr_));
    const std::size_t want = std::min<std::size_t>(sizeof(hdr_), (std::size_t)fsz);
    if (std::fread(&hdr_, 1, want, f_) != want) return fail("short read of LAS header");

    if (std::memcmp(hdr_.magic, "LASF", 4) != 0) return fail("not a LAS file (magic != LASF)");
    if (hdr_.version_major != 1 || hdr_.version_minor < 2 || hdr_.version_minor > 4)
        return fail("unsupported LAS version " + std::to_string(hdr_.version_major) + "." +
                    std::to_string(hdr_.version_minor) + " (1.2-1.4 supported)");
    if (hdr_.header_size < kLASHeader12Bytes || hdr_.header_size > fsz)
        return fail("invalid header_size " + std::to_string(hdr_.header_size));

    // Header bytes the declared header_size does not cover are not part of it.
    const uint16_t hs = hdr_.header_size;
    if (hs < sizeof(hdr_))
        std::memset(reinterpret_cast<uint8_t*>(&hdr_) + hs, 0, sizeof(hdr_) - hs);

    if (hdr_.point_data_format & 0x80)
        return fail("LAZ (compressed) input is not supported: decompress to LAS first");
    const uint8_t fmt = hdr_.point_data_format;
    FormatInfo fi;
    if (!format_info(fmt, fi)) return fail("unknown point data format " + std::to_string(fmt));
    if (hdr_.point_data_record_length < fi.min_len)
        return fail("point record length " + std::to_string(hdr_.point_data_record_length) +
                    " is shorter than format " + std::to_string(fmt) + " requires");
    if (hdr_.offset_to_point_data < hs || hdr_.offset_to_point_data > fsz)
        return fail("invalid offset_to_point_data");
    if (hdr_.scale_x == 0.0 || hdr_.scale_y == 0.0 || hdr_.scale_z == 0.0 ||
        !std::isfinite(hdr_.scale_x + hdr_.scale_y + hdr_.scale_z +
                       hdr_.offset_x + hdr_.offset_y + hdr_.offset_z))
        return fail("invalid scale/offset (zero or non-finite)");

    // LAS 1.4 keeps the true count in the 64-bit field; the legacy field is 0
    // for formats 6+ (and for any file over 4 G points).
    count_ = hdr_.legacy_point_count;
    if (hdr_.version_minor >= 4 && hs >= sizeof(hdr_) && count_ == 0)
        count_ = hdr_.extended_point_count;

    const unsigned long long need =
        (unsigned long long)count_ * hdr_.point_data_record_length + hdr_.offset_to_point_data;
    if (need > (unsigned long long)fsz)
        return fail("file is truncated: header declares " + std::to_string(count_) +
                    " points but the file holds fewer");

    stride_ = hdr_.point_data_record_length;
    rgb_offset_ = fi.rgb_off;
    fmt_ = fmt;
    has_color_ = rgb_offset_ >= 0;
    cursor_ = 0;
    if (!seek64(f_, hdr_.offset_to_point_data, SEEK_SET))
        return fail("cannot seek to point data");
    return true;
}

bool LASReader::rewind() {
    if (!f_) return false;
    cursor_ = 0;
    return seek64(f_, hdr_.offset_to_point_data, SEEK_SET);
}

Bounds3d LASReader::bounds() const {
    Bounds3d b;
    b.min[0] = hdr_.min_x; b.max[0] = hdr_.max_x;
    b.min[1] = hdr_.min_y; b.max[1] = hdr_.max_y;
    b.min[2] = hdr_.min_z; b.max[2] = hdr_.max_z;
    return b;
}

std::size_t LASReader::read_into(RawPoint* dst, std::size_t max_points) {
    if (!f_ || !dst) return 0;
    constexpr std::size_t kBatch = 65536;       // records per fread
    std::size_t done = 0;
    const uint64_t take = std::min<uint64_t>(max_points, count_ - cursor_);
    while (done < take) {
        const std::size_t n = (std::size_t)std::min<uint64_t>(kBatch, take - done);
        raw_.resize(n * stride_);
        if (std::fread(raw_.data(), 1, raw_.size(), f_) != raw_.size()) {
            error_ = "short read of point records";
            break;
        }
        for (std::size_t i = 0; i < n; ++i) {
            const uint8_t* p = raw_.data() + i * stride_;
            RawPoint& o = dst[done + i];
            o.x = double(rd<int32_t>(p + 0)) * hdr_.scale_x + hdr_.offset_x;
            o.y = double(rd<int32_t>(p + 4)) * hdr_.scale_y + hdr_.offset_y;
            o.z = double(rd<int32_t>(p + 8)) * hdr_.scale_z + hdr_.offset_z;
            const uint8_t inten = uint8_t(rd<uint16_t>(p + 12) >> 8);
            o.intensity = inten;
            // Preserve the complete stored classification byte in every LAS point
            // format.  In legacy formats 0-5 the high bits are named flags by the
            // LAS specification, but Naksha/TerraScan workflows also use the packed
            // byte as an extended class code (for example 0x33 == class 51).  Masking
            // with 0x1F silently changed class 51 to class 19 during .lnt conversion.
            // This matches gui.data_loader's legacy-class reconstruction and keeps
            // the source byte unchanged throughout loading, rendering and saving.
            o.classification = (fmt_ >= 6) ? p[16] : p[15];
            if (rgb_offset_ >= 0) {
                o.r = uint8_t(rd<uint16_t>(p + rgb_offset_) >> 8);
                o.g = uint8_t(rd<uint16_t>(p + rgb_offset_ + 2) >> 8);
                o.b = uint8_t(rd<uint16_t>(p + rgb_offset_ + 4) >> 8);
            } else {
                o.r = o.g = o.b = inten;        // grey from intensity
            }
        }
        done += n;
    }
    cursor_ += done;
    return done;
}

bool LASReader::read_chunk(std::vector<RawPoint>& buffer, std::size_t max_points) {
    buffer.clear();
    if (!f_ || max_points == 0 || cursor_ >= count_) return false;
    buffer.resize((std::size_t)std::min<uint64_t>(max_points, count_ - cursor_));
    const std::size_t got = read_into(buffer.data(), buffer.size());
    buffer.resize(got);
    return got > 0;
}

bool convert_las_to_lnt(const std::string& las_path, const std::string& lnt_path,
                        const LNTBuildParams& options, LNTBuildStats* stats,
                        std::string* error) {
    LNTBuildStats local;
    LNTBuildStats& st = stats ? *stats : local;
    auto fail = [&](const std::string& m) {
        st = LNTBuildStats();
        st.error = m;
        if (error) *error = m;
        return false;
    };
    LASReader r;
    if (!r.open(las_path)) return fail(r.error());
    if (r.point_count() == 0) return fail("LAS file holds no points");

    // One allocation, sized from the header; decoded chunk by chunk in place.
    std::vector<RawPoint> pts;
    try { pts.resize((std::size_t)r.point_count()); }
    catch (...) { return fail("cannot allocate " + std::to_string(r.point_count()) + " points"); }
    constexpr std::size_t kChunk = 100000;
    std::size_t at = 0;
    while (at < pts.size()) {
        const std::size_t got = r.read_into(pts.data() + at, std::min(kChunk, pts.size() - at));
        if (got == 0) return fail(r.error().empty() ? "unexpected end of LAS data" : r.error());
        at += got;
    }
    st = LNTBuilder::build(pts.data(), pts.size(), lnt_path.c_str(), options);
    if (!st.ok && error) *error = st.error;
    return st.ok;
}

}  // namespace lnt
