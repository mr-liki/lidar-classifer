// LNTLASReader.h - streaming ASPRS LAS 1.2-1.4 reader feeding the .lnt builder.
//
// LAYOUT. The header below is the REAL ASPRS layout (offsets are asserted), which
// is not the order of a hand-written summary: the point format is a byte at 104,
// the record length a uint16 at 105, the legacy point count sits AFTER them at
// 107, and the offset to point data at 96 precedes them. A reader laid out in any
// other order parses synthetic files written by the same mistake and nothing real.
//
// SCOPE. Uncompressed LAS 1.2-1.4, point formats 0-10. LAZ (point format bit 7
// set) is detected and REJECTED with a clear error rather than misread:
// decompression needs LASzip, which this layer does not link.
//
// STREAMING. read_into()/read_chunk() pull records through one reusable byte
// buffer, so the file is never held whole. LNTBuilder::build() itself takes one
// contiguous RawPoint array (it radix-sorts all keys), so convert_las_to_lnt()
// sizes that array once from the header and decodes straight into it: no second
// copy of the dataset exists at any time.
//
// DEQUANTISATION. world = raw * scale + offset, in double. Colour is 16-bit in
// LAS and is reduced with >> 8 as specified (note: files that store 8-bit values
// in the 16-bit fields therefore decode to 0). Intensity is 16-bit and reduced
// the same way. Formats without colour (0, 1, 4, 6, 9) get grey = intensity.

#ifndef LNT_LAS_READER_H
#define LNT_LAS_READER_H

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "LNTBuilder.h"
#include "LNTTypes.h"

namespace lnt {

#pragma pack(push, 1)
struct LASHeader {
    char     magic[4];                 //   0  "LASF"
    uint16_t file_source_id;           //   4
    uint16_t global_encoding;          //   6
    uint8_t  guid[16];                 //   8
    uint8_t  version_major;            //  24
    uint8_t  version_minor;            //  25
    char     system_identifier[32];    //  26
    char     generating_software[32];  //  58
    uint16_t creation_day;             //  90
    uint16_t creation_year;            //  92
    uint16_t header_size;              //  94
    uint32_t offset_to_point_data;     //  96
    uint32_t num_vlrs;                 // 100
    uint8_t  point_data_format;        // 104
    uint16_t point_data_record_length; // 105
    uint32_t legacy_point_count;       // 107
    uint32_t legacy_points_by_return[5]; // 111
    double   scale_x, scale_y, scale_z;        // 131
    double   offset_x, offset_y, offset_z;     // 155
    double   max_x, min_x, max_y, min_y, max_z, min_z;  // 179
    // ---- 1.3 / 1.4 tail: valid only when header_size covers it ----
    uint64_t waveform_data_start;      // 227 (1.3+)
    uint64_t evlr_start;               // 235 (1.4)
    uint32_t num_evlrs;                // 243
    uint64_t extended_point_count;     // 247
    uint64_t extended_points_by_return[15];  // 255
};
#pragma pack(pop)

static_assert(offsetof(LASHeader, header_size) == 94, "LAS header_size offset");
static_assert(offsetof(LASHeader, offset_to_point_data) == 96, "LAS point offset");
static_assert(offsetof(LASHeader, point_data_format) == 104, "LAS format offset");
static_assert(offsetof(LASHeader, point_data_record_length) == 105, "LAS length offset");
static_assert(offsetof(LASHeader, legacy_point_count) == 107, "LAS count offset");
static_assert(offsetof(LASHeader, scale_x) == 131, "LAS scale offset");
static_assert(offsetof(LASHeader, offset_x) == 155, "LAS offset offset");
static_assert(offsetof(LASHeader, max_x) == 179, "LAS bounds offset");
static_assert(offsetof(LASHeader, extended_point_count) == 247, "LAS 1.4 count offset");
static_assert(sizeof(LASHeader) == 375, "LAS 1.4 header is 375 bytes");

static constexpr std::size_t kLASHeader12Bytes = 227;

class LASReader {
public:
    LASReader() = default;
    ~LASReader() { close(); }
    LASReader(const LASReader&) = delete;
    LASReader& operator=(const LASReader&) = delete;

    // Validates magic, version, format, scale, offsets and that the file really
    // holds point_count records. False + error() on any failure (incl. LAZ).
    bool open(const std::string& path);
    void close();

    bool ok() const { return f_ != nullptr; }
    const std::string& error() const { return error_; }
    const LASHeader& header() const { return hdr_; }

    uint64_t point_count() const { return count_; }
    uint64_t points_remaining() const { return count_ - cursor_; }
    bool has_color() const { return has_color_; }
    // Header-declared bounds as a Bounds3d (min/max per axis).
    Bounds3d bounds() const;

    // Decodes up to `max_points` records into dst; returns how many (0 at EOF
    // or on error - check error()).
    std::size_t read_into(RawPoint* dst, std::size_t max_points);

    // buffer is REPLACED by the next <= max_points points. False when exhausted.
    bool read_chunk(std::vector<RawPoint>& buffer, std::size_t max_points);

    // Restarts at the first point.
    bool rewind();

private:
    std::FILE* f_ = nullptr;
    LASHeader hdr_{};
    std::string error_;
    uint64_t count_ = 0, cursor_ = 0;
    uint32_t stride_ = 0;
    uint8_t fmt_ = 0;         // point data format (selects where the class lives)
    int rgb_offset_ = -1;     // byte offset of the 3 x uint16 colour, or -1
    bool has_color_ = false;
    std::vector<uint8_t> raw_;
};

using LNTBuilderOptions = LNTBuildParams;

// Converts a LAS file to a .lnt file. `stats`, if given, receives the build
// stats. Returns false with the reason in `error` / stats->error.
bool convert_las_to_lnt(const std::string& las_path, const std::string& lnt_path,
                        const LNTBuildParams& options = LNTBuildParams(),
                        LNTBuildStats* stats = nullptr, std::string* error = nullptr);

}  // namespace lnt

#endif  // LNT_LAS_READER_H
