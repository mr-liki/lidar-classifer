// Regression: a valid LAS larger than 2 GiB must not look like a tiny file.
#include <windows.h>
#include <winioctl.h>
#include <io.h>

#include <cstdio>
#include <cstring>

#include "include/LNTLASReader.h"

int main() {
    const char* path = "large_header_probe.las";
    std::FILE* file = std::fopen(path, "wb+");
    if (!file) return 2;
    DWORD ignored = 0;
    const HANDLE handle = reinterpret_cast<HANDLE>(_get_osfhandle(_fileno(file)));
    if (!DeviceIoControl(handle, FSCTL_SET_SPARSE, nullptr, 0,
                         nullptr, 0, &ignored, nullptr)) {
        std::fclose(file);
        std::remove(path);
        return 3;
    }
    lnt::LASHeader h{};
    std::memcpy(h.magic, "LASF", 4);
    h.version_major = 1;
    h.version_minor = 2;
    h.header_size = static_cast<uint16_t>(lnt::kLASHeader12Bytes);
    h.offset_to_point_data = h.header_size;
    h.point_data_format = 0;
    h.point_data_record_length = 20;
    h.legacy_point_count = 1;
    h.scale_x = h.scale_y = h.scale_z = 0.01;
    const unsigned char point[20] = {};
    const bool header_ok = std::fwrite(&h, 1, h.header_size, file) == h.header_size &&
                           std::fwrite(point, 1, sizeof(point), file) == sizeof(point);
    std::fflush(file);
    const bool large = _chsize_s(_fileno(file), 2ULL * 1024 * 1024 * 1024 + 4096) == 0;
    std::fclose(file);
    lnt::LASReader reader;
    const bool opened = header_ok && large && reader.open(path);
    lnt::RawPoint decoded{};
    const bool read = opened && reader.read_into(&decoded, 1) == 1;
    if (!opened) std::fprintf(stderr, "LAS open failed: %s\n", reader.error().c_str());
    reader.close();
    std::remove(path);
    return read ? 0 : 1;
}
