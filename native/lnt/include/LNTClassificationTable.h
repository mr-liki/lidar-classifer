#ifndef LNT_CLASSIFICATION_TABLE_H
#define LNT_CLASSIFICATION_TABLE_H

#include <array>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace lnt {

#pragma pack(push, 1)
struct PackedClassColor {
    uint8_t r, g, b, visible;
};
#pragma pack(pop)
static_assert(sizeof(PackedClassColor) == 4, "PackedClassColor must be exactly 4 bytes");

class ClassificationTable {
public:
    struct Rule {
        PackedClassColor color{128, 128, 128, 1};
        float weight = 1.0f;
        std::string description;
    };

    ClassificationTable() { rules_.fill(Rule{}); }

    bool load_from_ptc(const std::string& filepath) {
        std::ifstream file(filepath, std::ios::binary);
        if (!file) { error_ = "cannot open PTC: " + filepath; return false; }
        std::array<Rule, 256> next;
        next.fill(Rule{});
        std::string line, header;
        unsigned header_line = 0, line_no = 0, count = 0;
        auto trim = [](std::string s) {
            const auto first = s.find_first_not_of(" \t\r\n");
            if (first == std::string::npos) return std::string();
            return s.substr(first, s.find_last_not_of(" \t\r\n") - first + 1);
        };
        auto fields = [](const std::string& s) {
            std::vector<std::string> out;
            std::stringstream stream(s);
            std::string field;
            while (std::getline(stream, field, '\t')) out.push_back(field);
            return out;
        };
        auto integer = [&](const std::string& s, int lo, int hi, int& value) {
            try {
                std::size_t used = 0;
                const int parsed = std::stoi(trim(s), &used);
                if (used != trim(s).size() || parsed < lo || parsed > hi) return false;
                value = parsed;
                return true;
            } catch (...) { return false; }
        };
        while (std::getline(file, line)) {
            ++line_no;
            const std::string clean = trim(line);
            if (clean.empty() || clean[0] == '#' || clean[0] == ';' ||
                clean.compare(0, 2, "//") == 0) continue;
            if (header.empty()) { header = line; header_line = line_no; continue; }
            const auto h = fields(header), d = fields(line);
            int code = -1, detail_code = -1, rgb[3] = {};
            if (h.size() < 2 || d.size() < 5 ||
                !integer(h[0], 0, 255, code) ||
                !integer(d[2], 0, 255, detail_code) || code != detail_code) {
                error_ = "invalid PTC record at line " + std::to_string(header_line);
                return false;
            }
            std::stringstream colors(d[3]);
            std::string component;
            for (int& channel : rgb) {
                if (!std::getline(colors, component, ',') ||
                    !integer(component, 0, 255, channel)) {
                    error_ = "invalid PTC RGB at line " + std::to_string(line_no);
                    return false;
                }
            }
            if (std::getline(colors, component, ',')) {
                error_ = "extra PTC RGB component at line " + std::to_string(line_no);
                return false;
            }
            const std::string show = trim(d[4]);
            if (show != "0" && show != "1") {
                error_ = "invalid PTC visibility at line " + std::to_string(line_no);
                return false;
            }
            float weight = 1.0f;
            if (d.size() > 5 && !trim(d[5]).empty()) {
                try {
                    std::size_t used = 0;
                    const std::string w = trim(d[5]);
                    weight = std::stof(w, &used);
                    if (used != w.size() || !std::isfinite(weight) || weight < 0.1f || weight > 12.0f)
                        throw std::invalid_argument("weight");
                } catch (...) {
                    error_ = "invalid PTC weight at line " + std::to_string(line_no);
                    return false;
                }
            }
            next[code] = Rule{{uint8_t(rgb[0]), uint8_t(rgb[1]), uint8_t(rgb[2]),
                               uint8_t(show == "1")}, weight, h[1]};
            header.clear();
            ++count;
        }
        if (!file.eof() || !header.empty() || count == 0) {
            error_ = "incomplete or empty PTC";
            return false;
        }
        rules_ = std::move(next);
        error_.clear();
        return true;
    }

    void export_gpu_lut(PackedClassColor out_lut[256]) const {
        if (!out_lut) return;
        for (unsigned i = 0; i < 256; ++i) out_lut[i] = rules_[i].color;
    }
    const Rule& rule(uint8_t code) const { return rules_[code]; }
    const std::string& error() const { return error_; }

private:
    std::array<Rule, 256> rules_{};
    std::string error_;
};

} // namespace lnt
namespace LNT = lnt;
#endif
