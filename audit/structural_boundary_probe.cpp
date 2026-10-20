// Test bridge for comparing the native cache with the original numpy algorithm.
#include "naksha/StructuralBoundary.hpp"
extern "C" __declspec(dllexport) int boundary_probe(
    const float* xyz, const uint8_t* cls, std::size_t n, uint8_t* out, int mode) {
    try {
        naksha::StructuralBoundary b;
        if (mode == 0) {
            b.positions(0, xyz, n);
            if (cls) b.classification(0, cls, n);
        } else {
            const std::size_t split = n / 2;
            // Reverse upload order deliberately; the result must still follow
            // arena order, including each cell's final class representative.
            b.positions(split, xyz + split*3, n-split);
            b.positions(0, xyz, split);
            if (cls) b.classification(0, cls, n);
            if (mode == 2 && n > 30) {
                // Partial class-only edits cross the two tile buffers.
                auto snapshot = b;
                const auto oldFlags = snapshot.compute(n);
                std::vector<uint8_t> patch(n/2, 9);
                b.classification(n/4, patch.data(), patch.size());
                if (snapshot.compute(n) != oldFlags) return 0;
            }
            if (mode == 3 && n > 30) {
                // Slot reuse splits both chunks, invalidates their old class
                // metadata, and retains the untouched points on either side.
                std::vector<float> patch(xyz+n/4*3, xyz+(n/4+n/2)*3);
                for (std::size_t i=2; i<patch.size(); i+=3) patch[i] += 4;
                b.positions(n/4, patch.data(), n/2);
                if (cls) b.classification(n/4, cls+n/4, n/2);
            }
        }
        if (mode == 4 || mode == 5) {
            auto resident = b;
            const auto first = mode == 4 ? 0 : n/4;
            b.restrictToRanges({{static_cast<uint32_t>(first),static_cast<uint32_t>(n/2)}});
            for (const auto& r : b.ranges()) if (!resident.matches(r)) return 0;
        }
        const auto flags = b.compute(n, mode == 6);
        std::copy(flags.begin(), flags.end(), out);
        b.markClean();
        if (b.dirty()) return 0;
        b.clear();
        if (!b.dirty() || b.compute(n) != std::vector<uint8_t>(n,0)) return 0;
        return 1;
    } catch (...) { return 0; }
}
