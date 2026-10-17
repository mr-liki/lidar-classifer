#pragma once
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include "naksha/Buffer.hpp"
#include "naksha/RenderOrigin.hpp"
#include "naksha/core/vulkan/VulkanPipelineManager.h"

namespace naksha {
using namespace vulkan;
class Renderer;

// ---------------------------------------------------------------------------
// Vector overlay renderer (SNT / CAD linework).
//
// WHY THIS EXISTS. Point clouds and surfaces have GPU paths; vector
// attachments did not. Every SNT/DXF entity was re-stroked by the transparent
// Qt overlay on a 33 ms repaint, so a 1.1M-entity file cost a full CPU
// world->screen transform plus one QPainter stroke per shape, every frame. This
// renderer uploads the linework ONCE as GPU buffers and draws ~one call per
// (file, layer, colour) group with ZERO per-frame CPU cost - the same
// upload-once / draw-many contract PointCloudRenderer and SurfaceRenderer keep.
//
// GEOMETRY CONTRACT
//   One interleaved vertex arena: float32 render-space XYZ (12 B) + RGBA8
//   colour (4 B) = 16 B per vertex. One uint32 index arena whose strips are
//   separated by PRIMITIVE-RESTART (0xFFFFFFFF), so a group's whole linework is
//   ONE vkCmdDrawIndexed. Closed polylines repeat their first index, matching
//   the VTK _build_line_actor path this replaces.
//
// DRAW-RANGE VISIBILITY
//   Layer/file visibility is a list of (indexFirst, indexCount) windows into
//   the resident index arena. Toggling one is a range-list change - it never
//   re-uploads a vertex or an index, exactly like nkv_set_point_draw_ranges.
//
// PIPELINE
//   VK_PRIMITIVE_TOPOLOGY_LINE_STRIP with primitive restart enabled, depth
//   test AND depth write OFF (matching the always-on-top Qt overlay semantics
//   this replaces: overlay linework is drawn after opaque geometry and is never
//   occluded by it), and DYNAMIC line width so a lineweight change is a
//   command, never a pipeline rebuild. wideLines is enabled by VulkanDevice
//   when the device supports it (the T400 reports [1.0, 64.0]).
// ---------------------------------------------------------------------------
class VectorOverlayRenderer {
public:
    // Interleaved vertex stride: 12 B position + 4 B RGBA8.
    static constexpr std::size_t kVertexStride = 16;
    // Primitive-restart index written between strips in the index arena.
    static constexpr uint32_t kRestartIndex = 0xFFFFFFFFu;

    VectorOverlayRenderer() = default;
    ~VectorOverlayRenderer();

    bool Initialize(Renderer& renderer);
    void Shutdown();
    bool IsInitialized() const { return initialized_; }

    // ---- Stable-offset arena -------------------------------------------------
    // Reserves a fixed vertex/index capacity ONCE. Strips are then written at
    // caller-chosen STABLE offsets and a draw range selects what is drawn, so a
    // visibility change never touches a buffer.
    //
    //   0 = failed (no renderer / zero capacity / allocation failure)
    //   1 = ready (nothing to re-upload)
    //   2 = (re)allocated - every earlier strip is INVALID and must be uploaded again
    // Same contract as PointCloudRenderer::ReserveArena.
    int Reserve(std::size_t vertexCapacity, std::size_t indexCapacity);

    // Uploads strips that SHARE the arena. `xyzWorldF64` is WORLD float64
    // (vCount*3), converted against the renderer's pinned render origin exactly
    // as the point path does; `rgb8` is vCount*3 bytes of authored RGB;
    // `indices` is iCount uint32 with kRestartIndex separators between strips.
    // Returns bytes==0 when nothing was uploaded (no arena, or past capacity).
    UploadStats UploadStrips(std::size_t vFirst, std::size_t vCount,
                             const double* xyzWorldF64, const uint8_t* rgb8,
                             std::size_t iFirst, std::size_t iCount,
                             const uint32_t* indices);

    // Index windows drawn this frame: one vkCmdDrawIndexed per range. An EMPTY
    // list draws nothing (an unfiltered whole-arena draw would rasterise the
    // unwritten gaps between groups).
    void SetDrawRanges(const uint32_t* firsts, const uint32_t* counts,
                       uint32_t rangeCount) {
        drawRanges_.clear();
        if (!firsts || !counts || rangeCount == 0) return;
        drawRanges_.reserve(rangeCount);
        for (uint32_t i = 0; i < rangeCount; ++i)
            drawRanges_.push_back({firsts[i], counts[i], 0.0f});
    }
    // Same, but with an EXPLICIT pixel width per range (`widths` may be NULL to
    // fall back to the global SetLineWidth value). Line width is dynamic state,
    // so Record() sets it immediately before each range's draw call - a
    // per-layer-group width therefore costs one extra command per group and
    // still never touches a buffer or rebuilds the pipeline. This is what makes
    // an SNT block outline drawn at its authored lineweight (1.0 + lw*0.5)
    // possible instead of one file-wide width for every layer.
    void SetDrawRangesWithWidths(const uint32_t* firsts, const uint32_t* counts,
                                 const float* widths, uint32_t rangeCount) {
        drawRanges_.clear();
        if (!firsts || !counts || rangeCount == 0) return;
        drawRanges_.reserve(rangeCount);
        for (uint32_t i = 0; i < rangeCount; ++i) {
            Range r;
            r.first = firsts[i];
            r.count = counts[i];
            r.width = (widths && widths[i] > 0.0f) ? widths[i] : 0.0f;
            drawRanges_.push_back(r);
        }
    }

    void ClearDrawRanges() { drawRanges_.clear(); }

    // ---- Style (push/dynamic only; never touches a buffer) -------------------
    // Clamped to the device's lineWidthRange: a vkCmdSetLineWidth outside that
    // range is an invalid command, and above 1.0 it also requires wideLines.
    // The clamped bounds are resolved at Initialize() from the physical device,
    // so this is safe on hardware whose range is not the T400's [1, 64].
    void SetLineWidth(float pixels) { lineWidthPx_ = ClampLineWidth(pixels); }
    float GetLineWidth() const { return lineWidthPx_; }
    float GetLineWidthMin() const { return lineWidthMin_; }
    float GetLineWidthMax() const { return lineWidthMax_; }
    float ClampLineWidth(float pixels) const;

    // Drops the arenas and the draw ranges, keeping the pipeline alive. Used by
    // nkv_clear (which drops point/surface buffers but keeps the renderer).
    void Clear();

    bool IsLoaded() const { return vertices_.IsValid() && indices_.IsValid(); }

    // ---- Counters (log/perf discipline) --------------------------------------
    uint64_t GetUploadCount() const { return uploads_; }
    uint64_t GetDrawCallCount() const { return drawCalls_; }
    uint64_t GetDrawnIndexCount() const { return drawnIndices_; }
    VkDeviceSize GetResidentBytes() const { return residentBytes_; }
    std::size_t GetVertexCapacity() const { return vertexCapacity_; }
    std::size_t GetIndexCapacity() const { return indexCapacity_; }
    uint32_t GetDrawRangeCount() const {
        return static_cast<uint32_t>(drawRanges_.size());
    }
    // out[0] upload calls, [1] drawn indices last frame, [2] draw calls last
    // frame, [3] resident bytes, [4] vertex capacity, [5] index capacity,
    // [6] draw range count.
    static constexpr uint32_t kStatWordCount = 7;
    void GetStats(uint64_t* out, uint32_t count) const;

    // Records the overlay pass into the CURRENT frame's command buffer. Must be
    // called inside the frame's main render pass, after the opaque passes.
    // Unreferenced frameSlot is kept for signature parity with the sibling
    // renderers (timestamps are deliberately NOT written here: the query pool's
    // point/surface pairs are already published and reusing a slot would skew
    // the surface GPU timing this pass is measured against).
    void Record(VkCommandBuffer cmd, uint32_t frameSlot);

    // Called by Renderer when the swapchain format changed (pipeline rebuild).
    void OnSwapchainRebuild();

private:
    bool CreatePipeline();

    Renderer* renderer_ = nullptr;
    PipelineConfig pipelineCfg_;
    VkPipeline pipeline_ = VK_NULL_HANDLE;
    VkPipelineLayout pipelineLayout_ = VK_NULL_HANDLE;
    ShaderModule vs_ = {}, fs_ = {};

    GPUBuffer vertices_;
    GPUBuffer indices_;
    std::size_t vertexCapacity_ = 0;
    std::size_t indexCapacity_ = 0;
    // One draw call each: an (indexFirst, indexCount) window plus an optional
    // per-range pixel width (0 = use the global lineWidthPx_).
    struct Range {
        uint32_t first = 0;
        uint32_t count = 0;
        float width = 0.0f;
    };
    std::vector<Range> drawRanges_;

    float lineWidthPx_ = 1.0f;
    float lineWidthMin_ = 1.0f;
    float lineWidthMax_ = 1.0f;   // 1.0 unless the device enables wideLines
    uint64_t uploads_ = 0;
    uint64_t drawCalls_ = 0;
    uint64_t drawnIndices_ = 0;
    VkDeviceSize residentBytes_ = 0;
    bool initialized_ = false;
};

} // namespace naksha
