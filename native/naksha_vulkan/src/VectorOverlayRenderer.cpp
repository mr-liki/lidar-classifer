#include "naksha/VectorOverlayRenderer.hpp"
#include "naksha/Renderer.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>

namespace naksha {

VectorOverlayRenderer::~VectorOverlayRenderer() { Shutdown(); }

bool VectorOverlayRenderer::Initialize(Renderer& renderer) {
    renderer_ = &renderer;

    // Resolve the usable dynamic-line-width range ONCE, from the device. Above
    // 1.0 line width is only legal with the wideLines feature, and never
    // outside limits.lineWidthRange - so a device without wideLines pins the
    // whole pass to 1.0 px instead of emitting an invalid command.
    const DeviceReport& report = renderer.GetContext().GetReport();
    lineWidthMin_ = (report.minLineWidth > 0.0f) ? report.minLineWidth : 1.0f;
    lineWidthMax_ = (report.maxLineWidth > 0.0f) ? report.maxLineWidth : 1.0f;
    if (renderer.GetContext().CoreDevice().GetEnabledFeatures().wideLines != VK_TRUE)
        lineWidthMax_ = 1.0f;
    if (lineWidthMax_ < lineWidthMin_) lineWidthMax_ = lineWidthMin_;
    lineWidthPx_ = lineWidthMin_;

    vs_ = renderer.Shaders().LoadSPIRV(
        renderer.cfg_.shaderDirectory + "/vector.vert.spv", VK_SHADER_STAGE_VERTEX_BIT);
    fs_ = renderer.Shaders().LoadSPIRV(
        renderer.cfg_.shaderDirectory + "/vector.frag.spv", VK_SHADER_STAGE_FRAGMENT_BIT);
    if (vs_.module == VK_NULL_HANDLE || fs_.module == VK_NULL_HANDLE) {
        fprintf(stderr, "[VectorOverlayRenderer] vector.vert/frag.spv missing - "
                        "vector overlay disabled (the .spv files are loaded from the DLL directory)\n");
        return false;
    }

    // No push constants: the camera arrives through the shared frame UBO (set 0)
    // and line width through dynamic state, so a style change is never a rebuild.
    pipelineLayout_ = renderer.Pipelines().CreatePipelineLayout(
        {renderer.GetFrameSetLayout()}, {});
    if (pipelineLayout_ == VK_NULL_HANDLE) return false;

    if (!CreatePipeline()) return false;
    initialized_ = true;
    return true;
}

bool VectorOverlayRenderer::CreatePipeline() {
    if (!renderer_ || pipelineLayout_ == VK_NULL_HANDLE) return false;

    // Interleaved vertex arena: location 0 position (12 B) + location 1 RGBA8
    // colour (4 B) in ONE 16-byte binding.
    pipelineCfg_ = PipelineConfig{};
    pipelineCfg_.SetDefaults();
    // BOTH attributes come from binding 0: the arena interleaves position and
    // colour in one 16-byte vertex, so the colour is at offset 12 of the SAME
    // binding (binding 1 exists only on the two-buffer surface path).
    pipelineCfg_.vertexBindings = {{0, kVertexStride, VK_VERTEX_INPUT_RATE_VERTEX}};
    pipelineCfg_.vertexAttributes = {
        {0, 0, VK_FORMAT_R32G32B32_SFLOAT, 0},
        {1, 0, VK_FORMAT_R8G8B8A8_UNORM, 12},
    };
    pipelineCfg_.vertexInput.vertexBindingDescriptionCount = 1;
    pipelineCfg_.vertexInput.pVertexBindingDescriptions = pipelineCfg_.vertexBindings.data();
    pipelineCfg_.vertexInput.vertexAttributeDescriptionCount = 2;
    pipelineCfg_.vertexInput.pVertexAttributeDescriptions = pipelineCfg_.vertexAttributes.data();

    pipelineCfg_.inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_LINE_STRIP;
    // Each group is ONE draw call because 0xFFFFFFFF separates its strips.
    pipelineCfg_.inputAssembly.primitiveRestartEnable = VK_TRUE;

    pipelineCfg_.rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
    pipelineCfg_.rasterizer.cullMode = VK_CULL_MODE_NONE;
    pipelineCfg_.rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    // Dynamic line width: the value used at draw time comes from
    // vkCmdSetLineWidth, so a lineweight change never rebuilds the pipeline.
    // Requires the wideLines device feature (enabled by VulkanDevice when
    // supported) for widths > 1.
    pipelineCfg_.rasterizer.lineWidth = 1.0f;

    // Overlay semantics, identical to the Qt overlay this replaces: linework is
    // drawn AFTER the opaque passes and is never occluded or clipped by them, so
    // depth test and depth WRITE are both off. Depth attachment state stays
    // declared so the pipeline remains compatible with the frame's render pass.
    pipelineCfg_.depthStencil.depthTestEnable = VK_FALSE;
    pipelineCfg_.depthStencil.depthWriteEnable = VK_FALSE;
    pipelineCfg_.depthStencil.depthCompareOp = VK_COMPARE_OP_ALWAYS;

    // Opaque lines: the colour is the authored CAD byte, and blending it would
    // make overlapping segments a different shade than a single stroke.
    pipelineCfg_.colorBlendAttachment.blendEnable = VK_FALSE;
    pipelineCfg_.colorBlending.attachmentCount = 1;
    pipelineCfg_.colorBlending.pAttachments = &pipelineCfg_.colorBlendAttachment;
    pipelineCfg_.colorFormat = VK_FORMAT_B8G8R8A8_SRGB;
    pipelineCfg_.depthFormat = VK_FORMAT_D32_SFLOAT;

    pipelineCfg_.dynamicStates = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR,
                                  VK_DYNAMIC_STATE_LINE_WIDTH};
    pipelineCfg_.dynamicState = {};
    pipelineCfg_.dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    pipelineCfg_.dynamicState.dynamicStateCount =
        static_cast<uint32_t>(pipelineCfg_.dynamicStates.size());
    pipelineCfg_.dynamicState.pDynamicStates = pipelineCfg_.dynamicStates.data();

    pipeline_ = renderer_->Pipelines().CreateGraphicsPipeline(
        pipelineLayout_, {vs_, fs_}, pipelineCfg_, renderer_->GetRenderPass(), 0);
    if (pipeline_ == VK_NULL_HANDLE) {
        fprintf(stderr, "[VectorOverlayRenderer] LINE_STRIP pipeline creation failed\n");
        return false;
    }
    return true;
}

void VectorOverlayRenderer::Shutdown() {
    // Deliberately NOT guarded on initialized_: a partial Initialize() (shaders
    // loaded but the pipeline refused) still owns shader modules and a pipeline
    // layout, and returning early here would leak them.
    initialized_ = false;
    drawRanges_.clear();
    auto& allocator = VulkanAllocator::Get();
    if (renderer_) {
        // Pipelines/buffers may still be referenced by the last recorded frame.
        renderer_->WaitIdle();
        if (pipeline_ != VK_NULL_HANDLE) renderer_->Pipelines().DestroyPipeline(pipeline_);
        if (pipelineLayout_ != VK_NULL_HANDLE)
            renderer_->Pipelines().DestroyPipelineLayout(pipelineLayout_);
        if (vs_.module != VK_NULL_HANDLE)
            renderer_->Shaders().DestroyShaderModule(vs_.module);
        if (fs_.module != VK_NULL_HANDLE)
            renderer_->Shaders().DestroyShaderModule(fs_.module);
    }
    pipeline_ = VK_NULL_HANDLE;
    pipelineLayout_ = VK_NULL_HANDLE;
    vs_ = {}; fs_ = {};
    allocator.DestroyBuffer(vertices_);
    allocator.DestroyBuffer(indices_);
    vertices_ = {}; indices_ = {};
    vertexCapacity_ = indexCapacity_ = 0;
    residentBytes_ = 0;
    uploads_ = 0; drawCalls_ = 0; drawnIndices_ = 0;
    renderer_ = nullptr;
}

void VectorOverlayRenderer::Clear() {
    drawRanges_.clear();
    uploads_ = 0; drawCalls_ = 0; drawnIndices_ = 0;
    if (renderer_) renderer_->WaitIdle();
    auto& allocator = VulkanAllocator::Get();
    allocator.DestroyBuffer(vertices_);
    allocator.DestroyBuffer(indices_);
    vertices_ = {}; indices_ = {};
    vertexCapacity_ = indexCapacity_ = 0;
    residentBytes_ = 0;
}

int VectorOverlayRenderer::Reserve(std::size_t vertexCapacity, std::size_t indexCapacity) {
    if (!renderer_ || vertexCapacity == 0 || indexCapacity == 0) return 0;
    const bool bothResident = vertices_.IsValid() && indices_.IsValid();
    // Already big enough: nothing to re-upload, so the caller's earlier strips
    // stay valid and their offsets do not move.
    if (bothResident && vertexCapacity_ >= vertexCapacity &&
        indexCapacity_ >= indexCapacity) {
        return 1;
    }
    const int rc = bothResident ? 2 : 1;

    auto& allocator = VulkanAllocator::Get();
    allocator.DestroyBuffer(vertices_);
    allocator.DestroyBuffer(indices_);
    vertices_ = allocator.CreateBuffer(
        static_cast<VkDeviceSize>(vertexCapacity) * kVertexStride,
        VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
        VMA_MEMORY_USAGE_GPU_ONLY);
    indices_ = allocator.CreateBuffer(
        static_cast<VkDeviceSize>(indexCapacity) * sizeof(uint32_t),
        VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
        VMA_MEMORY_USAGE_GPU_ONLY);
    if (!vertices_.IsValid() || !indices_.IsValid()) {
        allocator.DestroyBuffer(vertices_);
        allocator.DestroyBuffer(indices_);
        vertices_ = {}; indices_ = {};
        vertexCapacity_ = indexCapacity_ = 0;
        residentBytes_ = 0;
        fprintf(stderr, "[VectorOverlayRenderer] arena allocation failed "
                        "(vertices=%zu indices=%zu)\n", vertexCapacity, indexCapacity);
        return 0;
    }
    vertexCapacity_ = vertexCapacity;
    indexCapacity_ = indexCapacity;
    residentBytes_ = vertices_.size + indices_.size;
    uploads_ = 0;
    drawRanges_.clear();
    return rc;
}

UploadStats VectorOverlayRenderer::UploadStrips(
        std::size_t vFirst, std::size_t vCount,
        const double* xyzWorldF64, const uint8_t* rgb8,
        std::size_t iFirst, std::size_t iCount,
        const uint32_t* indices) {
    UploadStats stats;
    if (!initialized_ || !renderer_) return stats;
    if (!vertices_.IsValid() || !indices_.IsValid()) return stats;
    if (vCount == 0 && iCount == 0) return stats;
    if (vFirst + vCount > vertexCapacity_) return stats;   // never write past the arena
    if (iFirst + iCount > indexCapacity_) return stats;
    if (vCount > 0 && (!xyzWorldF64 || !rgb8)) return stats;
    if (iCount > 0 && !indices) return stats;

    const VkDeviceSize vBytes = static_cast<VkDeviceSize>(vCount) * kVertexStride;
    const VkDeviceSize iBytes = static_cast<VkDeviceSize>(iCount) * sizeof(uint32_t);
    const VkDeviceSize total = vBytes + iBytes;

    // Floating-origin rule, same as the point path: world float64 -> render
    // float32 against the renderer's pinned origin, in one linear pass.
    std::vector<float> render;
    const double t0 = NowMs();
    if (vCount > 0) {
        render.resize(vCount * 3);
        RenderOrigin::WorldToRenderF64ToF32(xyzWorldF64, render.data(), vCount,
                                            renderer_->GetOrigin());
    }
    const double t1 = NowMs();

    uint8_t* dst = renderer_->GetUploadContext().Stage(total);
    if (!dst) return stats;
    const double t2 = NowMs();

    std::vector<CopyRegion> regions;
    if (vCount > 0) {
        // Interleave render-space XYZ + RGBA8 (alpha 255 = opaque) into the
        // arena's 16-byte vertex layout.
        for (std::size_t i = 0; i < vCount; ++i) {
            uint8_t* v = dst + i * kVertexStride;
            std::memcpy(v, render.data() + i * 3, 12);
            v[12] = rgb8[i * 3 + 0];
            v[13] = rgb8[i * 3 + 1];
            v[14] = rgb8[i * 3 + 2];
            v[15] = 255;
        }
        CopyRegion r;
        r.dst = vertices_.buffer;
        r.dstOffset = static_cast<VkDeviceSize>(vFirst) * kVertexStride;
        r.srcOffset = 0;
        r.size = vBytes;
        regions.push_back(r);
    }
    if (iCount > 0) {
        std::memcpy(dst + vBytes, indices, static_cast<std::size_t>(iBytes));
        CopyRegion r;
        r.dst = indices_.buffer;
        r.dstOffset = static_cast<VkDeviceSize>(iFirst) * sizeof(uint32_t);
        r.srcOffset = vBytes;
        r.size = iBytes;
        regions.push_back(r);
    }
    if (regions.empty()) return stats;

    // ONE staged submit for the vertex + index regions of this call.
    stats = renderer_->GetUploadContext().Execute(regions, t1 - t0, t2 - t1);
    if (stats.bytes > 0) ++uploads_;
    return stats;
}

void VectorOverlayRenderer::GetStats(uint64_t* out, uint32_t count) const {
    if (!out) return;
    uint64_t values[kStatWordCount] = {
        uploads_,
        drawnIndices_,
        drawCalls_,
        static_cast<uint64_t>(residentBytes_),
        static_cast<uint64_t>(vertexCapacity_),
        static_cast<uint64_t>(indexCapacity_),
        static_cast<uint64_t>(drawRanges_.size()),
    };
    const uint32_t n = count < kStatWordCount ? count : kStatWordCount;
    for (uint32_t i = 0; i < n; ++i) out[i] = values[i];
}

float VectorOverlayRenderer::ClampLineWidth(float px) const {
    if (!(px > 0.0f)) px = lineWidthMin_;
    if (px < lineWidthMin_) px = lineWidthMin_;
    if (px > lineWidthMax_) px = lineWidthMax_;
    return px;
}

void VectorOverlayRenderer::Record(VkCommandBuffer cmd, uint32_t frameSlot) {
    (void)frameSlot;
    // "Last frame" telemetry: reset first so a frame with no ranges reports 0
    // instead of keeping the previous frame's counts.
    drawCalls_ = 0;
    drawnIndices_ = 0;
    if (!initialized_ || pipeline_ == VK_NULL_HANDLE || !renderer_) return;
    if (!vertices_.IsValid() || !indices_.IsValid()) return;
    if (drawRanges_.empty()) return;   // nothing visible: draw nothing at all

    // The pipeline declares VIEWPORT and SCISSOR dynamic (SetDefaults), so this
    // pass OWNS the state it uses: a dynamic state the pipeline requires but the
    // command buffer never set makes the draw invalid (validation VUID 07831 /
    // 07832) and draws nothing at all. Same full-extent viewport the sibling
    // renderers set in their own Record().
    VkExtent2D ext = renderer_->GetExtent();
    VkViewport vp{0.0f, 0.0f, static_cast<float>(ext.width),
                  static_cast<float>(ext.height), 0.0f, 1.0f};
    VkRect2D sc{{0, 0}, {ext.width, ext.height}};
    vkCmdSetViewport(cmd, 0, 1, &vp);
    vkCmdSetScissor(cmd, 0, 1, &sc);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_);
    vkCmdSetLineWidth(cmd, ClampLineWidth(lineWidthPx_));
    VkDescriptorSet frameSet = renderer_->GetFrameDescriptorSet();
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout_,
                            0, 1, &frameSet, 0, nullptr);
    VkDeviceSize vbOffset = 0;
    vkCmdBindVertexBuffers(cmd, 0, 1, &vertices_.buffer, &vbOffset);
    vkCmdBindIndexBuffer(cmd, indices_.buffer, 0, VK_INDEX_TYPE_UINT32);

    for (const auto& range : drawRanges_) {
        if (range.count == 0) continue;
        // Per-group width when the caller supplied one: dynamic state, so this
        // is one command per group and never a pipeline rebuild.
        if (range.width > 0.0f) vkCmdSetLineWidth(cmd, ClampLineWidth(range.width));
        vkCmdDrawIndexed(cmd, range.count, 1, range.first, 0, 0);
        ++drawCalls_;
        drawnIndices_ += range.count;
    }
}

void VectorOverlayRenderer::OnSwapchainRebuild() {
    if (!initialized_ || !renderer_) return;
    renderer_->Pipelines().DestroyPipeline(pipeline_);
    pipeline_ = VK_NULL_HANDLE;
    CreatePipeline();
}

} // namespace naksha
