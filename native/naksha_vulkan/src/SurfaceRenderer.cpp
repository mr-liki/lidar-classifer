#include "naksha/SurfaceRenderer.hpp"
#include "naksha/Renderer.hpp"
#include <cstring>
#include <cmath>
#include <algorithm>
#include <string>
#include <limits>

namespace naksha {

SurfaceRenderer::~SurfaceRenderer() { Shutdown(); }

void SurfaceLodBuffers::Release() {
    if (cellPool != VK_NULL_HANDLE) vkDestroyDescriptorPool(cellDevice, cellPool, nullptr);
    cellPool = VK_NULL_HANDLE; cellSet = VK_NULL_HANDLE;
    VulkanAllocator::Get().DestroyBuffer(cellColors);
    cellColors = {};
    VulkanAllocator::Get().DestroyBuffer(positions);
    VulkanAllocator::Get().DestroyBuffer(normals);
    VulkanAllocator::Get().DestroyBuffer(colors);
    VulkanAllocator::Get().DestroyBuffer(indices);
    positions = {}; normals = {}; colors = {}; indices = {};
}

void SurfaceResourceSet::ReleaseBuffers() {
    VulkanAllocator::Get().DestroyBuffer(coveragePositions);
    VulkanAllocator::Get().DestroyBuffer(coverageColors);
    coveragePositions={};coverageColors={};coverageCount=0;
    for (uint32_t i = 0; i < kSurfaceLodCount; ++i) {
        lods[i].Release();
        visibleTiles[i].clear();
    }
    lodMeshes.clear();
    gpuBytes = 0;
}

static VkVertexInputBindingDescription surfBindings[3] = {
    {0, sizeof(float) * 3, VK_VERTEX_INPUT_RATE_VERTEX},
    {1, sizeof(float) * 3, VK_VERTEX_INPUT_RATE_VERTEX},
    {2, sizeof(uint8_t) * 4, VK_VERTEX_INPUT_RATE_VERTEX},
};
static VkVertexInputAttributeDescription surfAttributes[3] = {
    {0, 0, VK_FORMAT_R32G32B32_SFLOAT, 0},
    {1, 1, VK_FORMAT_R32G32B32_SFLOAT, 0},
    {2, 2, VK_FORMAT_R8G8B8A8_UNORM, 0},
};

bool SurfaceRenderer::Initialize(Renderer& renderer) {
    renderer_ = &renderer;
    wireframeSupported_ = renderer.GetContext().GetReport().fillModeNonSolid != VK_FALSE;
    // Resolve the safe VRAM budget once, here, so EstimateRequiredBytes() /
    // WouldAcceptUpload() can answer before the first upload happens.
    // 70% of the device-local heap leaves room for the swapchain, depth,
    // staging and driver overhead while a swap has ACTIVE and PENDING both
    // resident.
    const uint64_t vram = renderer.GetContext().GetReport().vramBytes;
    if (vram > 0) vramBudgetBytes_ = static_cast<double>(vram) * 0.70;

    std::string vsPath = renderer.cfg_.shaderDirectory + "/surface.vert.spv";
    std::string fsPath = renderer.cfg_.shaderDirectory + "/surface.frag.spv";
    vs_ = renderer.Shaders().LoadSPIRV(vsPath, VK_SHADER_STAGE_VERTEX_BIT);
    fsSolid_ = renderer.Shaders().LoadSPIRV(fsPath, VK_SHADER_STAGE_FRAGMENT_BIT);
    fsWireframe_ = fsSolid_;
    if (vs_.module == VK_NULL_HANDLE || fsSolid_.module == VK_NULL_HANDLE) return false;

    VkPushConstantRange pcr{};
    pcr.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
    pcr.offset = 0;
    pcr.size = sizeof(PushConstants);
    pipelineLayout_ = renderer.Pipelines().CreatePipelineLayout(
        {renderer.GetFrameSetLayout()}, {pcr});
    if (pipelineLayout_ == VK_NULL_HANDLE) return false;

    pipelineCfg_ = PipelineConfig{};
    pipelineCfg_.SetDefaults();
    pipelineCfg_.vertexBindings = {surfBindings[0], surfBindings[1], surfBindings[2]};
    pipelineCfg_.vertexAttributes = {surfAttributes[0], surfAttributes[1], surfAttributes[2]};
    pipelineCfg_.vertexInput.vertexBindingDescriptionCount = 3;
    pipelineCfg_.vertexInput.pVertexBindingDescriptions = surfBindings;
    pipelineCfg_.vertexInput.vertexAttributeDescriptionCount = 3;
    pipelineCfg_.vertexInput.pVertexAttributeDescriptions = surfAttributes;
    pipelineCfg_.inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    pipelineCfg_.inputAssembly.primitiveRestartEnable = VK_FALSE;
    // No back-face culling, matching the VTK side this renderer must agree
    // with: gui/shading_display.py explicitly calls BackfaceCullingOff()/
    // FrontfaceCullingOff() on every shaded-mesh actor. It is also required
    // for correctness here: a triangle wound counter-clockwise as seen from
    // above (the CPU mesh builder's convention, and VTK's front face in a
    // y-up screen space) has a NEGATIVE signed area in Vulkan framebuffer
    // coordinates, whose y axis points down - with CULL_MODE_BACK_BIT the
    // entire terrain would be culled away and the viewport would show only
    // the clear colour.
    pipelineCfg_.rasterizer.cullMode = VK_CULL_MODE_NONE;
    pipelineCfg_.rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    pipelineCfg_.depthStencil.depthCompareOp = VK_COMPARE_OP_LESS;
    pipelineCfg_.colorBlendAttachment.blendEnable = VK_FALSE;
    pipelineCfg_.colorBlending.attachmentCount = 1;
    pipelineCfg_.colorBlending.pAttachments = &pipelineCfg_.colorBlendAttachment;
    pipelineCfg_.colorFormat = VK_FORMAT_B8G8R8A8_SRGB;
    pipelineCfg_.depthFormat = VK_FORMAT_D32_SFLOAT;

    std::vector<ShaderModule> solid{vs_, fsSolid_};
    pipelineSolid_ = renderer.Pipelines().CreateGraphicsPipeline(
        pipelineLayout_, solid, pipelineCfg_, renderer.GetRenderPass(), 0);

    // PASS 2 overlay pipeline: identical to the base pipeline except for the
    // depth comparison. The overlay draws the SAME triangles (bit-identical
    // vertex positions, built from the same face rows), so with the base
    // pipeline's VK_COMPARE_OP_LESS every overlay fragment failed against the
    // depth the base pass had just written and the mixed-class barycentric blend
    // never appeared on screen. LESS_OR_EQUAL lets the later draw win on the
    // shared depth, which is the same intent as the polygon offset VTK applies
    // to these actors in gui/shading_display.py.
    pipelineCfg_.depthStencil.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;
    pipelineSolidOverlay_ = renderer.Pipelines().CreateGraphicsPipeline(
        pipelineLayout_, solid, pipelineCfg_, renderer.GetRenderPass(), 0);

    pipelineCfg_.rasterizer.polygonMode = VK_POLYGON_MODE_LINE;
    pipelineCfg_.rasterizer.lineWidth = 1.0f;
    std::vector<ShaderModule> wire{vs_, fsWireframe_};
    pipelineWireframe_ = renderer.Pipelines().CreateGraphicsPipeline(
        pipelineLayout_, wire, pipelineCfg_, renderer.GetRenderPass(), 0);

    if (renderer.GetContext().CoreDevice().GetEnabledFeatures().geometryShader) {
        indexedVs_ = renderer.Shaders().LoadSPIRV(renderer.cfg_.shaderDirectory + "/surface_indexed.vert.spv", VK_SHADER_STAGE_VERTEX_BIT);
        indexedFs_ = renderer.Shaders().LoadSPIRV(renderer.cfg_.shaderDirectory + "/surface_indexed.frag.spv", VK_SHADER_STAGE_FRAGMENT_BIT);
        cellLayout_ = renderer.Descriptors().CreateLayout({{0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_FRAGMENT_BIT}});
        VkPushConstantRange indexedPush{VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(PushConstants)};
        indexedLayout_ = renderer.Pipelines().CreatePipelineLayout({renderer.GetFrameSetLayout(), cellLayout_}, {indexedPush});
        PipelineConfig indexed = pipelineCfg_;
        indexed.rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
        indexed.depthStencil.depthCompareOp = VK_COMPARE_OP_LESS;
        indexed.vertexBindings = {surfBindings[0]};
        indexed.vertexAttributes = {surfAttributes[0]};
        indexed.vertexInput.vertexBindingDescriptionCount = 1;
        indexed.vertexInput.pVertexBindingDescriptions = surfBindings;
        indexed.vertexInput.vertexAttributeDescriptionCount = 1;
        indexed.vertexInput.pVertexAttributeDescriptions = surfAttributes;
        indexedPipeline_ = renderer.Pipelines().CreateGraphicsPipeline(indexedLayout_, {indexedVs_, indexedFs_}, indexed, renderer.GetRenderPass(), 0);
        if (wireframeSupported_) {
            indexed.rasterizer.polygonMode = VK_POLYGON_MODE_LINE;
            indexedWireframePipeline_ = renderer.Pipelines().CreateGraphicsPipeline(indexedLayout_, {indexedVs_, indexedFs_}, indexed, renderer.GetRenderPass(), 0);
        }
    }
    // Vertex colours reuse the indexed layout; the unused cell descriptor is
    // harmless and avoids changing the stable facet pipeline.
    indexedVcVs_ = renderer.Shaders().LoadSPIRV(renderer.cfg_.shaderDirectory + "/surface_indexed_vc.vert.spv", VK_SHADER_STAGE_VERTEX_BIT);
    indexedVcFs_ = renderer.Shaders().LoadSPIRV(renderer.cfg_.shaderDirectory + "/surface_indexed_vc.frag.spv", VK_SHADER_STAGE_FRAGMENT_BIT);
    if (indexedLayout_ != VK_NULL_HANDLE) {
        PipelineConfig vc = pipelineCfg_;
        vc.rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
        vc.depthStencil.depthCompareOp = VK_COMPARE_OP_LESS;
        vc.vertexBindings = {{0,12,VK_VERTEX_INPUT_RATE_VERTEX},{1,4,VK_VERTEX_INPUT_RATE_VERTEX}};
        vc.vertexAttributes = {{0,0,VK_FORMAT_R32G32B32_SFLOAT,0},{1,1,VK_FORMAT_R8G8B8A8_UNORM,0}};
        vc.vertexInput.vertexBindingDescriptionCount = 2;
        vc.vertexInput.pVertexBindingDescriptions = vc.vertexBindings.data();
        vc.vertexInput.vertexAttributeDescriptionCount = 2;
        vc.vertexInput.pVertexAttributeDescriptions = vc.vertexAttributes.data();
        indexedVcPipeline_ = renderer.Pipelines().CreateGraphicsPipeline(indexedLayout_, {indexedVcVs_,indexedVcFs_}, vc, renderer.GetRenderPass(),0);
        vc.inputAssembly.topology=VK_PRIMITIVE_TOPOLOGY_POINT_LIST;
        vc.depthStencil.depthCompareOp=VK_COMPARE_OP_LESS_OR_EQUAL;
        coveragePipeline_=renderer.Pipelines().CreateGraphicsPipeline(indexedLayout_,{indexedVcVs_,indexedVcFs_},vc,renderer.GetRenderPass(),0);
    }
    // Compact crisp Shaded-Class pipeline: the fragment stage reads gl_PrimitiveID,
    // which needs the geometryShader feature (same requirement as indexedPipeline_).
    if (renderer.GetContext().CoreDevice().GetEnabledFeatures().geometryShader) {
        crispIdxVs_ = renderer.Shaders().LoadSPIRV(renderer.cfg_.shaderDirectory + "/surface_crisp_indexed.vert.spv", VK_SHADER_STAGE_VERTEX_BIT);
        crispIdxFs_ = renderer.Shaders().LoadSPIRV(renderer.cfg_.shaderDirectory + "/surface_crisp_indexed.frag.spv", VK_SHADER_STAGE_FRAGMENT_BIT);
        if (crispIdxVs_.module != VK_NULL_HANDLE && crispIdxFs_.module != VK_NULL_HANDLE) {
            crispIdxSetLayout_ = renderer.Descriptors().CreateLayout({
                {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_FRAGMENT_BIT},
                {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_FRAGMENT_BIT},
                {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_FRAGMENT_BIT}});
            crispIdxLayout_ = renderer.Pipelines().CreatePipelineLayout(
                {renderer.GetFrameSetLayout(), crispIdxSetLayout_}, {pcr});
            crispIdxPool_ = renderer.Descriptors().CreatePool({{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3}}, 1);
            if (crispIdxPool_ != VK_NULL_HANDLE && crispIdxSetLayout_ != VK_NULL_HANDLE)
                crispIdxSet_ = renderer.Descriptors().AllocateSet(crispIdxPool_, crispIdxSetLayout_);
            PipelineConfig c = pipelineCfg_;
            c.rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
            c.depthStencil.depthCompareOp = VK_COMPARE_OP_LESS;
            c.vertexBindings = {{0, 12, VK_VERTEX_INPUT_RATE_VERTEX}, {1, 4, VK_VERTEX_INPUT_RATE_VERTEX}};
            c.vertexAttributes = {{0, 0, VK_FORMAT_R32G32B32_SFLOAT, 0}, {1, 1, VK_FORMAT_R8G8B8A8_UNORM, 0}};
            c.vertexInput.vertexBindingDescriptionCount = 2;
            c.vertexInput.pVertexBindingDescriptions = c.vertexBindings.data();
            c.vertexInput.vertexAttributeDescriptionCount = 2;
            c.vertexInput.pVertexAttributeDescriptions = c.vertexAttributes.data();
            if (crispIdxLayout_ != VK_NULL_HANDLE && crispIdxSet_ != VK_NULL_HANDLE)
                crispIdxPipeline_ = renderer.Pipelines().CreateGraphicsPipeline(
                    crispIdxLayout_, {crispIdxVs_, crispIdxFs_}, c, renderer.GetRenderPass(), 0);
        }
        if (crispIdxPipeline_ == VK_NULL_HANDLE)
            fprintf(stderr, "[SurfaceRenderer] compact crisp pipeline unavailable "
                            "(surface_crisp_indexed.*.spv missing?) - using the two-pass soup\n");
    }
    initialized_ = true;
    return (pipelineSolid_ != VK_NULL_HANDLE && pipelineSolidOverlay_ != VK_NULL_HANDLE);
}

void SurfaceRenderer::Shutdown() {
    if (!initialized_) return;
    initialized_ = false;
    ReleaseCrispIndexed();
    if (renderer_) {
        renderer_->Pipelines().DestroyPipeline(crispIdxPipeline_); crispIdxPipeline_ = VK_NULL_HANDLE;
        if (crispIdxLayout_ != VK_NULL_HANDLE) renderer_->Pipelines().DestroyPipelineLayout(crispIdxLayout_);
        crispIdxLayout_ = VK_NULL_HANDLE;
        if (crispIdxPool_ != VK_NULL_HANDLE) renderer_->Descriptors().DestroyPool(crispIdxPool_);
        crispIdxPool_ = VK_NULL_HANDLE; crispIdxSet_ = VK_NULL_HANDLE;
        if (crispIdxSetLayout_ != VK_NULL_HANDLE) renderer_->Descriptors().DestroyLayout(crispIdxSetLayout_);
        crispIdxSetLayout_ = VK_NULL_HANDLE;
        renderer_->Shaders().DestroyShaderModule(crispIdxVs_.module);
        renderer_->Shaders().DestroyShaderModule(crispIdxFs_.module);
        crispIdxVs_ = {}; crispIdxFs_ = {};
        renderer_->Pipelines().DestroyPipeline(pipelineSolid_);
        renderer_->Pipelines().DestroyPipeline(indexedPipeline_);
        renderer_->Pipelines().DestroyPipeline(indexedVcPipeline_);
        renderer_->Pipelines().DestroyPipeline(coveragePipeline_); coveragePipeline_=VK_NULL_HANDLE;
        renderer_->Shaders().DestroyShaderModule(indexedVcVs_.module);
        renderer_->Shaders().DestroyShaderModule(indexedVcFs_.module);
        indexedVcPipeline_ = VK_NULL_HANDLE; indexedVcVs_ = {}; indexedVcFs_ = {};
        renderer_->Pipelines().DestroyPipeline(indexedWireframePipeline_);
        renderer_->Pipelines().DestroyPipelineLayout(indexedLayout_);
        renderer_->Descriptors().DestroyLayout(cellLayout_);
        renderer_->Shaders().DestroyShaderModule(indexedVs_.module);
        renderer_->Shaders().DestroyShaderModule(indexedFs_.module);
        renderer_->Pipelines().DestroyPipeline(pipelineSolidOverlay_);
        renderer_->Pipelines().DestroyPipeline(pipelineWireframe_);
        renderer_->Pipelines().DestroyPipelineLayout(pipelineLayout_);
        renderer_->Shaders().DestroyShaderModule(vs_.module);
        // fsWireframe_ is the same handle as fsSolid_ (aliased in
        // Initialize() - there's only one fragment shader, shared by both
        // pipelines), so destroying it a second time here was a double-free
        // that the validation layer flagged as "invalid"/"couldn't find".
        renderer_->Shaders().DestroyShaderModule(fsSolid_.module);
    }
    VulkanAllocator::Get().DestroyBuffer(positions_);
    VulkanAllocator::Get().DestroyBuffer(normals_);
    VulkanAllocator::Get().DestroyBuffer(colors_);
    VulkanAllocator::Get().DestroyBuffer(indices_);
    // Tiled resource sets (ACTIVE / PENDING / retired). Shutdown is the one
    // place a device idle is legitimate: the device is going away.
    if (active_) { active_->ReleaseBuffers(); active_.reset(); }
    if (pending_) { pending_->ReleaseBuffers(); pending_.reset(); }
    for (auto& r : retired_) { if (r) r->ReleaseBuffers(); }
    retired_.clear();
    VulkanAllocator::Get().DestroyBuffer(positions2_);
    VulkanAllocator::Get().DestroyBuffer(normals2_);
    VulkanAllocator::Get().DestroyBuffer(colors2_);
    VulkanAllocator::Get().DestroyBuffer(indices2_);
    // GPU class-colour LUT (Phase C): 1 KB, own lifetime.
    VulkanAllocator::Get().DestroyBuffer(classLut_);
    classLut_ = {};
    positions_ = {}; normals_ = {}; colors_ = {}; indices_ = {};
    positions2_ = {}; normals2_ = {}; colors2_ = {}; indices2_ = {};
    overlayIndexCount_ = 0; crispHybridActive_ = false;
    pipelineSolid_ = VK_NULL_HANDLE; pipelineWireframe_ = VK_NULL_HANDLE;
    pipelineSolidOverlay_ = VK_NULL_HANDLE;
    pipelineLayout_ = VK_NULL_HANDLE;
    vs_ = {}; fsSolid_ = {}; fsWireframe_ = {};
    renderer_ = nullptr;
}

UploadStats SurfaceRenderer::UploadMesh(const float* positions, const float* normals,
                                              const uint8_t* colorsRGB,
                                              std::size_t vertexCount,
                                              const uint32_t* indices, std::size_t indexCount) {
    UploadStats stats;
    ReleaseCrispIndexed();
    vertexCount_ = vertexCount;
    triangleCount_ = indexCount / 3;
    indexCount_ = indexCount;

    auto uploadOne = [&](GPUBuffer& buf, VkBufferUsageFlags extra,
                             VkDeviceSize bytes, const void* data) {
        UploadStats s;
        if (bytes == 0) return s;
        if (!buf.IsValid() || buf.size < bytes) {
            VulkanAllocator::Get().DestroyBuffer(buf);
            buf = VulkanAllocator::Get().CreateBuffer(
                bytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT | extra,
                VMA_MEMORY_USAGE_GPU_ONLY);
            if (!buf.IsValid()) return s;
        }
        uint8_t* dst = renderer_->GetUploadContext().Stage(bytes);
        if (!dst) return s;
        double t2 = NowMs();
        std::memcpy(dst, data, bytes);
        double t3 = NowMs();
        CopyRegion reg{buf.buffer, 0, 0, bytes};
        s = renderer_->GetUploadContext().Execute({reg}, 0.0, t3 - t2);
        stats.Accumulate(s);
        return s;
    };

    uploadOne(positions_, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
              vertexCount * 3 * sizeof(float), positions);
    uploadOne(normals_, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
              vertexCount * 3 * sizeof(float), normals);
    uploadOne(colors_, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
              vertexCount * 4, colorsRGB);
    if (indexCount > 0) {
        uploadOne(indices_, VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
                  indexCount * sizeof(uint32_t), indices);
    }

    crispHybridActive_ = false;   // legacy single mesh: never the crisp formula
    meshUploads_++;
    return stats;
}

bool SurfaceRenderer::UpdateFaceColors(const uint8_t* rgb, std::size_t faceCount) {
    if (!initialized_ || !rgb || active_ || pending_ || tilesActive_ ||
        crispHybridActive_ || !colors_.IsValid() || faceCount != triangleCount_ ||
        vertexCount_ != faceCount * 3) return false;
    const std::size_t bytes = faceCount * 3 * 4;
    auto* dst = renderer_->GetUploadContext().Stage(bytes);
    if (!dst) return false;
    for (std::size_t f = 0; f < faceCount; ++f) {
        for (std::size_t v = 0; v < 3; ++v) {
            auto* c = dst + (f * 3 + v) * 4;
            c[0] = rgb[f*3]; c[1] = rgb[f*3+1]; c[2] = rgb[f*3+2]; c[3] = 255;
        }
    }
    return renderer_->GetUploadContext().Execute(
        {{colors_.buffer, 0, 0, bytes}}, 0.0, 0.0).bytes == bytes;
}

UploadStats SurfaceRenderer::UploadShadedClassMesh(
        const float* basePositions, const float* baseNormals, const uint8_t* baseColorsRGB,
        std::size_t baseVertexCount, const uint32_t* baseIndices, std::size_t baseIndexCount,
        const float* overlayPositions, const float* overlayNormals, const uint8_t* overlayColorsRGB,
        std::size_t overlayVertexCount, const uint32_t* overlayIndices, std::size_t overlayIndexCount) {
    UploadStats stats;
    ReleaseCrispIndexed();

    auto uploadOne = [&](GPUBuffer& buf, VkBufferUsageFlags extra,
                             VkDeviceSize bytes, const void* data) {
        UploadStats s;
        if (bytes == 0) return s;
        if (!buf.IsValid() || buf.size < bytes) {
            VulkanAllocator::Get().DestroyBuffer(buf);
            buf = VulkanAllocator::Get().CreateBuffer(
                bytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT | extra,
                VMA_MEMORY_USAGE_GPU_ONLY);
            if (!buf.IsValid()) return s;
        }
        uint8_t* dst = renderer_->GetUploadContext().Stage(bytes);
        if (!dst) return s;
        double t2 = NowMs();
        std::memcpy(dst, data, bytes);
        double t3 = NowMs();
        CopyRegion reg{buf.buffer, 0, 0, bytes};
        s = renderer_->GetUploadContext().Execute({reg}, 0.0, t3 - t2);
        stats.Accumulate(s);
        return s;
    };

    // PASS 1 base - reuses the same positions_/normals_/colors_/indices_
    // buffers UploadMesh() uses, so Record()'s existing bind/draw code for
    // the "single mesh" surface_mode.py path needs no changes.
    vertexCount_ = baseVertexCount;
    indexCount_ = baseIndexCount;
    triangleCount_ = baseIndexCount / 3;
    uploadOne(positions_, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
              baseVertexCount * 3 * sizeof(float), basePositions);
    uploadOne(normals_, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
              baseVertexCount * 3 * sizeof(float), baseNormals);
    uploadOne(colors_, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
              baseVertexCount * 4, baseColorsRGB);
    if (baseIndexCount > 0) {
        uploadOne(indices_, VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
                  baseIndexCount * sizeof(uint32_t), baseIndices);
    }

    // PASS 2 overlay - separate buffers, only the mixed-class faces.
    overlayIndexCount_ = overlayIndexCount;
    if (overlayVertexCount > 0) {
        uploadOne(positions2_, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                  overlayVertexCount * 3 * sizeof(float), overlayPositions);
        uploadOne(normals2_, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                  overlayVertexCount * 3 * sizeof(float), overlayNormals);
        uploadOne(colors2_, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                  overlayVertexCount * 4, overlayColorsRGB);
    }
    if (overlayIndexCount > 0) {
        uploadOne(indices2_, VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
                  overlayIndexCount * sizeof(uint32_t), overlayIndices);
    }

    crispHybridActive_ = true;
    meshUploads_++;
    return stats;
}

// ---- GPU class-colour LUT (Phase C) --------------------------------------
//
// A palette change must be a 1 KB upload and nothing else. This deliberately
// does NOT call UploadMesh/UploadShadedClassMesh, does not touch
// positions_/normals_/indices_, and does NOT increment meshUploads_ -- so the
// "palette change is not a geometry change" invariant is provable from
// GetMeshUploadCount() alone.
UploadStats SurfaceRenderer::SetClassColorLut(const uint8_t* colorsRGB) {
    UploadStats stats;
    if (!colorsRGB || !initialized_) return stats;

    // Expand RGB -> RGBA8 (alpha 255) so the shader indexes with a byte stride.
    std::vector<uint8_t> rgba(kClassLutEntries * 4);
    for (std::size_t i = 0; i < kClassLutEntries; ++i) {
        rgba[i * 4 + 0] = colorsRGB[i * 3 + 0];
        rgba[i * 4 + 1] = colorsRGB[i * 3 + 1];
        rgba[i * 4 + 2] = colorsRGB[i * 3 + 2];
        rgba[i * 4 + 3] = 255;
    }
    const std::size_t bytes = rgba.size();
    if (!classLut_.IsValid()) {
        classLut_ = VulkanAllocator::Get().CreateBuffer(
            bytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            VMA_MEMORY_USAGE_GPU_ONLY);
        if (!classLut_.IsValid()) return stats;
    }
    uint8_t* dst = renderer_->GetUploadContext().Stage(bytes);
    if (!dst) return stats;
    const double t2 = NowMs();
    std::memcpy(dst, rgba.data(), bytes);
    const double t3 = NowMs();
    CopyRegion reg{classLut_.buffer, 0, 0, bytes};
    stats = renderer_->GetUploadContext().Execute({reg}, 0.0, t3 - t2);
    classLutUpdates_++;
    lutRevision_++;
    return stats;
}

void SurfaceRenderer::SetShadingParityParams(int colorMode, int debugStage,
                                             float ambientFloor,
                                             float baseLightElevationDeg) {
    parityColorMode_ = (colorMode != 0) ? 1 : 0;
    parityDebugStage_ = (debugStage < 0) ? 0 : debugStage;
    parityAmbientFloor_ = ambientFloor;
    parityBaseElevationDeg_ = baseLightElevationDeg;
    parityParamUpdates_++;
}

void SurfaceRenderer::SetCrispShadingParams(float azimuthDeg, float sharpnessRaw, float ambient,
                                             float keyIntensity, float fillIntensity) {
    crispAzimuthDeg_ = azimuthDeg;
    crispSharpnessRaw_ = sharpnessRaw;
    crispAmbient_ = ambient;
    crispKeyIntensity_ = keyIntensity;
    crispFillIntensity_ = fillIntensity;
    shadeParamUpdates_++;
}

void SurfaceRenderer::SetShadingMode(ShadingMode mode) {
    if (shadingMode_ != mode) modeSwitches_++;
    shadingMode_ = mode;
}
ShadingMode SurfaceRenderer::GetShadingMode() const { return shadingMode_; }

void SurfaceRenderer::SetLightDirection(float x, float y, float z) {
    lightDir_[0] = x; lightDir_[1] = y; lightDir_[2] = z;
}
void SurfaceRenderer::SetAmbient(float ambient) { ambient_ = ambient; }
void SurfaceRenderer::SetElevationRange(float zMin, float zMax) {
    zMin_ = zMin; zMax_ = zMax;
}

// ---------------------------------------------------------------------------
// Tiled / LOD-aware upload with an atomic ACTIVE <-> PENDING swap
// ---------------------------------------------------------------------------
namespace {

// Uploads one LOD level's four arrays in a SINGLE staging batch + copy
// submit. Returns false if any buffer could not be created.
bool UploadOneLevel(Renderer& renderer, SurfaceLodBuffers& dst,
                    const SurfaceLodMesh& mesh, UploadStats& stats) {
    const VkDeviceSize posBytes = mesh.positions.size() * sizeof(float);
    const VkDeviceSize nrmBytes = mesh.normals.size() * sizeof(float);
    const VkDeviceSize colBytes = mesh.colors.size() * sizeof(uint8_t);
    const VkDeviceSize idxBytes = mesh.indices.size() * sizeof(uint32_t);
    const VkDeviceSize total = posBytes + nrmBytes + colBytes + idxBytes;
    if (total == 0) return false;

    dst.positions = VulkanAllocator::Get().CreateBuffer(
        posBytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
        VMA_MEMORY_USAGE_GPU_ONLY);
    dst.normals = VulkanAllocator::Get().CreateBuffer(
        nrmBytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
        VMA_MEMORY_USAGE_GPU_ONLY);
    dst.colors = VulkanAllocator::Get().CreateBuffer(
        colBytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
        VMA_MEMORY_USAGE_GPU_ONLY);
    dst.indices = VulkanAllocator::Get().CreateBuffer(
        idxBytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
        VMA_MEMORY_USAGE_GPU_ONLY);
    if (!dst.IsValid()) { dst.Release(); return false; }

    // The staging buffer is one grow-only allocation, so all four regions are
    // packed back to back and copied in a single submit.
    uint8_t* stage = renderer.GetUploadContext().Stage(total);
    if (!stage) { dst.Release(); return false; }
    std::memcpy(stage + 0, mesh.positions.data(), posBytes);
    std::memcpy(stage + posBytes, mesh.normals.data(), nrmBytes);
    std::memcpy(stage + posBytes + nrmBytes, mesh.colors.data(), colBytes);
    std::memcpy(stage + posBytes + nrmBytes + colBytes, mesh.indices.data(), idxBytes);

    const std::vector<CopyRegion> regions{
        {dst.positions.buffer, 0, 0,                             posBytes},
        {dst.normals.buffer,   0, posBytes,                      nrmBytes},
        {dst.colors.buffer,    0, posBytes + nrmBytes,           colBytes},
        {dst.indices.buffer,   0, posBytes + nrmBytes + colBytes, idxBytes},
    };
    stats.Accumulate(renderer.GetUploadContext().Execute(regions, 0.0, 0.0));
    return true;
}

} // namespace

SurfaceRenderer::UploadTiledResult SurfaceRenderer::UploadTiledMesh(
        const float* positions, const float* normals, const uint8_t* colors,
        std::size_t triangleCount, uint32_t targetTileCount, float lodCellMeters,
        bool isPreview, uint64_t datasetRevision, uint64_t surfaceRevision) {
    UploadTiledResult res;
    if (!renderer_ || !initialized_ || !positions || triangleCount == 0) return res;

    const std::size_t vertexCount = triangleCount * 3;
    const std::size_t indexCount = triangleCount * 3;

    // ---- CPU side: tile level 0, then cluster the coarse levels ------------
    // Level 0 is a PERMUTATION of the input soup, so the final visual result
    // is identical to the untiled upload - tiling changes only the ORDER.
    std::vector<SurfaceLodMesh> levels(kSurfaceLodCount);
    levels[0].positions.assign(positions, positions + vertexCount * 3);
    levels[0].normals = normals ? std::vector<float>(normals, normals + vertexCount * 3)
                                : std::vector<float>(vertexCount * 3, 0.0f);
    levels[0].colors = colors ? std::vector<uint8_t>(colors, colors + vertexCount * 4)
                              : std::vector<uint8_t>(vertexCount * 4, 255);
    levels[0].vertexCount = vertexCount;
    levels[0].triangleCount = triangleCount;
    levels[0].cellSizeMeters = lodCellMeters;

    // The soup's implicit index is the identity; materialise it once.
    std::vector<uint32_t> identity(indexCount);
    for (std::size_t i = 0; i < indexCount; ++i) identity[i] = static_cast<uint32_t>(i);
    BuildSurfaceTiles(levels[0].positions.data(), vertexCount,
                      identity.data(), indexCount, targetTileCount,
                      levels[0].indices, levels[0].tiles);
    if (levels[0].indices.empty() || levels[0].tiles.empty()) {
        fprintf(stderr, "[SurfaceRenderer] tiling produced no tiles (%zu tris)\n",
                triangleCount);
        return res;
    }

    for (uint32_t lod = 1; lod < kSurfaceLodCount; ++lod) {
        // 4x then 16x the base cell size: a 2x then 4x reduction per axis,
        // which gives the projected-error test real levels to choose between
        // rather than a single binary choice.
        const float cell = lodCellMeters * (4.0f * static_cast<float>(lod * lod));
        if (!(cell > 0.0f)) continue;
        SurfaceLodMesh coarse;
        if (!ClusterSurfaceLod(levels[0].positions.data(), vertexCount,
                               levels[0].indices.data(), levels[0].indices.size(),
                               levels[0].colors.data(), cell, coarse)) {
            continue;   // level stays empty; LOD selection skips it
        }
        // Re-tile the coarse mesh so it is independently cullable too.
        BuildSurfaceTiles(coarse.positions.data(), coarse.vertexCount,
                          coarse.indices.data(), coarse.indices.size(),
                          targetTileCount, coarse.indices, coarse.tiles);
        levels[lod] = std::move(coarse);
    }

    // ---- Memory budget (Part 11) ------------------------------------------
    // An atomic swap means ACTIVE and PENDING coexist, so the check must
    // account for BOTH plus the incoming set. Refusing here keeps the current
    // surface on screen instead of failing a VMA allocation mid-copy.
    VkDeviceSize needed = 0;
    for (uint32_t lod = 0; lod < kSurfaceLodCount; ++lod) {
        const SurfaceLodMesh& m = levels[lod];
        if (m.triangleCount == 0) continue;
        needed += static_cast<VkDeviceSize>(m.positions.size() * sizeof(float));
        needed += static_cast<VkDeviceSize>(m.normals.size() * sizeof(float));
        needed += static_cast<VkDeviceSize>(m.colors.size() * sizeof(uint8_t));
        needed += static_cast<VkDeviceSize>(m.indices.size() * sizeof(uint32_t));
    }
    const uint64_t vramTotal = renderer_->GetContext().GetReport().vramBytes;
    if (vramBudgetBytes_ <= 0.0 && vramTotal > 0) {
        vramBudgetBytes_ = static_cast<double>(vramTotal) * 0.70;
    }
    const double resident = static_cast<double>(GetActiveBytes()) +
                            static_cast<double>(GetPendingBytes());
    if (vramBudgetBytes_ > 0.0 && (resident + static_cast<double>(needed)) > vramBudgetBytes_) {
        memoryRejected_ = true;
        res.memoryRejected = true;
        fprintf(stderr,
                "[SurfaceRenderer] upload REJECTED by VRAM budget: resident "
                "%.0f MB + new %.0f MB > budget %.0f MB (70%% of %.0f MB device)\n",
                resident / 1048576.0, static_cast<double>(needed) / 1048576.0,
                vramBudgetBytes_ / 1048576.0, static_cast<double>(vramTotal) / 1048576.0);
        return res;
    }
    memoryRejected_ = false;

    auto set = std::make_unique<SurfaceResourceSet>();
    set->vertexCount = vertexCount;
    set->indexCount = indexCount;
    set->triangleCount = triangleCount;
    set->isPreview = isPreview;
    set->datasetRevision = datasetRevision;
    set->surfaceRevision = surfaceRevision;
    set->totalTiles = static_cast<uint32_t>(levels[0].tiles.size());

    for (uint32_t lod = 0; lod < kSurfaceLodCount; ++lod) {
        if (levels[lod].triangleCount == 0) continue;
        if (!UploadOneLevel(*renderer_, set->lods[lod], levels[lod], res.stats)) {
            set->ReleaseBuffers();
            return res;
        }
        set->gpuBytes += set->lods[lod].Bytes();
        res.lodTriangles[lod] = levels[lod].triangleCount;
    }
    if (!set->IsValid()) { set->ReleaseBuffers(); return res; }

    set->lodMeshes = std::move(levels);
    res.tileCount = set->totalTiles;

    // Supersede any previous pending set; it is retired, not destroyed here.
    if (pending_) retired_.push_back(std::move(pending_));
    pending_ = std::move(set);
    // A tiled upload is presented by the tiled branch of Record(), which reads
    // crispHybridActive_ as "force fragment mode 4". Leaving the flag set would
    // re-light this mesh with the Shaded-Class formula - double-shading a mesh
    // whose colours are already baked on the CPU (the elevation ramp).
    crispHybridActive_ = false;
    meshUploads_++;

    const VkDeviceSize peak = GetActiveBytes() + GetPendingBytes();
    if (peak > peakBytesDuringSwap_) peakBytesDuringSwap_ = peak;
    res.peakBytesDuringSwap = static_cast<double>(peak);
    res.ok = true;
    return res;
}

// ---------------------------------------------------------------------------
// Atomic swap, retirement, culling and telemetry
// ---------------------------------------------------------------------------
SurfaceRenderer::UploadTiledResult SurfaceRenderer::UploadIndexedBlocks(
        const float* positions, std::size_t vertexCount, const uint32_t* indices,
        std::size_t indexCount, const uint32_t* cellColors,
        const uint32_t* blockCounts, uint32_t blockCount,
        uint64_t datasetRevision, uint64_t surfaceRevision, const uint32_t* vertexColors) {
    UploadTiledResult result;
    if (!initialized_ || indexedPipeline_ == VK_NULL_HANDLE || !positions || !indices ||
        !cellColors || !blockCounts || !vertexCount || !indexCount || indexCount % 3) return result;
    if (vertexColors && indexedVcPipeline_ == VK_NULL_HANDLE) return result;
    uint64_t count = 0;
    for (uint32_t i = 0; i < blockCount; ++i) count += blockCounts[i];
    if (count != indexCount) return result;
    for (std::size_t i = 0; i < indexCount; ++i) if (indices[i] >= vertexCount) return result;
    const VkDeviceSize posBytes = vertexCount * 12, idxBytes = indexCount * 4, cellBytes = indexCount / 3 * 4;
    const VkDeviceSize colorBytes = vertexColors ? vertexCount * 4 : 0;
    const VkDeviceSize required = posBytes + idxBytes + cellBytes + colorBytes;
    if (GetActiveBytes() + GetPendingBytes() + required > vramBudgetBytes_) {
        memoryRejected_ = result.memoryRejected = true; return result;
    }
    auto set = std::make_unique<SurfaceResourceSet>();
    set->vertexColored = vertexColors != nullptr;
    set->indexedCells = true; set->datasetRevision = datasetRevision; set->surfaceRevision = surfaceRevision;
    set->vertexCount = vertexCount; set->indexCount = indexCount; set->triangleCount = indexCount / 3;
    SurfaceLodMesh mesh;
    mesh.positions.assign(positions, positions + vertexCount * 3);
    mesh.indices.assign(indices, indices + indexCount);
    mesh.vertexCount = vertexCount; mesh.triangleCount = indexCount / 3;
    uint32_t first = 0;
    for (uint32_t i = 0; i < blockCount; ++i) {
        SurfaceTile tile; tile.firstIndex = first; tile.indexCount = blockCounts[i];
        tile.triangleCount = blockCounts[i] / 3; tile.residency = 1;
        if (tile.indexCount) {
            for (int k = 0; k < 3; ++k) tile.bmin[k] = tile.bmax[k] = positions[indices[first] * 3 + k];
            for (uint32_t j = first; j < first + tile.indexCount; ++j)
                for (int k = 0; k < 3; ++k) {
                    float value = positions[indices[j] * 3 + k];
                    tile.bmin[k] = std::min(tile.bmin[k], value); tile.bmax[k] = std::max(tile.bmax[k], value);
                }
            mesh.tiles.push_back(tile);
        }
        first += tile.indexCount;
    }
    set->lodMeshes.resize(kSurfaceLodCount); set->lodMeshes[0] = std::move(mesh);
    set->totalTiles = static_cast<uint32_t>(set->lodMeshes[0].tiles.size());
    auto& dst = set->lods[0]; auto& allocator = VulkanAllocator::Get();
    dst.positions = allocator.CreateBuffer(posBytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
    dst.indices = allocator.CreateBuffer(idxBytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
    dst.cellColors = allocator.CreateBuffer(cellBytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
    if (!dst.IsValid() || !dst.cellColors.IsValid()) { set->ReleaseBuffers(); return result; }
    if (vertexColors) {
        dst.colors = allocator.CreateBuffer(colorBytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
        if (!dst.colors.IsValid()) { set->ReleaseBuffers(); return result; }
    }
    auto* staging = renderer_->GetUploadContext().Stage(required);
    if (!staging) { set->ReleaseBuffers(); return result; }
    std::memcpy(staging, positions, posBytes); std::memcpy(staging + posBytes, indices, idxBytes);
    std::memcpy(staging + posBytes + idxBytes, cellColors, cellBytes);
    if (vertexColors) {
        std::memcpy(staging + posBytes + idxBytes + cellBytes, vertexColors, colorBytes);
        result.stats.Accumulate(renderer_->GetUploadContext().Execute({
            {dst.positions.buffer,0,0,posBytes}, {dst.indices.buffer,0,posBytes,idxBytes},
            {dst.cellColors.buffer,0,posBytes+idxBytes,cellBytes},
            {dst.colors.buffer,0,posBytes+idxBytes+cellBytes,colorBytes}},0.0,0.0));
    } else {
    result.stats.Accumulate(renderer_->GetUploadContext().Execute({
        {dst.positions.buffer, 0, 0, posBytes}, {dst.indices.buffer, 0, posBytes, idxBytes},
        {dst.cellColors.buffer, 0, posBytes + idxBytes, cellBytes}}, 0.0, 0.0));
    }
    dst.cellDevice = renderer_->GetContext().GetDevice();
    dst.cellPool = renderer_->Descriptors().CreatePool({{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1}}, 1);
    dst.cellSet = renderer_->Descriptors().AllocateSet(dst.cellPool, cellLayout_);
    if (dst.cellSet == VK_NULL_HANDLE) { set->ReleaseBuffers(); return result; }
    renderer_->Descriptors().UpdateBuffer(dst.cellSet, 0, dst.cellColors.buffer, cellBytes, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
    set->gpuBytes = required;
    if (pending_) retired_.push_back(std::move(pending_));
    pending_ = std::move(set); crispHybridActive_ = false;  // tiled path, see UploadTiledMesh
    meshUploads_++; memoryRejected_ = false;
    result.ok = true; result.tileCount = pending_->totalTiles; result.lodTriangles[0] = indexCount / 3;
    peakBytesDuringSwap_ = std::max<uint64_t>(peakBytesDuringSwap_, GetActiveBytes() + GetPendingBytes());
    return result;
}

bool SurfaceRenderer::UploadPendingCoverage(const float* positions,const uint32_t* colors,std::size_t count,
        uint64_t datasetRevision,uint64_t surfaceRevision) {
    if(!initialized_ || coveragePipeline_==VK_NULL_HANDLE || !positions || !colors || !count || count>UINT32_MAX) return false;
    auto refuse=[&]() { if(pending_) retired_.push_back(std::move(pending_)); return false; };
    const VkDeviceSize posBytes=count*12,colorBytes=count*4,required=posBytes+colorBytes;
    if(GetActiveBytes()+GetPendingBytes()+required+16>vramBudgetBytes_) {
        memoryRejected_=true;return refuse();
    }
    if(!pending_) {
        // A cut containing only isolated/collinear points still has a valid
        // atomic resource set. Dummy indexed buffers are never drawn.
        pending_=std::make_unique<SurfaceResourceSet>();
        pending_->indexedCells=true;pending_->datasetRevision=datasetRevision;
        pending_->surfaceRevision=surfaceRevision;pending_->lodMeshes.resize(kSurfaceLodCount);
        auto& a=VulkanAllocator::Get();
        pending_->lods[0].positions=a.CreateBuffer(12,VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,VMA_MEMORY_USAGE_GPU_ONLY);
        pending_->lods[0].indices=a.CreateBuffer(4,VK_BUFFER_USAGE_INDEX_BUFFER_BIT,VMA_MEMORY_USAGE_GPU_ONLY);
        pending_->gpuBytes=16;
        if(!pending_->IsValid()) return refuse();
    }
    if(pending_->datasetRevision!=datasetRevision||pending_->surfaceRevision!=surfaceRevision) return refuse();
    auto& a=VulkanAllocator::Get();
    pending_->coveragePositions=a.CreateBuffer(posBytes,VK_BUFFER_USAGE_TRANSFER_DST_BIT|VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,VMA_MEMORY_USAGE_GPU_ONLY);
    pending_->coverageColors=a.CreateBuffer(colorBytes,VK_BUFFER_USAGE_TRANSFER_DST_BIT|VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,VMA_MEMORY_USAGE_GPU_ONLY);
    if(!pending_->coveragePositions.IsValid()||!pending_->coverageColors.IsValid()) return refuse();
    auto* staging=renderer_->GetUploadContext().Stage(required);
    if(!staging) return refuse();
    std::memcpy(staging,positions,posBytes);std::memcpy(staging+posBytes,colors,colorBytes);
    renderer_->GetUploadContext().Execute({{pending_->coveragePositions.buffer,0,0,posBytes},
        {pending_->coverageColors.buffer,0,posBytes,colorBytes}},0.0,0.0);
    pending_->coverageCount=count;pending_->gpuBytes+=required;
    peakBytesDuringSwap_=std::max<uint64_t>(peakBytesDuringSwap_,GetActiveBytes()+GetPendingBytes());
    return true;
}

bool SurfaceRenderer::ActivatePendingIfFresh(bool* discardedPending) {
    if (discardedPending) *discardedPending = false;
    if (!pending_) return false;

    // Stale guard (Part 10): a pending set built for a different dataset or a
    // different surface generation must never be promoted. Discard it through
    // the normal retirement path so its buffers are freed safely.
    if (pending_->datasetRevision != currentDatasetRevision_ ||
        pending_->surfaceRevision != currentSurfaceRevision_) {
        staleDiscards_++;
        if (discardedPending) *discardedPending = true;
        fprintf(stderr,
                "[SurfaceRenderer] DISCARDED stale pending surface "
                "(dataset %llu->%llu, surface %llu->%llu)\n",
                (unsigned long long)pending_->datasetRevision,
                (unsigned long long)currentDatasetRevision_,
                (unsigned long long)pending_->surfaceRevision,
                (unsigned long long)currentSurfaceRevision_);
        retired_.push_back(std::move(pending_));
        return false;
    }

    // The outgoing ACTIVE set may still be referenced by an in-flight command
    // buffer, so it is retired, never destroyed here. The swap itself is two
    // pointer moves: there is no frame in which neither set is drawable, which
    // is what makes the transition atomic with respect to presentation.
    if (active_) retired_.push_back(std::move(active_));
    active_ = std::move(pending_);
    // A tiled/indexed set now presents; the compact crisp mesh must not linger in VRAM.
    if (crispIndexedActive_ || crispIndices_.IsValid()) ReleaseCrispIndexed();
    // Start at the coarsest level that actually has resident geometry. The
    // indexed-block path materialises only level 0, so unconditionally
    // selecting kSurfaceLodCount-1 left selectedLod pointing at an EMPTY level
    // whenever CullVisibleTiles() returned before recomputing it - the draw
    // loop then found no tiles and the Surface silently produced zero indexed
    // draws while still reporting itself ACTIVE.
    uint32_t resident = 0;
    const uint32_t levelCount = static_cast<uint32_t>(active_->lodMeshes.size());
    for (uint32_t lod = 0; lod < levelCount && lod < kSurfaceLodCount; ++lod) {
        if (active_->lodMeshes[lod].triangleCount != 0 && active_->lods[lod].IsValid()) {
            resident = lod;
        }
    }
    active_->selectedLod = resident;
    swapCount_++;
    return true;
}

void SurfaceRenderer::RetireCompleted() {
    if (retired_.empty() || !renderer_) return;
    // A set is safe to destroy once the frame that last used it has provably
    // completed. Renderer::IsFrameRetired() is backed by the per-slot in-flight
    // FENCES the frame loop already waits on, so this costs no device-wide
    // stall - which is exactly what a vkDeviceWaitIdle() per swap would cost.
    for (std::size_t i = retired_.size(); i-- > 0;) {
        if (renderer_->IsFrameRetired(retired_[i]->lastUsedFrame)) {
            retired_[i]->ReleaseBuffers();
            retired_.erase(retired_.begin() + static_cast<std::ptrdiff_t>(i));
        }
    }
}

void SurfaceRenderer::SetInteractionState(bool moving, double nowMs) {
    if (moving) {
        interactionMoving_ = true;
        lastInteractionMs_ = nowMs;
        // While the camera is in motion the interactive floor is enforced, so a
        // 53M-triangle SLOW surface can never be the navigation geometry.
        if (requestedLod_ < movingLodFloor_) requestedLod_ = movingLodFloor_;
    } else {
        interactionMoving_ = false;
        lastInteractionMs_ = nowMs;
        requestedLod_ = 0;   // settled: the final quality is allowed back
    }
}

uint64_t SurfaceRenderer::EstimateRequiredBytes(std::size_t triangleCount) const {
    // A triangle soup is 3 vertices of (pos 12 + normal 12 + colour 4) = 28 B,
    // i.e. 84 B per triangle, plus 3 uint32 indices = 12 B. The coarse LOD
    // levels are clustered to roughly 1/16 and 1/256 of that, which is the same
    // ratio the real build produces, so a small constant is added for them.
    const double perTri = 84.0 + 12.0;
    const double lodFactor = 1.0 + 1.0 / 16.0 + 1.0 / 256.0;
    return static_cast<uint64_t>(static_cast<double>(triangleCount) * perTri * lodFactor);
}

bool SurfaceRenderer::WouldAcceptUpload(std::size_t triangleCount) const {
    if (vramBudgetBytes_ <= 0.0) return true;   // no budget known: do not block
    const double resident = static_cast<double>(GetActiveBytes()) +
                            static_cast<double>(GetPendingBytes());
    return (resident + static_cast<double>(EstimateRequiredBytes(triangleCount)))
           <= vramBudgetBytes_;
}

SurfaceState SurfaceRenderer::GetState() const {
    if (pending_) return SurfaceState::FinalPending;
    if (!active_) return SurfaceState::Empty;
    return active_->isPreview ? SurfaceState::PreviewActive : SurfaceState::FinalActive;
}

uint32_t SurfaceRenderer::GetTotalTileCount() const {
    if (tilesActive_) return uint32_t(tileSlots_.size());
    if(crispIndexedActive_)return crispTiles_.empty()?1:uint32_t(crispTiles_.size());
    // The tile COUNT is per level: each LOD is re-tiled independently (LOD1/LOD2
    // have far fewer, smaller triangles, so the same target tile budget splits
    // into a different number of runs). Reporting level 0's count against the
    // SELECTED level's visible count produced nonsense like "176 visible of
    // 170 total", so the total is read from the level actually being drawn.
    if (!active_ || active_->lodMeshes.empty()) return 0;
    const uint32_t lod = active_->selectedLod;
    return static_cast<uint32_t>(active_->lodMeshes[lod].tiles.size());
}

uint32_t SurfaceRenderer::GetVisibleTileCount() const {
    if (tilesActive_) return uint32_t(tileDrawnTiles_);
    if(crispIndexedActive_)return crispVisibleTiles_;
    return active_ ? active_->visibleTilesLastFrame : 0;
}

uint64_t SurfaceRenderer::GetTotalTriangleCount() const {
    // Reported for the level actually being drawn, so "submitted / total" is a
    // meaningful ratio. Culling and LOD only ever reduce the drawn set, and the
    // per-level counts are available separately via GetLodTriangleCount().
    //
    // When no tiled set is resident, the drawn mesh IS the single resident one
    // (UploadMesh / UploadShadedClassMesh - the Shaded-Class crisp path), whose
    // count lives in triangleCount_. Returning 0 there reported "no surface"
    // for a mesh that was on screen: the .lnt Shading view showed 0 triangles
    // the moment it started using the crisp path.
    if (tilesActive_) return tileF_.used;
    if (!active_ || active_->lodMeshes.empty()) return triangleCount_;
    return static_cast<uint64_t>(active_->lodMeshes[active_->selectedLod].triangleCount);
}

uint64_t SurfaceRenderer::GetVisibleTriangleCount() const {
    if (tilesActive_) return tileDrawnFaces_;
    if(crispIndexedActive_)return crispSubmittedTriangles_;
    return active_ ? active_->submittedTrianglesLastFrame : 0;
}

uint32_t SurfaceRenderer::GetSelectedLod() const {
    return active_ ? active_->selectedLod : 0;
}

uint64_t SurfaceRenderer::GetLodTriangleCount(uint32_t lod) const {
    if (!active_ || lod >= kSurfaceLodCount) return 0;
    return static_cast<uint64_t>(active_->lodMeshes[lod].triangleCount);
}

VkDeviceSize SurfaceRenderer::GetActiveBytes() const {
    return active_ ? active_->gpuBytes : 0;
}

VkDeviceSize SurfaceRenderer::GetPendingBytes() const {
    return pending_ ? pending_->gpuBytes : 0;
}

VkDeviceSize SurfaceRenderer::GetLodCacheBytes() const {
    // Coarse levels are the cache: level 0 is the engineering surface, so
    // everything above it is the resident LOD cache.
    if (!active_) return 0;
    VkDeviceSize sum = 0;
    for (uint32_t lod = 1; lod < kSurfaceLodCount; ++lod) {
        sum += active_->lods[lod].Bytes();
    }
    return sum;
}

// Rebuilds the visible-tile list for this frame from the live camera. Called
// once per frame, never per tile: the frustum and the projected scale are both
// derived from the CURRENT MVP here and then reused by every tile test.
void SurfaceRenderer::CullVisibleTiles(double nowMs) {
    for (uint32_t lod = 0; lod < kSurfaceLodCount; ++lod) {
        active_->visibleTiles[lod].clear();
    }
    if (!active_ || active_->lodMeshes.empty()) return;

    // ---- Idle refine (Part 8) --------------------------------------------
    // After the debounce window the requested (final) level is allowed back.
    if (interactionMoving_ && (nowMs - lastInteractionMs_) >= idleRefineMs_) {
        interactionMoving_ = false;
        requestedLod_ = 0;
    }

    float mvp[16];
    renderer_->GetLastMvp(mvp);
    ExtractFrustumPlanes(mvp, lastFrustum_);

    const Camera& cam = renderer_->GetCamera();
    const VkExtent2D ext = renderer_->GetExtent();
    lastScale_ = MakeProjectedScale(
        cam.IsOrthographic(), cam.GetParallelScale(),
        cam.GetFovYDegrees(), static_cast<double>(ext.height));

    // Highest level with resident geometry is the finest available.
    uint32_t finest = 0;
    bool any = false;
    for (uint32_t lod = 0; lod < kSurfaceLodCount; ++lod) {
        if (active_->lodMeshes[lod].triangleCount != 0 && active_->lods[lod].IsValid()) {
            finest = lod;
            any = true;
        }
    }
    if (!any) return;

    // ---- Frustum cull EVERY level independently (Part 7) -----------------
    // Each level is culled against its OWN tile bounds. Doing it per level
    // (rather than reusing level 0's partition) is what keeps the levels'
    // draws self-consistent: a tile is only drawn with triangles that are
    // actually inside the bounds that were tested, so culling can never clip
    // geometry that is on screen.
    for (uint32_t lod = 0; lod <= finest; ++lod) {
        const SurfaceLodMesh& m = active_->lodMeshes[lod];
        if (m.triangleCount == 0 || !active_->lods[lod].IsValid()) continue;
        std::vector<uint32_t>& vis = active_->visibleTiles[lod];
        vis.reserve(m.tiles.size());
        for (std::size_t t = 0; t < m.tiles.size(); ++t) {
            if (AabbVisible(lastFrustum_, m.tiles[t].bmin, m.tiles[t].bmax)) {
                vis.push_back(static_cast<uint32_t>(t));
            }
        }
    }

    // ---- Pick the single level to draw (Part 8) --------------------------
    // ONE level is drawn per frame, not a per-tile mix. Mixing levels whose
    // tile partitions differ is exactly what produces T-junction cracks at
    // the seams, and the brief requires no holes and no duplicate triangles.
    // The level is chosen by PROJECTED ERROR: the coarsest level whose
    // geometric error still fits inside the pixel budget at this zoom.
    const uint32_t lodFloor = interactionMoving_ ? movingLodFloor_ : requestedLod_;
    const double pxPerMeter = lastScale_.isOrtho
        ? lastScale_.orthoPixelsPerMeter
        : lastScale_.perspPixelsPerMeterAt1m / 1000.0;
    const double err = active_->lodMeshes[0].cellSizeMeters * pxPerMeter;
    // While moving, the interactive floor wins outright.
    uint32_t chosen = interactionMoving_ ? lodFloor : 0;
    if (!interactionMoving_ && err <= static_cast<double>(lodErrorPixelLimit_)) {
        for (uint32_t lod = 1; lod <= finest; ++lod) {
            const double e = active_->lodMeshes[lod].cellSizeMeters * pxPerMeter;
            if (e <= static_cast<double>(lodErrorPixelLimit_)) { chosen = lod; break; }
        }
    }
    if (chosen > finest) chosen = finest;
    if (active_->lodMeshes[chosen].triangleCount == 0) chosen = 0;

    const std::vector<uint32_t>& vis = active_->visibleTiles[chosen];
    uint64_t submittedTris = 0;
    for (uint32_t t : vis) {
        const std::vector<SurfaceTile>& tiles = active_->lodMeshes[chosen].tiles;
        if (t < tiles.size()) submittedTris += tiles[t].triangleCount;
    }

    active_->selectedLod = chosen;
    active_->visibleTilesLastFrame = static_cast<uint32_t>(vis.size());
    active_->submittedTrianglesLastFrame = submittedTris;
    lastCullingValid_ = true;
}

void SurfaceRenderer::Record(VkCommandBuffer cmd, uint32_t frameSlot) {
    // Tiled path (Parts 6-9): cull + pick a LOD from the live camera, then
    // issue ONE vkCmdDrawIndexed per visible tile. No upload happens here, so
    // panning and zooming cost only CPU culling plus fewer draws.
    if (active_ && active_->IsValid() && !active_->lodMeshes.empty()) {
        CullVisibleTiles(NowMs());
        const uint32_t lod = active_->selectedLod;
        const SurfaceLodMesh& mesh = active_->lodMeshes[lod];
        const SurfaceLodBuffers& buffers = active_->lods[lod];
        if (!buffers.IsValid() || (mesh.tiles.empty() && !active_->coverageCount)) {
            RetireCompleted();
            return;
        }

        VkExtent2D ext = renderer_->GetExtent();
        VkViewport vp{0.0f, 0.0f, static_cast<float>(ext.width),
                         static_cast<float>(ext.height), 0.0f, 1.0f};
        VkRect2D sc{{0, 0}, {ext.width, ext.height}};
        vkCmdSetViewport(cmd, 0, 1, &vp);
        vkCmdSetScissor(cmd, 0, 1, &sc);

        if (active_->indexedCells) {
            VkDeviceSize offset = 0;
            vkCmdBindVertexBuffers(cmd, 0, 1, &buffers.positions.buffer, &offset);
            if (active_->vertexColored)
                vkCmdBindVertexBuffers(cmd, 1, 1, &buffers.colors.buffer, &offset);
            vkCmdBindIndexBuffer(cmd, buffers.indices.buffer, 0, VK_INDEX_TYPE_UINT32);
            VkDescriptorSet sets[] = {renderer_->GetFrameDescriptorSet(), buffers.cellSet};
            if(active_->indexCount) {
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                active_->vertexColored ? indexedVcPipeline_ :
                (shadingMode_ == ShadingMode::Wireframe ? indexedWireframePipeline_ : indexedPipeline_));
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, indexedLayout_, 0, 2, sets, 0, nullptr);
            }
            PushConstants indexedPc{}; FillCrispPushConstants(indexedPc);
            vkCmdPushConstants(cmd, indexedLayout_, VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(indexedPc), &indexedPc);
            uint64_t submitted = 0;
            for (uint32_t index : active_->visibleTiles[lod]) {
                const auto& tile = mesh.tiles[index]; uint32_t firstTriangle = tile.firstIndex / 3;
                vkCmdPushConstants(cmd, indexedLayout_, VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(firstTriangle), &firstTriangle);
                vkCmdDrawIndexed(cmd, tile.indexCount, 1, tile.firstIndex, 0, 0);
                indexedDrawCalls_++;
                submitted += tile.triangleCount;
            }
            if(active_->coverageCount) {
                VkBuffer coverageBuffers[]={active_->coveragePositions.buffer,active_->coverageColors.buffer};
                VkDeviceSize offsets[]={0,0};
                vkCmdBindPipeline(cmd,VK_PIPELINE_BIND_POINT_GRAPHICS,coveragePipeline_);
                PushConstants coveragePc{};
                FillCrispPushConstants(coveragePc);
                vkCmdPushConstants(cmd,indexedLayout_,VK_SHADER_STAGE_FRAGMENT_BIT,0,sizeof(coveragePc),&coveragePc);
                vkCmdBindVertexBuffers(cmd,0,2,coverageBuffers,offsets);
                vkCmdBindDescriptorSets(cmd,VK_PIPELINE_BIND_POINT_GRAPHICS,indexedLayout_,0,1,sets,0,nullptr);
                vkCmdDraw(cmd,uint32_t(active_->coverageCount),1,0,0);
            }
            active_->submittedTrianglesLastFrame = submitted;
            active_->lastUsedFrame = renderer_->GetFrameCounter(); drawCalls_++;
            renderer_->WriteTimestamp(cmd, frameSlot, Renderer::kTsSurfaceAfter);
            RetireCompleted(); return;
        }
        VkDeviceSize offsets[] = {0, 0, 0};
        VkBuffer buffers3[] = {buffers.positions.buffer, buffers.normals.buffer,
                                buffers.colors.buffer};
        vkCmdBindVertexBuffers(cmd, 0, 3, buffers3, offsets);
        vkCmdBindIndexBuffer(cmd, buffers.indices.buffer, 0, VK_INDEX_TYPE_UINT32);
        VkDescriptorSet frameSet = renderer_->GetFrameDescriptorSet();
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                        pipelineLayout_, 0, 1, &frameSet, 0, nullptr);

        PushConstants pc{};
        float len = std::sqrt(lightDir_[0] * lightDir_[0] +
                              lightDir_[1] * lightDir_[1] +
                              lightDir_[2] * lightDir_[2]);
        const float inv = (len > 1e-12f) ? 1.0f / len : 1.0f;
        pc.lightDirAmbient[0] = lightDir_[0] * inv;
        pc.lightDirAmbient[1] = lightDir_[1] * inv;
        pc.lightDirAmbient[2] = lightDir_[2] * inv;
        pc.lightDirAmbient[3] = ambient_;
        pc.elevRangeMode[0] = zMin_;
        pc.elevRangeMode[1] = zMax_;
        pc.elevRangeMode[2] = crispHybridActive_
            ? 4.0f
            : static_cast<float>(static_cast<int>(shadingMode_));
        pc.elevRangeMode[3] = parityBaseElevationDeg_;
        pc.shadeParamsA[0] = crispAzimuthDeg_;
        pc.shadeParamsA[1] = crispSharpnessRaw_;
        pc.shadeParamsA[2] = crispAmbient_;
        pc.shadeParamsA[3] = parityAmbientFloor_;
        std::copy(elevationColor_, elevationColor_ + 4, pc.elevationColor);
        pc.shadeParamsB[0] = crispKeyIntensity_;
        pc.shadeParamsB[1] = crispFillIntensity_;
        pc.shadeParamsB[2] = static_cast<float>(parityDebugStage_);
        pc.shadeParamsB[3] = static_cast<float>(parityColorMode_);

        renderer_->WriteTimestamp(cmd, frameSlot, Renderer::kTsSurfaceBefore);
        VkPipeline pipeline = (shadingMode_ == ShadingMode::Wireframe && !crispHybridActive_)
                                  ? pipelineWireframe_ : pipelineSolid_;
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
        vkCmdPushConstants(cmd, pipelineLayout_,
                              VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                              0, sizeof(pc), &pc);

        // ONE DRAW PER VISIBLE TILE. Each tile is a contiguous [firstIndex,
        // +indexCount) run of this level's index buffer, so no rebinding and no
        // re-upload is needed - only the draw list changes with the camera.
        uint64_t submitted = 0;
        for (uint32_t t : active_->visibleTiles[lod]) {
            if (t >= mesh.tiles.size()) continue;
            const SurfaceTile& tile = mesh.tiles[t];
            if (tile.indexCount == 0) continue;
            vkCmdDrawIndexed(cmd, tile.indexCount, 1, tile.firstIndex, 0, 0);
            indexedDrawCalls_++;
            submitted += tile.triangleCount;
        }
        drawCalls_++;
        active_->submittedTrianglesLastFrame = submitted;
        // Mark this set as in use THIS frame so retirement waits for its fence.
        active_->lastUsedFrame = renderer_->GetFrameCounter();
        renderer_->WriteTimestamp(cmd, frameSlot, Renderer::kTsSurfaceAfter);
        RetireCompleted();
        return;
    }

    if (initialized_ && crispIndexedActive_) {
        RecordCrispIndexed(cmd, frameSlot);
        return;
    }
    if (!initialized_ || vertexCount_ == 0) return;
    VkExtent2D ext = renderer_->GetExtent();
    VkViewport vp{0.0f, 0.0f, static_cast<float>(ext.width),
                     static_cast<float>(ext.height), 0.0f, 1.0f};
    VkRect2D sc{{0, 0}, {ext.width, ext.height}};
    vkCmdSetViewport(cmd, 0, 1, &vp);
    vkCmdSetScissor(cmd, 0, 1, &sc);

    // offsets[] previously had only 1 element for 3 bound buffers, so
    // vkCmdBindVertexBuffers() read offsets[1]/offsets[2] out of bounds.
    VkDeviceSize offsets[] = {0, 0, 0};
    VkBuffer buffers[] = {positions_.buffer, normals_.buffer, colors_.buffer};
    vkCmdBindVertexBuffers(cmd, 0, 3, buffers, offsets);
    vkCmdBindIndexBuffer(cmd, indices_.buffer, 0, VK_INDEX_TYPE_UINT32);
    VkDescriptorSet frameSet = renderer_->GetFrameDescriptorSet();
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                    pipelineLayout_, 0, 1,
                                    &frameSet, 0, nullptr);

    PushConstants pc{};
    float len = std::sqrt(lightDir_[0] * lightDir_[0] +
                             lightDir_[1] * lightDir_[1] +
                             lightDir_[2] * lightDir_[2]);
    const float inv = (len > 1e-12f) ? 1.0f / len : 1.0f;
    pc.lightDirAmbient[0] = lightDir_[0] * inv;
    pc.lightDirAmbient[1] = lightDir_[1] * inv;
    pc.lightDirAmbient[2] = lightDir_[2] * inv;
    pc.lightDirAmbient[3] = ambient_;
    pc.elevRangeMode[0] = zMin_;
    pc.elevRangeMode[1] = zMax_;
    pc.elevRangeMode[2] = crispHybridActive_
        ? 4.0f  // PASS 1: crisp base, see surface.frag mode==4
        : static_cast<float>(static_cast<int>(shadingMode_));
    pc.elevRangeMode[3] = parityBaseElevationDeg_;   // crisp modes: base light elevation
    pc.shadeParamsA[0] = crispAzimuthDeg_;
    pc.shadeParamsA[1] = crispSharpnessRaw_;
    pc.shadeParamsA[2] = crispAmbient_;
    pc.shadeParamsA[3] = parityAmbientFloor_;         // < 0 -> shader default
    std::copy(elevationColor_, elevationColor_ + 4, pc.elevationColor);
    pc.shadeParamsB[0] = crispKeyIntensity_;
    pc.shadeParamsB[1] = crispFillIntensity_;
    pc.shadeParamsB[2] = static_cast<float>(parityDebugStage_);
    pc.shadeParamsB[3] = static_cast<float>(parityColorMode_);

    renderer_->WriteTimestamp(cmd, frameSlot, Renderer::kTsSurfaceBefore);
    VkPipeline pipeline = (shadingMode_ == ShadingMode::Wireframe && !crispHybridActive_)
                              ? pipelineWireframe_ : pipelineSolid_;
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
    vkCmdPushConstants(cmd, pipelineLayout_,
                          VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                          0, sizeof(pc), &pc);
    vkCmdDrawIndexed(cmd, static_cast<uint32_t>(indexCount_), 1, 0, 0, 0);
    drawCalls_++;

    // PASS 2: mixed-class barycentric overlay (only when the crisp-hybrid
    // Shaded Class path is active and there are mixed faces to draw).
    if (crispHybridActive_ && overlayIndexCount_ > 0 && positions2_.IsValid()) {
        // PASS 2 draws the same triangles as PASS 1, so it must use the
        // LESS_OR_EQUAL pipeline (see Initialize) or its fragments are all
        // depth-rejected and the mixed-class blend silently disappears.
        VkPipeline overlayPipeline = (pipelineSolidOverlay_ != VK_NULL_HANDLE)
                                         ? pipelineSolidOverlay_ : pipeline;
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, overlayPipeline);
        VkDeviceSize offsets2[] = {0, 0, 0};
        VkBuffer buffers2[] = {positions2_.buffer, normals2_.buffer, colors2_.buffer};
        vkCmdBindVertexBuffers(cmd, 0, 3, buffers2, offsets2);
        vkCmdBindIndexBuffer(cmd, indices2_.buffer, 0, VK_INDEX_TYPE_UINT32);
        pc.elevRangeMode[2] = 5.0f;  // PASS 2: mixed overlay, see surface.frag mode==5
        vkCmdPushConstants(cmd, pipelineLayout_,
                              VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                              0, sizeof(pc), &pc);
        vkCmdDrawIndexed(cmd, static_cast<uint32_t>(overlayIndexCount_), 1, 0, 0, 0);
        overlayDrawCalls_++;
    }
    renderer_->WriteTimestamp(cmd, frameSlot, Renderer::kTsSurfaceAfter);
}

std::size_t SurfaceRenderer::GetIndexCount() const {
    if (active_ && !active_->lodMeshes.empty()) {
        return active_->lodMeshes[active_->selectedLod].indices.size();
    }
    return indexCount_;
}
std::size_t SurfaceRenderer::GetVertexCount() const {
    // A tiled resource set is the modern path and its vertex count lives on
    // the set, not in the legacy vertexCount_ member.
    if (active_) return active_->vertexCount;
    return vertexCount_;
}
std::size_t SurfaceRenderer::GetTriangleCount() const {
    if (active_) return active_->triangleCount;
    return triangleCount_;
}
void SurfaceRenderer::Clear() {
    vertexCount_ = 0;
    triangleCount_ = 0;
    indexCount_ = 0;
    overlayIndexCount_ = 0;
    crispIndexedActive_ = false;
    crispHybridActive_ = false;
}

bool SurfaceRenderer::IsLoaded() const {
    return (active_ && active_->IsValid()) || vertexCount_ > 0 || crispIndexedActive_;
}
uint64_t SurfaceRenderer::GetMeshUploadCount() const { return meshUploads_; }
uint64_t SurfaceRenderer::GetModeSwitchCount() const { return modeSwitches_; }
VkDeviceSize SurfaceRenderer::GetGpuBytes() const {
    // Includes every resident set (active + pending + retired) so the memory
    // report reflects what the driver is actually holding, which is what the
    // budget check has to stay under during a swap.
    VkDeviceSize total = positions_.size + normals_.size + colors_.size + indices_.size +
                         positions2_.size + normals2_.size + colors2_.size + indices2_.size;
    total += crispPositions_.size + crispColors_.size + crispIndices_.size +
             crispCovPositions_.size + crispCovColors_.size;
    total += tilePositions_.size + tileColors_.size + tileIndices_.size +
             tileCovPositions_.size + tileCovColors_.size;
    if (active_) total += active_->gpuBytes;
    if (pending_) total += pending_->gpuBytes;
    for (const auto& r : retired_) if (r) total += r->gpuBytes;
    return total;
}
bool SurfaceRenderer::IsWireframeSupported() const { return wireframeSupported_; }

void SurfaceRenderer::OnSwapchainRebuild() {
    if (!initialized_ || !renderer_) return;
    renderer_->Pipelines().DestroyPipeline(pipelineSolid_);
    renderer_->Pipelines().DestroyPipeline(pipelineSolidOverlay_);
    renderer_->Pipelines().DestroyPipeline(pipelineWireframe_);
    pipelineSolid_ = VK_NULL_HANDLE;
    pipelineSolidOverlay_ = VK_NULL_HANDLE;
    pipelineWireframe_ = VK_NULL_HANDLE;
    std::vector<ShaderModule> solid{vs_, fsSolid_};
    pipelineCfg_.depthStencil.depthCompareOp = VK_COMPARE_OP_LESS;
    pipelineSolid_ = renderer_->Pipelines().CreateGraphicsPipeline(
        pipelineLayout_, solid, pipelineCfg_, renderer_->GetRenderPass(), 0);
    // Same LESS_OR_EQUAL overlay pipeline as Initialize() (see its note): the
    // depth comparison of pipelineCfg_ is part of the created pipeline, so a
    // swapchain rebuild has to recreate both variants from their own state.
    pipelineCfg_.depthStencil.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;
    pipelineSolidOverlay_ = renderer_->Pipelines().CreateGraphicsPipeline(
        pipelineLayout_, solid, pipelineCfg_, renderer_->GetRenderPass(), 0);
    pipelineCfg_.rasterizer.polygonMode = VK_POLYGON_MODE_LINE;
    std::vector<ShaderModule> wire{vs_, fsWireframe_};
    pipelineWireframe_ = renderer_->Pipelines().CreateGraphicsPipeline(
        pipelineLayout_, wire, pipelineCfg_, renderer_->GetRenderPass(), 0);
}

// ---- compact crisp Shaded-Class mesh ---------------------------------------------

void SurfaceRenderer::ReleaseSoupBuffers() {
    auto& a = VulkanAllocator::Get();
    a.DestroyBuffer(positions_); a.DestroyBuffer(normals_);
    a.DestroyBuffer(colors_); a.DestroyBuffer(indices_);
    a.DestroyBuffer(positions2_); a.DestroyBuffer(normals2_);
    a.DestroyBuffer(colors2_); a.DestroyBuffer(indices2_);
    positions_ = {}; normals_ = {}; colors_ = {}; indices_ = {};
    positions2_ = {}; normals2_ = {}; colors2_ = {}; indices2_ = {};
    vertexCount_ = 0; indexCount_ = 0; triangleCount_ = 0; overlayIndexCount_ = 0;
    crispHybridActive_ = false;
}

void SurfaceRenderer::ReleaseCrispIndexed() {
    ReleaseTiles();                 // every other surface path also ends the tile arena
    crispTiles_.clear();
    crispVisibleTiles_=0;crispSubmittedTriangles_=0;
    if (!crispPositions_.IsValid() && !crispIndices_.IsValid() && !crispColors_.IsValid() &&
        !crispCovPositions_.IsValid() && !crispCovColors_.IsValid()) {
        crispIndexedActive_ = false;
        return;
    }
    // Rare transition (mode / path change): a frame in flight may still read these.
    if (renderer_) renderer_->WaitIdle();
    auto& a = VulkanAllocator::Get();
    a.DestroyBuffer(crispPositions_); a.DestroyBuffer(crispColors_); a.DestroyBuffer(crispIndices_);
    a.DestroyBuffer(crispCovPositions_); a.DestroyBuffer(crispCovColors_);
    crispPositions_ = {}; crispColors_ = {}; crispIndices_ = {};
    crispCovPositions_ = {}; crispCovColors_ = {};
    crispIndexCount_ = crispCoverageCount_ = 0;
    crispIndexedActive_ = false;
}

bool SurfaceRenderer::UploadCrispIndexedMesh(const double* positions, uint64_t vertexCount,
                                             const int32_t* faces, uint64_t faceCount,
                                             const uint8_t* vertexClass, const uint8_t* lut,
                                             const double* covPositions, const uint8_t* covRGB,
                                             uint64_t covCount, std::string* error) {
    auto fail = [&](const char* why) { if (error) *error = why; return false; };
    if (!initialized_ || crispIdxPipeline_ == VK_NULL_HANDLE) return fail("compact crisp pipeline unavailable");
    if (!positions || !faces || !vertexClass || !lut || vertexCount == 0 || faceCount == 0)
        return fail("empty compact crisp mesh");
    if (vertexCount > UINT32_MAX || faceCount * 3 > UINT32_MAX || covCount > UINT32_MAX)
        return fail("compact crisp mesh exceeds 32-bit indices");
    if (covCount && (!covPositions || !covRGB)) return fail("coverage arrays missing");

    const VkDeviceSize posBytes = vertexCount * 12, colBytes = vertexCount * 4;
    const VkDeviceSize idxBytes = faceCount * 12;
    const VkDeviceSize covPosBytes = covCount * 12, covColBytes = covCount * 4;
    const VkDeviceSize required = posBytes + colBytes + idxBytes + covPosBytes + covColBytes;
    const VkPhysicalDeviceProperties props =
        renderer_->GetContext().CoreDevice().GetPhysicalDeviceInfo().GetProperties();
    if (idxBytes > props.limits.maxStorageBufferRange || posBytes > props.limits.maxStorageBufferRange)
        return fail("compact crisp mesh exceeds maxStorageBufferRange");

    // ONE full-detail mesh on the GPU at a time: release the previous surface (every
    // path) first. The call is synchronous, so the next frame already shows the new one.
    renderer_->WaitIdle();
    ReleaseCrispIndexed();
    ReleaseSoupBuffers();
    RetireTiledSets();
    for (auto& r : retired_) if (r) r->ReleaseBuffers();
    retired_.clear();
    if (required > vramBudgetBytes_) {
        memoryRejected_ = true;
        return fail("compact crisp mesh exceeds the VRAM budget");
    }

    auto& a = VulkanAllocator::Get();
    const VkBufferUsageFlags dstUse = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    crispPositions_ = a.CreateBuffer(posBytes, dstUse | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT |
                                     VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
    crispColors_ = a.CreateBuffer(colBytes, dstUse | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT |
                                  VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
    crispIndices_ = a.CreateBuffer(idxBytes, dstUse | VK_BUFFER_USAGE_INDEX_BUFFER_BIT |
                                   VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
    if (covCount) {
        crispCovPositions_ = a.CreateBuffer(covPosBytes, dstUse | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
        crispCovColors_ = a.CreateBuffer(covColBytes, dstUse | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
    }
    if (!crispPositions_.IsValid() || !crispColors_.IsValid() || !crispIndices_.IsValid() ||
        (covCount && (!crispCovPositions_.IsValid() || !crispCovColors_.IsValid()))) {
        ReleaseCrispIndexed();
        memoryRejected_ = true;
        return fail("compact crisp mesh VRAM allocation failed");
    }

    // Stream every array through a bounded staging window (48 MiB: a multiple of 12
    // and 4 bytes), so host-visible staging never grows with the mesh.
    const RenderOrigin& origin = renderer_->GetOrigin();
    constexpr VkDeviceSize kChunk = 48ull * 1024 * 1024;
    auto stream = [&](GPUBuffer& dst, VkDeviceSize bytes, const auto& fill) {
        for (VkDeviceSize off = 0; off < bytes; off += kChunk) {
            const VkDeviceSize len = std::min(kChunk, bytes - off);
            uint8_t* st = renderer_->GetUploadContext().Stage(len);
            if (!st) return false;
            fill(st, off, len);
            renderer_->GetUploadContext().Execute({{dst.buffer, off, 0, len}}, 0.0, 0.0);
        }
        return true;
    };
    auto positionFill = [&](const double* src) {
        return [&origin, src](uint8_t* st, VkDeviceSize off, VkDeviceSize len) {
            float* out = reinterpret_cast<float*>(st);
            const uint64_t first = off / 12, count = len / 12;
            for (uint64_t i = 0; i < count; ++i) {
                double r[3];
                const double* w = src + (first + i) * 3;
                origin.WorldToRender(w[0], w[1], w[2], r);
                out[i * 3 + 0] = float(r[0]); out[i * 3 + 1] = float(r[1]); out[i * 3 + 2] = float(r[2]);
            }
        };
    };
    bool ok = stream(crispPositions_, posBytes, positionFill(positions));
    ok = ok && stream(crispColors_, colBytes, [&](uint8_t* st, VkDeviceSize off, VkDeviceSize len) {
        const uint64_t first = off / 4, count = len / 4;
        for (uint64_t i = 0; i < count; ++i) {
            const uint8_t c = vertexClass[first + i];
            st[i * 4 + 0] = lut[c * 3 + 0]; st[i * 4 + 1] = lut[c * 3 + 1];
            st[i * 4 + 2] = lut[c * 3 + 2]; st[i * 4 + 3] = 255;
        }
    });
    bool indicesValid = true;
    ok = ok && stream(crispIndices_, idxBytes, [&](uint8_t* st, VkDeviceSize off, VkDeviceSize len) {
        const int32_t* src = faces + off / 4;
        const uint64_t count = len / 4;
        for (uint64_t i = 0; i < count; ++i)
            if (src[i] < 0 || uint64_t(src[i]) >= vertexCount) indicesValid = false;
        std::memcpy(st, src, len);
    });
    if (ok && covCount) {
        ok = stream(crispCovPositions_, covPosBytes, positionFill(covPositions));
        ok = ok && stream(crispCovColors_, covColBytes, [&](uint8_t* st, VkDeviceSize off, VkDeviceSize len) {
            const uint64_t first = off / 4, count = len / 4;
            for (uint64_t i = 0; i < count; ++i) {
                const uint8_t* c = covRGB + (first + i) * 3;
                st[i * 4 + 0] = c[0]; st[i * 4 + 1] = c[1]; st[i * 4 + 2] = c[2]; st[i * 4 + 3] = 255;
            }
        });
    }
    if (!ok || !indicesValid) {
        ReleaseCrispIndexed();
        return fail(!indicesValid ? "compact crisp face index out of range" : "compact crisp staging failed");
    }
    renderer_->Descriptors().UpdateBuffer(crispIdxSet_, 0, crispIndices_.buffer, idxBytes, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
    renderer_->Descriptors().UpdateBuffer(crispIdxSet_, 1, crispPositions_.buffer, posBytes, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
    renderer_->Descriptors().UpdateBuffer(crispIdxSet_, 2, crispColors_.buffer, colBytes, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
    crispIndexCount_ = faceCount * 3;
    crispCoverageCount_ = covCount;
    crispIndexedActive_ = true;
    triangleCount_ = faceCount;
    memoryRejected_ = false;
    meshUploads_++;
    fprintf(stderr, "[SurfaceRenderer] compact crisp mesh: %llu vertices, %llu triangles, "
                    "%llu coverage points, %.0f MB VRAM\n",
            (unsigned long long)vertexCount, (unsigned long long)faceCount,
            (unsigned long long)covCount, double(required) / (1024.0 * 1024.0));
    return true;
}

bool SurfaceRenderer::SetCrispIndexBlocks(const double* positions,uint64_t vertexCount,
        const int32_t* faces,const uint32_t* counts,uint32_t count) {
    if(crispIndexedActive_&&count==0){crispTiles_.clear();return true;}
    if(!crispIndexedActive_||!positions||!faces||!counts||!count)return false;
    uint64_t sum=0;for(uint32_t i=0;i<count;++i){if(counts[i]%3)return false;sum+=counts[i];}
    if(sum!=crispIndexCount_)return false;
    std::vector<SurfaceTile> tiles;tiles.reserve(count);uint32_t first=0;
    const auto& origin=renderer_->GetOrigin();
    for(uint32_t b=0;b<count;++b){
        SurfaceTile tile;tile.firstIndex=first;tile.indexCount=counts[b];tile.triangleCount=counts[b]/3;tile.residency=1;
        for(int a=0;a<3;++a){tile.bmin[a]=std::numeric_limits<float>::infinity();tile.bmax[a]=-tile.bmin[a];}
        for(uint32_t i=0;i<counts[b];++i){
            int32_t v=faces[first+i];if(v<0||uint64_t(v)>=vertexCount)return false;
            const double* p=positions+uint64_t(v)*3;double r[3];origin.WorldToRender(p[0],p[1],p[2],r);
            for(int a=0;a<3;++a){float x=float(r[a]);if(!std::isfinite(x))return false;tile.bmin[a]=std::min(tile.bmin[a],x);tile.bmax[a]=std::max(tile.bmax[a],x);}
        }
        first+=counts[b];if(!counts[b])continue;
        for(int a=0;a<3;++a){float pad=std::max(0.001f,(tile.bmax[a]-tile.bmin[a])*0.00001f);tile.bmin[a]-=pad;tile.bmax[a]+=pad;}
        tiles.push_back(tile);
    }
    crispTiles_=std::move(tiles);
    fprintf(stderr,"[SurfaceRenderer] full-detail navigation: %zu conservative tiles, no mesh reduction\n",crispTiles_.size());
    return true;
}

void SurfaceRenderer::FillCrispPushConstants(PushConstants& pc) const {
    pc = PushConstants{};
    float len = std::sqrt(lightDir_[0] * lightDir_[0] + lightDir_[1] * lightDir_[1] +
                          lightDir_[2] * lightDir_[2]);
    const float inv = (len > 1e-12f) ? 1.0f / len : 1.0f;
    pc.lightDirAmbient[0] = lightDir_[0] * inv;
    pc.lightDirAmbient[1] = lightDir_[1] * inv;
    pc.lightDirAmbient[2] = lightDir_[2] * inv;
    pc.lightDirAmbient[3] = ambient_;
    pc.elevRangeMode[0] = zMin_;
    pc.elevRangeMode[1] = zMax_;
    pc.elevRangeMode[2] = 4.0f;   // the shader picks 4 (pure) or 5 (mixed) per triangle
    pc.elevRangeMode[3] = parityBaseElevationDeg_;
    pc.shadeParamsA[0] = crispAzimuthDeg_;
    pc.shadeParamsA[1] = crispSharpnessRaw_;
    pc.shadeParamsA[2] = crispAmbient_;
    pc.shadeParamsA[3] = parityAmbientFloor_;
    std::copy(elevationColor_, elevationColor_ + 4, pc.elevationColor);
    pc.shadeParamsB[0] = crispKeyIntensity_;
    pc.shadeParamsB[1] = crispFillIntensity_;
    pc.shadeParamsB[2] = static_cast<float>(parityDebugStage_);
    pc.shadeParamsB[3] = static_cast<float>(parityColorMode_);
}

void SurfaceRenderer::RecordCrispIndexed(VkCommandBuffer cmd, uint32_t frameSlot) {
    if (tilesActive_) { RecordTiles(cmd, frameSlot); return; }
    VkExtent2D ext = renderer_->GetExtent();
    VkViewport vp{0.0f, 0.0f, static_cast<float>(ext.width), static_cast<float>(ext.height), 0.0f, 1.0f};
    VkRect2D sc{{0, 0}, {ext.width, ext.height}};
    vkCmdSetViewport(cmd, 0, 1, &vp);
    vkCmdSetScissor(cmd, 0, 1, &sc);
    renderer_->WriteTimestamp(cmd, frameSlot, Renderer::kTsSurfaceBefore);

    VkDescriptorSet sets[2] = {renderer_->GetFrameDescriptorSet(), crispIdxSet_};
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, crispIdxPipeline_);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, crispIdxLayout_, 0, 2, sets, 0, nullptr);
    PushConstants pc;
    FillCrispPushConstants(pc);
    vkCmdPushConstants(cmd, crispIdxLayout_, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                       0, sizeof(pc), &pc);
    VkBuffer vbs[] = {crispPositions_.buffer, crispColors_.buffer};
    VkDeviceSize offsets[] = {0, 0};
    vkCmdBindVertexBuffers(cmd, 0, 2, vbs, offsets);
    vkCmdBindIndexBuffer(cmd, crispIndices_.buffer, 0, VK_INDEX_TYPE_UINT32);
    if(crispTiles_.empty()) {
        crispVisibleTiles_=1;crispSubmittedTriangles_=crispIndexCount_/3;
        vkCmdDrawIndexed(cmd, static_cast<uint32_t>(crispIndexCount_), 1, 0, 0, 0);
        drawCalls_++;
    } else {
        crispVisibleTiles_=0;crispSubmittedTriangles_=0;
        float mvp[16];renderer_->GetLastMvp(mvp);FrustumPlanes fr;ExtractFrustumPlanes(mvp,fr);
        uint32_t first=0,count=0;
        auto flush=[&](){if(count){vkCmdDrawIndexed(cmd,count,1,first,0,first/3);drawCalls_++;count=0;}};
        // Full-resolution triangles only. Adjacent visible tiles share a draw;
        // firstInstance preserves absolute triangle identity in the fragment shader.
        for(const auto& tile:crispTiles_) {
            if(!AabbVisible(fr,tile.bmin,tile.bmax)){flush();continue;}
            ++crispVisibleTiles_;crispSubmittedTriangles_+=tile.triangleCount;
            if(count&&first+count!=tile.firstIndex)flush();
            if(!count)first=tile.firstIndex;
            count+=tile.indexCount;
        }
        flush();
    }

    // Points that no kept triangle references: drawn, never dropped.
    if (crispCoverageCount_ && coveragePipeline_ != VK_NULL_HANDLE) {
        VkBuffer cbs[] = {crispCovPositions_.buffer, crispCovColors_.buffer};
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, coveragePipeline_);
        PushConstants coveragePc{};
        FillCrispPushConstants(coveragePc);
        vkCmdPushConstants(cmd, indexedLayout_, VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(coveragePc), &coveragePc);
        vkCmdBindVertexBuffers(cmd, 0, 2, cbs, offsets);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, indexedLayout_, 0, 1, sets, 0, nullptr);
        vkCmdDraw(cmd, static_cast<uint32_t>(crispCoverageCount_), 1, 0, 0);
    }
    renderer_->WriteTimestamp(cmd, frameSlot, Renderer::kTsSurfaceAfter);
}

// ---- viewport-LOD surface tiles ---------------------------------------------------

void SurfaceRenderer::RangeAllocator::Reset(uint64_t cap) {
    free.clear();
    if (cap) free[0] = cap;
    capacity = cap;
    used = 0;
}

bool SurfaceRenderer::RangeAllocator::Alloc(uint64_t n, uint64_t& offset) {
    if (n == 0) { offset = 0; return true; }
    for (auto it = free.begin(); it != free.end(); ++it) {
        if (it->second < n) continue;
        offset = it->first;
        const uint64_t rest = it->second - n;
        free.erase(it);
        if (rest) free[offset + n] = rest;
        used += n;
        return true;
    }
    return false;
}

void SurfaceRenderer::RangeAllocator::Free(uint64_t offset, uint64_t n) {
    if (n == 0) return;
    used -= n;
    uint64_t start = offset, len = n;
    auto next = free.lower_bound(offset);
    if (next != free.begin()) {
        auto prev = std::prev(next);
        if (prev->first + prev->second == start) { start = prev->first; len += prev->second; free.erase(prev); }
    }
    if (next != free.end() && start + len == next->first) { len += next->second; free.erase(next); }
    free[start] = len;
}

void SurfaceRenderer::ReleaseTiles() {
    const bool any = tilesActive_ || tilePositions_.IsValid() || tileIndices_.IsValid();
    if (!any) return;
    // Rare transition (mode / path change): a frame in flight may still read these.
    if (renderer_) renderer_->WaitIdle();
    auto& a = VulkanAllocator::Get();
    a.DestroyBuffer(tilePositions_); a.DestroyBuffer(tileColors_); a.DestroyBuffer(tileIndices_);
    a.DestroyBuffer(tileCovPositions_); a.DestroyBuffer(tileCovColors_);
    tilePositions_ = {}; tileColors_ = {}; tileIndices_ = {}; tileCovPositions_ = {}; tileCovColors_ = {};
    tileSlots_.clear();
    tileRetired_.clear();
    tileDraw_.clear();
    tileV_.Reset(0); tileF_.Reset(0); tileC_.Reset(0);
    tileDrawnTiles_ = tileDrawnFaces_ = tileDrawnCoverage_ = tileDraws_ = 0;
    if (tilesActive_) crispIndexedActive_ = false;
    tilesActive_ = false;
}

bool SurfaceRenderer::ConfigureTiles(uint64_t vcap, uint64_t fcap, uint64_t ccap, std::string* error) {
    auto fail = [&](const char* why) { if (error) *error = why; return false; };
    if (!initialized_ || crispIdxPipeline_ == VK_NULL_HANDLE || coveragePipeline_ == VK_NULL_HANDLE)
        return fail("compact crisp pipeline unavailable");
    ccap = std::max<uint64_t>(ccap, 1);
    if (vcap == 0 || fcap == 0 || vcap > UINT32_MAX || fcap * 3 > UINT32_MAX || ccap > UINT32_MAX)
        return fail("invalid surface tile arena capacity");
    const VkPhysicalDeviceProperties props =
        renderer_->GetContext().CoreDevice().GetPhysicalDeviceInfo().GetProperties();
    if (fcap * 12 > props.limits.maxStorageBufferRange || vcap * 12 > props.limits.maxStorageBufferRange)
        return fail("surface tile arena exceeds maxStorageBufferRange");
    renderer_->WaitIdle();
    ReleaseCrispIndexed();          // also ends a previous arena
    ReleaseSoupBuffers();
    RetireTiledSets();
    for (auto& r : retired_) if (r) r->ReleaseBuffers();
    retired_.clear();
    const double bytes = double(vcap) * 16.0 + double(fcap) * 12.0 + double(ccap) * 16.0;
    if (vramBudgetBytes_ > 0.0 && bytes > vramBudgetBytes_) return fail("surface tile arena exceeds the VRAM budget");
    auto& a = VulkanAllocator::Get();
    const VkBufferUsageFlags dst = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    tilePositions_ = a.CreateBuffer(vcap * 12, dst | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
    tileColors_ = a.CreateBuffer(vcap * 4, dst | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
    tileIndices_ = a.CreateBuffer(fcap * 12, dst | VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
    tileCovPositions_ = a.CreateBuffer(ccap * 12, dst | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
    tileCovColors_ = a.CreateBuffer(ccap * 4, dst | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
    if (!tilePositions_.IsValid() || !tileColors_.IsValid() || !tileIndices_.IsValid() ||
        !tileCovPositions_.IsValid() || !tileCovColors_.IsValid()) {
        a.DestroyBuffer(tilePositions_); a.DestroyBuffer(tileColors_); a.DestroyBuffer(tileIndices_);
        a.DestroyBuffer(tileCovPositions_); a.DestroyBuffer(tileCovColors_);
        tilePositions_ = {}; tileColors_ = {}; tileIndices_ = {}; tileCovPositions_ = {}; tileCovColors_ = {};
        memoryRejected_ = true;
        return fail("surface tile arena VRAM allocation failed");
    }
    renderer_->Descriptors().UpdateBuffer(crispIdxSet_, 0, tileIndices_.buffer, fcap * 12, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
    renderer_->Descriptors().UpdateBuffer(crispIdxSet_, 1, tilePositions_.buffer, vcap * 12, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
    renderer_->Descriptors().UpdateBuffer(crispIdxSet_, 2, tileColors_.buffer, vcap * 4, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
    tileV_.Reset(vcap); tileF_.Reset(fcap); tileC_.Reset(ccap);
    tilesActive_ = true;
    crispIndexedActive_ = true;
    triangleCount_ = 0;
    memoryRejected_ = false;
    fprintf(stderr, "[SurfaceRenderer] surface tile arena: %llu vertices, %llu triangles, %llu coverage points, %.0f MB VRAM\n",
            (unsigned long long)vcap, (unsigned long long)fcap, (unsigned long long)ccap, bytes / (1024.0 * 1024.0));
    return true;
}

void SurfaceRenderer::RecycleTileRanges(bool all) {
    size_t keep = 0;
    for (size_t i = 0; i < tileRetired_.size(); ++i) {
        const RetiredRanges& r = tileRetired_[i];
        if (all || renderer_->IsFrameRetired(r.frame)) {
            tileV_.Free(r.vbase, r.vcount); tileF_.Free(r.fbase, r.fcount); tileC_.Free(r.cbase, r.ccount);
        } else tileRetired_[keep++] = r;
    }
    tileRetired_.resize(keep);
}

uint32_t SurfaceRenderer::EvictTiles(const uint64_t* keys, uint32_t count) {
    uint32_t n = 0;
    for (uint32_t i = 0; keys && i < count; ++i) {
        auto it = tileSlots_.find(keys[i]);
        if (it == tileSlots_.end()) continue;
        const TileSlot& s = it->second;
        // A frame already recorded may still read these ranges: reuse waits for it.
        tileRetired_.push_back({s.vbase, s.vcount, s.fbase, s.fcount, s.cbase, s.ccount, renderer_->GetFrameCounter()});
        tileSlots_.erase(it);
        ++n;
    }
    tileEvictions_ += n;
    return n;
}

uint32_t SurfaceRenderer::UploadTiles(TileUploadDesc* tiles, uint32_t count, std::string* error) {
    for (uint32_t i = 0; i < count; ++i) tiles[i].result = 0;
    if (!tilesActive_) { if (error) *error = "surface tile arena not configured"; return 0; }
    RecycleTileRanges(false);
    struct Plan { TileUploadDesc* d; TileSlot slot; };
    std::vector<Plan> plans;
    plans.reserve(count);
    VkDeviceSize total = 0;
    for (uint32_t i = 0; i < count; ++i) {
        TileUploadDesc& d = tiles[i];
        const bool facesOk = d.faceCount == 0 || (d.faces && d.vertexCount);
        const bool vertsOk = d.vertexCount == 0 || (d.positions && d.vertexClass);
        const bool covOk = d.coverageCount == 0 || (d.coveragePositions && d.coverageClass);
        if ((d.vertexCount == 0 && d.coverageCount == 0) || !facesOk || !vertsOk || !covOk) {
            if (error) *error = "invalid surface tile arrays";
            continue;
        }
        bool indicesOk = true;
        for (uint64_t k = 0; k < uint64_t(d.faceCount) * 3; ++k)
            if (d.faces[k] < 0 || uint32_t(d.faces[k]) >= d.vertexCount) { indicesOk = false; break; }
        if (!indicesOk) { if (error) *error = "surface tile face index out of range"; continue; }
        if (tileSlots_.count(d.key)) EvictTiles(&d.key, 1);
        Plan p{&d, {}};
        p.slot.vcount = d.vertexCount; p.slot.fcount = d.faceCount; p.slot.ccount = d.coverageCount;
        auto allocate = [&]() {
            uint64_t v = 0, f = 0, c = 0;
            if (!tileV_.Alloc(p.slot.vcount, v)) return false;
            if (!tileF_.Alloc(p.slot.fcount, f)) { tileV_.Free(v, p.slot.vcount); return false; }
            if (!tileC_.Alloc(p.slot.ccount, c)) { tileV_.Free(v, p.slot.vcount); tileF_.Free(f, p.slot.fcount); return false; }
            p.slot.vbase = v; p.slot.fbase = f; p.slot.cbase = c;
            return true;
        };
        bool ok = allocate();
        if (!ok && !tileRetired_.empty()) {
            renderer_->WaitIdle();        // every retired range is now provably unused
            RecycleTileRanges(true);
            ok = allocate();
        }
        if (!ok) { d.result = -1; continue; }
        total += p.slot.vcount * 16 + p.slot.fcount * 12 + p.slot.ccount * 16;
        plans.push_back(std::move(p));
    }
    if (plans.empty()) return 0;
    uint8_t* st = renderer_->GetUploadContext().Stage(total);
    if (!st) {
        for (auto& p : plans) { tileV_.Free(p.slot.vbase, p.slot.vcount); tileF_.Free(p.slot.fbase, p.slot.fcount); tileC_.Free(p.slot.cbase, p.slot.ccount); }
        if (error) *error = "surface tile staging failed";
        return 0;
    }
    const RenderOrigin& origin = renderer_->GetOrigin();
    std::vector<CopyRegion> regions;
    VkDeviceSize cursor = 0;
    auto region = [&](GPUBuffer& dst, VkDeviceSize dstOffset, VkDeviceSize size) {
        if (size) regions.push_back({dst.buffer, dstOffset, cursor, size});
        cursor += size;
    };
    for (auto& p : plans) {
        TileUploadDesc& d = *p.d;
        TileSlot& s = p.slot;
        for (int a = 0; a < 3; ++a) { s.bmin[a] = std::numeric_limits<float>::infinity(); s.bmax[a] = -s.bmin[a]; }
        auto positions = [&](const double* src, uint64_t n) {
            float* out = reinterpret_cast<float*>(st + cursor);
            for (uint64_t i = 0; i < n; ++i) {
                double r[3];
                origin.WorldToRender(src[i * 3], src[i * 3 + 1], src[i * 3 + 2], r);
                for (int a = 0; a < 3; ++a) {
                    const float x = float(r[a]);
                    out[i * 3 + a] = x;
                    s.bmin[a] = std::min(s.bmin[a], x); s.bmax[a] = std::max(s.bmax[a], x);
                }
            }
        };
        positions(d.positions, s.vcount);
        region(tilePositions_, s.vbase * 12, s.vcount * 12);
        for (uint64_t i = 0; i < s.vcount; ++i) {
            const uint8_t c = d.vertexClass[i];
            uint8_t* o = st + cursor + i * 4;
            o[0] = tileLut_[c * 3]; o[1] = tileLut_[c * 3 + 1]; o[2] = tileLut_[c * 3 + 2]; o[3] = 255;
        }
        region(tileColors_, s.vbase * 4, s.vcount * 4);
        {
            uint32_t* out = reinterpret_cast<uint32_t*>(st + cursor);
            const uint32_t base = uint32_t(s.vbase);
            for (uint64_t i = 0; i < s.fcount * 3; ++i) out[i] = base + uint32_t(d.faces[i]);
        }
        region(tileIndices_, s.fbase * 12, s.fcount * 12);
        positions(d.coveragePositions, s.ccount);
        region(tileCovPositions_, s.cbase * 12, s.ccount * 12);
        for (uint64_t i = 0; i < s.ccount; ++i) {
            const uint8_t c = d.coverageClass[i];
            uint8_t* o = st + cursor + i * 4;
            for (int k = 0; k < 3; ++k)
                o[k] = uint8_t(std::min(255.0f, std::max(0.0f, float(tileLut_[c * 3 + k]) * tileShade_ + 0.5f)));
            o[3] = 255;
        }
        region(tileCovColors_, s.cbase * 4, s.ccount * 4);
        // Same conservative pad as the monolithic block bounds.
        for (int a = 0; a < 3; ++a) {
            const float pad = std::max(0.001f, (s.bmax[a] - s.bmin[a]) * 0.00001f);
            s.bmin[a] -= pad; s.bmax[a] += pad;
        }
        s.cls.assign(d.vertexClass, d.vertexClass + s.vcount);
        if (s.ccount) s.covCls.assign(d.coverageClass, d.coverageClass + s.ccount);
    }
    renderer_->GetUploadContext().Execute(regions, 0.0, 0.0);
    for (auto& p : plans) {
        tileSlots_[p.d->key] = std::move(p.slot);
        p.d->result = 1;
    }
    tileUploads_ += plans.size();
    tileUploadBytes_ += total;
    return uint32_t(plans.size());
}

bool SurfaceRenderer::SetTilePalette(const uint8_t* lut, float shade, std::string* error) {
    if (lut) std::memcpy(tileLut_, lut, sizeof(tileLut_));
    tileShade_ = shade;
    if (!tilesActive_ || tileSlots_.empty()) return true;
    VkDeviceSize total = 0;
    for (const auto& kv : tileSlots_) total += (kv.second.vcount + kv.second.ccount) * 4;
    if (!total) return true;
    uint8_t* st = renderer_->GetUploadContext().Stage(total);
    if (!st) { if (error) *error = "surface tile staging failed"; return false; }
    std::vector<CopyRegion> regions;
    VkDeviceSize cursor = 0;
    for (const auto& kv : tileSlots_) {
        const TileSlot& s = kv.second;
        for (uint64_t i = 0; i < s.vcount; ++i) {
            const uint8_t c = s.cls[i];
            uint8_t* o = st + cursor + i * 4;
            o[0] = tileLut_[c * 3]; o[1] = tileLut_[c * 3 + 1]; o[2] = tileLut_[c * 3 + 2]; o[3] = 255;
        }
        if (s.vcount) regions.push_back({tileColors_.buffer, s.vbase * 4, cursor, s.vcount * 4});
        cursor += s.vcount * 4;
        for (uint64_t i = 0; i < s.ccount; ++i) {
            const uint8_t c = s.covCls[i];
            uint8_t* o = st + cursor + i * 4;
            for (int k = 0; k < 3; ++k)
                o[k] = uint8_t(std::min(255.0f, std::max(0.0f, float(tileLut_[c * 3 + k]) * tileShade_ + 0.5f)));
            o[3] = 255;
        }
        if (s.ccount) regions.push_back({tileCovColors_.buffer, s.cbase * 4, cursor, s.ccount * 4});
        cursor += s.ccount * 4;
    }
    renderer_->GetUploadContext().Execute(regions, 0.0, 0.0);
    ++tileRecolors_;
    return true;
}

bool SurfaceRenderer::PatchTileClasses(uint64_t key, const uint32_t* vi, const uint8_t* vc, uint32_t nv,
                                       const uint32_t* ci, const uint8_t* cc, uint32_t nc, std::string* error) {
    auto fail = [&](const char* why) { if (error) *error = why; return false; };
    if (!tilesActive_) return fail("surface tile arena not configured");
    auto it = tileSlots_.find(key);
    if (it == tileSlots_.end()) return fail("surface tile is not resident");
    TileSlot& s = it->second;
    if ((nv && (!vi || !vc)) || (nc && (!ci || !cc))) return fail("invalid tile class patch arrays");
    for (uint32_t i = 0; i < nv; ++i) if (vi[i] >= s.vcount) return fail("tile vertex index out of range");
    for (uint32_t i = 0; i < nc; ++i) if (ci[i] >= s.ccount) return fail("tile coverage index out of range");
    for (uint32_t i = 0; i < nv; ++i) s.cls[vi[i]] = vc[i];
    for (uint32_t i = 0; i < nc; ++i) s.covCls[ci[i]] = cc[i];
    // One copy per contiguous run of changed entries (sorted, de-duplicated).
    auto runs = [](const uint32_t* idx, uint32_t n) {
        if (!n) return std::vector<std::pair<uint32_t, uint32_t>>{};
        std::vector<uint32_t> v(idx, idx + n);
        std::sort(v.begin(), v.end());
        v.erase(std::unique(v.begin(), v.end()), v.end());
        std::vector<std::pair<uint32_t, uint32_t>> r;          // (first, count)
        for (uint32_t x : v) {
            if (!r.empty() && r.back().first + r.back().second == x) ++r.back().second;
            else r.push_back({x, 1});
        }
        return r;
    };
    const auto vr = runs(vi, nv), cr = runs(ci, nc);
    VkDeviceSize total = 0;
    for (const auto& r : vr) total += VkDeviceSize(r.second) * 4;
    for (const auto& r : cr) total += VkDeviceSize(r.second) * 4;
    if (!total) return true;
    renderer_->WaitIdle();            // colours are read by frames in flight
    uint8_t* st = renderer_->GetUploadContext().Stage(total);
    if (!st) return fail("surface tile staging failed");
    std::vector<CopyRegion> regions;
    regions.reserve(vr.size() + cr.size());
    VkDeviceSize cursor = 0;
    for (const auto& r : vr) {
        for (uint32_t i = 0; i < r.second; ++i) {
            const uint8_t c = s.cls[r.first + i];
            uint8_t* o = st + cursor + i * 4;
            o[0] = tileLut_[c * 3]; o[1] = tileLut_[c * 3 + 1]; o[2] = tileLut_[c * 3 + 2]; o[3] = 255;
        }
        regions.push_back({tileColors_.buffer, (s.vbase + r.first) * 4, cursor, VkDeviceSize(r.second) * 4});
        cursor += VkDeviceSize(r.second) * 4;
    }
    for (const auto& r : cr) {
        for (uint32_t i = 0; i < r.second; ++i) {
            const uint8_t c = s.covCls[r.first + i];
            uint8_t* o = st + cursor + i * 4;
            for (int k = 0; k < 3; ++k)
                o[k] = uint8_t(std::min(255.0f, std::max(0.0f, float(tileLut_[c * 3 + k]) * tileShade_ + 0.5f)));
            o[3] = 255;
        }
        regions.push_back({tileCovColors_.buffer, (s.cbase + r.first) * 4, cursor, VkDeviceSize(r.second) * 4});
        cursor += VkDeviceSize(r.second) * 4;
    }
    renderer_->GetUploadContext().Execute(regions, 0.0, 0.0, true);
    ++tileClassPatches_;
    return true;
}

bool SurfaceRenderer::SetTileDrawList(const TileDrawItem* items, uint32_t count) {
    if (!tilesActive_) return false;
    if (items && count) tileDraw_.assign(items, items + count);
    else tileDraw_.clear();
    return true;
}

void SurfaceRenderer::GetTileStats(uint64_t* out, uint32_t count) const {
    const uint64_t v[16] = {tileSlots_.size(), tileV_.used, tileV_.capacity, tileF_.used, tileF_.capacity,
                            tileC_.used, tileC_.capacity, tileDrawnTiles_, tileDrawnFaces_, tileDrawnCoverage_,
                            tileUploads_, tileUploadBytes_,
                            uint64_t(tilePositions_.size + tileColors_.size + tileIndices_.size +
                                     tileCovPositions_.size + tileCovColors_.size),
                            tileEvictions_, tileDraws_, tileRecolors_};
    for (uint32_t i = 0; out && i < count; ++i) out[i] = i < 16 ? v[i] : 0;
}

void SurfaceRenderer::RecordTiles(VkCommandBuffer cmd, uint32_t frameSlot) {
    VkExtent2D ext = renderer_->GetExtent();
    VkViewport vp{0.0f, 0.0f, static_cast<float>(ext.width), static_cast<float>(ext.height), 0.0f, 1.0f};
    VkRect2D sc{{0, 0}, {ext.width, ext.height}};
    vkCmdSetViewport(cmd, 0, 1, &vp);
    vkCmdSetScissor(cmd, 0, 1, &sc);
    renderer_->WriteTimestamp(cmd, frameSlot, Renderer::kTsSurfaceBefore);
    tileDrawnTiles_ = tileDrawnFaces_ = tileDrawnCoverage_ = tileDraws_ = 0;
    float mvp[16];
    renderer_->GetLastMvp(mvp);
    FrustumPlanes fr;
    ExtractFrustumPlanes(mvp, fr);
    std::vector<std::pair<uint64_t, uint64_t>> faces, coverage, underlay;   // absolute (first, count)
    faces.reserve(tileDraw_.size());
    for (const TileDrawItem& item : tileDraw_) {
        auto it = tileSlots_.find(item.key);
        if (it == tileSlots_.end()) continue;
        const TileSlot& s = it->second;
        if (!AabbVisible(fr, s.bmin, s.bmax)) continue;
        if (item.faceCount && uint64_t(item.firstFace) + item.faceCount <= s.fcount)
            (item.pad & 1u ? underlay : faces).push_back({s.fbase + item.firstFace, item.faceCount});
        if (item.coverage && s.ccount) coverage.push_back({s.cbase, s.ccount});
        if (item.coverage) ++tileDrawnTiles_;
    }
    auto merge = [](std::vector<std::pair<uint64_t, uint64_t>>& r) {
        std::sort(r.begin(), r.end());
        size_t out = 0;
        for (size_t i = 0; i < r.size(); ++i) {
            if (out && r[out - 1].first + r[out - 1].second == r[i].first) r[out - 1].second += r[i].second;
            else r[out++] = r[i];
        }
        r.resize(out);
    };
    merge(faces);
    merge(coverage);
    merge(underlay);
    VkDeviceSize offsets[] = {0, 0};
    if (!faces.empty() || !underlay.empty()) {
        VkDescriptorSet sets[2] = {renderer_->GetFrameDescriptorSet(), crispIdxSet_};
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, crispIdxPipeline_);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, crispIdxLayout_, 0, 2, sets, 0, nullptr);
        PushConstants pc;
        FillCrispPushConstants(pc);
        vkCmdPushConstants(cmd, crispIdxLayout_, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(pc), &pc);
        VkBuffer vbs[] = {tilePositions_.buffer, tileColors_.buffer};
        vkCmdBindVertexBuffers(cmd, 0, 2, vbs, offsets);
        vkCmdBindIndexBuffer(cmd, tileIndices_.buffer, 0, VK_INDEX_TYPE_UINT32);
        // firstInstance carries the absolute first triangle, so gl_PrimitiveID + base is
        // the triangle's index in the shared index arena (see surface_crisp_indexed.vert).
        for (const auto& r : faces) {
            vkCmdDrawIndexed(cmd, uint32_t(r.second * 3), 1, uint32_t(r.first * 3), 0, uint32_t(r.first));
            tileDrawnFaces_ += r.second;
            ++tileDraws_;
            drawCalls_++;
        }
        // Keep sparse ground covered, behind every detailed surface fragment.
        if (!underlay.empty()) {
            vp.minDepth = vp.maxDepth = 0.99999994f;
            vkCmdSetViewport(cmd, 0, 1, &vp);
            for (const auto& r : underlay) {
                vkCmdDrawIndexed(cmd, uint32_t(r.second * 3), 1, uint32_t(r.first * 3), 0, uint32_t(r.first));
                tileDrawnFaces_ += r.second; ++tileDraws_; ++drawCalls_;
            }
            vp.minDepth = 0.0f; vp.maxDepth = 1.0f;
            vkCmdSetViewport(cmd, 0, 1, &vp);
        }
    }
    if (!coverage.empty() && coveragePipeline_ != VK_NULL_HANDLE) {
        VkDescriptorSet frame = renderer_->GetFrameDescriptorSet();
        VkBuffer cbs[] = {tileCovPositions_.buffer, tileCovColors_.buffer};
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, coveragePipeline_);
        PushConstants coveragePc{};
        FillCrispPushConstants(coveragePc);
        vkCmdPushConstants(cmd, indexedLayout_, VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(coveragePc), &coveragePc);
        vkCmdBindVertexBuffers(cmd, 0, 2, cbs, offsets);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, indexedLayout_, 0, 1, &frame, 0, nullptr);
        for (const auto& r : coverage) {
            vkCmdDraw(cmd, uint32_t(r.second), 1, uint32_t(r.first), 0);
            tileDrawnCoverage_ += r.second;
            ++tileDraws_;
        }
    }
    renderer_->WriteTimestamp(cmd, frameSlot, Renderer::kTsSurfaceAfter);
}

} // namespace naksha
