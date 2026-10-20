#pragma once
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <unordered_map>
#include <string>
#include <vector>

#include "naksha/Buffer.hpp"
#include "naksha/SurfaceTiling.hpp"
#include "naksha/core/vulkan/VulkanFrameManager.h"
#include "naksha/core/vulkan/VulkanPipelineManager.h"

namespace naksha {
using namespace vulkan;

enum class ShadingMode : uint32_t {
    Flat = 0,
    Smooth = 1,
    Elevation = 2,
    Wireframe = 3
};

// ---------------------------------------------------------------------------
// Atomic surface swap states (Part 9).
//
// EMPTY          nothing resident
// PREVIEW_ACTIVE a coarse/interactive resource set is on screen
// FINAL_PENDING  a final-quality set is uploaded but not yet swapped in
// FINAL_ACTIVE   the final-quality set is on screen
// ---------------------------------------------------------------------------
enum class SurfaceState : uint32_t {
    Empty = 0,
    PreviewActive = 1,
    FinalPending = 2,
    FinalActive = 3,
};

// GPU buffers for ONE LOD level. Each level is a self-contained mesh: the
// clustered coarse levels have their own compact vertex arrays, so they need
// their own vertex buffers as well as their own index buffer.
struct SurfaceLodBuffers {
    GPUBuffer positions;
    GPUBuffer normals;
    GPUBuffer colors;
    GPUBuffer indices;
    GPUBuffer cellColors;
    VkDescriptorPool cellPool = VK_NULL_HANDLE;
    VkDescriptorSet cellSet = VK_NULL_HANDLE;
    VkDevice cellDevice = VK_NULL_HANDLE;
    bool IsValid() const { return positions.IsValid() && indices.IsValid(); }
    VkDeviceSize Bytes() const {
        return positions.size + normals.size + colors.size + indices.size + cellColors.size;
    }
    void Release();
};

// One complete, independently-owned GPU resource set. ACTIVE and PENDING are
// separate instances of this, so a new final mesh can be uploaded, validated
// and only then swapped in without ever touching the set currently on screen.
struct SurfaceResourceSet {
    // One buffer set per LOD level. Level 0 is the final mesh; 1 and 2 are
    // the clustered coarse levels. Each is fully tiled and independently
    // cullable, which is what lets a pan/zoom change the drawn triangle count
    // with no geometry upload at all.
    SurfaceLodBuffers lods[kSurfaceLodCount];
    std::vector<SurfaceLodMesh> lodMeshes;   // CPU copy of the tiled levels
    // Per level, the tile indices that survived this frame's frustum test.
    // Rebuilt every frame from the live camera - never from an upload.
    std::vector<uint32_t> visibleTiles[kSurfaceLodCount];
    std::size_t vertexCount = 0;
    std::size_t indexCount = 0;
    std::size_t triangleCount = 0;
    uint32_t selectedLod = 0;                // level actually drawn last frame
    bool isPreview = false;                  // coarse stand-in, not final
    bool indexedCells = false;
    bool vertexColored = false;
    GPUBuffer coveragePositions, coverageColors;
    std::size_t coverageCount = 0;
    // Stale-guard identity. A pending set whose revisions no longer match the
    // renderer's current ones is discarded instead of swapped in.
    uint64_t datasetRevision = 0;
    uint64_t surfaceRevision = 0;
    // Frame in which this set was last submitted. Retirement waits for the
    // frame fence to prove that frame finished - never a device-wide idle.
    uint64_t lastUsedFrame = VulkanFrameManager::kNeverSubmitted;
    // Free/dispatch stats for the memory report.
    VkDeviceSize gpuBytes = 0;
    uint32_t totalTiles = 0;
    uint32_t visibleTilesLastFrame = 0;
    uint64_t submittedTrianglesLastFrame = 0;

    bool IsValid() const { return lods[0].IsValid(); }
    void ReleaseBuffers();
};

class Renderer;

// ---------------------------------------------------------------------------
// Surface/mesh renderer.
//
// Mesh arrives as a single UploadMesh() call (positions, normals, RGB
// colors, indices). Shading mode, light direction, ambient and elevation
// range are set via push constants (or, for wireframe, a pipeline
// switch). None of these change the GPU buffers - meshUploadCount
// increments exactly once per mesh and never again until the next
// UploadMesh().
// ---------------------------------------------------------------------------
class SurfaceRenderer {
public:
    SurfaceRenderer() = default;
    ~SurfaceRenderer();

    bool Initialize(Renderer& renderer);
    void Shutdown();
    void Clear();
    bool IsInitialized() const { return initialized_; }

    // Upload a single mesh. position/normal/color are N,3 C-contiguous
    // float32; indices are count uint32. Creates/replaces all buffers
    // in one staged->GPU copy batch.
    UploadStats UploadMesh(const float* positions, const float* normals,
                           const uint8_t* colorsRGB,
                           std::size_t vertexCount,
                           const uint32_t* indices, std::size_t indexCount);

    // Recolour the resident legacy triangle soup without uploading geometry.
    bool UpdateFaceColors(const uint8_t* rgb, std::size_t faceCount);

    // ---- State changes (push constants / pipeline switch, no upload) ----
    void SetShadingMode(ShadingMode mode);
    ShadingMode GetShadingMode() const;
    void SetLightDirection(float x, float y, float z);
    void SetAmbient(float ambient);     // 0..1
    void SetElevationRange(float zMin, float zMax);

    // ---- GPU class-colour LUT (Phase C) -----------------------------------
    //
    // A palette change must re-upload ONLY these 256*4 bytes. It must NOT
    // touch positions, normals, indices or the per-vertex class-id stream, and
    // it must NOT increment the mesh-upload counter. The LUT lives in its own
    // GPU buffer and is bound to the surface descriptor set, so the fragment
    // shader resolves colour = LUT[vertex_class_id] at draw time.
    //
    // colorsRGB is 256*3 uint8 (RGB); it is expanded to 256*4 RGBA here so the
    // shader can index it with a byte stride without a second lookup.
    UploadStats SetClassColorLut(const uint8_t* colorsRGB);
    uint64_t GetClassLutUpdateCount() const { return classLutUpdates_; }

    // Record draw commands into cmd (current frame's command buffer).
    // Writes the surface timestamp pair and binds set 0 (frame UBO).
    void Record(VkCommandBuffer cmd, uint32_t frameSlot);

    // ---- Tiled / culled / LOD-aware upload (Parts 6-9) --------------------
    //
    // Uploads a mesh as a TILE + LOD resource set. Unlike UploadMesh() this
    // never overwrites the live GPU buffers: it builds a fresh PENDING set,
    // and ActivatePendingIfFresh() promotes it to ACTIVE at a frame boundary.
    //
    // positions/normals/colors are a flat triangle soup in RENDER space
    // (3 vertices per triangle, exactly what nkv_set_surface already builds).
    // `lodCellMeters` is the base vertex spacing used to derive the coarse
    // levels; pass 0 to disable LOD generation.
    //
    // `isPreview` marks a coarse interactive stand-in. A preview can never
    // silently become the final engineering surface: it is reported through
    // GetState() as PREVIEW_ACTIVE until a real final set is uploaded.
    struct UploadTiledResult {
        bool ok = false;
        UploadStats stats;
        uint32_t tileCount = 0;
        std::size_t lodTriangles[kSurfaceLodCount] = {0, 0, 0};
        bool memoryRejected = false;   // budget refused the upload
        double peakBytesDuringSwap = 0.0;
    };
    UploadTiledResult UploadTiledMesh(const float* positions, const float* normals,
                                      const uint8_t* colors,
                                      std::size_t triangleCount,
                                      uint32_t targetTileCount,
                                      float lodCellMeters,
                                      bool isPreview,
                                      uint64_t datasetRevision,
                                      uint64_t surfaceRevision);
    UploadTiledResult UploadIndexedBlocks(const float* positions, std::size_t vertexCount,
                                          const uint32_t* indices, std::size_t indexCount,
                                          const uint32_t* cellColors,
                                          const uint32_t* blockCounts, uint32_t blockCount,
                                          uint64_t datasetRevision, uint64_t surfaceRevision,
                                          const uint32_t* vertexColors = nullptr);

    bool UploadPendingCoverage(const float* positions, const uint32_t* colors, std::size_t count,
                               uint64_t datasetRevision, uint64_t surfaceRevision);
    void DiscardPending() { if (pending_) retired_.push_back(std::move(pending_)); }
    // The Shaded-Class (crisp) mesh is ONE resident buffer set drawn through the
    // legacy branch of Record(), and that branch is only reachable while no
    // tiled (LOD/culling) set is live. Uploading it must therefore retire the
    // tiled sets: otherwise a view that showed the tiled elevation surface and
    // then switched back to Shaded Classification keeps drawing the elevation
    // mesh and the crisp mesh never appears on screen. The sets are RETIRED
    // (fence-deferred), never destroyed here, so a frame still in flight cannot
    // lose the buffers it is reading.
    void RetireTiledSets() {
        if (active_) retired_.push_back(std::move(active_));
        if (pending_) retired_.push_back(std::move(pending_));
    }
    // Promotes PENDING -> ACTIVE when its revisions still match the current
    // ones. Called once per frame from nkv_render, at the frame boundary.
    // Returns true when a swap happened. A stale pending set is retired
    // instead of promoted, and `discardedPending` is set.
    bool ActivatePendingIfFresh(bool* discardedPending = nullptr);

    // Destroys any resource set whose last submitted frame has provably
    // completed. Uses the frame fences, never vkDeviceWaitIdle().
    void RetireCompleted();

    // Rebuilds the visible-tile list from the CURRENT camera and picks the LOD
    // to draw. Called once per frame from Record(). Pure CPU: it touches no
    // GPU buffer, so panning/zooming never triggers an upload.
    void CullVisibleTiles(double nowMs);

    void SetInteractionState(bool moving, double nowMs);
    // Bytes UploadTiledMesh() would need for a mesh of `triangleCount`
    // triangles, and whether the budget would accept it right now. Exposed so
    // the refusal decision can be reported for meshes too large to materialise
    // on the host (the SLOW-quality case) without allocating anything.
    uint64_t EstimateRequiredBytes(std::size_t triangleCount) const;
    bool WouldAcceptUpload(std::size_t triangleCount) const;
    double GetVramBudgetBytes() const { return vramBudgetBytes_; }
    // Overrides the safe VRAM budget; 0 restores the 70%-of-device default.
    void SetVramBudgetBytes(double bytes) { vramBudgetBytes_ = bytes; }
    bool IsInteractionMoving() const { return interactionMoving_; }
    // Minimum LOD floor while the camera is moving; the requested level is
    // restored automatically once the idle timer expires.
    void SetMovingLodFloor(uint32_t lod) { movingLodFloor_ = lod; }
    void SetIdleRefineMs(double ms) { idleRefineMs_ = ms; }
    void SetLodErrorPixelLimit(float px) { lodErrorPixelLimit_ = px; }

    SurfaceState GetState() const;
    // Stale-guard revisions. Bumping the dataset/surface revision invalidates
    // any PENDING set built against the old one, so it is discarded instead of
    // swapped in.
    void SetDatasetRevision(uint64_t r) { currentDatasetRevision_ = r; }
    void SetSurfaceRevision(uint64_t r) { currentSurfaceRevision_ = r; }
    uint64_t GetDatasetRevision() const { return currentDatasetRevision_; }
    uint64_t GetSurfaceRevision() const { return currentSurfaceRevision_; }
    // Culling/LOD telemetry for the reports.
    uint32_t GetTotalTileCount() const;
    uint32_t GetVisibleTileCount() const;
    uint64_t GetTotalTriangleCount() const;
    uint64_t GetVisibleTriangleCount() const;
    uint32_t GetSelectedLod() const;
    // Resident triangle count of one LOD level (0 = final, 1 = medium,
    // 2 = interaction). This is what proves the coarse levels are real,
    // measured geometry rather than a label.
    uint64_t GetLodTriangleCount(uint32_t lod) const;
    uint64_t GetSwapCount() const { return swapCount_; }
    uint64_t GetStaleDiscardCount() const { return staleDiscards_; }
    VkDeviceSize GetActiveBytes() const;
    VkDeviceSize GetPendingBytes() const;
    VkDeviceSize GetLodCacheBytes() const;
    uint64_t GetPeakBytesDuringSwap() const { return peakBytesDuringSwap_; }
    // True when the last budget check refused an upload for lack of VRAM.
    bool WasMemoryRejected() const { return memoryRejected_; }

    std::size_t GetVertexCount() const;
    // Resident index count for the level actually being DRAWN. Exposed so the
    // GPU-memory breakdown can report a measured index-buffer size instead of
    // an estimate, and so a caller asking "how many triangles is on screen"
    // gets the drawn level rather than a stale legacy count. The per-level
    // breakdown is available through GetLodTriangleCount().
    std::size_t GetIndexCount() const;
    std::size_t GetTriangleCount() const;
    bool IsLoaded() const;
    uint64_t GetMeshUploadCount() const; // increments ONLY on UploadMesh
    uint64_t GetModeSwitchCount() const;
    VkDeviceSize GetGpuBytes() const;
    bool IsWireframeSupported() const;

    // ---- Shaded Class (crisp hybrid) path --------------------------------
    // Real GLSL port of gui/shading_display.py's crisp-hybrid formula
    // (see CHECKPOINTS.md for the Python -> GLSL expression map). Two
    // persistent buffer sets, at most two draw calls per frame:
    //   PASS 1 (base, reuses positions_/normals_/colors_/indices_): ALL
    //     faces, unlit hillshade computed in-shader from the face normal +
    //     azimuth/sharpness/ambient (mirrors _crisp_shade_chunk exactly).
    //   PASS 2 (overlay, positions2_/normals2_/colors2_/indices2_): ONLY
    //     mixed-class faces, true per-vertex class colors (GPU rasterizer
    //     barycentrically interpolates them - no CPU blending), lit by a
    //     key+fill two-light approximation of
    //     _configure_nakshatech_color_blend_lighting.
    // SetCrispShadingParams() is push-constant only - never touches these
    // buffers. UploadShadedClassMesh() is the only thing that increments
    // meshUploads_ / touches GPU buffers for this path.
    UploadStats UploadShadedClassMesh(
        const float* basePositions, const float* baseNormals, const uint8_t* baseColorsRGB,
        std::size_t baseVertexCount, const uint32_t* baseIndices, std::size_t baseIndexCount,
        const float* overlayPositions, const float* overlayNormals, const uint8_t* overlayColorsRGB,
        std::size_t overlayVertexCount, const uint32_t* overlayIndices, std::size_t overlayIndexCount);

    // ---- Compact crisp Shaded-Class mesh (surface.frag built with CRISP_INDEXED) ----
    // The same crisp-hybrid look as UploadShadedClassMesh, drawn ONCE from shared
    // vertices: float3 position + RGBA8 class colour per vertex (16 B) and one
    // uint32 index buffer (12 B per triangle) - about 28 B per point, against
    // ~190 B per point for the two-pass triangle soup. The fragment shader fetches
    // each triangle's vertices by gl_PrimitiveID for the flat normal and the
    // pure/mixed split. Unmeshed points (coverage) are drawn as points, so no point
    // of the cut is ever missing. Positions are world doubles, converted to render
    // space with the renderer's current origin and streamed through the staging
    // buffer in bounded chunks. The previous surface is released FIRST (after a
    // device idle) so the GPU never holds two full-detail meshes at once.
    bool HasCrispIndexed() const { return crispIdxPipeline_ != VK_NULL_HANDLE; }
    bool UploadCrispIndexedMesh(const double* positions, uint64_t vertexCount,
                                const int32_t* faces, uint64_t faceCount,
                                const uint8_t* vertexClass, const uint8_t* classLutRGB,
                                const double* coveragePositions, const uint8_t* coverageRGB,
                                uint64_t coverageCount, std::string* error);
    bool IsCrispIndexedActive() const { return crispIndexedActive_; }
    bool SetCrispIndexBlocks(const double* positions,uint64_t vertexCount,
                            const int32_t* faces,const uint32_t* counts,uint32_t count);

    // ---- Viewport-LOD surface tiles (same compact crisp look and shaders) ----------
    // Independently resident tiles share one vertex arena (float3 + RGBA8 class colour),
    // one index arena (absolute uint32 indices, so gl_PrimitiveID + firstInstance stay
    // the absolute triangle identity the fragment shader reads) and one coverage arena.
    // Tiles are uploaded in batches through ONE staging submit, evicted individually
    // (their ranges are reused only after every frame that may read them has retired),
    // frustum-culled per tile and drawn from a host-supplied list of tile-local triangle
    // ranges; adjacent ranges share a draw. Configuring the arena releases every other
    // surface resource; any other surface upload releases the arena.
    struct TileUploadDesc {
        uint64_t key;
        const double* positions;
        const uint8_t* vertexClass;
        const int32_t* faces;            // local vertex indices
        const double* coveragePositions;
        const uint8_t* coverageClass;
        uint32_t vertexCount, faceCount, coverageCount;
        int32_t result;                  // out: 1 resident, 0 rejected, -1 arena full
    };
    struct TileDrawItem {
        uint64_t key;
        uint32_t firstFace, faceCount;   // tile-local triangle range
        uint32_t coverage, pad;          // pad bit 0: background interpolation TIN
    };
    bool ConfigureTiles(uint64_t vertexCapacity, uint64_t faceCapacity, uint64_t coverageCapacity,
                        std::string* error);
    void ReleaseTiles();
    bool TilesActive() const { return tilesActive_; }
    uint32_t UploadTiles(TileUploadDesc* tiles, uint32_t count, std::string* error);
    uint32_t EvictTiles(const uint64_t* keys, uint32_t count);
    bool SetTilePalette(const uint8_t* lutRgb, float coverageShade, std::string* error);
    // Point class edits on one resident tile: rewrites only the listed vertex/coverage
    // colours (tile-local indices) from the current palette. Geometry is unchanged.
    bool PatchTileClasses(uint64_t key, const uint32_t* vertexIndex, const uint8_t* vertexClass,
                          uint32_t vertexCount, const uint32_t* coverageIndex,
                          const uint8_t* coverageClass, uint32_t coverageCount, std::string* error);
    bool SetTileDrawList(const TileDrawItem* items, uint32_t count);
    void GetTileStats(uint64_t* out, uint32_t count) const;

    void SetCrispHybridActive(bool active) { crispHybridActive_ = active; }
    bool GetCrispHybridActive() const { return crispHybridActive_; }
    void SetCrispShadingParams(float azimuthDeg, float sharpnessRaw, float ambient,
                                float keyIntensity, float fillIntensity);

    // ---- Shading parity with the VTK/CPU path (gui/shading_display.py) ----
    // colorMode        1 = the shader pre-compensates the swapchain's sRGB
    //                      encode so the framebuffer holds exactly the byte
    //                      VTK would have written (shipping default);
    //                  0 = the previous raw linear write.
    // debugStage       0 = final shaded colour, 1 = face normal, 2 = lighting
    //                  factor, 3 = class colour, 4 = raw N.L (staged isolation
    //                  of the pipeline; see shaders/surface.frag).
    // ambientFloor     _crisp_blend_ambient_floor(app); < 0 = in-shader default.
    // baseLightElevationDeg _shading_fixed_light_elevation(app); <= 0 = default.
    // Push-constant only - never touches a GPU buffer.
    void SetShadingParityParams(int colorMode, int debugStage,
                                float ambientFloor, float baseLightElevationDeg);
    void SetElevationColor(bool on, float z0, float z1, float ramp) {
        elevationColor_[0] = on ? 1.0f : 0.0f;
        elevationColor_[1] = z0; elevationColor_[2] = z1; elevationColor_[3] = ramp;
    }
    uint64_t GetParityParamUpdateCount() const { return parityParamUpdates_; }

    uint64_t GetShadeParamUpdateCount() const { return shadeParamUpdates_; }
    std::size_t GetOverlayTriangleCount() const { return overlayIndexCount_ / 3; }
    // Base-mesh and mixed-face-overlay draw submissions. "Shaded Class is on
    // screen" is exactly: meshUploads == 1 AND these counts advancing per frame.
    uint64_t GetDrawCallCount() const { return drawCalls_; }
    uint64_t GetIndexedDrawCallCount() const { return indexedDrawCalls_; }
    uint64_t GetOverlayDrawCallCount() const { return overlayDrawCalls_; }

    // Called by Renderer when the swapchain format changed (pipeline
    // rebuild with a new color attachment format).
    void OnSwapchainRebuild();

private:
    Renderer* renderer_ = nullptr;
    PipelineConfig pipelineCfg_;

    struct PushConstants {
        float lightDirAmbient[4];   // xyz = light dir, w = ambient
        float elevRangeMode[4];     // x = zMin, y = zMax, z = mode,
                                    // w = base light elevation (crisp modes 4/5)
        // Crisp-hybrid Shaded Class params (mode 4/5 only; unused by 0-3).
        float shadeParamsA[4];      // x = azimuthDeg, y = sharpnessRaw(0-999),
                                    // z = ambient, w = ambient floor
        float elevationColor[4];  // enabled, world zMin, world zMax, ramp
        float shadeParamsB[4];      // x = keyIntensity, y = fillIntensity,
                                    // z = debug stage, w = colour mode
    };

    float elevationColor_[4]{};
    GPUBuffer positions_;
    GPUBuffer normals_;
    GPUBuffer colors_;
    GPUBuffer indices_;

    // PASS 2 (mixed-face barycentric overlay) buffers - separate from the
    // PASS 1 base buffers above so a shading-parameter change never has to
    // touch either set.
    GPUBuffer positions2_;
    GPUBuffer normals2_;
    GPUBuffer colors2_;
    GPUBuffer indices2_;
    std::size_t overlayIndexCount_ = 0;
    bool crispHybridActive_ = false;

    // Viewport-LOD tile arena (ConfigureTiles / UploadTiles / RecordTiles).
    struct RangeAllocator {
        std::map<uint64_t, uint64_t> free;   // offset -> length
        uint64_t capacity = 0, used = 0;
        void Reset(uint64_t cap);
        bool Alloc(uint64_t n, uint64_t& offset);
        void Free(uint64_t offset, uint64_t n);
    };
    struct TileSlot {
        uint64_t vbase = 0, vcount = 0, fbase = 0, fcount = 0, cbase = 0, ccount = 0;
        float bmin[3]{}, bmax[3]{};
        std::vector<uint8_t> cls, covCls;    // kept for palette recolouring
    };
    struct RetiredRanges { uint64_t vbase, vcount, fbase, fcount, cbase, ccount, frame; };
    void RecordTiles(VkCommandBuffer cmd, uint32_t frameSlot);
    void RecycleTileRanges(bool all);
    bool tilesActive_ = false;
    RangeAllocator tileV_, tileF_, tileC_;
    GPUBuffer tilePositions_, tileColors_, tileIndices_, tileCovPositions_, tileCovColors_;
    std::unordered_map<uint64_t, TileSlot> tileSlots_;
    std::vector<RetiredRanges> tileRetired_;
    std::vector<TileDrawItem> tileDraw_;
    uint8_t tileLut_[768]{};
    float tileShade_ = 1.0f;
    uint64_t tileDrawnTiles_ = 0, tileDrawnFaces_ = 0, tileDrawnCoverage_ = 0, tileDraws_ = 0;
    uint64_t tileUploads_ = 0, tileUploadBytes_ = 0, tileEvictions_ = 0, tileRecolors_ = 0;
    uint64_t tileClassPatches_ = 0;

    // Compact crisp mesh (UploadCrispIndexedMesh) and its pipeline.
    void RecordCrispIndexed(VkCommandBuffer cmd, uint32_t frameSlot);
    void ReleaseCrispIndexed();
    void ReleaseSoupBuffers();
    void FillCrispPushConstants(PushConstants& pc) const;
    GPUBuffer crispPositions_, crispColors_, crispIndices_;
    GPUBuffer crispCovPositions_, crispCovColors_;
    uint64_t crispIndexCount_ = 0, crispCoverageCount_ = 0;
    bool crispIndexedActive_ = false;
    std::vector<SurfaceTile> crispTiles_;
    uint32_t crispVisibleTiles_=0;
    uint64_t crispSubmittedTriangles_=0;
    VkPipeline crispIdxPipeline_ = VK_NULL_HANDLE;
    VkPipelineLayout crispIdxLayout_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout crispIdxSetLayout_ = VK_NULL_HANDLE;
    VkDescriptorPool crispIdxPool_ = VK_NULL_HANDLE;
    VkDescriptorSet crispIdxSet_ = VK_NULL_HANDLE;
    ShaderModule crispIdxVs_ = {}, crispIdxFs_ = {};

    float crispAzimuthDeg_ = 45.0f;
    float crispSharpnessRaw_ = 45.0f;
    float crispAmbient_ = 0.25f;
    float crispKeyIntensity_ = 0.85f;
    float crispFillIntensity_ = 0.18f;
    uint64_t shadeParamUpdates_ = 0;

    // ---- GPU class-colour LUT (Phase C) -----------------------------------
    // 256 entries x RGBA8 = 1024 B. Deliberately its OWN buffer so a palette
    // change is a single 1 KB upload that touches neither the geometry buffers
    // above nor the mesh-upload counter.
    GPUBuffer classLut_;
    uint64_t classLutUpdates_ = 0;
    uint64_t lutRevision_ = 0;   // bumped on every LUT write
    static constexpr std::size_t kClassLutEntries = 256;

    // Shading-parity state (see SetShadingParityParams). Defaults are the
    // shipping byte-parity behaviour so a client that never calls the setter
    // still matches the VTK viewport instead of silently reverting to the
    // brighter raw-linear write.
    int parityColorMode_ = 1;
    int parityDebugStage_ = 0;
    float parityAmbientFloor_ = -1.0f;          // < 0 -> shader default (0.08)
    float parityBaseElevationDeg_ = 0.0f;       // <= 0 -> shader default (45)
    uint64_t parityParamUpdates_ = 0;

    VkPipeline pipelineSolid_ = VK_NULL_HANDLE;
    VkPipeline indexedPipeline_ = VK_NULL_HANDLE;
    VkPipeline indexedVcPipeline_ = VK_NULL_HANDLE;
    VkPipeline coveragePipeline_ = VK_NULL_HANDLE;
    ShaderModule indexedVcVs_ = {}, indexedVcFs_ = {};
    VkPipeline indexedWireframePipeline_ = VK_NULL_HANDLE;
    VkPipelineLayout indexedLayout_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout cellLayout_ = VK_NULL_HANDLE;
    ShaderModule indexedVs_ = {}, indexedFs_ = {};
    // PASS 2 uses LESS_OR_EQUAL, not LESS: the overlay is the same triangles as
    // the base (bit-identical vertex positions), so with LESS every overlay
    // fragment was rejected against the base's own depth and the entire
    // mixed-class barycentric blend disappeared. VTK renders that overlay with
    // ResolveCoincidentTopology polygon offset for the same reason (see
    // gui/shading_display.py _build_static_multiclass_blend_overlays).
    VkPipeline pipelineSolidOverlay_ = VK_NULL_HANDLE;
    VkPipeline pipelineWireframe_ = VK_NULL_HANDLE;
    VkPipelineLayout pipelineLayout_ = VK_NULL_HANDLE;
    ShaderModule vs_ = {};
    ShaderModule fsSolid_ = {};
    ShaderModule fsWireframe_ = {};

    std::size_t vertexCount_ = 0;
    std::size_t indexCount_ = 0;
    std::size_t triangleCount_ = 0;
    ShadingMode shadingMode_ = ShadingMode::Flat;
    float zMin_ = 0.0f;
    float zMax_ = 1.0f;
    float lightDir_[3] = {0.0f, 0.0f, 1.0f};
    float ambient_ = 0.3f;
    uint64_t meshUploads_ = 0;
    uint64_t modeSwitches_ = 0;
    uint64_t drawCalls_ = 0;
    uint64_t indexedDrawCalls_ = 0;
    uint64_t overlayDrawCalls_ = 0;
    bool wireframeSupported_ = false;
    bool initialized_ = false;

    // ---- Tiled / culled / LOD state (Parts 6-9) ---------------------------
    //
    // active_ is what the draw path uses. pending_ is a fully uploaded but not
    // yet promoted set. retired_ holds sets whose GPU work has provably
    // finished and which can be destroyed on the next RetireCompleted().
    // This ACTIVE/PENDING split is what removes the old behaviour where a new
    // surface destroyed the live buffers before the swap.
    std::unique_ptr<SurfaceResourceSet> active_;
    std::unique_ptr<SurfaceResourceSet> pending_;
    std::vector<std::unique_ptr<SurfaceResourceSet>> retired_;

    // Revisions the stale guard compares PENDING against. Bumped by the C API
    // when the dataset or the surface generation changes underneath us.
    uint64_t currentDatasetRevision_ = 0;
    uint64_t currentSurfaceRevision_ = 0;

    // Interaction / LOD policy. Moving forces at least `movingLodFloor_`;
    // after `idleRefineMs_` of stillness the requested level is allowed back.
    bool interactionMoving_ = false;
    double lastInteractionMs_ = 0.0;
    double idleRefineMs_ = 400.0;
    uint32_t movingLodFloor_ = 2;      // interaction/coarse while dragging
    uint32_t requestedLod_ = 0;        // what a settled camera may use
    float lodErrorPixelLimit_ = 1.0f;  // geometric error budget, in pixels

    // Safe VRAM budget (Part 11). 70% of the device-local heap leaves room for
    // the swapchain, depth, staging and driver overhead during a swap.
    double vramBudgetBytes_ = 0.0;
    VkDeviceSize peakBytesDuringSwap_ = 0;
    bool memoryRejected_ = false;
    uint64_t swapCount_ = 0;
    uint64_t staleDiscards_ = 0;
    // Cached for the current frame: the frustum and projected scale are
    // derived from the live MVP once per Record(), never per tile.
    FrustumPlanes lastFrustum_;
    ProjectedScale lastScale_;
    bool lastCullingValid_ = false;
};

} // namespace naksha
