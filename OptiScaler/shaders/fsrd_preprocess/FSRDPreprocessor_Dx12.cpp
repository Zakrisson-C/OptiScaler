#include "pch.h"
#include "FSRDPreprocessor_Dx12.h"
#include "FSRDShaderUtils.h"
#include "FSRDShaderData.h"
#include "FSRDDiagnostics.h"
#include "precompile/FSRDInputConv_Shader.h"
#include "precompile/FSRDFloorSeed_Shader.h"
#include "precompile/FSRDFloor_Shader.h"
#include "precompile/FSRDOutputComp_Shader.h"
#include "precompile/FSRDLeak_Shader.h"
#include "precompile/FSRDSplitFit_Shader.h"
#include "precompile/FSRDSssBlur_Shader.h"

#include "dx12/ffx_api_dx12.h"
#include "fsr-rr/ffx_denoiser.h"

#include <d3dcompiler.h>
#include <d3d12.h>
#include <DirectXPackedVector.h>
#include <stdexcept>
#include <vector>
#include <string>
#include <array>
#include <algorithm>
#include <cmath>
#include <cstring>

#pragma comment(lib, "d3dcompiler.lib")

using Microsoft::WRL::ComPtr;
using namespace DirectX;
using namespace FSRD;

constexpr UINT kBackBufferCount = 3;

constexpr UINT kThreadGroupSizeX = 8;
constexpr UINT kThreadGroupSizeY = 8;

constexpr D3D12_RESOURCE_STATES kSrvState =
    D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE | D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
constexpr D3D12_RESOURCE_STATES kUavState = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;

namespace FSRDFormats
{
// ffxDispatchDescDenoiserIndirectDiffuse / ffxDispatchDescDenoiserIndirectSpecular signal
// textures. Radiance/FusedAlbedo (the old Mode 1 fused-signal formats) removed - transplant,
// 22 Sep, no successor in denoiser 1.2 (transplant plan §6e/6f).
constexpr DXGI_FORMAT SpecRadiance = DXGI_FORMAT_R16G16B16A16_FLOAT;
constexpr DXGI_FORMAT DiffRadiance = DXGI_FORMAT_R16G16B16A16_FLOAT;

// ffxDispatchDescDenoiser
constexpr DXGI_FORMAT Motion = DXGI_FORMAT_R16G16B16A16_FLOAT;
constexpr DXGI_FORMAT Normals = DXGI_FORMAT_R10G10B10A2_UNORM;
constexpr DXGI_FORMAT SpecAlbedo = DXGI_FORMAT_R8G8B8A8_UNORM;
constexpr DXGI_FORMAT DiffAlbedo = DXGI_FORMAT_R8G8B8A8_UNORM;
constexpr DXGI_FORMAT LinearDepth = DXGI_FORMAT_R32_FLOAT;

constexpr DXGI_FORMAT SkipSignal = DXGI_FORMAT_R16G16B16A16_FLOAT;

constexpr DXGI_FORMAT OutputBuffer1 = DXGI_FORMAT_R16G16B16A16_FLOAT;
constexpr DXGI_FORMAT OutputBuffer2 = DXGI_FORMAT_R16G16B16A16_FLOAT;

// A/B, audit finding 3 (24 Sep): both albedo textures at half precision, so remodulation in the
// composition multiplies by (nearly) the same albedo the packing shader divided by.
constexpr DXGI_FORMAT Albedo16 = DXGI_FORMAT_R16G16B16A16_FLOAT;

// Pixel probe records (see FSRDDiagnostics.h)
constexpr DXGI_FORMAT Probe = DXGI_FORMAT_R32G32B32A32_FLOAT;

// SSS separation history (26 Sep): accumulated SSS contribution rgb, linear depth a
constexpr DXGI_FORMAT SssHistory = DXGI_FORMAT_R16G16B16A16_FLOAT;

// Additive light split (27 Sep): specular share rgb + linear depth a (history), intercept rgb (view)
constexpr DXGI_FORMAT SplitShare = DXGI_FORMAT_R16G16B16A16_FLOAT;
constexpr DXGI_FORMAT SplitIntercept = DXGI_FORMAT_R16G16B16A16_FLOAT;

// SSS re-blur (28 Sep): the denoised colour before the blur, the horizontal pass results, the fit terms
constexpr DXGI_FORMAT SssSource = DXGI_FORMAT_R16G16B16A16_FLOAT;
constexpr DXGI_FORMAT SssBlurH = DXGI_FORMAT_R16G16B16A16_FLOAT;
constexpr DXGI_FORMAT SssFitH = DXGI_FORMAT_R16G16B16A16_FLOAT;
constexpr DXGI_FORMAT SssFitTiles = DXGI_FORMAT_R32G32B32A32_FLOAT;
} // namespace FSRDFormats

namespace FSRDProbe
{
constexpr UINT64 AlignUp(UINT64 value, UINT64 alignment) { return (value + alignment - 1) & ~(alignment - 1); }

// Readback buffer layout per ring slot: the whole probe texture, then one window per box.
constexpr UINT kTexRowPitch = FSRD::Probe::kSlotCount * 4 * sizeof(float);
constexpr UINT64 kTexBytes = UINT64(kTexRowPitch) * FSRD::Probe::kMaxRecords;
constexpr UINT kBoxRowPitch = D3D12_TEXTURE_DATA_PITCH_ALIGNMENT; // 9 px x 16 bytes fits in one
constexpr UINT kBoxMaxSide = 2 * FSRD::Probe::kMaxRadius + 1;
constexpr UINT64 kBoxBase = AlignUp(kTexBytes, D3D12_TEXTURE_DATA_PLACEMENT_ALIGNMENT);
constexpr UINT64 kBoxStride = AlignUp(UINT64(kBoxRowPitch) * kBoxMaxSide, D3D12_TEXTURE_DATA_PLACEMENT_ALIGNMENT);
constexpr UINT64 kReadbackBytes = kBoxBase + kBoxStride * FSRD::Probe::BoxCount;

static_assert(kTexRowPitch % D3D12_TEXTURE_DATA_PITCH_ALIGNMENT == 0, "Probe row pitch must be 256 aligned");
static_assert(kBoxMaxSide * 16 <= kBoxRowPitch, "Probe box row does not fit its pitch");

static UINT BytesPerPixel(DXGI_FORMAT format)
{
    switch (format)
    {
    case DXGI_FORMAT_R32G32B32A32_FLOAT:
        return 16;
    case DXGI_FORMAT_R16G16B16A16_FLOAT:
        return 8;
    case DXGI_FORMAT_R8G8B8A8_UNORM:
        return 4;
    default:
        return 0;
    }
}

// Decodes one texel of the formats the probe copies. False for anything else.
static bool DecodeTexel(DXGI_FORMAT format, const uint8_t* texel, float out[4])
{
    switch (format)
    {
    case DXGI_FORMAT_R32G32B32A32_FLOAT:
        memcpy(out, texel, 4 * sizeof(float));
        return true;

    case DXGI_FORMAT_R16G16B16A16_FLOAT:
    {
        DirectX::PackedVector::HALF halves[4];
        memcpy(halves, texel, sizeof(halves));

        for (int c = 0; c < 4; c++)
            out[c] = DirectX::PackedVector::XMConvertHalfToFloat(halves[c]);

        return true;
    }

    case DXGI_FORMAT_R8G8B8A8_UNORM:
        for (int c = 0; c < 4; c++)
            out[c] = float(texel[c]) / 255.0f;

        return true;

    default:
        return false;
    }
}
} // namespace FSRDProbe

struct ComputeState
{
    ID3D12Device* m_pDev = nullptr;

    ComPtr<ID3D12RootSignature> m_rootSig;
    ComPtr<ID3D12PipelineState> m_pso;
    std::vector<FrameDescriptorHeap> m_frameHeaps;

    ComPtr<ID3D12Resource> m_constUploadBuffer;
    byte* m_cbMappedData = nullptr;
    UINT m_cbSlotSize = 0;
    UINT m_cbCurrentFrameIndex = 0;
    UINT backBufferCount = kBackBufferCount;

    ~ComputeState()
    {
        if (m_constUploadBuffer && m_cbMappedData)
        {
            m_constUploadBuffer->Unmap(0, nullptr);
            m_cbMappedData = nullptr;
        }
    }

    void Initialize(ID3D12Device* pDev, std::span<const byte> bytecode, UINT cbDataSize, UINT numSrvs, UINT numUavs,
                    LPCWSTR cbName, UINT backBufferCount = kBackBufferCount)
    {
        m_pDev = pDev;
        this->backBufferCount = backBufferCount;

        // Create Root Signature
        ThrowIfFailed(m_pDev->CreateRootSignature(0, bytecode.data(), bytecode.size(), IID_PPV_ARGS(&m_rootSig)),
                      "Failed to create Root Signature");

        // Create PSO
        D3D12_COMPUTE_PIPELINE_STATE_DESC psoDesc = {};
        psoDesc.pRootSignature = m_rootSig.Get();
        psoDesc.CS = { bytecode.data(), bytecode.size() };
        ThrowIfFailed(m_pDev->CreateComputePipelineState(&psoDesc, IID_PPV_ARGS(&m_pso)), "Failed to create PSO");

        // Create Constant Buffer Upload Heap
        m_cbSlotSize = AlignTo256(cbDataSize);
        const UINT bufferSize = m_cbSlotSize * backBufferCount;

        D3D12_HEAP_PROPERTIES heapProps = { D3D12_HEAP_TYPE_UPLOAD };
        D3D12_RESOURCE_DESC bufferDesc = {};
        bufferDesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        bufferDesc.Width = bufferSize;
        bufferDesc.Height = 1;
        bufferDesc.DepthOrArraySize = 1;
        bufferDesc.MipLevels = 1;
        bufferDesc.SampleDesc.Count = 1;
        bufferDesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        bufferDesc.Flags = D3D12_RESOURCE_FLAG_NONE;

        ThrowIfFailed(m_pDev->CreateCommittedResource(&heapProps, D3D12_HEAP_FLAG_NONE, &bufferDesc,
                                                      D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
                                                      IID_PPV_ARGS(&m_constUploadBuffer)),
                      "Failed to create Constant Buffer");

        m_constUploadBuffer->SetName(cbName);
        D3D12_RANGE readRange = { 0, 0 };
        ThrowIfFailed(m_constUploadBuffer->Map(0, &readRange, reinterpret_cast<void**>(&m_cbMappedData)),
                      "Failed to map Constant Buffer");

        m_frameHeaps.resize(backBufferCount);

        // Create Descriptor Heaps
        for (auto& heap : m_frameHeaps)
        {
            if (!heap.Initialize(m_pDev, numSrvs, numUavs, 0, 0))
                throw std::runtime_error("Failed to initialize FrameDescriptorHeap");
        }
    }

    void Dispatch(ID3D12GraphicsCommandList* cmdList, std::span<const byte> cbData,
                  std::span<ID3D12Resource* const> inputs, std::span<const MipChainDesc> inputMips,
                  std::span<ID3D12Resource*> output, std::span<const UINT> outputMips, XMFLOAT2 outDim,
                  bool autoBarrierOutput = true)
    {
        if (!cmdList)
            return;

        ScopedSkipHeapCapture skipHeapCapture {};

        // Constant Buffer Updates
        const UINT currentFrame = m_cbCurrentFrameIndex;
        const UINT currentOffset = currentFrame * m_cbSlotSize;
        memcpy(m_cbMappedData + currentOffset, cbData.data(), cbData.size());

        D3D12_GPU_VIRTUAL_ADDRESS cbAddress = m_constUploadBuffer->GetGPUVirtualAddress() + currentOffset;
        m_cbCurrentFrameIndex = (m_cbCurrentFrameIndex + 1) % backBufferCount;

        // Transitions SRV -> UAV
        if (autoBarrierOutput)
            AddBarriers(cmdList, output, outputMips, kSrvState, kUavState);

        // Update descriptors
        FrameDescriptorHeap& currentHeap = m_frameHeaps[currentFrame];
        CreateSRVs(m_pDev, currentHeap, inputs, inputMips);
        CreateUAVs(m_pDev, currentHeap, output, outputMips);

        // Configure pipeline
        cmdList->SetPipelineState(m_pso.Get());
        cmdList->SetComputeRootSignature(m_rootSig.Get());

        ID3D12DescriptorHeap* heaps[] = { currentHeap.GetHeapCSU() };
        cmdList->SetDescriptorHeaps(1, heaps);
        cmdList->SetComputeRootConstantBufferView(0, cbAddress);

        // SRV table
        cmdList->SetComputeRootDescriptorTable(1, currentHeap.GetTableGPUStart());

        // UAV table
        CD3DX12_GPU_DESCRIPTOR_HANDLE uavTable = currentHeap.GetTableGPUStart();
        uavTable.Offset((UINT) inputs.size(),
                        m_pDev->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV));
        cmdList->SetComputeRootDescriptorTable(2, uavTable);

        // Dispatch
        const UINT dimX = ((UINT) outDim.x + (kThreadGroupSizeX - 1)) / kThreadGroupSizeX;
        const UINT dimY = ((UINT) outDim.y + (kThreadGroupSizeY - 1)) / kThreadGroupSizeY;
        cmdList->Dispatch(dimX, dimY, 1);

        // Transition the UAVs back to SRV
        if (autoBarrierOutput)
            AddBarriers(cmdList, output, outputMips, kUavState, kSrvState);
    }

    void Dispatch(ID3D12GraphicsCommandList* cmdList, std::span<const byte> cbData,
                  std::span<ID3D12Resource* const> inputs, std::span<ID3D12Resource*> output, XMFLOAT2 outDim,
                  bool autoBarrierOutput = true)
    {
        Dispatch(cmdList, cbData, inputs, {}, output, {}, outDim, autoBarrierOutput);
    }
};

// Private implementation
struct FSRDPreprocessor_Dx12::Impl
{
    ID3D12Device* m_pDev = nullptr;

    ComputeState m_floorSeedShader;
    ComputeState m_floorFilterShader;
    ComputeState m_convShader;
    ComputeState m_compShader;
    ComputeState m_leakShader;     // texture leak views (26 Sep), debug only
    ComputeState m_splitFitShader; // additive light split (27 Sep)
    ComputeState m_sssBlurShader;  // SSS re-blur (28 Sep)

    UINT m_maxWidth = 0;
    UINT m_maxHeight = 0;

    // Output Targets
    // Internal storage
    Conversion::Output m_out;
    ComPtr<ID3D12Resource> m_LinearDepth;
    ComPtr<ID3D12Resource> m_PrevLinearDepth; // copy of last frame's m_LinearDepth (24 Sep)
    ComPtr<ID3D12Resource> m_outputBuffer1;
    ComPtr<ID3D12Resource> m_outputBuffer2;

    // Floor filter
    ID3D12Resource* m_smoothFloor;

    // SSS separation history (26 Sep): ping-pong pair, rgb = accumulated SSS contribution, a = linear
    // depth (0 = no SSS material). m_sssRead is last frame's.
    ComPtr<ID3D12Resource> m_sssHistory[2];
    uint32_t m_sssRead = 0;
    bool m_sssWasActive = false; // false = history invalid, the next active frame starts it afresh

    // Additive light split (27 Sep): ping-pong specular share history and the intercept for the view.
    // m_splitCurrent is the share written this frame (nullptr = split off this frame).
    ComPtr<ID3D12Resource> m_splitHistory[2];
    ComPtr<ID3D12Resource> m_splitIntercept;
    uint32_t m_splitRead = 0;
    bool m_splitWasActive = false;
    ID3D12Resource* m_splitCurrent = nullptr;

    // A/B, audit finding 3: albedo textures created as RGBA16_FLOAT
    bool m_albedo16 = false;

    // SSS re-blur (28 Sep, FSRDSssBlur.hlsl). Created on first use at the current maximum render size, released on
    // a resize; a feature that never turns the re-blur on never allocates them.
    ComPtr<ID3D12Resource> m_sssSource;   // denoised colour before the blur, SSS pixels (composition u1)
    ComPtr<ID3D12Resource> m_sssBlurH;    // horizontal pass: blurred denoised colour
    ComPtr<ID3D12Resource> m_sssFitH;     // horizontal pass: blurred noisy colour (fit only)
    ComPtr<ID3D12Resource> m_sssFitTiles; // fit terms, one texel per 8x8 group (fit only)
    bool m_sssCreateFailed = false;       // don't retry every frame; reset on a resize
    bool m_sssFitCreateFailed = false;

    // Fit readback ring: each slot is mapped kSssFitRing frames after its copy was recorded (no fence, the same
    // margin as the pixel probe).
    struct SssFitSlot
    {
        ComPtr<ID3D12Resource> readback;
        bool pending = false;
        UINT tilesX = 0;
        UINT tilesY = 0;
        UINT rowPitch = 0;
        float radiusMm = 0.0f;
        float falloff = 0.0f;
    };

    static constexpr UINT kSssFitRing = 6;
    std::array<SssFitSlot, kSssFitRing> m_sssFitSlots;
    uint64_t m_sssFitFrame = 0;
    uint64_t m_sssFitReadouts = 0;

    // Running average of the per-pixel fit means (g d, d d, g g), restarted when radius or falloff change
    double m_sssFitAverage[3] = {};
    uint32_t m_sssFitSamples = 0;
    float m_sssFitAvgRadius = -1.0f;
    float m_sssFitAvgFalloff = -1.0f;

    // Pixel probe (24 Sep) --------------------------------------------------------------------
    //
    // The packing shader writes one record of intermediate values per pixel of a small window
    // into m_probeTex (layout: FSRD::Probe::Slot). The texture is copied into a readback buffer
    // straight after packing, and the same window of the denoiser outputs, the composition output
    // and the stored albedos after composition. Each copy is mapped kRingSize frames later,
    // reduced to mean / min / max and published to FSRD::Diagnostics for the menu.
    struct ProbeSlot
    {
        ComPtr<ID3D12Resource> readback;
        bool pending = false;  // probe texture copy recorded, not read back yet
        bool hasBoxes = false; // box copies recorded too (composition ran)
        uint64_t frame = 0;
        int centerX = 0;
        int centerY = 0;
        int radius = 0;
        uint32_t renderWidth = 0;
        uint32_t renderHeight = 0;
        uint32_t convFlags = 0;
        DXGI_FORMAT boxFormats[FSRD::Probe::BoxCount] = {};
    };

    ComPtr<ID3D12Resource> m_probeTex; // set only once all probe resources exist
    bool m_probeCreateFailed = false;
    std::array<ProbeSlot, FSRD::Probe::kRingSize> m_probeSlots;
    ProbeSlot* m_probeCurrent = nullptr; // slot being recorded this frame; nullptr = probe off
    uint64_t m_probeFrame = 0;
    int m_probeAverageFrames = 1;

    // Temporal average state, reset whenever the window or the flags change
    FSRD::ProbeReadout m_probeReadout;
    float m_probeSlotMeanSq[FSRD::Probe::kSlotCount][4] = {};
    float m_probeBoxMeanSq[FSRD::Probe::BoxCount][4] = {};

    void Initialize(std::span<const byte> blSeedByteCode, std::span<const byte> blPyramidByteCode,
                    std::span<const byte> convByteCode, std::span<const byte> compByteCode,
                    std::span<const byte> leakByteCode, std::span<const byte> splitFitByteCode,
                    std::span<const byte> sssBlurByteCode)
    {
        ScopedSkipHeapCapture skipHeapCapture {};

        LOG_DEBUG("Creating FSRD interop shaders...");

        m_floorSeedShader.Initialize(m_pDev, blSeedByteCode, sizeof(FloorSeed::Constants), FloorSeed::Input::kCount,
                                     FloorSeed::Output::kCount, L"FSRD_FloorSeed_Constants",
                                     FloorSeed::kBackBufferCount);
        m_floorFilterShader.Initialize(m_pDev, blPyramidByteCode, sizeof(FloorFilter::Constants),
                                       FloorFilter::Input::kCount, FloorFilter::Output::kCount,
                                       L"FSRD_FloorFilter_Constants", FloorFilter::kBackBufferCount);
        m_convShader.Initialize(m_pDev, convByteCode, sizeof(Conversion::Constants), Conversion::Input::kCount,
                                Conversion::kUavCount, L"FSRD_Conv_Constants", Conversion::kBackBufferCount);
        m_compShader.Initialize(m_pDev, compByteCode, sizeof(Composition::Constants), Composition::Input::kCount,
                                Composition::kOutputCount, L"FSRD_Comp_Constants", Composition::kBackBufferCount);
        m_leakShader.Initialize(m_pDev, leakByteCode, sizeof(Leak::Constants), Leak::Input::kCount, Leak::kOutputCount,
                                L"FSRD_Leak_Constants", Leak::kBackBufferCount);
        m_splitFitShader.Initialize(m_pDev, splitFitByteCode, sizeof(SplitFit::Constants), SplitFit::Input::kCount,
                                    SplitFit::kOutputCount, L"FSRD_SplitFit_Constants", SplitFit::kBackBufferCount);
        m_sssBlurShader.Initialize(m_pDev, sssBlurByteCode, sizeof(SssBlur::Constants), SssBlur::Input::kCount,
                                   SssBlur::kOutputCount, L"FSRD_SssBlur_Constants", SssBlur::kBackBufferCount);

        LOG_DEBUG("FSRD interop shaders and resources initialized.");
    }

    void SetMaxRenderSize(UINT width, UINT height, bool albedo16)
    {
        CreateProbeResources();

        if (m_maxWidth == width && m_maxHeight == height && m_albedo16 == albedo16)
            return;

        m_maxWidth = width;
        m_maxHeight = height;
        m_albedo16 = albedo16;

        auto CreateTex = [&](DXGI_FORMAT fmt, LPCWSTR name, UINT mipLevels = 1)
        { return CreateTexture2D(m_pDev, width, height, fmt, name, kSrvState, mipLevels); };

        auto& outResources = m_out.Resources;
        outResources.Motion = CreateTex(FSRDFormats::Motion, L"FSR_Conv_Motion");
        outResources.Normals = CreateTex(FSRDFormats::Normals, L"FSR_Conv_Normals");
        outResources.SpecAlbedo =
            CreateTex(albedo16 ? FSRDFormats::Albedo16 : FSRDFormats::SpecAlbedo, L"FSR_Conv_SpecAlbedo");
        outResources.DiffAlbedo =
            CreateTex(albedo16 ? FSRDFormats::Albedo16 : FSRDFormats::DiffAlbedo, L"FSR_Conv_DiffAlbedo");
        outResources.SkipSignal = CreateTex(FSRDFormats::SkipSignal, L"FSR_Conv_SkipSignal");

        m_LinearDepth = CreateTex(FSRDFormats::LinearDepth, L"FSR_Conv_LinearDepth");
        m_PrevLinearDepth = CreateTex(FSRDFormats::LinearDepth, L"FSR_Conv_PrevLinearDepth");

        m_smoothFloor = nullptr;

        // Scratch buffers: floor ping-pong, then the denoiser's outputs. (25 Sep: these were created twice in a
        // row; the first pair was released straight away, a wasted allocation per feature creation.)
        m_outputBuffer1 = CreateTex(FSRDFormats::OutputBuffer1, L"FSR_Conv_OutputBuffer1");
        m_outputBuffer2 = CreateTex(FSRDFormats::OutputBuffer2, L"FSR_Conv_OutputBuffer2");

        // Transplant, 22 Sep: unconditional now - denoiser 1.2 only has the split signal shape,
        // so these are always allocated (transplant plan §6e/6f).
        outResources.SpecRadiance = CreateTex(FSRDFormats::SpecRadiance, L"FSR_Conv_SpecRadiance");
        outResources.DiffRadiance = CreateTex(FSRDFormats::DiffRadiance, L"FSR_Conv_DiffRadiance");

        // SSS separation history (26 Sep)
        m_sssHistory[0] = CreateTex(FSRDFormats::SssHistory, L"FSR_Conv_SssHistoryA");
        m_sssHistory[1] = CreateTex(FSRDFormats::SssHistory, L"FSR_Conv_SssHistoryB");
        m_sssRead = 0;
        m_sssWasActive = false;

        // Additive light split (27 Sep)
        m_splitHistory[0] = CreateTex(FSRDFormats::SplitShare, L"FSR_Conv_SplitShareA");
        m_splitHistory[1] = CreateTex(FSRDFormats::SplitShare, L"FSR_Conv_SplitShareB");
        m_splitIntercept = CreateTex(FSRDFormats::SplitIntercept, L"FSR_Conv_SplitIntercept");
        m_splitRead = 0;
        m_splitWasActive = false;
        m_splitCurrent = nullptr;

        // SSS re-blur (28 Sep): recreated at the new size on next use
        ReleaseSssBlurResources();
    }

    void ReleaseSssBlurResources()
    {
        m_sssSource.Reset();
        m_sssBlurH.Reset();
        m_sssFitH.Reset();
        m_sssFitTiles.Reset();
        m_sssCreateFailed = false;
        m_sssFitCreateFailed = false;

        for (auto& slot : m_sssFitSlots)
        {
            slot.readback.Reset();
            slot.pending = false;
        }
    }

    // SSS re-blur (28 Sep). Returns false if the textures can't be created; the re-blur then stays off (and the
    // failure is logged once, until the next resize). The fit's own resources are optional on top: m_sssFitTiles
    // stays null if they fail.
    bool EnsureSssBlurResources(bool fit)
    {
        if (!m_maxWidth || !m_maxHeight)
            return false;

        ScopedSkipHeapCapture skipHeapCapture {};

        if (!m_sssBlurH)
        {
            if (m_sssCreateFailed)
                return false;

            try
            {
                m_sssSource = CreateTexture2D(m_pDev, m_maxWidth, m_maxHeight, FSRDFormats::SssSource,
                                              L"FSR_Conv_SssSource", kSrvState);
                m_sssBlurH = CreateTexture2D(m_pDev, m_maxWidth, m_maxHeight, FSRDFormats::SssBlurH,
                                             L"FSR_Conv_SssBlurH", kSrvState);
            }
            catch (const std::exception& err)
            {
                m_sssSource.Reset();
                m_sssBlurH.Reset();
                m_sssCreateFailed = true;
                LOG_WARN("FSR-RR SSS re-blur unavailable: {}", err.what());
                return false;
            }
        }

        if (fit && !m_sssFitTiles && !m_sssFitCreateFailed)
        {
            try
            {
                const UINT tilesX = (m_maxWidth + SssBlur::kGroupSize - 1) / SssBlur::kGroupSize;
                const UINT tilesY = (m_maxHeight + SssBlur::kGroupSize - 1) / SssBlur::kGroupSize;
                const UINT64 rowPitch = FSRDProbe::AlignUp(UINT64(tilesX) * 16, D3D12_TEXTURE_DATA_PITCH_ALIGNMENT);

                ComPtr<ID3D12Resource> fitH = CreateTexture2D(m_pDev, m_maxWidth, m_maxHeight, FSRDFormats::SssFitH,
                                                              L"FSR_Conv_SssFitH", kSrvState);
                ComPtr<ID3D12Resource> tiles = CreateTexture2D(m_pDev, tilesX, tilesY, FSRDFormats::SssFitTiles,
                                                               L"FSR_Conv_SssFitTiles", kSrvState);

                D3D12_HEAP_PROPERTIES heapProps = { D3D12_HEAP_TYPE_READBACK };
                D3D12_RESOURCE_DESC bufferDesc = {};
                bufferDesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
                bufferDesc.Width = rowPitch * tilesY;
                bufferDesc.Height = 1;
                bufferDesc.DepthOrArraySize = 1;
                bufferDesc.MipLevels = 1;
                bufferDesc.SampleDesc.Count = 1;
                bufferDesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
                bufferDesc.Flags = D3D12_RESOURCE_FLAG_NONE;

                std::array<ComPtr<ID3D12Resource>, kSssFitRing> readbacks;

                for (auto& readback : readbacks)
                {
                    ThrowIfFailed(m_pDev->CreateCommittedResource(&heapProps, D3D12_HEAP_FLAG_NONE, &bufferDesc,
                                                                  D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                                                  IID_PPV_ARGS(&readback)),
                                  "Failed to create SSS fit readback buffer");

                    readback->SetName(L"FSR_Conv_SssFitReadback");
                }

                for (size_t i = 0; i < readbacks.size(); i++)
                {
                    m_sssFitSlots[i].readback = readbacks[i];
                    m_sssFitSlots[i].pending = false;
                }

                m_sssFitH = fitH;
                m_sssFitTiles = tiles;
            }
            catch (const std::exception& err)
            {
                m_sssFitCreateFailed = true;
                LOG_WARN("FSR-RR SSS re-blur fit unavailable: {}", err.what());
            }
        }

        return true;
    }

    // depthOnly (25 Sep): the floor is off, so only the linear depth is written; no median, no gradient, and
    // no floor for the packing shader (a null SRV reads as zero).
    void DispatchFloorSeed(ID3D12GraphicsCommandList* cmdList, const ConversionDesc& desc, bool depthOnly)
    {
        const XMFLOAT2 dispatchSize = { desc.RenderSize.x, desc.RenderSize.y };
        const bool isDepthLinear = (desc.Flags & (uint32_t) ConvFlags::IsDepthLinear);
        ID3D12Resource* inColor = desc.Resources.InColor;

        const uint32_t seedFlags = isDepthLinear ? uint32_t(FloorSeed::Flags::LinearDepth) : 0u;

        if (depthOnly)
        {
            FloorSeed::Constants constants = { .InvProjMatrix = desc.InvProjMatrix,
                                               .RenderSize = desc.RenderSize,
                                               .NearPlane = desc.NearPlane,
                                               .FarPlane = desc.FarPlane,
                                               .Flags = seedFlags | uint32_t(FloorSeed::Flags::DepthOnly) };
            const auto cbData = GetAsByteSpan(constants);

            FloorSeed::Input in = { .Resources = { .InColor = inColor,
                                                   .InNormals = desc.Resources.InNormals,
                                                   .InDepth = desc.Resources.InDepth } };

            FloorSeed::Output out = { .Resources = { .OutColor = m_outputBuffer1.Get(),
                                                     .OutLinearDepth = m_LinearDepth.Get(),
                                                     .OutDepthGradient = m_out.Resources.Motion.Get() } };

            m_floorSeedShader.Dispatch(cmdList, cbData, in.AsArray, out.AsArray, dispatchSize);

            m_smoothFloor = nullptr;
            return;
        }

        for (int i = 0; i < FloorSeed::kPasses; i++)
        {
            FloorSeed::Constants constants = { .InvProjMatrix = desc.InvProjMatrix,
                                               .RenderSize = desc.RenderSize,
                                               .NearPlane = desc.NearPlane,
                                               .FarPlane = desc.FarPlane,
                                               .Flags = seedFlags };
            const auto cbData = GetAsByteSpan(constants);

            // Create median filtered raw color before cross bilateral filtering
            // Write to mip chain at top level
            FloorSeed::Input in = { .Resources = { .InColor = inColor,
                                                   .InNormals = desc.Resources.InNormals,
                                                   .InDepth = desc.Resources.InDepth } };

            FloorSeed::Output out = { .Resources = { .OutColor = m_outputBuffer1.Get(),
                                                     .OutLinearDepth = m_LinearDepth.Get(),
                                                     .OutDepthGradient = m_out.Resources.Motion.Get() } };

            m_floorSeedShader.Dispatch(cmdList, cbData, in.AsArray, out.AsArray, dispatchSize);

            std::swap(m_outputBuffer1, m_outputBuffer2);
            inColor = m_outputBuffer2.Get();
        }

        m_smoothFloor = m_outputBuffer2.Get();
    }

    void DispatchFloorFilter(ID3D12GraphicsCommandList* cmdList, const ConversionDesc& desc)
    {
        static uint32_t frameIndex = 0;
        const XMFLOAT2 dispatchSize = { desc.RenderSize.x, desc.RenderSize.y };

        // Tukey biweight: W = ( 1 - ( (center - tap) * scale )^2 )^2
        // scale = 2^(i + 1) / norm
        float rcpCrossNorm = (1.0f / 0.5f);
        float rcpLumNorm = (1e-2f / 0.3f);

        // A/B, audit finding 4 (24 Sep). After the seed's swap, m_smoothFloor == m_outputBuffer2, so
        // pass 0 below reads and writes the same texture (a race between groups, and an SRV read of a
        // resource in UAV state). Swapping first makes pass 0 write into the other buffer. The swap
        // after the loop restores the frame's usual buffer parity, so the denoiser keeps writing its
        // specular and diffuse outputs into the same physical textures as without the fix.
        const bool noAlias = desc.FloorNoAlias;

        if (noAlias)
            std::swap(m_outputBuffer1, m_outputBuffer2);

        for (int i = 0; i < FloorFilter::kPasses; i++)
        {
            // The detail residual is only meaningful once the wavelet has reached its full
            // support - re-injecting it on every pass would compound it kPasses times.
            const bool isFinalPass = (i == (FloorFilter::kPasses - 1));

            FloorFilter::Constants constants = { .DstTexSize = desc.RenderSize,
                                                 .RcpCrossBlNorm = rcpCrossNorm,
                                                 .RcpSelfBlNorm = rcpLumNorm,
                                                 .StepSize = 1 << i,
                                                 .FrameIndex = frameIndex,
                                                 .DetailBoost = isFinalPass ? desc.FloorDetailBoost : 0.0f,
                                                 .NormalSharpness = desc.FloorNormalSharpness,
                                                 .AlbedoGuideStrength = desc.FloorAlbedoGuide,
                                                 .LumSymmetry = desc.FloorLumSymmetry,
                                                 .GrazingSharpness = desc.FloorGrazingSharpness };
            const auto cbData = GetAsByteSpan(constants);

            FloorFilter::Input in = { .Resources = { .InColor = m_smoothFloor,
                                                     .InLinearDepth = m_LinearDepth.Get(),
                                                     .InDepthGradient = m_out.Resources.Motion.Get(),
                                                     .InDiffAlbedo = desc.Resources.InDiffAlbedo } };

            FloorFilter::Output out = { .Resources = { .OutColor = m_outputBuffer2.Get() } };

            m_floorFilterShader.Dispatch(cmdList, cbData, in.AsArray, out.AsArray, dispatchSize);

            std::swap(m_outputBuffer1, m_outputBuffer2);
            m_smoothFloor = m_outputBuffer1.Get();
        }

        // m_smoothFloor keeps pointing at the final pass's output either way.
        if (noAlias)
            std::swap(m_outputBuffer1, m_outputBuffer2);

        frameIndex++;
    }

    void DispatchPackingShader(ID3D12GraphicsCommandList* cmdList, const ConversionDesc& desc)
    {
        const XMFLOAT2 dispatchSize = { desc.RenderSize.x, desc.RenderSize.y };

        // Prepare inputs for packing and format conversion
        Conversion::Input in = {};
        memcpy_s(in.AsArray, sizeof(in.AsArray), desc.Resources.AsArray, sizeof(desc.Resources.AsArray));
        in.Resources.InDepth = m_LinearDepth.Get();

        Conversion::Constants packConstants = { .InvViewMatrix = desc.InvViewMatrix,
                                                .InvProjMatrix = desc.InvProjMatrix,
                                                .PrevViewMatrix = desc.PrevViewMatrix,
                                                .RenderSize = desc.RenderSize,
                                                .NearPlane = desc.NearPlane,
                                                .FarPlane = desc.FarPlane,
                                                .FloorIsolation = desc.FloorIsolation,
                                                .Flags = desc.Flags,
                                                .BiasMaskStrength = desc.BiasMaskStrength,
                                                .FloorSoftMin = desc.FloorSoftMin,
                                                .RoughnessExponent = desc.RoughnessExponent,
                                                .HitDistScale = desc.HitDistScale,
                                                .FloorSpecGuard = desc.FloorSpecGuard,
                                                .SplitPriorStrength = desc.SplitPriorStrength,
                                                .RoughnessProbe = desc.RoughnessProbe,
                                                .FloorSpecGuardFadeStart = desc.FloorSpecGuardFadeStart,
                                                .FloorSpecGuardFadeEnd = desc.FloorSpecGuardFadeEnd,
                                                .FireflyClampK = desc.FireflyClampK,
                                                .SssSeparation = desc.SssSeparation,
                                                .ProjMatrix = desc.ProjMatrix,
                                                .SssHistoryAlpha = std::clamp(desc.SssHistoryAlpha, 0.01f, 1.0f) };

        in.Resources.InBlurColor = m_smoothFloor;
        in.Resources.InPrevLinearDepth = m_PrevLinearDepth.Get();
        in.Resources.InSSSGuide = desc.InSSSGuide;

        // SSS separation (26 Sep): read last frame's history, write this frame's into the other one. The
        // first active frame (after switching on or a resize) starts the history afresh.
        const bool sssActive = desc.SssSeparation > 0.0f && desc.InSSSGuide != nullptr;
        in.Resources.InSssHistory = m_sssHistory[m_sssRead].Get();

        // Additive light split (27 Sep): the specular share DispatchSplitFit wrote this frame, if it ran
        in.Resources.InSplitFit = m_splitCurrent;
        packConstants.AdditiveSeparation = (m_splitCurrent != nullptr) ? desc.AdditiveSeparation : 0.0f;

        if (!sssActive)
            packConstants.SssSeparation = 0.0f;
        else if (!m_sssWasActive)
            packConstants.Flags |= UINT(ConvFlags::SssReset);

        // Texture leak views (26 Sep) measure what the denoiser receives, and the additive light views (27 Sep)
        // show the split fit's outputs, so the packing shader runs its normal path on those frames;
        // DispatchConversion draws the view afterwards.
        if (IsPostPackingView(desc))
            packConstants.Flags &= ~UINT(ConvFlags::DebugModeMask);

        // A null SRV reads as zero, so the shader is safe without this, but the flag keeps
        // the "no mask provided" case explicit and visible in the debug views.
        if (desc.Resources.InBiasMask != nullptr)
            packConstants.Flags |= UINT(ConvFlags::HasBiasMask);

        // Pixel probe window, chosen in BeginProbeFrame. No slot this frame = no probe writes.
        if (m_probeCurrent != nullptr)
        {
            packConstants.ProbeCenterX = m_probeCurrent->centerX;
            packConstants.ProbeCenterY = m_probeCurrent->centerY;
            packConstants.ProbeRadius = m_probeCurrent->radius;
        }
        else
        {
            packConstants.Flags &= ~UINT(ConvFlags::Probe);
        }

        // u0-u6: the denoiser inputs, u7: the probe texture (always bound, written only with the flag),
        // u8: this frame's SSS history (always bound, written only while the separation is on)
        std::array<ID3D12Resource*, Conversion::kUavCount> uavs {};
        std::copy(std::begin(m_out.AsRawArray), std::end(m_out.AsRawArray), uavs.begin());
        uavs[Conversion::kProbeUav] = m_probeTex.Get();
        uavs[Conversion::kSssHistoryUav] = m_sssHistory[1 - m_sssRead].Get();

        const std::span<const byte> convCBData((const byte*) &packConstants, sizeof(packConstants));
        m_convShader.Dispatch(cmdList, convCBData, in.AsArray, uavs, dispatchSize, true);

        if (sssActive)
            m_sssRead = 1 - m_sssRead;

        m_sssWasActive = sssActive;
    }

    static bool IsLeakView(const ConversionDesc& desc)
    {
        const uint32_t debugMode = desc.Flags & uint32_t(ConvFlags::DebugModeMask);
        return debugMode == uint32_t(ConvFlags::DebugLeakSpecular) ||
               debugMode == uint32_t(ConvFlags::DebugLeakDiffuse);
    }

    // Views drawn after the packing shader ran its normal path (26/27 Sep)
    static bool IsPostPackingView(const ConversionDesc& desc)
    {
        const uint32_t debugMode = desc.Flags & uint32_t(ConvFlags::DebugModeMask);
        return IsLeakView(desc) || debugMode == uint32_t(ConvFlags::DebugAdditiveLight) ||
               debugMode == uint32_t(ConvFlags::DebugSpecularShare);
    }

    // Copies a view over SpecRadiance, the texture a conversion debug view shows. The denoiser is bypassed
    // on those frames, so nothing else reads SpecRadiance. Both must be in SRV state.
    void ShowTexture(ID3D12GraphicsCommandList* cmdList, ID3D12Resource* view)
    {
        ID3D12Resource* shown = m_out.Resources.SpecRadiance.Get();

        if (view == nullptr || shown == nullptr)
            return;

        AddBarrier(cmdList, view, kSrvState, D3D12_RESOURCE_STATE_COPY_SOURCE);
        AddBarrier(cmdList, shown, kSrvState, D3D12_RESOURCE_STATE_COPY_DEST);
        cmdList->CopyResource(shown, view);
        AddBarrier(cmdList, view, D3D12_RESOURCE_STATE_COPY_SOURCE, kSrvState);
        AddBarrier(cmdList, shown, D3D12_RESOURCE_STATE_COPY_DEST, kSrvState);
    }

    // Additive light split (27 Sep, FSRDSplitFit.hlsl). Runs after the floor seed (it needs this frame's linear
    // depth) and before the packing shader, which reads the specular share it writes.
    void DispatchSplitFit(ID3D12GraphicsCommandList* cmdList, const ConversionDesc& desc)
    {
        const bool active = desc.AdditiveSeparation > 0.0f;

        if (!active)
        {
            m_splitCurrent = nullptr;
            m_splitWasActive = false;
            return;
        }

        SplitFit::Constants constants = { .DstTexSize = desc.RenderSize,
                                          .Strength = std::clamp(desc.AdditiveSeparation, 0.0f, 1.0f),
                                          .HistoryAlpha = std::clamp(desc.AdditiveHistoryAlpha, 0.01f, 1.0f),
                                          .Flags = m_splitWasActive ? 0u : uint32_t(SplitFit::Flags::Reset) };

        SplitFit::Input in = { .Resources = { .InColor = desc.Resources.InColor,
                                              .InDiffAlbedo = desc.Resources.InDiffAlbedo,
                                              .InSpecAlbedo = desc.Resources.InSpecAlbedo,
                                              .InLinearDepth = m_LinearDepth.Get(),
                                              .InNormals = desc.Resources.InNormals,
                                              .InMotionVectors = desc.Resources.InMotionVectors,
                                              .InHistory = m_splitHistory[m_splitRead].Get() } };

        ID3D12Resource* written = m_splitHistory[1 - m_splitRead].Get();
        std::array<ID3D12Resource*, SplitFit::kOutputCount> out { written, m_splitIntercept.Get() };

        const XMFLOAT2 dispatchSize = { desc.RenderSize.x, desc.RenderSize.y };
        m_splitFitShader.Dispatch(cmdList, GetAsByteSpan(constants), in.AsArray, out, dispatchSize);

        m_splitCurrent = written;
        m_splitRead = 1 - m_splitRead;
        m_splitWasActive = true;
    }

    // Texture leak view (26 Sep): FSRDLeak.hlsl reads the packing shader's outputs and draws into the
    // scratch buffer, which is then copied over SpecRadiance - the texture a conversion debug view shows.
    // The denoiser is bypassed on those frames, so nothing else reads SpecRadiance.
    void DispatchLeakView(ID3D12GraphicsCommandList* cmdList, const ConversionDesc& desc)
    {
        const uint32_t debugMode = desc.Flags & uint32_t(ConvFlags::DebugModeMask);
        const bool diffuse = debugMode == uint32_t(ConvFlags::DebugLeakDiffuse);

        Leak::Constants constants = { .DstTexSize = desc.RenderSize,
                                      .Flags = diffuse ? uint32_t(Leak::Flags::Diffuse) : 0u };

        Leak::Input in = { .Resources = { .InSpecRadiance = m_out.Resources.SpecRadiance.Get(),
                                          .InDiffRadiance = m_out.Resources.DiffRadiance.Get(),
                                          .InSpecAlbedo = m_out.Resources.SpecAlbedo.Get(),
                                          .InDiffAlbedo = m_out.Resources.DiffAlbedo.Get(),
                                          .InLinearDepth = m_LinearDepth.Get(),
                                          .InNormals = m_out.Resources.Normals.Get() } };

        std::array<ID3D12Resource*, Leak::kOutputCount> out { m_outputBuffer1.Get() };

        const XMFLOAT2 dispatchSize = { desc.RenderSize.x, desc.RenderSize.y };
        m_leakShader.Dispatch(cmdList, GetAsByteSpan(constants), in.AsArray, out, dispatchSize);

        ShowTexture(cmdList, m_outputBuffer1.Get());
    }

    void DispatchConversion(ID3D12GraphicsCommandList* cmdList, const ConversionDesc& desc)
    {
        if (!cmdList || !m_maxWidth)
            return;

        // Reads back the probe copy recorded kRingSize frames ago and picks this frame's window
        BeginProbeFrame(desc);

        // Filtered raster lighting estimate.
        //
        // 25 Sep: with Floor Isolation 0 the packing shader multiplies the floor by zero, so it isn't computed:
        // the seed pass writes linear depth only and the five a-trous passes are skipped. The one thing the
        // floor still fed at 0 was the SkipSignal alpha, the luminance reference of the Correlation Bias raw
        // blend, where it added the a-trous floor's light although the output contains none of it (audit
        // finding 2); with no floor that alpha is zero. The floor debug views still get a floor.
        const uint32_t debugMode = desc.Flags & uint32_t(ConvFlags::DebugModeMask);
        const bool floorView =
            debugMode == uint32_t(ConvFlags::DebugFloorVariance) || debugMode == uint32_t(ConvFlags::DebugFloorColor);
        const bool floorActive = desc.FloorIsolation > 0.0f || floorView;

        DispatchFloorSeed(cmdList, desc, !floorActive);

        if (floorActive)
            DispatchFloorFilter(cmdList, desc);

        // Additive light split (27 Sep): specular share for the packing shader
        DispatchSplitFit(cmdList, desc);

        // DLSS-RR to FSR-RR conversion
        DispatchPackingShader(cmdList, desc);

        // Keep this frame's linear depth for next frame's object-motion depth delta (24 Sep). Copied
        // every frame, so switching the A/B on never reads stale depth. Both stay in SRV state.
        AddBarrier(cmdList, m_LinearDepth.Get(), kSrvState, D3D12_RESOURCE_STATE_COPY_SOURCE);
        AddBarrier(cmdList, m_PrevLinearDepth.Get(), kSrvState, D3D12_RESOURCE_STATE_COPY_DEST);
        cmdList->CopyResource(m_PrevLinearDepth.Get(), m_LinearDepth.Get());
        AddBarrier(cmdList, m_LinearDepth.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE, kSrvState);
        AddBarrier(cmdList, m_PrevLinearDepth.Get(), D3D12_RESOURCE_STATE_COPY_DEST, kSrvState);

        // The probe texture is back in SRV state after the dispatch
        RecordProbeTexCopy(cmdList);

        // Texture leak views (26 Sep), while the scratch buffers are still in SRV state, and the additive light
        // views (27 Sep)
        const uint32_t viewMode = desc.Flags & uint32_t(ConvFlags::DebugModeMask);

        if (IsLeakView(desc))
            DispatchLeakView(cmdList, desc);
        else if (viewMode == uint32_t(ConvFlags::DebugAdditiveLight))
            ShowTexture(cmdList, m_splitCurrent != nullptr ? m_splitIntercept.Get() : nullptr);
        else if (viewMode == uint32_t(ConvFlags::DebugSpecularShare))
            ShowTexture(cmdList, m_splitCurrent);

        // Transition output buffers to UAV after last composition pass or first init.
        // The denoiser will be writing to these.
        AddBarrier(cmdList, m_outputBuffer1.Get(), kSrvState, kUavState);
        AddBarrier(cmdList, m_outputBuffer2.Get(), kSrvState, kUavState);
    }

    void DispatchComposition(ID3D12GraphicsCommandList* cmdList, const CompositionDesc& desc)
    {
        if (!cmdList || !m_maxWidth)
            return;

        auto& outResources = m_out.Resources;
        Composition::Input inputs = {};
        Composition::Constants constants = { .DstTexSize = desc.DstTexSize,
                                             .CorrelationBias = desc.CorrelationBias,
                                             .Flags = UINT(desc.Flags) };

        // SSS re-blur (28 Sep). Runs on the normal composition and on its own fit view; any other composition
        // view shows something else, so it stays off there.
        const uint32_t compDebug = desc.Flags & uint32_t(CompFlags::DebugModeMask);
        const bool sssView = compDebug == uint32_t(CompFlags::DebugSssFit);
        const bool otherView = (desc.Flags & uint32_t(CompFlags::Debug)) != 0 && !sssView;
        const bool sssUsable =
            !otherView && desc.InSSSGuide != nullptr && desc.InRawColor != nullptr && desc.FocalLengthPx > 0.0f;

        bool sssApply = sssUsable && desc.SssApply;
        bool sssFit = sssUsable && (desc.SssFit || sssView);

        if ((sssApply || sssFit) && !EnsureSssBlurResources(sssFit))
        {
            sssApply = false;
            sssFit = false;
        }

        sssFit = sssFit && m_sssFitTiles != nullptr;

        if (sssApply)
            constants.Flags |= uint32_t(CompFlags::SssResynth);

        // Transition denoiser output buffers to SRV for composition
        std::array<ID3D12Resource*, 2> buffers = { m_outputBuffer1.Get(), m_outputBuffer2.Get() };
        AddBarriers(cmdList, buffers, kUavState, kSrvState);

        // Transplant, 22 Sep: unconditional now - always the split-signal shape (transplant plan
        // §6e/6f). CompFlags::Mode2Signal removed; FSRDOutputComp.hlsl no longer branches on it
        // either (confirmed by reading it fresh this pass, not assumed - it wasn't in the
        // original plan's §6f scope list, but Composition::Input's already-generic two-signal
        // shape was the tell).
        inputs.Resources = { .InDenoisedSignal1 = m_outputBuffer1.Get(),
                             .InAlbedo1 = outResources.SpecAlbedo.Get(),
                             .InDenoisedSignal2 = m_outputBuffer2.Get(),
                             .InAlbedo2 = outResources.DiffAlbedo.Get(),
                             .InSkipSignal = outResources.SkipSignal.Get(),
                             .InRawColor = desc.InRawColor,
                             .InColorBeforeParticles = desc.InColorBeforeParticles,
                             .InSSSGuide = desc.InSSSGuide };

        std::array<ID3D12Resource*, Composition::kOutputCount> uavs { m_out.Resources.Motion.Get(),
                                                                      sssApply ? m_sssSource.Get() : nullptr };
        const std::span<const byte> cbData((const byte*) &constants, sizeof(constants));
        const XMFLOAT2 dstDim = { constants.DstTexSize.x, constants.DstTexSize.y };

        m_compShader.Dispatch(cmdList, cbData, inputs.AsArray, uavs, dstDim, true);

        if (sssApply || sssFit)
            DispatchSssBlur(cmdList, desc, sssApply, sssFit, sssView);

        // Denoiser outputs, composition output and stored albedos are all in SRV state here
        RecordProbeBoxCopies(cmdList);
    }

    // SSS re-blur (28 Sep, FSRDSssBlur.hlsl): horizontal pass into m_sssBlurH (and m_sssFitH), vertical pass over
    // the composition output (and the fit terms into m_sssFitTiles).
    void DispatchSssBlur(ID3D12GraphicsCommandList* cmdList, const CompositionDesc& desc, bool apply, bool fit,
                         bool view)
    {
        const float radiusMm = std::clamp(desc.SssRadiusMm, 0.05f, 50.0f);
        const float falloff = std::clamp(desc.SssFalloff, 0.0f, 1.0f);
        const float sigmaM = radiusMm * 1e-3f;

        // Red scatters furthest in skin; at falloff 1 green and blue are close to the usual separable-SSS profile.
        SssBlur::Constants constants = {
            .DstTexSize = desc.DstTexSize,
            .ChannelScale = { 1.0f, 1.0f + (0.4f - 1.0f) * falloff, 1.0f + (0.3f - 1.0f) * falloff },
            .SigmaScale = sigmaM * desc.FocalLengthPx,
            .Strength = std::clamp(desc.SssStrength, 0.0f, 2.0f),
            .DepthTolerance = 6.0f * sigmaM,
        };

        uint32_t flags = 0;

        if (apply)
            flags |= uint32_t(SssBlur::Flags::Apply);

        if (fit)
            flags |= uint32_t(SssBlur::Flags::Fit);

        const XMFLOAT2 dispatchSize = { desc.DstTexSize.x, desc.DstTexSize.y };

        // Horizontal
        {
            constants.Flags = flags;

            SssBlur::Input in = { .Resources = { .InBlurSrc = m_sssSource.Get(),
                                                 .InFitSrc = nullptr,
                                                 .InColor = desc.InRawColor,
                                                 .InSSSGuide = desc.InSSSGuide,
                                                 .InLinearDepth = m_LinearDepth.Get(),
                                                 .InColorBeforeParticles = nullptr,
                                                 .InSssSource = nullptr } };

            std::array<ID3D12Resource*, SssBlur::kOutputCount> out { m_sssBlurH.Get(),
                                                                     fit ? m_sssFitH.Get() : nullptr };

            m_sssBlurShader.Dispatch(cmdList, GetAsByteSpan(constants), in.AsArray, out, dispatchSize);
        }

        // Vertical
        {
            constants.Flags = flags | uint32_t(SssBlur::Flags::Vertical) | (view ? uint32_t(SssBlur::Flags::View) : 0u);

            SssBlur::Input in = { .Resources = { .InBlurSrc = m_sssBlurH.Get(),
                                                 .InFitSrc = fit ? m_sssFitH.Get() : nullptr,
                                                 .InColor = desc.InRawColor,
                                                 .InSSSGuide = desc.InSSSGuide,
                                                 .InLinearDepth = m_LinearDepth.Get(),
                                                 .InColorBeforeParticles = desc.InColorBeforeParticles,
                                                 .InSssSource = m_sssSource.Get() } };

            std::array<ID3D12Resource*, SssBlur::kOutputCount> out { m_out.Resources.Motion.Get(),
                                                                     fit ? m_sssFitTiles.Get() : nullptr };

            m_sssBlurShader.Dispatch(cmdList, GetAsByteSpan(constants), in.AsArray, out, dispatchSize);
        }

        if (fit)
            RecordSssFitCopy(cmdList, desc, radiusMm, falloff);
    }

    // Copies this frame's fit terms into the next readback slot, after reading that slot's previous copy
    // (recorded kSssFitRing frames ago).
    void RecordSssFitCopy(ID3D12GraphicsCommandList* cmdList, const CompositionDesc& desc, float radiusMm,
                          float falloff)
    {
        SssFitSlot& slot = m_sssFitSlots[m_sssFitFrame % kSssFitRing];
        m_sssFitFrame++;

        if (slot.pending)
        {
            ReadSssFitSlot(slot);
            slot.pending = false;
        }

        if (!slot.readback)
            return;

        const UINT width = UINT(desc.DstTexSize.x);
        const UINT height = UINT(desc.DstTexSize.y);
        const UINT tilesX = (width + SssBlur::kGroupSize - 1) / SssBlur::kGroupSize;
        const UINT tilesY = (height + SssBlur::kGroupSize - 1) / SssBlur::kGroupSize;
        const D3D12_RESOURCE_DESC tilesDesc = m_sssFitTiles->GetDesc();

        if (tilesX == 0 || tilesY == 0 || tilesX > tilesDesc.Width || tilesY > tilesDesc.Height)
            return;

        slot.tilesX = tilesX;
        slot.tilesY = tilesY;
        slot.rowPitch = UINT(FSRDProbe::AlignUp(UINT64(tilesX) * 16, D3D12_TEXTURE_DATA_PITCH_ALIGNMENT));
        slot.radiusMm = radiusMm;
        slot.falloff = falloff;

        AddBarrier(cmdList, m_sssFitTiles.Get(), kSrvState, D3D12_RESOURCE_STATE_COPY_SOURCE);

        D3D12_TEXTURE_COPY_LOCATION dst = {};
        dst.pResource = slot.readback.Get();
        dst.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
        dst.PlacedFootprint.Offset = 0;
        dst.PlacedFootprint.Footprint.Format = FSRDFormats::SssFitTiles;
        dst.PlacedFootprint.Footprint.Width = tilesX;
        dst.PlacedFootprint.Footprint.Height = tilesY;
        dst.PlacedFootprint.Footprint.Depth = 1;
        dst.PlacedFootprint.Footprint.RowPitch = slot.rowPitch;

        D3D12_TEXTURE_COPY_LOCATION src = {};
        src.pResource = m_sssFitTiles.Get();
        src.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        src.SubresourceIndex = 0;

        const D3D12_BOX box = { 0, 0, 0, tilesX, tilesY, 1 };
        cmdList->CopyTextureRegion(&dst, 0, 0, 0, &src, &box);

        AddBarrier(cmdList, m_sssFitTiles.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE, kSrvState);
        slot.pending = true;
    }

    // Sums the fit terms of one readback and publishes the least-squares strength and residual. The per-pixel
    // means are averaged over readouts (a plain mean for the first ten, then 10% per readout), restarting when the
    // radius or the falloff changes.
    void ReadSssFitSlot(SssFitSlot& slot)
    {
        if (!slot.readback || slot.tilesX == 0 || slot.tilesY == 0)
            return;

        void* mapped = nullptr;
        const D3D12_RANGE readRange = { 0, SIZE_T(slot.rowPitch) * slot.tilesY };

        if (FAILED(slot.readback->Map(0, &readRange, &mapped)) || mapped == nullptr)
            return;

        double sums[4] = {};

        for (UINT y = 0; y < slot.tilesY; y++)
        {
            const float* row =
                reinterpret_cast<const float*>(static_cast<const byte*>(mapped) + SIZE_T(slot.rowPitch) * y);

            for (UINT x = 0; x < slot.tilesX; x++)
            {
                for (UINT c = 0; c < 4; c++)
                {
                    const float value = row[4 * x + c];

                    if (std::isfinite(value))
                        sums[c] += value;
                }
            }
        }

        const D3D12_RANGE writeRange = { 0, 0 };
        slot.readback->Unmap(0, &writeRange);

        m_sssFitReadouts++;

        const double pixels = sums[3];

        if (pixels >= 1.0)
        {
            const double means[3] = { sums[0] / pixels, sums[1] / pixels, sums[2] / pixels };

            if (m_sssFitSamples == 0 || slot.radiusMm != m_sssFitAvgRadius || slot.falloff != m_sssFitAvgFalloff)
            {
                std::copy(std::begin(means), std::end(means), std::begin(m_sssFitAverage));
                m_sssFitSamples = 1;
                m_sssFitAvgRadius = slot.radiusMm;
                m_sssFitAvgFalloff = slot.falloff;
            }
            else
            {
                m_sssFitSamples++;
                const double weight = std::max(1.0 / double(m_sssFitSamples), 0.1);

                for (int i = 0; i < 3; i++)
                    m_sssFitAverage[i] += weight * (means[i] - m_sssFitAverage[i]);
            }
        }

        const double gd = m_sssFitAverage[0];
        const double dd = m_sssFitAverage[1];
        const double gg = m_sssFitAverage[2];
        const double r2 = (dd > 0.0 && gg > 0.0) ? (gd * gd) / (dd * gg) : 0.0;

        auto& diagnostics = FSRD::Diagnostics::Instance();
        std::scoped_lock lock(diagnostics.Mutex);
        auto& readout = diagnostics.SssFit;

        readout.valid = m_sssFitSamples > 0;
        readout.readouts = m_sssFitReadouts;
        readout.pixels = uint64_t(std::max(pixels, 0.0));
        readout.emaSamples = m_sssFitSamples;
        readout.radiusMm = m_sssFitAvgRadius;
        readout.falloff = m_sssFitAvgFalloff;
        readout.strength = (dd > 0.0) ? float(gd / dd) : 0.0f;
        readout.error = float(std::sqrt(std::max(1.0 - r2, 0.0)));
    }

    // Pixel probe ---------------------------------------------------------------------------------

    // The probe is optional: if its resources can't be created, it stays off and FSR-RR runs as before.
    // m_probeTex is only set once every readback buffer exists, and the attempt is made once.
    void CreateProbeResources()
    {
        if (m_probeTex || m_probeCreateFailed)
            return;

        ScopedSkipHeapCapture skipHeapCapture {};

        try
        {
            ComPtr<ID3D12Resource> probeTex = CreateTexture2D(m_pDev, FSRD::Probe::kSlotCount, FSRD::Probe::kMaxRecords,
                                                              FSRDFormats::Probe, L"FSR_Conv_Probe", kSrvState);

            D3D12_HEAP_PROPERTIES heapProps = { D3D12_HEAP_TYPE_READBACK };
            D3D12_RESOURCE_DESC bufferDesc = {};
            bufferDesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
            bufferDesc.Width = FSRDProbe::kReadbackBytes;
            bufferDesc.Height = 1;
            bufferDesc.DepthOrArraySize = 1;
            bufferDesc.MipLevels = 1;
            bufferDesc.SampleDesc.Count = 1;
            bufferDesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
            bufferDesc.Flags = D3D12_RESOURCE_FLAG_NONE;

            std::array<ComPtr<ID3D12Resource>, FSRD::Probe::kRingSize> readbacks;

            for (auto& readback : readbacks)
            {
                ThrowIfFailed(m_pDev->CreateCommittedResource(&heapProps, D3D12_HEAP_FLAG_NONE, &bufferDesc,
                                                              D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                                              IID_PPV_ARGS(&readback)),
                              "Failed to create probe readback buffer");

                readback->SetName(L"FSR_Conv_ProbeReadback");
            }

            for (size_t i = 0; i < readbacks.size(); i++)
                m_probeSlots[i].readback = readbacks[i];

            m_probeTex = probeTex;
        }
        catch (const std::exception& err)
        {
            m_probeCreateFailed = true;
            LOG_WARN("FSR-RR pixel probe unavailable: {}", err.what());
        }
    }

    // Called once per frame before the conversion passes.
    void BeginProbeFrame(const ConversionDesc& desc)
    {
        m_probeCurrent = nullptr;

        if (!m_probeTex)
            return;

        ProbeSlot& slot = m_probeSlots[m_probeFrame % FSRD::Probe::kRingSize];
        m_probeFrame++;
        m_probeAverageFrames = std::max(desc.ProbeAverageFrames, 1);

        // Recorded kRingSize frames ago, so the GPU is done with it (see kRingSize)
        if (slot.pending)
        {
            ReadProbeSlot(slot);
            slot.pending = false;
            slot.hasBoxes = false;
        }

        const int width = int(desc.RenderSize.x);
        const int height = int(desc.RenderSize.y);

        if (!(desc.Flags & uint32_t(ConvFlags::Probe)) || width <= 0 || height <= 0)
            return;

        const int radius =
            std::clamp(desc.ProbeRadius, 0, std::min({ FSRD::Probe::kMaxRadius, (width - 1) / 2, (height - 1) / 2 }));
        const int centerX = int(std::lround(double(desc.ProbeU) * width));
        const int centerY = int(std::lround(double(desc.ProbeV) * height));

        slot.centerX = std::clamp(centerX, radius, width - 1 - radius);
        slot.centerY = std::clamp(centerY, radius, height - 1 - radius);
        slot.radius = radius;
        slot.renderWidth = uint32_t(width);
        slot.renderHeight = uint32_t(height);
        slot.convFlags = desc.Flags;
        slot.frame = m_probeFrame;
        slot.hasBoxes = false;

        m_probeCurrent = &slot;
    }

    void RecordProbeTexCopy(ID3D12GraphicsCommandList* cmdList)
    {
        if (m_probeCurrent == nullptr)
            return;

        AddBarrier(cmdList, m_probeTex.Get(), kSrvState, D3D12_RESOURCE_STATE_COPY_SOURCE);

        D3D12_TEXTURE_COPY_LOCATION dst = {};
        dst.pResource = m_probeCurrent->readback.Get();
        dst.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
        dst.PlacedFootprint.Offset = 0;
        dst.PlacedFootprint.Footprint.Format = FSRDFormats::Probe;
        dst.PlacedFootprint.Footprint.Width = FSRD::Probe::kSlotCount;
        dst.PlacedFootprint.Footprint.Height = FSRD::Probe::kMaxRecords;
        dst.PlacedFootprint.Footprint.Depth = 1;
        dst.PlacedFootprint.Footprint.RowPitch = FSRDProbe::kTexRowPitch;

        D3D12_TEXTURE_COPY_LOCATION src = {};
        src.pResource = m_probeTex.Get();
        src.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        src.SubresourceIndex = 0;

        cmdList->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);

        AddBarrier(cmdList, m_probeTex.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE, kSrvState);
        m_probeCurrent->pending = true;
    }

    void RecordProbeBoxCopies(ID3D12GraphicsCommandList* cmdList)
    {
        if (m_probeCurrent == nullptr || !m_probeCurrent->pending)
            return;

        std::array<ID3D12Resource*, FSRD::Probe::BoxCount> sources {};
        sources[FSRD::Probe::DenoisedSpec] = m_outputBuffer1.Get();
        sources[FSRD::Probe::DenoisedDiff] = m_outputBuffer2.Get();
        sources[FSRD::Probe::Composited] = m_out.Resources.Motion.Get();
        sources[FSRD::Probe::SpecAlbedoTex] = m_out.Resources.SpecAlbedo.Get();
        sources[FSRD::Probe::DiffAlbedoTex] = m_out.Resources.DiffAlbedo.Get();

        for (auto* source : sources)
        {
            if (source == nullptr)
                return;
        }

        ProbeSlot& slot = *m_probeCurrent;
        const UINT side = UINT(2 * slot.radius + 1);
        const UINT left = UINT(slot.centerX - slot.radius);
        const UINT top = UINT(slot.centerY - slot.radius);
        const D3D12_BOX box = { left, top, 0, left + side, top + side, 1 };

        AddBarriers(cmdList, sources, kSrvState, D3D12_RESOURCE_STATE_COPY_SOURCE);

        for (UINT b = 0; b < FSRD::Probe::BoxCount; b++)
        {
            const D3D12_RESOURCE_DESC sourceDesc = sources[b]->GetDesc();
            slot.boxFormats[b] = sourceDesc.Format;

            // Only formats the readback can decode, and only a box that lies inside the texture
            // (they are the render size, but never trust that for a copy that could fault).
            const bool fits = UINT64(box.right) <= sourceDesc.Width && box.bottom <= sourceDesc.Height;

            if (FSRDProbe::BytesPerPixel(sourceDesc.Format) == 0 || !fits)
            {
                slot.boxFormats[b] = DXGI_FORMAT_UNKNOWN;
                continue;
            }

            const DXGI_FORMAT format = sourceDesc.Format;

            D3D12_TEXTURE_COPY_LOCATION dst = {};
            dst.pResource = slot.readback.Get();
            dst.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
            dst.PlacedFootprint.Offset = FSRDProbe::kBoxBase + FSRDProbe::kBoxStride * b;
            dst.PlacedFootprint.Footprint.Format = format;
            dst.PlacedFootprint.Footprint.Width = side;
            dst.PlacedFootprint.Footprint.Height = side;
            dst.PlacedFootprint.Footprint.Depth = 1;
            dst.PlacedFootprint.Footprint.RowPitch = FSRDProbe::kBoxRowPitch;

            D3D12_TEXTURE_COPY_LOCATION src = {};
            src.pResource = sources[b];
            src.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
            src.SubresourceIndex = 0;

            cmdList->CopyTextureRegion(&dst, 0, 0, 0, &src, &box);
        }

        AddBarriers(cmdList, sources, D3D12_RESOURCE_STATE_COPY_SOURCE, kSrvState);
        slot.hasBoxes = true;
    }

    // Folds one frame's window into mean / min / max and the running temporal average.
    template <typename Fetch>
    void ReduceProbeWindow(FSRD::Probe::Stats& stats, float (&meanSq)[4], uint32_t count, bool resetAverage,
                           Fetch fetch)
    {
        float sum[4] = {};
        float lo[4] = { INFINITY, INFINITY, INFINITY, INFINITY };
        float hi[4] = { -INFINITY, -INFINITY, -INFINITY, -INFINITY };

        for (uint32_t i = 0; i < count; i++)
        {
            float value[4] = {};
            fetch(i, value);

            for (int c = 0; c < 4; c++)
            {
                sum[c] += value[c];
                lo[c] = std::min(lo[c], value[c]);
                hi[c] = std::max(hi[c], value[c]);
            }
        }

        // First sample initialises the average; after that a running mean over at most
        // m_probeAverageFrames readouts.
        const uint32_t samples = std::min<uint32_t>(m_probeReadout.emaSamples, uint32_t(m_probeAverageFrames));
        const float alpha = resetAverage ? 1.0f : 1.0f / float(samples);

        for (int c = 0; c < 4; c++)
        {
            const float mean = count > 0 ? sum[c] / float(count) : 0.0f;
            stats.mean[c] = mean;
            stats.min[c] = count > 0 ? lo[c] : 0.0f;
            stats.max[c] = count > 0 ? hi[c] : 0.0f;

            stats.emaMean[c] += alpha * (mean - stats.emaMean[c]);
            meanSq[c] += alpha * (mean * mean - meanSq[c]);
            stats.emaStd[c] = std::sqrt(std::max(meanSq[c] - stats.emaMean[c] * stats.emaMean[c], 0.0f));
        }
    }

    void ReadProbeSlot(const ProbeSlot& slot)
    {
        const SIZE_T readEnd = SIZE_T(slot.hasBoxes ? FSRDProbe::kReadbackBytes : FSRDProbe::kTexBytes);
        const D3D12_RANGE readRange = { 0, readEnd };
        void* mapped = nullptr;

        if (FAILED(slot.readback->Map(0, &readRange, &mapped)) || mapped == nullptr)
            return;

        const auto* bytes = static_cast<const uint8_t*>(mapped);
        auto& out = m_probeReadout;

        // A new window or new flags (A/B switches, debug view) restart the temporal average, so a
        // toggle never shows a blend of before and after. A new averaging length only changes the
        // weight of the next readouts.
        const bool resetAverage = !out.valid || out.centerX != slot.centerX || out.centerY != slot.centerY ||
                                  out.radius != slot.radius || out.convFlags != slot.convFlags ||
                                  out.hasBoxes != slot.hasBoxes;

        out.emaSamples = resetAverage ? 1 : out.emaSamples + 1;

        const uint32_t side = uint32_t(2 * slot.radius + 1);
        const uint32_t records = side * side;

        for (uint32_t s = 0; s < FSRD::Probe::kSlotCount; s++)
        {
            ReduceProbeWindow(out.slots[s], m_probeSlotMeanSq[s], records, resetAverage,
                              [&](uint32_t record, float value[4])
                              {
                                  const uint8_t* row = bytes + uint64_t(record) * FSRDProbe::kTexRowPitch;
                                  memcpy(value, row + s * 4 * sizeof(float), 4 * sizeof(float));
                              });
        }

        for (uint32_t b = 0; b < FSRD::Probe::BoxCount; b++)
        {
            const DXGI_FORMAT format = slot.boxFormats[b];
            const UINT bpp = FSRDProbe::BytesPerPixel(format);
            out.boxFormats[b] = uint32_t(format);

            if (!slot.hasBoxes || bpp == 0)
                continue;

            const uint8_t* boxBase = bytes + FSRDProbe::kBoxBase + FSRDProbe::kBoxStride * b;

            ReduceProbeWindow(out.boxes[b], m_probeBoxMeanSq[b], records, resetAverage,
                              [&](uint32_t pixel, float value[4])
                              {
                                  const uint32_t x = pixel % side;
                                  const uint32_t y = pixel / side;
                                  FSRDProbe::DecodeTexel(format, boxBase + y * FSRDProbe::kBoxRowPitch + x * bpp,
                                                         value);
                              });
        }

        const D3D12_RANGE noWrite = { 0, 0 };
        slot.readback->Unmap(0, &noWrite);

        out.valid = true;
        out.hasBoxes = slot.hasBoxes;
        out.frame = slot.frame;
        out.centerX = slot.centerX;
        out.centerY = slot.centerY;
        out.radius = slot.radius;
        out.renderWidth = slot.renderWidth;
        out.renderHeight = slot.renderHeight;
        out.convFlags = slot.convFlags;

        auto& diagnostics = FSRD::Diagnostics::Instance();
        std::scoped_lock lock(diagnostics.Mutex);
        diagnostics.Probe = out;
    }

    void Blit(ID3D12GraphicsCommandList* cmdList, ID3D12Resource* srcTex, ID3D12Resource* dstTex, XMFLOAT2 dstDim)
    {
        XMFLOAT2 srcDim = {};
        D3D12_RESOURCE_DESC srcDesc = srcTex->GetDesc();
        srcDim.x = (float) srcDesc.Width;
        srcDim.y = (float) srcDesc.Height;

        if (dstDim.x == 0 || dstDim.y == 0)
        {
            D3D12_RESOURCE_DESC dstDesc = dstTex->GetDesc();
            dstDim.x = (float) dstDesc.Width;
            dstDim.y = (float) dstDesc.Height;
        }

        if (!cmdList || dstDim.x == 0.0f)
            return;

        Composition::Input inputs = {};
        inputs.Resources.InDenoisedSignal1 = srcTex;

        const Composition::Constants constants = {
            .DstTexSize = { dstDim.x, dstDim.y, (1.0f / dstDim.x), (1.0f / dstDim.y) },
            .Flags = (UINT) CompFlags::RawSourceBlit | (UINT) CompFlags::ScaleSrc
        };

        // u1 (the SSS re-blur source, 28 Sep) gets a null descriptor: the blit never writes it
        std::array<ID3D12Resource*, Composition::kOutputCount> uavs { dstTex, nullptr };
        const std::span<const byte> cbData((const byte*) &constants, sizeof(constants));

        m_compShader.Dispatch(cmdList, cbData, inputs.AsArray, uavs, dstDim, false);
    }

    void SetDescResources(ffxDispatchDescHeader& signalHeader, ffxDispatchDescDenoiser& dispatchDesc)
    {
        auto& outResources = m_out.Resources;

        dispatchDesc.header = {
            .type = FFX_API_DISPATCH_DESC_TYPE_DENOISER,
            .pNext = &signalHeader // Link signal desc to main header
        };

        dispatchDesc.linearDepth =
            ffxApiGetResourceDX12(m_LinearDepth.Get(), FFX_API_RESOURCE_STATE_PIXEL_COMPUTE_READ);
        dispatchDesc.motionVectors =
            ffxApiGetResourceDX12(outResources.Motion.Get(), FFX_API_RESOURCE_STATE_PIXEL_COMPUTE_READ);
        dispatchDesc.normals =
            ffxApiGetResourceDX12(outResources.Normals.Get(), FFX_API_RESOURCE_STATE_PIXEL_COMPUTE_READ);
        dispatchDesc.specularAlbedo =
            ffxApiGetResourceDX12(outResources.SpecAlbedo.Get(), FFX_API_RESOURCE_STATE_PIXEL_COMPUTE_READ);
        dispatchDesc.diffuseAlbedo =
            ffxApiGetResourceDX12(outResources.DiffAlbedo.Get(), FFX_API_RESOURCE_STATE_PIXEL_COMPUTE_READ);
    }
};

// Public interface

FSRDPreprocessor_Dx12::FSRDPreprocessor_Dx12(std::string_view name, ID3D12Device* pDev)
    : m_impl(std::make_unique<Impl>()), m_InstanceName(name), m_IsInitialized(false)
{
    try
    {
        m_impl->m_pDev = pDev;
        m_impl->Initialize(GetAsByteSpan(FSRDFloorSeed_cso), GetAsByteSpan(FSRDFloor_cso),
                           GetAsByteSpan(FSRDInputConv_cso), GetAsByteSpan(FSRDOutputComp_cso),
                           GetAsByteSpan(FSRDLeak_cso), GetAsByteSpan(FSRDSplitFit_cso),
                           GetAsByteSpan(FSRDSssBlur_cso));
        m_IsInitialized = true;
    }
    catch (const std::exception& err)
    {
        LOG_ERROR("FSRD shaders failed to initialize. Details: {}", err.what());
    }
}

FSRDPreprocessor_Dx12::~FSRDPreprocessor_Dx12() = default;

bool FSRDPreprocessor_Dx12::IsInit() const { return m_IsInitialized; }

std::string_view FSRDPreprocessor_Dx12::GetName() const { return m_InstanceName; }

bool FSRDPreprocessor_Dx12::SetMaxRenderSize(UINT width, UINT height, bool albedo16)
{
    try
    {
        m_impl->SetMaxRenderSize(width, height, albedo16);
        return true;
    }
    catch (const std::exception& err)
    {
        LOG_ERROR("Failed to resize FSRD buffers. Details: {}", err.what());
    }

    return false;
}

bool FSRDPreprocessor_Dx12::DispatchConversion(ID3D12GraphicsCommandList* cmdList, const ConversionDesc& desc)
{
    try
    {
        m_impl->DispatchConversion(cmdList, desc);
        return true;
    }
    catch (const std::exception& err)
    {
        LOG_ERROR("FSRD input conversion failed. Details: {}", err.what());
    }

    return false;
}

void FSRDPreprocessor_Dx12::GetSignal(ffxDispatchDescDenoiserIndirectDiffuse& indirectDiffuseSignal,
                                      ffxDispatchDescDenoiserIndirectSpecular& indirectSpecularSignal,
                                      ffxDispatchDescDenoiser& dispatchDesc, bool declareActualStates) const
{
    // Transplant, 22 Sep: replaces the old Input1Signal/Input2Signals overloads. Denoiser 1.2 has
    // one small chainable struct per signal-flag bit instead of the old fused (Mode 1) / paired
    // (Mode 2) dispatch structs - transplant plan §6e/6f. No fusedAlbedo field anywhere in 1.2:
    // every signal demodulates against the shared specularAlbedo/diffuseAlbedo pair that
    // SetDescResources() already populates on the main dispatch desc below.
    auto& outResources = m_impl->m_out.Resources;

    // A/B, audit finding 5 (24 Sep). The signal resources were declared with ffxApiGetResourceDX12's
    // default state, COMPUTE_READ, for inputs and outputs alike. The inputs are actually in
    // NON_PIXEL | PIXEL_SHADER_RESOURCE (the packing dispatch's auto barrier) and the outputs in
    // UNORDERED_ACCESS (DispatchConversion's final barriers), which is what AMD's sample declares.
    // The SDK issues its own barriers from, and back to, the declared state.
    const uint32_t inputState =
        declareActualStates ? FFX_API_RESOURCE_STATE_PIXEL_COMPUTE_READ : FFX_API_RESOURCE_STATE_COMPUTE_READ;
    const uint32_t outputState =
        declareActualStates ? FFX_API_RESOURCE_STATE_UNORDERED_ACCESS : FFX_API_RESOURCE_STATE_COMPUTE_READ;

    indirectDiffuseSignal = { .header = { .type = FFX_API_DISPATCH_DESC_TYPE_DENOISER_INDIRECT_DIFFUSE },
                              .signal = { .input = ffxApiGetResourceDX12(outResources.DiffRadiance.Get(), inputState),
                                          .output =
                                              ffxApiGetResourceDX12(m_impl->m_outputBuffer2.Get(), outputState) } };

    indirectSpecularSignal = { .header = { .type = FFX_API_DISPATCH_DESC_TYPE_DENOISER_INDIRECT_SPECULAR,
                                           .pNext = &indirectDiffuseSignal.header },
                               .signal = { .input = ffxApiGetResourceDX12(outResources.SpecRadiance.Get(), inputState),
                                           .output =
                                               ffxApiGetResourceDX12(m_impl->m_outputBuffer1.Get(), outputState) } };

    // Chain: dispatchDesc -> indirectSpecularSignal -> indirectDiffuseSignal
    m_impl->SetDescResources(indirectSpecularSignal.header, dispatchDesc);
}

bool FSRDPreprocessor_Dx12::DispatchComposition(ID3D12GraphicsCommandList* cmdList, const CompositionDesc& desc)
{
    try
    {
        m_impl->DispatchComposition(cmdList, desc);
        return true;
    }
    catch (const std::exception& err)
    {
        LOG_ERROR("FSRD output composition failed. Details: {}", err.what());
    }

    return false;
}

ID3D12Resource* FSRDPreprocessor_Dx12::GetCompositionOutput() const { return m_impl->m_out.Resources.Motion.Get(); }

bool FSRDPreprocessor_Dx12::IsAlbedo16() const { return m_impl->m_albedo16; }

uint64_t FSRDPreprocessor_Dx12::GetGpuMemoryBytes() const
{
    if (!m_impl)
        return 0;

    const Impl& impl = *m_impl;
    uint64_t total = 0;

    // Width x height x texel size, buffers by width. Leaves out the 64 KB placement rounding, and doesn't go
    // through GetResourceAllocationInfo, which OptiScaler hooks.
    const auto Add = [&](ID3D12Resource* resource)
    {
        if (resource == nullptr)
            return;

        const D3D12_RESOURCE_DESC desc = resource->GetDesc();

        if (desc.Dimension == D3D12_RESOURCE_DIMENSION_BUFFER)
        {
            total += desc.Width;
            return;
        }

        uint64_t texelBytes = 0;

        switch (desc.Format)
        {
        case DXGI_FORMAT_R32G32B32A32_FLOAT:
            texelBytes = 16;
            break;
        case DXGI_FORMAT_R16G16B16A16_FLOAT:
            texelBytes = 8;
            break;
        case DXGI_FORMAT_R8G8B8A8_UNORM:
        case DXGI_FORMAT_R10G10B10A2_UNORM:
        case DXGI_FORMAT_R32_FLOAT:
            texelBytes = 4;
            break;
        default:
            texelBytes = 4;
            break;
        }

        total += desc.Width * desc.Height * desc.DepthOrArraySize * texelBytes;
    };

    for (ID3D12Resource* resource : impl.m_out.AsRawArray)
        Add(resource);

    Add(impl.m_LinearDepth.Get());
    Add(impl.m_PrevLinearDepth.Get());
    Add(impl.m_outputBuffer1.Get());
    Add(impl.m_outputBuffer2.Get());
    Add(impl.m_probeTex.Get());
    Add(impl.m_sssHistory[0].Get());
    Add(impl.m_sssHistory[1].Get());
    Add(impl.m_splitHistory[0].Get());
    Add(impl.m_splitHistory[1].Get());
    Add(impl.m_splitIntercept.Get());
    Add(impl.m_sssSource.Get());
    Add(impl.m_sssBlurH.Get());
    Add(impl.m_sssFitH.Get());
    Add(impl.m_sssFitTiles.Get());

    for (const auto& slot : impl.m_probeSlots)
        Add(slot.readback.Get());

    for (const auto& slot : impl.m_sssFitSlots)
        Add(slot.readback.Get());

    return total;
}

bool FSRDPreprocessor_Dx12::Blit(ID3D12GraphicsCommandList* cmdList, ID3D12Resource* srcTex, ID3D12Resource* dstTex,
                                 XMFLOAT2 dim) const

{
    try
    {
        m_impl->Blit(cmdList, srcTex, dstTex, dim);
        return true;
    }
    catch (const std::exception& err)
    {
        LOG_ERROR("FSRD blit failed. Details: {}", err.what());
    }

    return false;
}
