#include "pch.h"
#include <nvsdk_ngx_defs_dlssd.h>
#include <DirectXMath.h>
#include "NVNGX_Parameter.h"
#include "hooks/Streamline_Hooks.h"
#include "FSRDFeature_Dx12.h"
#include "shaders/fsrd_preprocess/FSRDPreprocessor_Dx12.h"
#include "shaders/fsrd_preprocess/FSRDDiagnostics.h"
#include "MathUtils.h"
#include "OptiTexts.h"
#include <atomic>
#include <format>

using namespace DirectX;
using namespace OptiMath;

using FSRDConvDesc = FSRDPreprocessor_Dx12::ConversionDesc;
using FSRDCompDesc = FSRDPreprocessor_Dx12::CompositionDesc;

/**
 * @brief Retrieves a matrix from the given parameter table. Matrices used by DLSS are in column-major
 * order, but DirectXMath operations assume row-major. Appropriate for passing to DirectX shaders, but not for
 * CPU-side operations without transposing.
 */
static bool TryGetNGXMatrixTranspose(const NVSDK_NGX_Parameter& ngxParams, const char* key, DirectX::XMMATRIX& outValue)
{
    float* pMat = nullptr;

    if (ngxParams.Get(key, (void**) &pMat) == NVSDK_NGX_Result_Success && pMat != nullptr)
    {
        memcpy_s(&outValue, sizeof(DirectX::XMMATRIX), pMat, sizeof(float) * 16);
        return true;
    }
    else
        return false;
}

/**
 * @brief Retrieves a matrix from the given parameter table and transposes it for CPU-side
 * operations with DirectXMath.
 */
static bool TryGetNGXMatrix(const NVSDK_NGX_Parameter& ngxParams, const char* key, DirectX::XMMATRIX& outValue)
{
    if (TryGetNGXMatrixTranspose(ngxParams, key, outValue))
    {
        outValue = XMMatrixTranspose(outValue);
        return true;
    }
    else
        return false;
}

template <typename T>
static bool TryGetLoggedResource(const NVSDK_NGX_Parameter& ngxParams, const char* key, T*& outValue)
{
    const bool success = TryGetNGXVoidPointer(ngxParams, key, outValue);

    if (success)
        LOG_DEBUG("{} exists..", key);
    else
        LOG_ERROR("{} is missing!!", key);

    return success;
}

/**
 * @brief Calculates vertical FOV according to: FOVv = 2 * arctan( 1 / M22 )
 * @param proj View to Clip / Perspective projection matrix
 * @return Vertical field of view in radians
 */
static float GetVertFovFromProjectionMatrixRad(const XMMATRIX& proj)
{
    return float(2.0 * (std::atan(1.0 / (double) proj.r[1].m128_f32[1])));
}

/**
 * @brief Calculates horizontal FOV according to: FOVh = 2 * arctan( 1 / M11 )
 * @param proj View to Clip / Perspective projection matrix
 * @return Horizontal field of view in radians
 */
static float GetHorzFovFromProjectionMatrixRad(const XMMATRIX& proj)
{
    return float(2.0 * (std::atan(1.0 / (double) proj.r[0].m128_f32[0])));
}

/**
 * @brief Calculates aspect ratio (width / height) as AR = M22 / M11
 * @param proj View to Clip / Perspective projection matrix
 * @return Aspect ratio as an fp32 decimal e.g. 1.778
 */
static float GetAspectRatioFromProjectionMatrix(const XMMATRIX& proj)
{
    return proj.r[1].m128_f32[1] / proj.r[0].m128_f32[0];
}

static XMFLOAT3 GetFloat3(const XMVECTOR& vec4)
{
    XMFLOAT3 vec3 = {};
    XMStoreFloat3(&vec3, vec4);
    return vec3;
}

static XMVECTOR GetColumn(const XMMATRIX& mat, int col)
{
    return { mat.r[0].m128_f32[col], mat.r[1].m128_f32[col], mat.r[2].m128_f32[col], 0 };
}

static void SetColumn(const XMVECTOR& vec, int col, XMMATRIX& mat)
{
    mat.r[0].m128_f32[col] = vec.m128_f32[0];
    mat.r[1].m128_f32[col] = vec.m128_f32[1];
    mat.r[2].m128_f32[col] = vec.m128_f32[2];
    mat.r[3].m128_f32[col] = vec.m128_f32[3];
}

static XMFLOAT3 GetFloat3Column(const XMMATRIX& mat, int col)
{
    return { mat.r[0].m128_f32[col], mat.r[1].m128_f32[col], mat.r[2].m128_f32[col] };
}

static FfxApiFloatCoords3D GetFloat3ColumnFFX(const XMMATRIX& mat, int col)
{
    return { mat.r[0].m128_f32[col], mat.r[1].m128_f32[col], mat.r[2].m128_f32[col] };
}

static FfxApiFloatCoords3D GetFloat3FFX(const XMVECTOR& vec4)
{
    FfxApiFloatCoords3D vec3 = {};
    XMStoreFloat3(reinterpret_cast<XMFLOAT3*>(&vec3), vec4);
    return vec3;
}

static const FfxApiFloatCoords3D& GetFloat3FFX(const XMFLOAT3& vec3)
{
    return *reinterpret_cast<const FfxApiFloatCoords3D*>(&vec3);
}

static ID3D12Resource* GetD3D12ResFromFFX(const FfxApiResource& resource)
{
    return static_cast<ID3D12Resource*>(resource.resource);
}

struct ViewPlanes
{
    float nearPlane;
    float farPlane;
    bool isInfinite;
    bool isRightHanded;
};

static ViewPlanes GetViewPlanes(const DirectX::XMMATRIX& projection, bool isInverted)
{
    ViewPlanes planes;
    // View to clip
    float A = projection.r[2].m128_f32[2];
    float B = projection.r[2].m128_f32[3];
    float W = projection.r[3].m128_f32[2];

    float infiniteCheckVal = isInverted ? A : (A - W);
    planes.isInfinite = std::abs(infiniteCheckVal) < 1e-6f;
    planes.isRightHanded = B < 0.0f;

    if (isInverted)
    {
        // Inverted: Near is at D=1, Far is at D=0
        // 1 = A/W + B/(n*W) -> n = B / (W - A)
        planes.nearPlane = std::abs(B / (W - A));

        // 0 = A/W + B/(f*W) -> f = -B / A
        planes.farPlane = std::abs(-B / A);
    }
    else
    {
        // Standard: Near is at D=0, Far is at D=1
        // 0 = A/W + B/(n*W) -> n = -B / A
        planes.nearPlane = std::abs(-B / A);

        // 1 = A/W + B/(f*W) -> f = B / (W - A)
        planes.farPlane = std::abs(B / (W - A));
    }

    return planes;
}

using FSRDConvFlags = FSRDPreprocessor_Dx12::ConvFlags;
using FSRDCompFlags = FSRDPreprocessor_Dx12::CompFlags;

enum class DebugModes : uint64_t
{
    None = 0,
    DenoiserBypass = 1,
    UpscalerBypass = 2,
    RawColor = 3,
    DlssBias = 4,
    DlssColorBeforeParticles = 5,
    DlssColorBeforeTransparency = 6,
    DlssTransparencyLayer = 7,
    FfxDebug = 8,

    ConversionDebug = FSRDConvFlags::Debug,
    ConversionDebugMask = FSRDConvFlags::DebugModeMask,

    // Not Mode-1-specific despite living on the same struct as the old fused-radiance concept:
    // this is "Debug flag set, no submode selected" (value == Debug itself), which the shader's
    // switch(GetDebugMode()) default case reads as "show combined denoised radiance" - demodColor
    // is populated by the split-signal (Mode 2) path too, just via a different formula. Confirmed
    // by reading FSRDInputConv.hlsl fresh this pass rather than trusting the earlier guess that
    // this was Mode-1-only (transplant plan §6f originally mislabeled it - corrected here).
    OutRadiance = FSRDConvFlags::DebugOutRadiance,

    InSpecHitDist = FSRDConvFlags::DebugInSpecHitDist,
    InMotion = FSRDConvFlags::DebugInMotion,
    InNormals = FSRDConvFlags::DebugInNormals,
    InRoughness = FSRDConvFlags::DebugInRoughness,
    InDiffAlbedo = FSRDConvFlags::DebugInDiffAlbedo,
    InSpecAlbedo = FSRDConvFlags::DebugInSpecAlbedo,

    // OutFusedAlbedo removed (transplant, 22 Sep): genuinely Mode-1-only - fusedAlbedo is never
    // assigned outside the removed Mode 1 branch in FSRDInputConv.hlsl, so this view would read
    // permanent black once Mode 1 is gone. Not repointed at anything; there's no equivalent
    // concept in the split-signal shape.
    OutLinearDepth = FSRDConvFlags::DebugOutLinearDepth,
    OutMotion = FSRDConvFlags::DebugOutMotion,
    OutNormals = FSRDConvFlags::DebugOutNormals,
    OutSpecAlbedo = FSRDConvFlags::DebugOutSpecAlbedo,
    OutDiffAlbedo = FSRDConvFlags::DebugOutDiffAlbedo,

    OutDepthDelta = FSRDConvFlags::DebugOutDepthDelta,
    NormDepth = FSRDConvFlags::DebugNormDepth,
    AlbedoError = FSRDConvFlags::DebugAlbedoError,

    FloorVariance = FSRDConvFlags::DebugFloorVariance,
    FloorColor = FSRDConvFlags::DebugFloorColor,

    InBiasMask = FSRDConvFlags::DebugInBiasMask,
    DemodGain = FSRDConvFlags::DebugDemodGain,
    HitDistGate = FSRDConvFlags::DebugHitDistGate,
    DenoiserFraction = FSRDConvFlags::DebugDenoiserFraction,
    SignalDelta = FSRDConvFlags::DebugSignalDelta,
    RoughnessProbe = FSRDConvFlags::DebugRoughnessProbe,
    EmissiveCheck = FSRDConvFlags::DebugEmissiveCheck,         // 24 Sep
    HitGateParts = FSRDConvFlags::DebugHitGateParts,           // 24 Sep
    MotionConsistency = FSRDConvFlags::DebugMotionConsistency, // 24 Sep

    CompositionDebugOffset = 16u,
    CompositionDebug = (uint64_t) FSRDCompFlags::Debug << CompositionDebugOffset,
    CompositionDebugMask = (uint64_t) FSRDCompFlags::DebugModeMask,

    Correlation = (uint64_t) FSRDCompFlags::DebugCorrelation << CompositionDebugOffset,
    SkipSignal = (uint64_t) FSRDCompFlags::DebugSkipSignal << CompositionDebugOffset,
    DenoiserOutput = (uint64_t) FSRDCompFlags::DebugDenoiserOutput << CompositionDebugOffset,
    Signal1 = (uint64_t) FSRDCompFlags::DebugSignal1 << CompositionDebugOffset,
    Signal2 = (uint64_t) FSRDCompFlags::DebugSignal2 << CompositionDebugOffset,
};

static FSRDConvFlags GetConvDebugFlags(DebugModes mode)
{
    uint32_t flags = uint32_t(mode);
    flags &= uint32_t(DebugModes::ConversionDebugMask);
    return FSRDConvFlags(flags);
}

static FSRDCompFlags GetCompDebugFlags(DebugModes mode)
{
    uint64_t flags = uint64_t(mode);
    flags >>= uint64_t(DebugModes::CompositionDebugOffset);
    flags &= uint64_t(DebugModes::CompositionDebugMask);
    return FSRDCompFlags(flags);
}

using ModeNamePair = std::pair<const char*, uint64_t>;
constexpr auto kDebugModes = std::to_array<ModeNamePair>({
    { "None", (uint64_t) DebugModes::None },
    { "DebugOverview", (uint64_t) DebugModes::FfxDebug },

    { "DenoiserBypass", (uint64_t) DebugModes::DenoiserBypass },
    { "UpscalerBypass", (uint64_t) DebugModes::UpscalerBypass },
    { "DenoiserOutput", (uint64_t) DebugModes::DenoiserOutput },
    { "SkipSignal", (uint64_t) DebugModes::SkipSignal },

    { "RawColor", (uint64_t) DebugModes::RawColor },
    { "DlssBias", (uint64_t) DebugModes::DlssBias },
    { "DlssColorBeforeParticles", (uint64_t) DebugModes::DlssColorBeforeParticles },
    { "DlssColorBeforeTransparency", (uint64_t) DebugModes::DlssColorBeforeTransparency },
    { "DlssTransparencyLayer", (uint64_t) DebugModes::DlssTransparencyLayer },

    { "InMotionVectors", (uint64_t) DebugModes::InMotion },
    { "InNormals", (uint64_t) DebugModes::InNormals },
    { "InRoughness", (uint64_t) DebugModes::InRoughness },
    { "InSpecHitDist", (uint64_t) DebugModes::InSpecHitDist },
    { "InDiffAlbedo", (uint64_t) DebugModes::InDiffAlbedo },
    { "InSpecAlbedo", (uint64_t) DebugModes::InSpecAlbedo },

    { "OutRadiance", (uint64_t) DebugModes::OutRadiance },
    { "OutLinearDepth", (uint64_t) DebugModes::OutLinearDepth },
    { "OutMotionVectors", (uint64_t) DebugModes::OutMotion },
    { "OutNormals", (uint64_t) DebugModes::OutNormals },
    { "OutSpecAlbedo", (uint64_t) DebugModes::OutSpecAlbedo },
    { "OutDiffAlbedo", (uint64_t) DebugModes::OutDiffAlbedo },
    { "OutDepthDelta", (uint64_t) DebugModes::OutDepthDelta },
    { "NormDepth", (uint64_t) DebugModes::NormDepth },

    { "AlbedoError", (uint64_t) DebugModes::AlbedoError },
    { "Correlation", (uint64_t) DebugModes::Correlation },

    { "FloorVariance", (uint64_t) DebugModes::FloorVariance },
    { "FloorColor", (uint64_t) DebugModes::FloorColor },

    { "InBiasMask", (uint64_t) DebugModes::InBiasMask },
    { "DemodGain", (uint64_t) DebugModes::DemodGain },
    { "HitDistGate", (uint64_t) DebugModes::HitDistGate },
    { "DenoiserFraction", (uint64_t) DebugModes::DenoiserFraction },
    { "SignalDelta", (uint64_t) DebugModes::SignalDelta },
    { "RoughnessProbe", (uint64_t) DebugModes::RoughnessProbe },
    { "EmissiveCheck", (uint64_t) DebugModes::EmissiveCheck },
    { "HitGateParts", (uint64_t) DebugModes::HitGateParts },
    { "MotionConsistency", (uint64_t) DebugModes::MotionConsistency },

    { "Signal1", (uint64_t) DebugModes::Signal1 },
    { "Signal2", (uint64_t) DebugModes::Signal2 },
});

bool FSRDFeatureDx12::s_isHWDepth = false;
bool FSRDFeatureDx12::s_isRoughnessPacked = false;

// FSR-RR runtime messages (24 Sep). The denoiser reports what it rejects - including everything
// "Enable Validation" (FFX_DENOISER_ENABLE_VALIDATION) checks - through the FFX message callback, and
// nothing was listening. Each distinct message is logged once and kept with a repeat count for the
// menu. After 100 distinct messages further ones are only counted, so a message that embeds a frame
// number cannot flood the log.
static void FsrdRuntimeMessage(uint32_t type, const wchar_t* message)
{
    static std::atomic<int> s_logged { 0 };
    const std::string text = message != nullptr ? wstring_to_string(std::wstring(message)) : std::string();

    if (!FSRD::Diagnostics::Instance().AddMessage(text, type))
        return;

    const int logged = ++s_logged;

    if (logged > 100)
    {
        if (logged == 101)
            LOG_WARN("FSR-RR runtime: over 100 distinct messages, further ones only counted in the menu");

        return;
    }

    if (type == FFX_API_MESSAGE_TYPE_ERROR)
        LOG_ERROR("FSR-RR runtime: {}", text);
    else
        LOG_WARN("FSR-RR runtime: {}", text);
}

bool FSRDFeatureDx12::DesiredAlbedo16()
{
    const auto& cfg = *Config::Instance();
    return cfg.FfxDenoiserAbActive.value_or_default() && cfg.FfxDenoiserAbAlbedo16.value_or_default();
}

FSRDFeatureDx12::FSRDFeatureDx12(uint32_t InHandleId, NVSDK_NGX_Parameter* InParameters)
    : FFXFeatureDx12(InHandleId, InParameters), IFeature(InHandleId, InParameters), _pDenoiserCtx(nullptr),
      _denoiserCtxDesc({}), _denoiserSettings({}), _convDesc({})
{
    // FFXFeatureDx12::SetParameters() is private, so its call in FFXFeatureDx12's own constructor
    // (chained through IFeature's mem-initializer) never runs for us: IFeature is a virtual base
    // shared by FFXFeature and IFeature_Dx12, so as the most-derived class WE are responsible for
    // initializing it directly, which means FFXFeatureDx12's own IFeature(..., SetParameters(...))
    // mem-initializer - argument expression included - is skipped entirely, not just its result
    // discarded. Replicate SetParameters()'s one side effect here instead.
    InParameters->Set("OptiScaler.SupportsUpscaleSize", true);

    _moduleLoaded = FfxApiProxy::IsDenoiserReady();

    if (_moduleLoaded)
        LOG_INFO("amd_fidelityfx_denoiser_dx12.dll methods loaded!");
    else
        LOG_ERROR("can't load amd_fidelityfx_denoiser_dx12.dll methods!");
}

FSRDFeatureDx12::~FSRDFeatureDx12()
{
    if (State::Instance().isShuttingDown)
        return;

    DestroyDenoiserContext();
}

bool FSRDFeatureDx12::InitFFX(const NVSDK_NGX_Parameter* InParameters)
{
    LOG_FUNC();

    // Init upscaler first - borrow some init boilerplate and some cfg
    if (FFXFeatureDx12::InitFFX(InParameters))
    {
        SetInit(false);

        LOG_DEBUG("FSR Ray Regeneration Initializing");
        _name = OptiTexts::FSR_RR_Name;

        if (int value; InParameters->Get(NVSDK_NGX_Parameter_Use_HW_Depth, &value) == NVSDK_NGX_Result_Success)
            s_isHWDepth = value == NVSDK_NGX_DLSS_Depth_Type_HW;

        if (int value; InParameters->Get(NVSDK_NGX_Parameter_DLSS_Roughness_Mode, &value) == NVSDK_NGX_Result_Success)
            s_isRoughnessPacked = value == NVSDK_NGX_DLSS_Roughness_Mode_Packed;

        LOG_INFO("DLSSD Flags HWDepth: {} - IsRoughnessPacked: {}", s_isHWDepth, s_isRoughnessPacked);

        if (!CreateDenoiserContext())
            return false;

        LOG_INFO("FSR Ray Regeneration Initialized");

        SetInit(true);
        return true;
    }

    return false;
}

uint32_t FSRDFeatureDx12::DesiredSignalFlags()
{
    // Transplant fix, 23 Sep. Phase 3 (plan §6e) mapped the shim's two demodulated channels to the INDIRECT
    // buckets on the reasoning that RR exists for ray-traced/GI content. But DLSS-RR's input is the whole
    // path-traced composite - direct lighting, contact shadows and all - and 1.1's Mode 2 took it as fused
    // (direct+indirect) specular/diffuse. 1.2 has no fused bucket. Worse, 1.2's indirect buckets read a ray
    // hit distance from alpha ("must be valid if the signal is active"); the shim has one for specular only
    // and writes 0 for diffuse, which tells the denoiser every diffuse ray hit something at zero distance.
    // Direct buckets ignore alpha. Default: diffuse -> DIRECT, specular -> INDIRECT (keeps hit-distance
    // reprojection for reflections). Both switchable from the menu for comparison.
    const auto& cfg = *Config::Instance();

    uint32_t flags = cfg.FfxDenoiserDiffuseAsDirect.value_or_default() ? FFX_DENOISER_SIGNAL_DIRECT_DIFFUSE
                                                                       : FFX_DENOISER_SIGNAL_INDIRECT_DIFFUSE;
    flags |= cfg.FfxDenoiserSpecularAsDirect.value_or_default() ? FFX_DENOISER_SIGNAL_DIRECT_SPECULAR
                                                                : FFX_DENOISER_SIGNAL_INDIRECT_SPECULAR;
    return flags;
}

uint32_t FSRDFeatureDx12::DesiredCreateFlags()
{
    const auto& cfg = *Config::Instance();
    uint32_t flags = 0;

    // FFX_DENOISER_ENABLE_DEBUGGING is a runtime create flag the SDK honours in any build (see CreateDenoiserContext).
    if (cfg.FfxDenoiserFsrDebugViews.value_or_default())
        flags |= FFX_DENOISER_ENABLE_DEBUGGING;

    // Plan §6a: bit 1 was ENABLE_DOMINANT_LIGHT in 1.1, ENABLE_VALIDATION in 1.2.
    if (cfg.FfxDenoiserValidation.value_or_default())
        flags |= FFX_DENOISER_ENABLE_VALIDATION;

    return flags;
}

bool FSRDFeatureDx12::CreateDenoiserContext()
{
    ScopedSkipSpoofingGlobal skipSpoofingGlobal {};
    auto& state = State::Instance();
    const auto& cfg = *Config::Instance();

    if (!QueryDenoiserVersions())
        return false;

    state.ffxDenoiserUpscalerVersion = Version();
    parse_version(state.ffxDenoiserVersionNames[cfg.FfxDenoiserIndex.value_or_default()]);

    ffxOverrideVersion vidOverride = { .header = { .type = FFX_API_DESC_TYPE_OVERRIDE_VERSION },
                                       .versionId =
                                           state.ffxDenoiserVersionIds[cfg.FfxDenoiserIndex.value_or_default()] };
    // Create context
    // Backend desc
    ffxCreateBackendDX12Desc backendDesc = 
    { 
        .header = 
        { 
            .type = FFX_API_CREATE_CONTEXT_DESC_TYPE_BACKEND_DX12,
            .pNext = &vidOverride.header // Chain override into backend desc
        },
        .device = Device
    };
    // Chain: ContextDesc -> BackendDesc -> OverrideVersion
    // Phase 3 note (transplant, 22 Sep): denoiser 1.2 has no .mode field - replaced by
    // signalFlags/checkerboardSignalFlags bitmasks (transplant plan §6e). Hardcoded to the split
    // indirect diffuse+specular signals per §6f (Mode 1 removed outright, Chris's call - the SDK
    // no longer offers a combined-signal option to select). No checkerboard reconstruction is
    // used anywhere in this shim.
    _denoiserCtxDesc = { .header = { .type = FFX_API_CREATE_CONTEXT_DESC_TYPE_DENOISER,
                                     // Chain backend desc into context desc
                                     .pNext = &backendDesc.header },
                         .version = FFX_DENOISER_VERSION,
                         .maxRenderSize = { RenderWidth(), RenderHeight() },
                         .signalFlags = DesiredSignalFlags(),
                         .checkerboardSignalFlags = 0,
                         .flags = DesiredCreateFlags() };

    LOG_INFO(
        "FSR-RR signals: diffuse -> {}, specular -> {}",
        (_denoiserCtxDesc.signalFlags & FFX_DENOISER_SIGNAL_DIRECT_DIFFUSE) ? "DIRECT_DIFFUSE" : "INDIRECT_DIFFUSE",
        (_denoiserCtxDesc.signalFlags & FFX_DENOISER_SIGNAL_DIRECT_SPECULAR) ? "DIRECT_SPECULAR" : "INDIRECT_SPECULAR");

    // FFX_DENOISER_ENABLE_DEBUGGING is a runtime create flag the SDK honours in any build.
    // It was previously wrapped in OptiScaler's own #ifdef _DEBUG, which kept FSR-RR's own
    // debug views - Virtual Hit Pos, View Centered Pos, Motion Vectors Z - unreachable in
    // Release. Virtual Hit Pos is the only view that shows the END of the
    // DLSS -> shim -> FSR-RR chain, so it is what settles hit-distance units and camera
    // parameters; InSpecHitDist only ever showed what the shim received.
    if (_denoiserCtxDesc.flags & FFX_DENOISER_ENABLE_DEBUGGING)
        LOG_INFO("FSR-RR denoiser debug views enabled (increases memory use)");

    // Transplant, 22 Sep (plan §6a/§7): bit 1 was FFX_DENOISER_ENABLE_DOMINANT_LIGHT in the 1.1 SDK
    // this fork shipped against; 1.2 repurposes it to FFX_DENOISER_ENABLE_VALIDATION -- exhaustive
    // internal validation of denoiser inputs, the single most valuable diagnostic this migration
    // unlocks. Exposed as its own menu toggle (menu_common.cpp) alongside FSR-RR Debug Views.
    if (_denoiserCtxDesc.flags & FFX_DENOISER_ENABLE_VALIDATION)
        LOG_INFO("FSR-RR denoiser validation enabled");

    // Create the denoiser context
    {
        ScopedSkipHeapCapture skipHeapCapture {};
        auto ret = FfxApiProxy::D3D12_CreateContext(&_pDenoiserCtx, &_denoiserCtxDesc.header, NULL);

        if (ret != FFX_API_RETURN_OK)
        {
            LOG_ERROR("_denoiserCtx error: {0}", FfxApiProxy::ReturnCodeToString(ret));
            return false;
        }

        _dispatchedOnContext = false;
    }

    // Runtime messages (24 Sep): route the denoiser's own errors and warnings, and what validation
    // finds, into the log and the menu. Same configure call FSR FG uses for its context.
    {
        ffxConfigureDescGlobalDebug1 debugDesc = {};
        debugDesc.header.type = FFX_API_CONFIGURE_DESC_TYPE_GLOBALDEBUG1;
        debugDesc.fpMessage = &FsrdRuntimeMessage;
        debugDesc.debugLevel = FFX_API_CONFIGURE_GLOBALDEBUG_LEVEL_VERBOSE;

        const ffxReturnCode_t code = FfxApiProxy::D3D12_Configure(&_pDenoiserCtx, &debugDesc.header);
        _messageCallbackOk = (code == FFX_API_RETURN_OK);

        if (_messageCallbackOk)
            LOG_INFO("FSR-RR runtime message callback installed");
        else
            LOG_WARN("FSR-RR runtime message callback not accepted: {}", FfxApiProxy::ReturnCodeToString(code));
    }

    // Query default settings
    SetDefaultConfiguration();

    // Create DLSS-RR to FSR-RR input converter
    FSRDConvShader = std::make_unique<FSRDPreprocessor_Dx12>("FSRD Converter", Device);

    if (!FSRDConvShader->IsInit())
        return false;

    // A/B, audit finding 3: the albedo textures' format is fixed at creation, so it is a create-time
    // option like the signal buckets (see UpdateSize).
    _albedo16AtCreate = DesiredAlbedo16();

    if (!FSRDConvShader->SetMaxRenderSize(_denoiserCtxDesc.maxRenderSize.width, _denoiserCtxDesc.maxRenderSize.height,
                                          _albedo16AtCreate))
        return false;

    if (_albedo16AtCreate)
        LOG_INFO("FSR-RR A/B: albedo textures created as RGBA16_FLOAT");

    return true;
}

bool FSRDFeatureDx12::QueryDenoiserVersions()
{
    ScopedSkipSpoofingGlobal skipSpoofingGlobal {};
    auto& state = State::Instance();

    // Get version count
    uint64_t versionCount = 0;
    ffxQueryDescGetVersions queryVersionsDesc = { .header = { .type = FFX_API_QUERY_DESC_TYPE_GET_VERSIONS },
                                                  .createDescType = FFX_API_EFFECT_ID_DENOISER,
                                                  .device = Device,
                                                  .outputCount = &versionCount };
    FfxApiProxy::D3D12_Query(nullptr, &queryVersionsDesc.header);

    state.ffxDenoiserVersionIds.resize(versionCount);
    state.ffxDenoiserVersionNames.resize(versionCount);

    state.ffxDenoiserDebugModes.clear();
    state.ffxDenoiserDebugModeNames.clear();

    for (const auto& mode : kDebugModes)
    {
        state.ffxDenoiserDebugModes.push_back(mode.second);
        state.ffxDenoiserDebugModeNames.emplace(mode.second, mode.first);
    }

    if (versionCount == 0)
    {
        LOG_ERROR("No FSR-RR denoisers were found.");
        return false;
    }
    else
        LOG_DEBUG("Found {} versions of FSR-RR", versionCount);

    LOG_DEBUG("Initialising FSR denoiser context");

    // Get version IDs
    queryVersionsDesc.versionIds = state.ffxDenoiserVersionIds.data();
    queryVersionsDesc.versionNames = state.ffxDenoiserVersionNames.data();
    FfxApiProxy::D3D12_Query(nullptr, &queryVersionsDesc.header);

    return true;
}

void FSRDFeatureDx12::DestroyDenoiserContext()
{
    if (_pDenoiserCtx != nullptr)
        FfxApiProxy::D3D12_DestroyContext(&_pDenoiserCtx, nullptr);
}

void FSRDFeatureDx12::UpdateSize()
{
    // FSR-RR doesn't currently have proper DRS support. The example implementation
    // reinits on resolution change as well.
    const bool sizeChanged = _denoiserCtxDesc.maxRenderSize.width != RenderWidth() ||
                             _denoiserCtxDesc.maxRenderSize.height != RenderHeight();

    // 23 Sep: create-time options (signal buckets, debug views, validation) take effect without a
    // resolution change. Not by destroying the context in place, though: that frees GPU resources
    // (context, converter buffers) that frames still in flight may be using. Hand it to OptiScaler's
    // own backend-change path instead, which recreates the whole feature and destroys the old one
    // on a 2 s delay (Util::DelayedDestroy) - the path already seen working for FSR-RR in the logs.
    const bool optionsChanged = _denoiserCtxDesc.signalFlags != DesiredSignalFlags() ||
                                _denoiserCtxDesc.flags != DesiredCreateFlags() ||
                                _albedo16AtCreate != DesiredAlbedo16();

    if (optionsChanged && !sizeChanged)
    {
        auto& state = State::Instance();

        if (!state.changeBackend[Handle()->Id])
        {
            LOG_INFO("FSR-RR create-time option changed, recreating the feature");
            state.newBackend = Upscaler::FSRD;
            state.changeBackend[Handle()->Id] = true;
        }
    }
    else if (sizeChanged)
    {
        LOG_INFO("Reinitializing FSR-RR for resolution change. "
                 "Previous: {} x {}, New: {} x {}",
                 _denoiserCtxDesc.maxRenderSize.width, _denoiserCtxDesc.maxRenderSize.height, RenderWidth(),
                 RenderHeight());

        DestroyDenoiserContext();
        CreateDenoiserContext();
    }
}

bool FSRDFeatureDx12::EvaluateInternal(ID3D12GraphicsCommandList* InCommandList, NVSDK_NGX_Parameter* InParameters)
{
    LOG_FUNC();

    if (!IsInited())
        return false;

    auto& state = State::Instance();
    auto& cfg = *Config::Instance();
    const auto& inParams = *InParameters;

    // 24 Sep: one step per game frame, whatever path this frame takes, so a frame on which the denoiser
    // doesn't run (bypass views, a failed conversion) still shows up as a jump and resets its history.
    _denoiserFrameIndex++;

    UpdateSize();

    const auto dbgMode = static_cast<DebugModes>(cfg.FfxDenoiserDebugMode.value_or_default());
    const bool isDebugVis = (uint32_t) dbgMode & (uint32_t) DebugModes::ConversionDebug;
    const bool isDebugComp = ((uint64_t) dbgMode & (uint64_t) DebugModes::CompositionDebug);
    const bool isFfxDebug = dbgMode == DebugModes::FfxDebug;
    const bool hasAnyDebug = (dbgMode != DebugModes::None);

    // Denoise is bypassed if we are debugging something OTHER than the final outputs
    const bool isDenoiseBypassed = !isFfxDebug && !isDebugComp && hasAnyDebug &&
                                   dbgMode != DebugModes::DenoiserOutput && dbgMode != DebugModes::UpscalerBypass;

    // Upscale is bypassed if we are in a debug mode that isn't the DenoiserBypass (final raw)
    const bool isUpscaleBypassed = hasAnyDebug && dbgMode != DebugModes::DenoiserBypass;

    // Phase 2 note (transplant, 22 Sep): the old RCAS->IsInit()/OutputScaler->IsInit() guards that
    // used to live here are gone - IFeature_Dx12::Evaluate() (the fixed entry point that calls this
    // EvaluateInternal() override) now runs those same IsInit() checks itself, generically, before
    // deciding whether to build its RCAS/OutputScaling/Magnifier post-process pipeline. Nothing to
    // replicate here.

    _isInReset = false;

    if (uint32_t value = 0; inParams.Get(NVSDK_NGX_Parameter_Reset, &value) == NVSDK_NGX_Result_Success)
        _isInReset = value > 0;

    // Troubleshooting snapshot for the menu (24 Sep). Describes the previous frame's dispatch, which
    // is close enough for a panel refreshed a few times a second.
    if (_frameCount % 15 == 0)
        PublishDiagnostics(inParams, isDenoiseBypassed, isUpscaleBypassed);

    _lastDispatchCode = -1;

    // Denoiser start
    ffxDispatchDescDenoiserIndirectDiffuse indirectDiffuseSignal = {};
    ffxDispatchDescDenoiserIndirectSpecular indirectSpecularSignal = {};
    ffxDispatchDescDenoiser denoiserDesc = {};
    bool isDenoiserReady = false;

    // Pull configuration and input buffers for DLSS-RR from the param table, convert and
    // repack input buffers into intermediate FSR-RR input buffers, and configure descriptors.
    // Phase 3 note (transplant, 22 Sep): no more Mode 1 / Mode 2 branch here - denoiser 1.2 only
    // has the split signal shape (transplant plan §6e/6f).
    if (!PrepareDenoiserInput(InCommandList, *InParameters, denoiserDesc, indirectDiffuseSignal,
                              indirectSpecularSignal))
        return false;

    // Dispatch denoiser
    if (!isDenoiseBypassed)
    {
        ffxDispatchDescDenoiserDebugView dispatchDebugView = {};

        // 23 Sep: only chain the SDK debug view into a context that was created with
        // FFX_DENOISER_ENABLE_DEBUGGING. Selecting the FfxDebug mode without the "FSR-RR Debug Views"
        // checkbox (or before the context had been recreated with it) sent a debug-view desc to a
        // non-debug context - the likely cause of the crash reported on 17c0ee85.
        const bool ctxHasDebugging = (_denoiserCtxDesc.flags & FFX_DENOISER_ENABLE_DEBUGGING) != 0;

        if (isFfxDebug && !ctxHasDebugging && !_warnedFfxDebugWithoutFlag)
        {
            _warnedFfxDebugWithoutFlag = true;
            LOG_WARN("FfxDebug view selected but the denoiser context has no debugging enabled - "
                     "tick \"FSR-RR Debug Views\" (skipping the debug view until then)");
        }

        if (isFfxDebug && ctxHasDebugging)
        {
            // Append rather than overwrite. The previous line clobbered whatever was already
            // chained at that node; harmless while this path never executed, not harmless now.
            ffxDispatchDescHeader* tail = denoiserDesc.header.pNext;

            while (tail != nullptr && tail->pNext != nullptr)
                tail = tail->pNext;

            if (tail != nullptr)
                tail->pNext = &dispatchDebugView.header;

            ID3D12Resource* dstTex;
            TryGetLoggedResource(inParams, NVSDK_NGX_Parameter_Output, dstTex);

            int debugViewport = cfg.FfxDenoiserFsrDebugViewport.value_or_default();

            if (debugViewport >= int(FFX_API_DENOISER_DEBUG_VIEW_MAX_VIEWPORTS))
                debugViewport = int(FFX_API_DENOISER_DEBUG_VIEW_MAX_VIEWPORTS) - 1;

            dispatchDebugView = { // Word-order swap from 1.1's FFX_API_DISPATCH_DESC_DEBUG_VIEW_TYPE_DENOISER, not a
                                  // semantic change (transplant plan §6i).
                                  .header = { .type = FFX_API_DISPATCH_DESC_TYPE_DENOISER_DEBUG_VIEW },
                                  .output = ffxApiGetResourceDX12(dstTex, FFX_API_RESOURCE_STATE_UNORDERED_ACCESS),
                                  .outputSize = { TargetWidth(), TargetHeight() },
                                  // The named SDK views are viewports, not modes: the enum is only
                                  // OVERVIEW / FULLSCREEN_VIEWPORT. -1 keeps the tiled overview, which shows
                                  // every viewport at once; 0..MAX-1 blows one up full screen.
                                  .mode = (debugViewport < 0)
                                              ? uint32_t(FFX_API_DENOISER_DEBUG_VIEW_MODE_OVERVIEW)
                                              : uint32_t(FFX_API_DENOISER_DEBUG_VIEW_MODE_FULLSCREEN_VIEWPORT),
                                  .viewportIndex = uint32_t(debugViewport < 0 ? 0 : debugViewport)
            };
        }

        // A/B (24 Sep): the transplant's frame index, which advanced twice per frame and made 1.2 reset its
        // history on every dispatch. For before/after comparison only.
        if (cfg.FfxDenoiserAbActive.value_or_default() && cfg.FfxDenoiserAbFrameIndexDoubled.value_or_default())
            denoiserDesc.frameIndex = (uint32_t) _frameCount;

        isDenoiserReady = DispatchDenoiser(InCommandList, denoiserDesc);

        if (!isDenoiserReady)
            return false;

        // Compose denoised signals
        FSRDCompDesc compDesc = { .DstTexSize = _convDesc.RenderSize,
                                  .CorrelationBias = cfg.FfxDenoiserCorrelationBias.value_or_default(),
                                  .Flags = (uint32_t) GetCompDebugFlags(dbgMode) };

        TryGetNGXVoidPointer(inParams, NVSDK_NGX_Parameter_Color, compDesc.InRawColor);
        TryGetNGXVoidPointer(inParams, NVSDK_NGX_Parameter_DLSSD_ColorBeforeParticles, compDesc.InColorBeforeParticles);

        if (!isFfxDebug && !FSRDConvShader->DispatchComposition(InCommandList, compDesc))
            return false;

        isDenoiserReady = true;
    }

    // Upscaler start
    if (!isUpscaleBypassed)
    {
        // Phase 2 note (transplant, 22 Sep): there's no separate ffxDispatchDescUpscale left to
        // build and hand off - FFXFeatureDx12::EvaluateInternal() builds its own internally from
        // InParameters and dispatches in one call (it also derives cameraFovAngleVertical and
        // frameTimeDelta itself from InParameters/config, so the old cross-assignment from
        // denoiserDesc doesn't need porting either - denoiserDesc no longer carries those fields in
        // 1.2 regardless, see the transplant plan §6g). Steer it onto the denoiser's composited
        // output the same way IFeature_Dx12::Evaluate() itself steers NVSDK_NGX_Parameter_Output
        // for its own post-process pipeline: substitute the parameter, call through, restore it.
        // Resource barriers, RCAS/OutputScaler/Magnifier post-process and their IsInit() guards are
        // all handled automatically now - the barriers inside this call, the rest by
        // IFeature_Dx12::Evaluate() once it gets control back, gated on OUR return value below.
        //
        // KNOWN GAP, not fixed here: the old code gated its explicit PostProcess(...) call on the
        // LOCAL isUpscalerReady result, but this function's own return statement (unchanged, a few
        // lines down) is `isDenoiserReady || isDenoiseBypassed` - it never reflected upscaler
        // success even in the original. IFeature_Dx12::Evaluate() now runs its post-process pipeline
        // based on THAT return value, so a failed upscaler dispatch here no longer skips
        // post-processing the way it used to (denoiser-only failures are unaffected: this function
        // already returns false directly in that case, well before this point). Narrow edge case -
        // flagged rather than silently changed, since fixing it means changing what this function's
        // return communicates to its own caller, which is its own decision to make deliberately.
        if (isDenoiserReady)
            InParameters->Set(NVSDK_NGX_Parameter_Color, FSRDConvShader->GetCompositionOutput());

        FFXFeatureDx12::EvaluateInternal(InCommandList, InParameters);

        if (isDenoiserReady)
            InParameters->Set(NVSDK_NGX_Parameter_Color, _convDesc.Resources.InColor);
    }
    else if (!isFfxDebug) // Debug visualization
    {
        ID3D12Resource* srcTex = nullptr;

        if (dbgMode == DebugModes::DlssColorBeforeParticles)
            TryGetNGXVoidPointer(inParams, NVSDK_NGX_Parameter_DLSSD_ColorBeforeParticles, srcTex);
        else if (dbgMode == DebugModes::DlssColorBeforeTransparency)
            TryGetNGXVoidPointer(inParams, NVSDK_NGX_Parameter_DLSSD_ColorBeforeTransparency, srcTex);
        else if (dbgMode == DebugModes::DlssTransparencyLayer)
            TryGetNGXVoidPointer(inParams, NVSDK_NGX_Parameter_DLSS_TransparencyLayer, srcTex);
        else if (dbgMode == DebugModes::DlssBias)
            TryGetNGXVoidPointer(inParams, NVSDK_NGX_Parameter_DLSS_Input_Bias_Current_Color_Mask, srcTex);
        else if (dbgMode == DebugModes::RawColor)
            TryGetNGXVoidPointer(inParams, NVSDK_NGX_Parameter_Color, srcTex);
        else if (isDebugVis)
            srcTex = GetD3D12ResFromFFX(indirectSpecularSignal.signal.input);
        else
            srcTex = FSRDConvShader->GetCompositionOutput();

        ID3D12Resource* dstTex;

        if (!srcTex || !TryGetLoggedResource(inParams, NVSDK_NGX_Parameter_Output, dstTex))
        {
            _frameCount++;
            return true;
        }

        FSRDConvShader->Blit(InCommandList, srcTex, dstTex);
    }

    _frameCount++;
    return isDenoiserReady || isDenoiseBypassed;
}

bool FSRDFeatureDx12::PrepareDenoiserInput(ID3D12GraphicsCommandList* InCommandList,
                                           const NVSDK_NGX_Parameter& inParams, ffxDispatchDescDenoiser& dispatchDesc,
                                           ffxDispatchDescDenoiserIndirectDiffuse& indirectDiffuseSignal,
                                           ffxDispatchDescDenoiserIndirectSpecular& indirectSpecularSignal)
{
    // cfg's only use here was the now-removed FSR-frame-time-delta resolution (see the comment
    // below dispatchDesc's initializer) - dropped rather than left dangling. slData was already
    // unused in this function before this pass; left as-is.
    const auto& slData = State::Instance().slLastConstants;

    // Gather DLSS-RR input buffers for conversion and repacking for FSR-RR
    if (!PrepareDenoiseConvInput(inParams))
        return false;

    if (!ConvertDenoiserBuffers(InCommandList))
        return false;

    // Camera position, from viewMatrix^-1's translation column
    const XMFLOAT3 camPos = GetFloat3Column(_invViewMatrix, 3);

    // Pack dispatch configuration
    //
    // Phase 3 note (transplant, 22 Sep): 1.1's decomposed cameraRight/cameraUp/cameraForward +
    // cameraAspectRatio/cameraNear/cameraFar/cameraFovAngleVertical fields don't exist in 1.2 -
    // replaced wholesale by .view/.projection matrices set below, plus .linearDepthBounds (new -
    // see the comment on it). deltaTime doesn't exist either, with no successor field anywhere on
    // this struct in 1.2 - the old FSR-frame-time-delta resolution that used to feed it is removed
    // below rather than ported, since it would have nothing left to write into (confirmed against
    // the real 1.2 header, not assumed). Transplant plan §6g/§6h.
    dispatchDesc = { .commandList = InCommandList,
                     .motionVectorScale = { 1.0f, 1.0f, 1.0f },
                     // Camera movement since last frame (PreviousPosition - CurrentPosition)
                     .cameraPositionDelta = { (_lastCamPos.x - camPos.x), (_lastCamPos.y - camPos.y),
                                              (_lastCamPos.z - camPos.z) },
                     // Absolute linear depth bounds - passthrough is enabled outside this range. Left
                     // zero-initialized, this silently disables the denoiser for the entire frame with no
                     // error (transplant plan §6h - the top silent hazard in this migration). Near/far are
                     // already in absolute linear-depth units (GetViewPlanes(), computed every frame in
                     // ConvertDenoiserBuffers() just above).
                     .linearDepthBounds = { .min = _convDesc.NearPlane, .max = _convDesc.FarPlane },
                     .renderSize = { RenderWidth(), RenderHeight() },
                     // 24 Sep: was (uint32_t) _frameCount, which advances twice per frame (see
                     // _denoiserFrameIndex): 'Frame index jump detected. Resetting...' on every
                     // dispatch, i.e. no temporal accumulation at all.
                     .frameIndex = _denoiserFrameIndex,
                     .flags = FFX_DENOISER_DISPATCH_NON_GAMMA_ALBEDO };

    // World-to-view and (unjittered) view-to-projection matrices.
    //
    // Transplant fix, 23 Sep: these MUST be transposed. _viewMatrix/_projMatrix are held in the
    // column-vector convention (row-major storage): TryGetNGXMatrix() transposes the raw NGX data
    // into it, GetFloat3Column(_invViewMatrix, 3) reads the camera position out of column 3,
    // the Streamline fallback builds _invViewMatrix from basis columns, and GetViewPlanes() reads
    // the projection's z row as [.., A, B] with W in row 3. The fork's working 1.1 build relied on
    // exactly that convention (camera basis vectors taken from columns). The 1.2 header wants
    // row-major storage with ROW vectors (its own example puts translation in row 3) and says
    // "row-major with column vectors: requires transpose". Phase 3 passed them untransposed on the
    // mistaken belief they were already DirectXMath row-vector matrices, which handed the denoiser
    // an inverted camera rotation, a misplaced translation and a scrambled projection.
    //
    // Still OPEN (transplant plan §6b/§6g): whether 1.2's *signed* linear depth needs a sign flip
    // for a right-handed view space. The shim writes positive depth, as 1.1 wanted. See the camera
    // convention log line below and the Motion Vectors Z / View Centered Pos debug views.
    XMMATRIX view = _viewMatrix;
    XMMATRIX proj = _projMatrix;

    // Diagnostic toggle, default off: z-flip S = diag(1,1,-1,1). In the column-vector convention
    // above, view' = S * view negates view-space z, and proj' = proj * S compensates so that
    // proj' * view' == proj * view (clip space, and therefore the image, is unchanged). The shim's
    // positive linear depth and its depth delta then match view' for a right-handed game.
    if (Config::Instance()->FfxDenoiserFlipViewZ.value_or_default())
    {
        const XMMATRIX flipZ = XMMatrixScaling(1.0f, 1.0f, -1.0f);
        view = XMMatrixMultiply(flipZ, view);
        proj = XMMatrixMultiply(proj, flipZ);
    }

    XMStoreFloat4x4(reinterpret_cast<XMFLOAT4X4*>(&dispatchDesc.view), XMMatrixTranspose(view));
    XMStoreFloat4x4(reinterpret_cast<XMFLOAT4X4*>(&dispatchDesc.projection), XMMatrixTranspose(proj));

    // Populate resources and link signal headers: dispatchDesc -> indirectSpecularSignal ->
    // indirectDiffuseSignal (see FSRDPreprocessor_Dx12::GetSignal).
    //
    // A/B, audit finding 5 (24 Sep): declare the signal resources in the states they are actually in.
    const auto& abCfg = *Config::Instance();
    _lastDeclaredStates =
        abCfg.FfxDenoiserAbActive.value_or_default() && abCfg.FfxDenoiserAbDeclaredStates.value_or_default();
    FSRDConvShader->GetSignal(indirectDiffuseSignal, indirectSpecularSignal, dispatchDesc, _lastDeclaredStates);

    // Retag to the buckets the context was created with (23 Sep, see DesiredSignalFlags). The direct and
    // indirect per-signal structs are layout-identical ({ header, FfxApiDenoiserSignal }), only the type differs.
    static_assert(sizeof(ffxDispatchDescDenoiserDirectDiffuse) == sizeof(ffxDispatchDescDenoiserIndirectDiffuse));
    static_assert(sizeof(ffxDispatchDescDenoiserDirectSpecular) == sizeof(ffxDispatchDescDenoiserIndirectSpecular));
    static_assert(offsetof(ffxDispatchDescDenoiserDirectDiffuse, signal) ==
                  offsetof(ffxDispatchDescDenoiserIndirectDiffuse, signal));
    static_assert(offsetof(ffxDispatchDescDenoiserDirectSpecular, signal) ==
                  offsetof(ffxDispatchDescDenoiserIndirectSpecular, signal));

    if (_denoiserCtxDesc.signalFlags & FFX_DENOISER_SIGNAL_DIRECT_DIFFUSE)
        indirectDiffuseSignal.header.type = FFX_API_DISPATCH_DESC_TYPE_DENOISER_DIRECT_DIFFUSE;
    if (_denoiserCtxDesc.signalFlags & FFX_DENOISER_SIGNAL_DIRECT_SPECULAR)
        indirectSpecularSignal.header.type = FFX_API_DISPATCH_DESC_TYPE_DENOISER_DIRECT_SPECULAR;

    if (_isInReset)
        dispatchDesc.flags |= FFX_DENOISER_DISPATCH_RESET;

    // Camera movement over the last frame, for the diagnostics panel (24 Sep): distance moved and the
    // angle between the previous and current forward axes (row 2 of world-to-view).
    {
        const XMVECTOR delta = XMVectorSet(dispatchDesc.cameraPositionDelta.x, dispatchDesc.cameraPositionDelta.y,
                                           dispatchDesc.cameraPositionDelta.z, 0.0f);
        _lastCamMove = XMVectorGetX(XMVector3Length(delta));

        const XMVECTOR fwd = XMVector3Normalize(_viewMatrix.r[2]);
        const XMVECTOR prevFwd = XMVector3Normalize(_prevViewMatrix.r[2]);
        const float cosTurn = std::clamp(XMVectorGetX(XMVector3Dot(fwd, prevFwd)), -1.0f, 1.0f);
        _lastCamTurnDeg = std::acos(cosTurn) * (180.0f / 3.14159265f);
    }

    // Update camera position for next frame
    _lastCamPos = camPos;

    // Motion Vector Scaling
    // Scaling must result in UV space vectors, unlike FSR/DLSS pixel space vectors
    float MVScaleX = 1.0f, MVScaleY = 1.0f;

    if (inParams.Get(NVSDK_NGX_Parameter_MV_Scale_X, &MVScaleX) == NVSDK_NGX_Result_Success &&
        inParams.Get(NVSDK_NGX_Parameter_MV_Scale_Y, &MVScaleY) == NVSDK_NGX_Result_Success)
    {
        dispatchDesc.motionVectorScale.x = MVScaleX / dispatchDesc.renderSize.width;
        dispatchDesc.motionVectorScale.y = MVScaleY / dispatchDesc.renderSize.height;
    }

    float jitterX = 0.0f, jitterY = 0.0f;
    inParams.Get(NVSDK_NGX_Parameter_Jitter_Offset_X, &jitterX);
    inParams.Get(NVSDK_NGX_Parameter_Jitter_Offset_Y, &jitterY);

    // Convert from pixel to NDC jitter. Inline AMD docs incorrectly claim this is "expressed in screen pixels".
    // The RR 1.0 and 1.1 reference implementations use NDC jitter. Fucking clowns.
    dispatchDesc.jitterOffsets.x = 2.0f * (jitterX / (float) RenderWidth());
    dispatchDesc.jitterOffsets.y = -2.0f * (jitterY / (float) RenderHeight());

    LOG_DEBUG("Jitter NDC [{:.6f}, {:.6f}]", dispatchDesc.jitterOffsets.x, dispatchDesc.jitterOffsets.y);

    return true;
}

bool FSRDFeatureDx12::PrepareDenoiseConvInput(const NVSDK_NGX_Parameter& inParams)
{
    const auto& slData = State::Instance().slLastConstants;

    // Gather DLSS-RR input buffers for conversion and repacking for FSR-RR
    bool isReady = true;

    // Standard TSR buffers
    if (!TryGetLoggedResource(inParams, NVSDK_NGX_Parameter_Color, _convDesc.Resources.InColor))
        isReady = false;
    if (!TryGetLoggedResource(inParams, NVSDK_NGX_Parameter_MotionVectors, _convDesc.Resources.InMotionVectors))
        isReady = false;
    if (!TryGetLoggedResource(inParams, NVSDK_NGX_Parameter_Depth, _convDesc.Resources.InDepth) && LowResMV())
        isReady = false;

    // DLSSD-specific buffers
    if (!TryGetLoggedResource(inParams, NVSDK_NGX_Parameter_GBuffer_Normals, _convDesc.Resources.InNormals))
        isReady = false;

    // If roughness is not packed into normals, then this texture is mandatory.
    // This value should be available in one of these two buffers in any DLSS-RR implementation.
    if (!s_isRoughnessPacked &&
        !TryGetLoggedResource(inParams, NVSDK_NGX_Parameter_GBuffer_Roughness, _convDesc.Resources.InRoughness))
    {
        LOG_WARN("Expected unpacked roughness buffer from DLSS-RR. Defaulting to packed roughness...");
        s_isRoughnessPacked = true;
    }

    if (!TryGetLoggedResource(inParams, NVSDK_NGX_Parameter_DiffuseAlbedo, _convDesc.Resources.InDiffAlbedo))
        isReady = false;

    if (!TryGetLoggedResource(inParams, NVSDK_NGX_Parameter_SpecularAlbedo, _convDesc.Resources.InSpecAlbedo))
        isReady = false;

    TryGetNGXVoidPointer(inParams, NVSDK_NGX_Parameter_DLSS_Input_Bias_Current_Color_Mask,
                         _convDesc.Resources.InBiasMask);

    // Optional. Specular hit distance can be used with mode-2 denoising to track movement inside reflections,
    // in addition to primary motion tracking for the surface and camera.
    TryGetLoggedResource(inParams, NVSDK_NGX_Parameter_DLSSD_SpecularHitDistance, _convDesc.Resources.InSpecHitDist);

    // 23 Sep: not consumed yet. Reported once so we know whether INDIRECT_DIFFUSE could be fed a real hit distance.
    if (!_loggedDiffuseHitDist)
    {
        _loggedDiffuseHitDist = true;
        ID3D12Resource* diffHitDist = nullptr;
        const bool hasDiffHitDist =
            TryGetNGXVoidPointer(inParams, NVSDK_NGX_Parameter_DLSSD_DiffuseHitDistance, diffHitDist) && diffHitDist;
        LOG_INFO("Game supplies DLSSD.DiffuseHitDistance: {}", hasDiffHitDist ? "yes" : "no");
    }

    // Get DLSSD matrices and derive related values
    // World to view/camera space (V)
    _prevViewMatrix = _viewMatrix;
    _viewMatrix = {};

    if (!TryGetNGXMatrix(inParams, NVSDK_NGX_Parameter_DLSS_WORLD_TO_VIEW_MATRIX, _viewMatrix))
    {
        if (StreamlineHooks::isSetConstantsHooked())
        {
            SetColumn(XMLoadFloat3((XMFLOAT3*) &slData.cameraRight), 0, _invViewMatrix);
            SetColumn(XMLoadFloat3((XMFLOAT3*) &slData.cameraUp), 1, _invViewMatrix);
            SetColumn(XMLoadFloat3((XMFLOAT3*) &slData.cameraFwd), 2, _invViewMatrix);
            SetColumn(XMLoadFloat3((XMFLOAT3*) &slData.cameraPos), 3, _invViewMatrix);
            _invViewMatrix.r[3].m128_f32[3] = 1.0f;

            _viewMatrix = XMMatrixInverse(nullptr, _invViewMatrix);
        }
        else
        {
            LOG_ERROR("View matrix missing! Denoiser not ready.");
            isReady = false;
        }
    }
    else
    {
        // Camera rotation and position
        _invViewMatrix = XMMatrixInverse(nullptr, _viewMatrix);
    }

    // Perspective projection matrix (P)
    _projMatrix = {};

    if (!TryGetNGXMatrix(inParams, NVSDK_NGX_Parameter_DLSS_VIEW_TO_CLIP_MATRIX, _projMatrix))
    {
        if (StreamlineHooks::isSetConstantsHooked())
        {
            if (slData.cameraFOV != sl::INVALID_FLOAT && slData.cameraNear != slData.cameraFar)
            {
                // These measurements are supposed to be in radians, but some titles supply degrees.
                // Valid FOV in radians never exceeds PI. Realistic FOV in degrees is basically never in the single
                // digits.
                const float fov = (slData.cameraFOV < 4.0f) ? slData.cameraFOV : GetRadiansFromDeg(slData.cameraFOV);
                const float nearPlane = slData.cameraNear;
                const float farPlane = slData.cameraFar;
                _isRightHanded = slData.cameraViewToClip[2].w < 0.0f;

                // Actual SL view to clip matrix isn't reliable. This is harder to fuck up.
                if (_isRightHanded)
                    _projMatrix = XMMatrixPerspectiveFovRH(fov, slData.cameraAspectRatio, nearPlane, farPlane);
                else
                    _projMatrix = XMMatrixPerspectiveFovLH(fov, slData.cameraAspectRatio, nearPlane, farPlane);

                _projMatrix = XMMatrixTranspose(_projMatrix);
            }
        }
        else
        {
            LOG_ERROR("Projection matrix missing! Denoiser not ready.");
            isReady = false;
        }
    }

    return isReady;
}

bool FSRDFeatureDx12::ConvertDenoiserBuffers(ID3D12GraphicsCommandList* InCommandList)
{
    const uint32_t dbgMode = (uint32_t) Config::Instance()->FfxDenoiserDebugMode.value_or_default();
    const auto& cfg = *Config::Instance();
    const auto& slData = State::Instance().slLastConstants;

    // Prepare input converter
    _convDesc.RenderSize = { (float) RenderWidth(), (float) RenderHeight(), 1.0f / (float) RenderWidth(),
                             1.0f / (float) RenderHeight() };
    _convDesc.Flags = (uint32_t) FSRDConvFlags::NonGammaAlbedo | (dbgMode & (uint32_t) FSRDConvFlags::DebugModeMask);
    _convDesc.FloorIsolation = cfg.FfxDenoiserFloorIsolation.value_or_default();
    _convDesc.BiasMaskStrength = cfg.FfxDenoiserBiasMaskStrength.value_or_default();
    _convDesc.FloorDetailBoost = cfg.FfxDenoiserFloorDetailBoost.value_or_default();
    _convDesc.FloorNormalSharpness = cfg.FfxDenoiserFloorNormalSharpness.value_or_default();
    _convDesc.FloorAlbedoGuide = cfg.FfxDenoiserFloorAlbedoGuide.value_or_default();
    _convDesc.FloorLumSymmetry = cfg.FfxDenoiserFloorLumSymmetry.value_or_default();
    _convDesc.FloorGrazingSharpness = cfg.FfxDenoiserFloorGrazingSharpness.value_or_default();
    _convDesc.FloorSoftMin = cfg.FfxDenoiserFloorSoftMin.value_or_default();
    _convDesc.RoughnessExponent = cfg.FfxDenoiserRoughnessExponent.value_or_default();
    _convDesc.HitDistScale = cfg.FfxDenoiserHitDistScale.value_or_default();
    _convDesc.FloorSpecGuard = cfg.FfxDenoiserFloorSpecGuard.value_or_default();
    _convDesc.FloorSpecGuardFadeStart = cfg.FfxDenoiserFloorSpecGuardFadeStart.value_or_default();
    _convDesc.FloorSpecGuardFadeEnd = cfg.FfxDenoiserFloorSpecGuardFadeEnd.value_or_default();
    _convDesc.SplitPriorStrength = cfg.FfxDenoiserSplitPrior.value_or_default();
    _convDesc.RoughnessProbe = cfg.FfxDenoiserRoughnessProbe.value_or_default();

    if (s_isRoughnessPacked)
        _convDesc.Flags |= (uint32_t) FSRDConvFlags::IsRoughnessPacked;

    // Troubleshooting (24 Sep). A/B switches for the pipeline audit's findings, all gated by the
    // master switch so a chosen set flips as one. Each bit off = previous behaviour.
    {
        const bool abActive = cfg.FfxDenoiserAbActive.value_or_default();
        const auto AbFlag = [abActive](const CustomOptional<bool>& option, FSRDConvFlags flag)
        { return (abActive && option.value_or_default()) ? (uint32_t) flag : 0u; };

        _convDesc.Flags |= AbFlag(cfg.FfxDenoiserAbNoEmissive, FSRDConvFlags::AbNoEmissive);
        _convDesc.Flags |= AbFlag(cfg.FfxDenoiserAbGateNoRoughness, FSRDConvFlags::AbGateNoRoughness);
        _convDesc.Flags |= AbFlag(cfg.FfxDenoiserAbGateNoBias, FSRDConvFlags::AbGateNoBias);
        _convDesc.Flags |= AbFlag(cfg.FfxDenoiserAbSoftMinNonNeg, FSRDConvFlags::AbSoftMinNonNeg);
        _convDesc.Flags |= AbFlag(cfg.FfxDenoiserAbSkipAlphaFinal, FSRDConvFlags::AbSkipAlphaFinal);
        _convDesc.Flags |= AbFlag(cfg.FfxDenoiserAbSkippedInactive, FSRDConvFlags::AbSkippedInactive);
        _convDesc.Flags |= AbFlag(cfg.FfxDenoiserAbCameraDepthDelta, FSRDConvFlags::AbCameraDepthDelta);
        _convDesc.FloorNoAlias = abActive && cfg.FfxDenoiserAbFloorNoAlias.value_or_default();

        // Pixel probe
        if (cfg.FfxDenoiserProbe.value_or_default())
            _convDesc.Flags |= (uint32_t) FSRDConvFlags::Probe;

        _convDesc.ProbeU = std::clamp(cfg.FfxDenoiserProbeX.value_or_default(), 0.0f, 1.0f);
        _convDesc.ProbeV = std::clamp(cfg.FfxDenoiserProbeY.value_or_default(), 0.0f, 1.0f);
        _convDesc.ProbeRadius = cfg.FfxDenoiserProbeRadius.value_or_default();
        _convDesc.ProbeAverageFrames = cfg.FfxDenoiserProbeAverage.value_or_default();
    }

    // Store in column major order for GPU
    XMStoreFloat4x4(&_convDesc.InvViewMatrix, XMMatrixTranspose(_invViewMatrix));

    // Inverse perspective projection
    const XMMATRIX invProjMatrix = XMMatrixInverse(nullptr, _projMatrix);
    XMStoreFloat4x4(&_convDesc.InvProjMatrix, XMMatrixTranspose(invProjMatrix));

    // Previous world to view for linear depth delta
    XMStoreFloat4x4(&_convDesc.PrevViewMatrix, XMMatrixTranspose(_prevViewMatrix));

    // Forward projection, same storage as the inverse above (motion consistency check, 24 Sep)
    XMStoreFloat4x4(&_convDesc.ProjMatrix, XMMatrixTranspose(_projMatrix));

    // Near and far planes
    const ViewPlanes planes = GetViewPlanes(_projMatrix, DepthInverted());
    _convDesc.NearPlane = planes.nearPlane;
    _convDesc.FarPlane = planes.farPlane;

    // Kept for the diagnostics panel (24 Sep) - the same numbers as the one-time "Camera:" log line
    _lastProjTerms[0] = _projMatrix.r[2].m128_f32[2];
    _lastProjTerms[1] = _projMatrix.r[2].m128_f32[3];
    _lastProjTerms[2] = _projMatrix.r[3].m128_f32[2];
    _lastInfiniteFar = planes.isInfinite;
    _lastRightHanded = _projMatrix.r[3].m128_f32[2] < 0.0f;

    // One-time camera convention report (transplant, 23 Sep) - evidence for the open signed-depth /
    // handedness question in PrepareDenoiserInput. W is the projection's w_clip = W * z_view term:
    // +1 means +z is in front of the camera (left-handed view space), -1 means -z (right-handed).
    if (!_loggedCameraConvention)
    {
        _loggedCameraConvention = true;
        LOG_INFO("Camera: proj A={} B={} W={} ({}-handed view space), near={} far={}{}, depthInverted={}, "
                 "camPos=({}, {}, {})",
                 _projMatrix.r[2].m128_f32[2], _projMatrix.r[2].m128_f32[3], _projMatrix.r[3].m128_f32[2],
                 _projMatrix.r[3].m128_f32[2] < 0.0f ? "right" : "left", planes.nearPlane, planes.farPlane,
                 planes.isInfinite ? " (infinite)" : "", DepthInverted(), _invViewMatrix.r[0].m128_f32[3],
                 _invViewMatrix.r[1].m128_f32[3], _invViewMatrix.r[2].m128_f32[3]);
    }

    if (!s_isHWDepth)
        _convDesc.Flags |= (uint32_t) FSRDConvFlags::IsDepthLinear;

    LOG_DEBUG("Distpaching FSRD Input Converter");

    // Dispatch resource converter. Outputs are automatically transitioned for reading.
    if (!FSRDConvShader->DispatchConversion(InCommandList, _convDesc))
        return false;

    return true;
}

bool FSRDFeatureDx12::DispatchDenoiser(ID3D12GraphicsCommandList* InCommandList,
                                       const ffxDispatchDescDenoiser& dispatchDesc)
{
    auto& state = State::Instance();
    const auto& cfg = *Config::Instance();

    // Runtime tuning values (24 Sep, audit finding 10). As before, Config's values are forced over the
    // SDK's defaults and only keys whose target changed are configured. With the SDK-defaults A/B
    // switch on, every key whose default query succeeded is taken back to denoiser 1.2's own default
    // instead; switching it off forces Config's values again. Indexed by FfxApiConfigureDenoiserKey - 1.
    const bool useSdkDefaults =
        cfg.FfxDenoiserAbActive.value_or_default() && cfg.FfxDenoiserAbSdkDefaults.value_or_default();

    const std::array<float, DenoiserConfiguration::kCount> configValues = {
        cfg.FfxDenoiserCrossBlNormStr.value_or_default(), cfg.FfxDenoiserStabilityBias.value_or_default(),
        cfg.FfxDenoiserMaxRadiance.value_or_default(),    cfg.FfxDenoiserRadianceClip.value_or_default(),
        cfg.FfxDenoiserGaussKernRelax.value_or_default(), cfg.FfxDenoiserDisocThreshold.value_or_default()
    };

    for (int i = 0; i < DenoiserConfiguration::kCount; i++)
    {
        const bool haveDefault = _sdkDefaultCodes[i] == FFX_API_RETURN_OK;
        const float target = (useSdkDefaults && haveDefault) ? _sdkDefaults.AsArray[i] : configValues[i];

        if (target == _denoiserSettings.AsArray[i])
            continue;

        const FfxApiConfigureDenoiserKey key = DenoiserConfiguration::GetIndexKey(i);
        _denoiserSettings.AsArray[i] = target;
        _applyCodes[i] = ApplyConfiguration(key);

        if (_applyCodes[i] != FFX_API_RETURN_OK)
            LOG_WARN("FSR-RR configure key {} = {} failed: {}", (int) key, target,
                     FfxApiProxy::ReturnCodeToString(_applyCodes[i]));
    }

    // Frame index continuity (24 Sep). Classifies each dispatch the way 1.2's "Frame index jump" check
    // would see it, so the panel can show whether the shim accounts for every warning.
    _dispatchCount++;

    if (dispatchDesc.flags & FFX_DENOISER_DISPATCH_RESET)
        _gameResetCount++;

    if (!_dispatchedOnContext)
    {
        _contextStartCount++;
        _dispatchedOnContext = true;
    }
    else if (dispatchDesc.frameIndex != _lastDispatchedIndex + 1)
    {
        _indexGapCount++;
        _lastGapFrames = dispatchDesc.frameIndex - _lastDispatchedIndex - 1;
        LOG_INFO("FSR-RR frame index gap: {} frame(s) without a denoiser dispatch before index {}", _lastGapFrames,
                 dispatchDesc.frameIndex);
    }

    _lastDispatchedIndex = dispatchDesc.frameIndex;

    LOG_DEBUG("Dispatching FSR-RR...");
    const ffxReturnCode_t result = FfxApiProxy::D3D12_Dispatch(&_pDenoiserCtx, &dispatchDesc.header);
    _lastDispatchCode = result;
    _lastDispatchFlags = dispatchDesc.flags;

    if (result != FFX_API_RETURN_OK)
    {
        LOG_ERROR("Dispatch error: {0}", FfxApiProxy::ReturnCodeToString(result));

        if (result == FFX_API_RETURN_ERROR_RUNTIME_ERROR)
        {
            LOG_WARN("Trying to recover by recreating the feature");
            state.newBackend = Upscaler::FSRD;
            state.changeBackend[Handle()->Id] = true;
        }

        return false;
    }

    return true;
}

void FSRDFeatureDx12::SetDefaultConfiguration()
{
    // 24 Sep: the queried defaults are kept apart from the values in force, with their return codes,
    // for the SDK-defaults A/B switch and the diagnostics panel. They were never logged before.
    for (int i = 0; i < DenoiserConfiguration::kCount; i++)
    {
        _sdkDefaultCodes[i] = SetDefaultConfiguration(DenoiserConfiguration::GetIndexKey(i));
        _sdkDefaults.AsArray[i] = _denoiserSettings.AsArray[i];
        _applyCodes[i] = -1;
    }

    LOG_INFO("FSR-RR 1.2 defaults: cross bilateral normal strength {} ({}), stability bias {} ({}), "
             "max radiance {} ({}), radiance clip std k {} ({}), gaussian kernel relaxation {} ({}), "
             "disocclusion threshold {} ({})",
             _sdkDefaults.m_CrossBilateralNormalStrength, FSRD::ReturnCodeName(_sdkDefaultCodes[0]),
             _sdkDefaults.m_StabilityBias, FSRD::ReturnCodeName(_sdkDefaultCodes[1]), _sdkDefaults.m_MaxRadiance,
             FSRD::ReturnCodeName(_sdkDefaultCodes[2]), _sdkDefaults.m_RadianceClipStdK,
             FSRD::ReturnCodeName(_sdkDefaultCodes[3]), _sdkDefaults.m_GaussianKernelRelaxation,
             FSRD::ReturnCodeName(_sdkDefaultCodes[4]), _sdkDefaults.m_DisocclusionThreshold,
             FSRD::ReturnCodeName(_sdkDefaultCodes[5]));
}

void FSRDFeatureDx12::PublishDiagnostics(const NVSDK_NGX_Parameter& inParams, bool denoiseBypassed,
                                         bool upscaleBypassed)
{
    const auto& state = State::Instance();
    const auto& cfg = *Config::Instance();

    FSRD::FrameInfo frame;
    frame.frame = _frameCount;
    frame.renderWidth = RenderWidth();
    frame.renderHeight = RenderHeight();
    frame.targetWidth = TargetWidth();
    frame.targetHeight = TargetHeight();

    frame.nearPlane = _convDesc.NearPlane;
    frame.farPlane = _convDesc.FarPlane;
    frame.infiniteFar = _lastInfiniteFar;
    frame.rightHanded = _lastRightHanded;
    frame.depthInverted = DepthInverted();
    frame.hwDepth = s_isHWDepth;
    frame.roughnessPacked = s_isRoughnessPacked;
    frame.reset = _isInReset;
    frame.fovVerticalDeg = GetVertFovFromProjectionMatrixRad(_projMatrix) * (180.0f / 3.14159265f);
    frame.projA = _lastProjTerms[0];
    frame.projB = _lastProjTerms[1];
    frame.projW = _lastProjTerms[2];
    frame.camPos[0] = _invViewMatrix.r[0].m128_f32[3];
    frame.camPos[1] = _invViewMatrix.r[1].m128_f32[3];
    frame.camPos[2] = _invViewMatrix.r[2].m128_f32[3];

    inParams.Get(NVSDK_NGX_Parameter_Jitter_Offset_X, &frame.jitterPx[0]);
    inParams.Get(NVSDK_NGX_Parameter_Jitter_Offset_Y, &frame.jitterPx[1]);
    inParams.Get(NVSDK_NGX_Parameter_MV_Scale_X, &frame.mvScale[0]);
    inParams.Get(NVSDK_NGX_Parameter_MV_Scale_Y, &frame.mvScale[1]);

    const int versionIndex = cfg.FfxDenoiserIndex.value_or_default();

    if (versionIndex >= 0 && versionIndex < (int) state.ffxDenoiserVersionNames.size() &&
        state.ffxDenoiserVersionNames[versionIndex] != nullptr)
        frame.denoiserVersion = state.ffxDenoiserVersionNames[versionIndex];

    frame.signalFlags = _denoiserCtxDesc.signalFlags;
    frame.createFlags = _denoiserCtxDesc.flags;
    frame.dispatchFlags = _lastDispatchFlags;
    frame.signalText = std::format(
        "diffuse -> {}, specular -> {}",
        (frame.signalFlags & FFX_DENOISER_SIGNAL_DIRECT_DIFFUSE) ? "DIRECT_DIFFUSE" : "INDIRECT_DIFFUSE",
        (frame.signalFlags & FFX_DENOISER_SIGNAL_DIRECT_SPECULAR) ? "DIRECT_SPECULAR" : "INDIRECT_SPECULAR");
    frame.createText =
        std::format("debugging {}, validation {}", (frame.createFlags & FFX_DENOISER_ENABLE_DEBUGGING) ? "on" : "off",
                    (frame.createFlags & FFX_DENOISER_ENABLE_VALIDATION) ? "on" : "off");
    frame.dispatchText = std::format("non-gamma albedo {}, reset {}",
                                     (frame.dispatchFlags & FFX_DENOISER_DISPATCH_NON_GAMMA_ALBEDO) ? "on" : "off",
                                     (frame.dispatchFlags & FFX_DENOISER_DISPATCH_RESET) ? "on" : "off");
    frame.lastDispatchCode = _lastDispatchCode;
    frame.denoiseBypassed = denoiseBypassed;
    frame.upscaleBypassed = upscaleBypassed;
    frame.declaredStatesFixed = _lastDeclaredStates;
    frame.albedo16 = FSRDConvShader != nullptr && FSRDConvShader->IsAlbedo16();
    frame.messageCallback = _messageCallbackOk;
    frame.camMove = _lastCamMove;
    frame.camTurnDeg = _lastCamTurnDeg;
    frame.dispatches = _dispatchCount;
    frame.indexGaps = _indexGapCount;
    frame.lastGapFrames = _lastGapFrames;
    frame.contextStarts = _contextStartCount;
    frame.gameResets = _gameResetCount;

    // Every DLSS-RR input the game hands over, consumed or not, with its format
    std::vector<FSRD::InputInfo> inputs;

    const auto AddInput = [&](const char* label, const char* key)
    {
        FSRD::InputInfo info;
        info.name = label;
        ID3D12Resource* resource = nullptr;

        if (TryGetNGXVoidPointer(inParams, key, resource) && resource != nullptr)
        {
            const D3D12_RESOURCE_DESC desc = resource->GetDesc();
            info.present = true;
            info.format = (uint32_t) desc.Format;
            info.width = desc.Width;
            info.height = desc.Height;
        }

        inputs.push_back(std::move(info));
    };

    AddInput("Color", NVSDK_NGX_Parameter_Color);
    AddInput("Depth", NVSDK_NGX_Parameter_Depth);
    AddInput("MotionVectors", NVSDK_NGX_Parameter_MotionVectors);
    AddInput("Normals", NVSDK_NGX_Parameter_GBuffer_Normals);
    AddInput("Roughness", NVSDK_NGX_Parameter_GBuffer_Roughness);
    AddInput("DiffuseAlbedo", NVSDK_NGX_Parameter_DiffuseAlbedo);
    AddInput("SpecularAlbedo", NVSDK_NGX_Parameter_SpecularAlbedo);
    AddInput("SpecularHitDistance", NVSDK_NGX_Parameter_DLSSD_SpecularHitDistance);
    AddInput("DiffuseHitDistance", NVSDK_NGX_Parameter_DLSSD_DiffuseHitDistance);
    AddInput("BiasCurrentColorMask", NVSDK_NGX_Parameter_DLSS_Input_Bias_Current_Color_Mask);
    AddInput("ColorBeforeParticles", NVSDK_NGX_Parameter_DLSSD_ColorBeforeParticles);
    AddInput("ColorBeforeTransparency", NVSDK_NGX_Parameter_DLSSD_ColorBeforeTransparency);
    AddInput("TransparencyLayer", NVSDK_NGX_Parameter_DLSS_TransparencyLayer);
    AddInput("Output", NVSDK_NGX_Parameter_Output);

    static constexpr std::array<const char*, DenoiserConfiguration::kCount> kTuningNames = {
        "Cross Bilateral Normal Strength", "Temporal Stability Bias",    "Max Radiance",
        "Radiance Clip Deviation",         "Gaussian Kernel Relaxation", "Disocclusion Threshold"
    };

    auto& diagnostics = FSRD::Diagnostics::Instance();
    std::scoped_lock lock(diagnostics.Mutex);

    diagnostics.Frame = std::move(frame);
    diagnostics.Inputs = std::move(inputs);

    for (int i = 0; i < DenoiserConfiguration::kCount; i++)
    {
        auto& entry = diagnostics.Tuning[i];
        entry.name = kTuningNames[i];
        entry.sdkDefault = _sdkDefaults.AsArray[i];
        entry.queryCode = _sdkDefaultCodes[i];
        entry.applied = _denoiserSettings.AsArray[i];
        entry.applyCode = _applyCodes[i];
    }
}

ffxReturnCode_t FSRDFeatureDx12::SetDefaultConfiguration(FfxApiConfigureDenoiserKey key)
{
    ffxQueryDescDenoiserGetDefaultKeyValue queryDesc = {
        .header = { .type = FFX_API_QUERY_DESC_TYPE_DENOISER_GET_DEFAULT_KEYVALUE },
        .key = (uint64_t) key,
        .count = 1u,
        .data = &_denoiserSettings.GetMember(key)
    };

    const ffxReturnCode_t code = FfxApiProxy::D3D12_Query(&_pDenoiserCtx, &queryDesc.header);
    return code;
}

ffxReturnCode_t FSRDFeatureDx12::ApplyConfiguration(FfxApiConfigureDenoiserKey key)
{
    ffxQueryDescDenoiserGetDefaultKeyValue configureDesc = {
        .header = { .type = FFX_API_CONFIGURE_DESC_TYPE_DENOISER_KEYVALUE },
        .key = (uint64_t) key,
        .count = 1u,
        .data = &_denoiserSettings.GetMember(key)
    };

    const ffxReturnCode_t code = FfxApiProxy::D3D12_Configure(&_pDenoiserCtx, &configureDesc.header);
    return code;
}