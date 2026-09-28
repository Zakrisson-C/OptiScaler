#pragma once
#include "FSRDShaderUtils.h"

namespace FSRD
{
namespace FloorSeed
{
constexpr UINT kPasses = 1;
constexpr UINT kBackBufferCount = std::max(3 * (kPasses + 1), 1u);

enum class Flags : uint32_t
{
    None = 0,
    LinearDepth = (1 << 0),
    DepthOnly = (1 << 1) // 25 Sep: floor off, write linear depth only
};

struct alignas(16) Constants
{
    XMFLOAT4X4 InvProjMatrix;
    XMFLOAT4 RenderSize;

    float NearPlane;
    float FarPlane;

    uint32_t Flags;
    float _Padding[1];
};

union Input
{
    struct Data
    {
        ID3D12Resource* InColor;
        ID3D12Resource* InNormals;
        ID3D12Resource* InDepth;
    };

    // The number of D3D12 resources in the struct
    static constexpr uint32_t kCount = sizeof(Data) / sizeof(ID3D12Resource*);

    Data Resources;

    ID3D12Resource* AsArray[kCount];
};

union Output
{
    struct Data
    {
        ID3D12Resource* OutColor;
        ID3D12Resource* OutLinearDepth;
        ID3D12Resource* OutDepthGradient;
    };

    // The number of D3D12 resources in the struct
    static constexpr uint32_t kCount = sizeof(Data) / sizeof(ID3D12Resource*);

    Data Resources;

    ID3D12Resource* AsArray[kCount];
};
} // namespace FloorSeed

namespace FloorFilter
{
constexpr UINT kPasses = 5;
constexpr UINT kBackBufferCount = std::max(3 * (kPasses + 1), 1u);

enum class Flags : uint32_t
{
    None = 0,
};

struct alignas(16) Constants
{
    XMFLOAT4 DstTexSize;

    float RcpCrossBlNorm;
    float RcpSelfBlNorm;

    int32_t StepSize;
    uint32_t FrameIndex;

    uint32_t Flags;

    float DetailBoost;         // Laplacian residual re-injection. Final pass only.
    float NormalSharpness;     // Exponent on the normal edge-stopping weight.
    float AlbedoGuideStrength; // Luminance edge-stop released on same-material taps.

    float LumSymmetry;      // 0 = centre-normalised luma delta, 1 = symmetric.
    float GrazingSharpness; // Extra normal edge-stop proportional to slope.

    float _Padding[2];
};

union Input
{
    struct Data
    {
        ID3D12Resource* InColor;
        ID3D12Resource* InLinearDepth;
        ID3D12Resource* InDepthGradient;
        ID3D12Resource* InDiffAlbedo; // Material guide - see GetAlbedoAgreement
    };

    // The number of D3D12 resources in the struct
    static constexpr uint32_t kCount = sizeof(Data) / sizeof(ID3D12Resource*);

    Data Resources;

    ID3D12Resource* AsArray[kCount];
};

union Output
{
    struct Data
    {
        ID3D12Resource* OutColor;
    };

    // The number of D3D12 resources in the struct
    static constexpr uint32_t kCount = sizeof(Data) / sizeof(ID3D12Resource*);

    Data Resources;

    ID3D12Resource* AsArray[kCount];
};
} // namespace FloorFilter

namespace Conversion
{
constexpr UINT kBackBufferCount = 3;

// Mode1Signal (ffxDispatchDescDenoiserInput1Signal) and Mode2Signal
// (ffxDispatchDescDenoiserInput2Signals) removed (transplant, 22 Sep): denoiser 1.2 has no
// combined-signal shape to give Mode1Signal a successor, and Mode2Signal's two members are
// now inlined directly into Output::Data below instead of held in a separate struct
// (transplant plan §6e/6f).

/**
 * @brief Constant buffer data passed to the conversion shader.
 */
struct alignas(16) Constants
{
    XMFLOAT4X4 InvViewMatrix;  // DLSSD WorldToView^1 - Camera matrix
    XMFLOAT4X4 InvProjMatrix;  // DLSSD ViewToClip^-1 - Projection
    XMFLOAT4X4 PrevViewMatrix; // DLSSD WorldToView from last frame

    XMFLOAT4 RenderSize;

    float NearPlane; // Near < Far - IsInverted flag accounts for inversion
    float FarPlane;  // Near < Far - IsInverted flag accounts for inversion

    float FloorIsolation;
    uint32_t Flags; // Dynamic configuration flags. See: ConfigFlags

    float BiasMaskStrength; // Scales InBiasMask when routing pixels into the floor
    float FloorSoftMin;     // Smoothing radius on the floor/raw clamp. 0 = exact min()

    float RoughnessExponent;  // Re-encodes roughness. 1.0 = bit-identical
    float HitDistScale;       // Scales specular ray length. 1.0 = bit-identical
    float FloorSpecGuard;     // Pulls the floor off mirrors. 0 = bit-identical
    float SplitPriorStrength; // Biases the Mode 2 split. 0 = bit-identical
    float RoughnessProbe;     // Target for the roughness null-probe debug view

    float _Padding;

    // Pixel probe window (24 Sep): centre pixel and half-width. Only read with ConvFlags::Probe.
    int32_t ProbeCenterX;
    int32_t ProbeCenterY;
    int32_t ProbeRadius;
    uint32_t _Padding2;

    // Specular guard distance fade (24 Sep), linear depth units. FadeEnd <= FadeStart = no fade.
    float FloorSpecGuardFadeStart;
    float FloorSpecGuardFadeEnd;

    // Firefly clamp (25 Sep): max ratio to the brightest neighbour. 0 = off (bit-identical).
    float FireflyClampK;

    // SSS separation (26 Sep): share of the SSS contribution routed around the denoiser. 0 = off.
    float SssSeparation;

    // Current view to clip, for the motion consistency check (24 Sep). Debug view / probe only.
    XMFLOAT4X4 ProjMatrix;

    // SSS separation (26 Sep): weight of the current frame in the SSS history. 1 = no history.
    float SssHistoryAlpha;

    // Additive light split (27 Sep): > 0 = specular share from the split fit. 0 = off (bit-identical).
    float AdditiveSeparation;
    float _Padding4[2];
};

// Matches the cbuffer layout DXC reports for CB_Packing in FSRDInputConv.hlsl.
static_assert(sizeof(Constants) == 368, "Conversion::Constants out of sync with CB_Packing");
static_assert(offsetof(Constants, ProbeCenterX) == 256, "Conversion::Constants out of sync with CB_Packing");
static_assert(offsetof(Constants, FloorSpecGuardFadeStart) == 272, "Conversion::Constants out of sync with CB_Packing");
static_assert(offsetof(Constants, FireflyClampK) == 280, "Conversion::Constants out of sync with CB_Packing");
static_assert(offsetof(Constants, SssSeparation) == 284, "Conversion::Constants out of sync with CB_Packing");
static_assert(offsetof(Constants, ProjMatrix) == 288, "Conversion::Constants out of sync with CB_Packing");
static_assert(offsetof(Constants, SssHistoryAlpha) == 352, "Conversion::Constants out of sync with CB_Packing");
static_assert(offsetof(Constants, AdditiveSeparation) == 356, "Conversion::Constants out of sync with CB_Packing");

union Input
{
    struct Data
    {
        ID3D12Resource* InColor;         // RGB - NVSDK_NGX_Parameter_Color - HDR or SDR
        ID3D12Resource* InDepth;         // R - NVSDK_NGX_Parameter_Depth - 24/32bits
        ID3D12Resource* InMotionVectors; // RG - NVSDK_NGX_Parameter_MotionVectors - RG16/RG32
        ID3D12Resource* InNormals;     // RGB: Normals, A: Roughness (Optional) - NVSDK_NGX_Parameter_GBuffer_Normals -
                                       // RGB16_FLOAT/RG32_FLOAT
        ID3D12Resource* InRoughness;   // R - May be packed in normals. NVSDK_NGX_Parameter_GBuffer_Roughness
        ID3D12Resource* InSpecHitDist; // R - NVSDK_NGX_Parameter_DLSSD_SpecularHitDistance - FP16/FP32
        ID3D12Resource* InDiffAlbedo;  // RGB - NVSDK_NGX_Parameter_GBuffer_DiffuseAlbedo - RGBA32
        ID3D12Resource* InSpecAlbedo;  // RGB - NVSDK_NGX_Parameter_GBuffer_SpecularAlbedo - RGBA32
        ID3D12Resource* InBiasMask;    // R8 - NVSDK_NGX_Parameter_DLSS_Input_Bias_Current_Color_Mask

        ID3D12Resource* InBlurColor;
        ID3D12Resource* InPrevLinearDepth; // t10, last frame's linear depth (24 Sep; was the unused InEdgeGuide)
        ID3D12Resource* InSSSGuide;        // t11, DLSSD SSS guide (25 Sep, optional)
        ID3D12Resource* InSssHistory;      // t12, last frame's SSS history (26 Sep)
        ID3D12Resource* InSplitFit;        // t13, specular share from the additive light split (27 Sep)
    };

    // The number of D3D12 resources in the struct
    static constexpr uint32_t kCount = sizeof(Data) / sizeof(ID3D12Resource*);

    Data Resources;

    ID3D12Resource* AsArray[kCount];
};

/**
 * @brief Output resources formatted for direct consumption by FSR Ray Regeneration.
 * All resources are automatically transitioned to SRV state after dispatch.
 */
union Output
{
    struct Data
    {
        // Transplant, 22 Sep: flattened from a Mode1Signal/Mode2Signal union - denoiser
        // 1.2 only has the split signal shape, so no union is needed any more (transplant
        // plan §6f).
        ComPtr<ID3D12Resource> SpecRadiance; // RGB: Noisy specular lighting A: Specular Ray Length - RGBA16_FLOAT
        ComPtr<ID3D12Resource> DiffRadiance; // RGB: Noisy diffuse lighting - RGBA16_FLOAT

        ComPtr<ID3D12Resource> Motion;  // RG: Standard TSR motion vectors, B: Linear Depth Delta (CurrentLinearDepth -
                                        // PrevLinearDepth) - RGBA16_FLOAT
        ComPtr<ID3D12Resource> Normals; // RG: Octahedrally encoded normals, B: Linear Roughness, A: Material Type
                                        // (Optional) - RGB10A2_UNORM
        ComPtr<ID3D12Resource> SpecAlbedo; // RGB: Specular Albedo, A: saturate(dot(Normal, ViewDir)) - RGBA8_UNORM
        ComPtr<ID3D12Resource> DiffAlbedo; // RGB: Diffuse Albedo, A: Metalness (heuristic approximate) - RGBA8_UNORM

        ComPtr<ID3D12Resource> SkipSignal;

        Data() {}
        ~Data() {}
    };

    Output()
    {
        for (auto& resource : AsArray)
            resource = ComPtr<ID3D12Resource>();
    }

    ~Output()
    {
        for (auto& resource : AsArray)
            resource.~ComPtr();
    }

    // The number of D3D12 resources in the struct
    static constexpr uint32_t kCount = sizeof(Data) / sizeof(ID3D12Resource*);

    Data Resources;

    ComPtr<ID3D12Resource> AsArray[kCount];

    ID3D12Resource* AsRawArray[kCount];
};

// UAVs bound to the packing shader: the Output resources above (u0-u6), then the pixel probe
// texture (u7, 24 Sep) and this frame's SSS history (u8, 26 Sep). Both are kept out of Output
// because Output is the denoiser's input set.
constexpr UINT kUavCount = Output::kCount + 2;
constexpr UINT kProbeUav = Output::kCount;
constexpr UINT kSssHistoryUav = Output::kCount + 1;
} // namespace Conversion

// Additive light split (27 Sep, FSRDSplitFit.hlsl): per-window fit of colour against albedo; its specular
// share (averaged over frames) replaces the reflectance ratio in the packing shader.
namespace SplitFit
{
constexpr UINT kBackBufferCount = 3;

enum class Flags : uint32_t
{
    None = 0,
    Reset = (1 << 0) // history invalid this frame
};

struct alignas(16) Constants
{
    XMFLOAT4 DstTexSize; // XY = size, ZW = 1 / size
    float Strength;      // scales the intercept; 0 = the old split
    float HistoryAlpha;  // weight of the current frame; 1 = no history
    uint32_t Flags;
    float _Padding;
};

static_assert(sizeof(Constants) == 32, "SplitFit::Constants out of sync with CB_SplitFit");

union Input
{
    struct Data
    {
        ID3D12Resource* InColor;
        ID3D12Resource* InDiffAlbedo;
        ID3D12Resource* InSpecAlbedo;
        ID3D12Resource* InLinearDepth;
        ID3D12Resource* InNormals;
        ID3D12Resource* InMotionVectors;
        ID3D12Resource* InHistory;
    };

    static constexpr uint32_t kCount = sizeof(Data) / sizeof(ID3D12Resource*);

    Data Resources;

    ID3D12Resource* AsArray[kCount];
};

constexpr UINT kOutputCount = 2; // share history (read by the packing shader), intercept (view)
} // namespace SplitFit

// Texture leak views (26 Sep, FSRDLeak.hlsl): debug only, one dispatch on frames that show one.
namespace Leak
{
constexpr UINT kBackBufferCount = 3;

enum class Flags : uint32_t
{
    None = 0,
    Diffuse = (1 << 0) // diffuse lobe; clear = specular
};

struct alignas(16) Constants
{
    XMFLOAT4 DstTexSize; // XY = size, ZW = 1 / size
    uint32_t Flags;
    float _Padding[3];
};

static_assert(sizeof(Constants) == 32, "Leak::Constants out of sync with CB_Leak");

union Input
{
    struct Data
    {
        ID3D12Resource* InSpecRadiance;
        ID3D12Resource* InDiffRadiance;
        ID3D12Resource* InSpecAlbedo;
        ID3D12Resource* InDiffAlbedo;
        ID3D12Resource* InLinearDepth;
        ID3D12Resource* InNormals;
    };

    static constexpr uint32_t kCount = sizeof(Data) / sizeof(ID3D12Resource*);

    Data Resources;

    ID3D12Resource* AsArray[kCount];
};

constexpr UINT kOutputCount = 1;
} // namespace Leak

// SSS re-blur (28 Sep, FSRDSssBlur.hlsl): after composition, a separable Gaussian of a world-space width over the
// pixels the game's SSS guide marks, applied to the denoised colour; optionally the same kernel on the noisy colour
// before the game's blur, to fit radius and strength against the guide.
namespace SssBlur
{
// Two dispatches per frame (horizontal, vertical), three frames in flight, with margin
constexpr UINT kBackBufferCount = 12;

enum class Flags : uint32_t
{
    None = 0,
    Vertical = (1 << 0), // second pass
    Apply = (1 << 1),    // blur the denoised colour, write it over the composition output
    Fit = (1 << 2),      // blur the noisy colour too, reduce the fit terms
    View = (1 << 3),     // vertical pass: draw the fit residual over the whole frame
};

struct alignas(16) Constants
{
    XMFLOAT4 DstTexSize;   // XY = size, ZW = 1 / size
    XMFLOAT3 ChannelScale; // sigma per channel relative to red
    float SigmaScale;      // sigma (m) * focal length (px)
    float Strength;        // 0 = no blur, 1 = full
    float DepthTolerance;  // m, before the 1% of depth the shader adds
    uint32_t Flags;
    float _Padding;
};

static_assert(sizeof(Constants) == 48, "SssBlur::Constants out of sync with CB_SssBlur");

union Input
{
    struct Data
    {
        ID3D12Resource* InBlurSrc;
        ID3D12Resource* InFitSrc;
        ID3D12Resource* InColor;
        ID3D12Resource* InSSSGuide;
        ID3D12Resource* InLinearDepth;
        ID3D12Resource* InColorBeforeParticles;
        ID3D12Resource* InSssSource;
    };

    static constexpr uint32_t kCount = sizeof(Data) / sizeof(ID3D12Resource*);

    Data Resources;

    ID3D12Resource* AsArray[kCount];
};

constexpr UINT kOutputCount = 2;

// One fit texel per thread group
constexpr UINT kGroupSize = 8;
} // namespace SssBlur

namespace Composition
{
constexpr UINT kBackBufferCount = 7;
constexpr UINT kOutputCount = 2; // output colour, SSS re-blur source (28 Sep, null unless the re-blur is on)

struct alignas(16) Constants
{
    XMFLOAT4 DstTexSize; // XY = Tex Size - ZW = 1 / XY

    float CorrelationBias; // Controls the contribution of stable elements to the final image
    uint32_t Flags;

    float _Padding[2];
};

/**
 * @brief Resources used for composition after denoising
 */
union Input
{
    struct Data
    {
        ID3D12Resource* InDenoisedSignal1;
        ID3D12Resource* InAlbedo1;

        ID3D12Resource* InDenoisedSignal2;
        ID3D12Resource* InAlbedo2;

        ID3D12Resource* InSkipSignal;
        ID3D12Resource* InRawColor;
        ID3D12Resource* InColorBeforeParticles; // NVSDK_NGX_Parameter_DLSSD_ColorBeforeParticles
        ID3D12Resource* InSSSGuide;             // DLSSD SSS guide (25 Sep, optional)
    };

    // The number of D3D12 resources in the struct
    static constexpr uint32_t kCount = sizeof(Data) / sizeof(ID3D12Resource*);

    Data Resources;

    ID3D12Resource* AsArray[kCount];
};
} // namespace Composition
} // namespace FSRD