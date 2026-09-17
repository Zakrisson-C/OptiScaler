#pragma once
#include "SysUtils.h"

#include <DirectXMath.h>
#include <cstdint>
#include <memory>
#include <string_view>

struct ID3D12Device;
struct ID3D12GraphicsCommandList;
struct ID3D12Resource;

struct ffxDispatchDescDenoiserInput1Signal;
struct ffxDispatchDescDenoiserInput2Signals;
struct ffxDispatchDescDenoiser;

/**
 * @brief Converts DLSS Ray Reconstruction inputs into the format expected by FSR Ray Regeneration,
 * and handles composition for FSR-RR outputs.
 */
class FSRDPreprocessor_Dx12
{
  public:

    enum class ConvFlags : uint32_t
    {
        None = 0,

        NonGammaAlbedo =        1 << 0, // If set, FFX_DENOISER_DISPATCH_NON_GAMMA_ALBEDO should ALSO be set
        IsDepthLinear =         1 << 1, // Interprets input depth as already linearized for view space calculations
        IsRoughnessPacked =     1 << 2, // Roughness = InNormals.A - NVSDK_NGX_DLSS_Roughness_Mode_Packed (Init param)
        Mode2Signal =           1 << 3, // Enables mode 2 denoiser outputs with discrete diffuse and specular lighting
        HasBiasMask =           1 << 4, // InBiasMask holds a real DLSS bias-current-color mask

        Debug =                 1 << 16, // Denoiser and upscaler bypassed for debug out if this is set
        DebugModeMask =         0xFF << 16,

        DebugOutRadiance =      Debug, // Default debug vis

        DebugInSpecHitDist =    1 << 17 | Debug,
        DebugInMotion =         2 << 17 | Debug,
        DebugInNormals =        3 << 17 | Debug,
        DebugInRoughness =      4 << 17 | Debug,
        DebugInDiffAlbedo =     5 << 17 | Debug,
        DebugInSpecAlbedo =     6 << 17 | Debug,

        DebugOutFusedAlbedo =   7 << 17 | Debug,
        DebugOutLinearDepth =   8 << 17 | Debug,
        DebugOutMotion =        9 << 17 | Debug,
        DebugOutNormals =       10 << 17 | Debug,
        DebugOutSpecAlbedo =    11 << 17 | Debug,
        DebugOutDiffAlbedo =    12 << 17 | Debug,

        DebugOutDepthDelta =    13 << 17 | Debug,
        DebugNormDepth =        14 << 17 | Debug,

        DebugAlbedoError =      15 << 17 | Debug,

        DebugFloorVariance =    16 << 17 | Debug,
        DebugFloorColor =       17 << 17 | Debug,

        DebugInBiasMask =       18 << 17 | Debug, // Bias mask as applied, after strength scaling
        DebugDemodGain =        19 << 17 | Debug, // 1 / albedo used as the demodulation divisor

        DebugHitDistGate =      20 << 17 | Debug, // canUseHitDist ramp, separate from the buffer it scales
        DebugDenoiserFraction = 21 << 17 | Debug, // Share of the pixel routed to the denoiser vs the skip signal
    };

    enum class CompFlags : uint32_t
    {
        None =                  0,
        RawSourceBlit =         1 << 0, // Bypass composition and write unmodified input
        ScaleSrc =              1 << 1, // Enable bilinear scaling to output
        Mode2Signal =           1 << 2,

        Debug =                 1 << 16,
        DebugModeMask =         0xFF << 16,

        DebugCorrelation =      1 << 17 | Debug,
        DebugSkipSignal =       2 << 17 | Debug,
        DebugDenoiserOutput =   3 << 17 | Debug,
        DebugSignal1 =          4 << 17 | Debug,
        DebugSignal2 =          5 << 17 | Debug,
    };

    /**
     * @brief Input resources matching DLSS Ray Reconstruction / NGX parameter names.
     * All resources must be in readable state before dispatch.
     */
    union InputResources
    {
        struct
        {
            ID3D12Resource* InColor; // RGB - NVSDK_NGX_Parameter_Color - HDR or SDR
            ID3D12Resource* InDepth; // R - NVSDK_NGX_Parameter_Depth - 24/32bits
            ID3D12Resource* InMotionVectors; // RG - NVSDK_NGX_Parameter_MotionVectors - RG16/RG32
            ID3D12Resource* InNormals; // RGB: Normals, A: Roughness (Optional) - NVSDK_NGX_Parameter_GBuffer_Normals - RGB16_FLOAT/RG32_FLOAT
            ID3D12Resource* InRoughness; // R - May be packed in normals. NVSDK_NGX_Parameter_GBuffer_Roughness
            ID3D12Resource* InSpecHitDist; // R - NVSDK_NGX_Parameter_DLSSD_SpecularHitDistance - FP16/FP32
            ID3D12Resource* InDiffAlbedo; // RGB - NVSDK_NGX_Parameter_GBuffer_DiffuseAlbedo - RGBA32
            ID3D12Resource* InSpecAlbedo; // RGB - NVSDK_NGX_Parameter_GBuffer_SpecularAlbedo - RGBA32
            ID3D12Resource* InBiasMask; // R8 - NVSDK_NGX_Parameter_DLSS_Input_Bias_Current_Color_Mask
        };

        ID3D12Resource* AsArray[9];
    };

    /**
     * @brief Configuration and resources required for generating FSR-RR inputs from DLSS-RR
     * inputs.
     */
    struct ConversionDesc
    {
        InputResources Resources;

        DirectX::XMFLOAT4X4 InvViewMatrix;     // DLSSD WorldToView^1 - Camera matrix
        DirectX::XMFLOAT4X4 InvProjMatrix;     // DLSSD ViewToClip^-1 - Projection
        DirectX::XMFLOAT4X4 PrevViewMatrix;    // DLSSD WorldToView from last frame

        DirectX::XMFLOAT4 RenderSize;    // XY: Resolution of inputs - ZW: 1.0 / Resolution

        float NearPlane; // Near < Far
        float FarPlane;  // Near < Far

        float FloorIsolation;

        // Fraction of the DLSS bias mask used to route flagged pixels (particles, alpha,
        // animated textures) around the denoiser via the floor / skip signal. 0 disables.
        float BiasMaskStrength;

        // Fraction of the floor filter's Laplacian luminance residual pushed back into the
        // floor on the final pass, to keep texture microcontrast out of the denoiser.
        float FloorDetailBoost;

        // Exponent on the floor filter's normal edge-stopping weight.
        float FloorNormalSharpness;

        // Fraction of the floor filter's luminance edge-stop released where diffuse albedo
        // says two taps sit on the same material, so shadows and reflections are blurred
        // through into the denoiser signal instead of preserved into the floor. 0 disables.
        float FloorAlbedoGuide;

        // Blends the floor's luminance normaliser from centre-only (0) to max(centre, tap)
        // (1). 0 reproduces the previous asymmetric behaviour.
        float FloorLumSymmetry;

        // Additional normal edge-stop exponent applied in proportion to screen space
        // surface slope. 0 disables.
        float FloorGrazingSharpness;

        // Smoothing radius on the min(raw, floor) clamp in the packing shader. 0 = exact min().
        float FloorSoftMin;

        uint32_t Flags; // Dynamic configuration flags. See: ConfigFlags
    };

    /**
     * @brief Configuration and external resources used for compositing denoiser output into final result.
     */
    struct CompositionDesc
    {
        DirectX::XMFLOAT4 DstTexSize; // XY = Tex Size - ZW = 1 / XY
        float CorrelationBias; // Enhances the contribution of stable elements to the final image
        uint32_t Flags;

        ID3D12Resource* InRawColor;
        ID3D12Resource* InColorBeforeParticles; // NVSDK_NGX_Parameter_DLSSD_ColorBeforeParticles (Optional)
    };

  public:

    FSRDPreprocessor_Dx12(std::string_view name, ID3D12Device* pDev, bool isMode2);

    ~FSRDPreprocessor_Dx12();

    /**
     * @brief Indicates whether the converter has been successfully initialized.
     */
    bool IsInit() const;

    /**
     * @brief Returns the name of the shader instance
     */
    std::string_view GetName() const;

    /**
     * @brief (Re)allocates internal resources to match the specified render resolution.
     * Must be called at least once before any Dispatch().
     * @return True if resize succeeded
     */
    bool SetMaxRenderSize(uint32_t width, uint32_t height);

    /**
     * @brief Executes the input conversion shader.
     * Input resources must already be in shader-readable state.
     * Output resources are automatically transitioned to SRV state upon completion.
     *
     * @param inputs Input resource pointers
     * @param constants Per-frame constant buffer values
     * @return True if dispatch completed successfully
     */
    bool DispatchConversion(ID3D12GraphicsCommandList* cmdList, const ConversionDesc& desc);

    /**
     * @brief Configures input/output resources after input conversion for FSR-RR with Mode-1 fused inputs.
     * Resources are transitioned to SRV state and valid until the next conversion or composition dispatch.
     * Must be re-acquired after each dispatch (lifetime managed internally).
     */
    void GetSignal(ffxDispatchDescDenoiserInput1Signal& signalDesc, ffxDispatchDescDenoiser& dispatchDesc) const;

    /**
     * @brief Configures input/output resources after input conversion for FSR-RR with Mode-2 discrete diffuse/specular color.
     * Resources are transitioned to SRV state and valid until the next conversion or composition dispatch.
     * Must be re-acquired after each dispatch (lifetime managed internally).
     */
    void GetSignal(ffxDispatchDescDenoiserInput2Signals& signalDesc, ffxDispatchDescDenoiser& dispatchDesc) const;

    /**
     * @brief Composes the denoised radiance from FSR-RR with the skip signal previously generated 
     * by the converter, and writes the result to the given destination texture.
     */
    bool DispatchComposition(ID3D12GraphicsCommandList* cmdList, const CompositionDesc& desc);

    /**
     * @brief Returns the output from the last composition dispatch. Valid until the next conversion dispatch.
     */
    ID3D12Resource* GetCompositionOutput() const;

    /**
     * @brief Copies the contents of the given source texture. Does not automatically set resource barriers.
     */
    bool Blit(ID3D12GraphicsCommandList* cmdList, ID3D12Resource* srcTex, ID3D12Resource* dstTex,
              DirectX::XMFLOAT2 dim = {}) const;

  private:
    struct Impl;

    std::unique_ptr<Impl> m_impl;
    std::string_view m_InstanceName;
    bool m_IsInitialized;
};
