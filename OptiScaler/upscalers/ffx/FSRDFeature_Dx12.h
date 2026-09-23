#pragma once
#include "FFXFeature_Dx12.h"
#include "shaders/fsrd_preprocess/FSRDPreprocessor_Dx12.h"
#include "fsr-rr/ffx_denoiser.h"
#include <DirectXMath.h>

/**
 * @brief Unfied denoiser-upscaler utilising AMD FSR Ray Regeneration and Super Resolution with
 * DLSS-RR inputs. Extends the FFX upscaler implementation (FSR 3.1+).
 *
 * Phase 2 note (transplant, 22 Sep): base class changed from FSR31FeatureDx12 (single inheritance)
 * to FFXFeatureDx12 (public FFXFeature, public IFeature_Dx12 - split inheritance). Name() is
 * dropped entirely: IFeature::Name() is non-virtual now (derives from GetUpscalerType(), which
 * FFXFeatureDx12 locks `final` to Upscaler::FFX) so there is nothing left to override - the
 * existing `_name = OptiTexts::FSR_RR_Name;` assignment in InitFFX (below) is untouched and still
 * the right way to carry the FSR-RR display string. Evaluate() is renamed to EvaluateInternal():
 * IFeature_Dx12::Evaluate() is now a fixed, non-overridable template-method entry point that runs
 * the shared RCAS/OutputScaling/Magnifier post-process pipeline and GPU timing around whatever
 * EvaluateInternal() does - see FSRDFeature_Dx12.cpp for how the old PrepareUpscalerInput /
 * DispatchUpscaler / PostProcess / SetConfigurableBarriers calls collapse into one call to
 * FFXFeatureDx12::EvaluateInternal().
 *
 * Phase 3 note (transplant, 22 Sep): denoiser 1.2 has no "Mode 1" (single combined signal) concept -
 * see the transplant plan §6e/6f. Mode 1 is removed outright (Chris's call, not preserved as a
 * preset); the shim now always dispatches the split indirect-diffuse/indirect-specular signals.
 */
class FSRDFeatureDx12 : public FFXFeatureDx12
{
  public:
    using FSRDConvDesc = FSRDPreprocessor_Dx12::ConversionDesc;

    FSRDFeatureDx12(uint32_t InHandleId, NVSDK_NGX_Parameter* InParameters);

    ~FSRDFeatureDx12();

    feature_version Version() override { return FFXFeatureDx12::Version(); }

    bool EvaluateInternal(ID3D12GraphicsCommandList* InCommandList, NVSDK_NGX_Parameter* InParameters) override;

  private:

    union DenoiserConfiguration
    {
        static constexpr uint32_t kCount = FFX_API_CONFIGURE_DENOISER_KEY_DISOCCLUSION_THRESHOLD;

        // Ordered by FfxApiConfigureDenoiserKey
        struct
        {
            float m_CrossBilateralNormalStrength;
            float m_StabilityBias;
            float m_MaxRadiance;
            float m_RadianceClipStdK;
            float m_GaussianKernelRelaxation;
            float m_DisocclusionThreshold;
        };

        float AsArray[kCount];

        static int GetKeyIndex(FfxApiConfigureDenoiserKey key) 
        {
            return std::clamp((int) key - 1, 0, (int)DenoiserConfiguration::kCount - 1);
        }

        static FfxApiConfigureDenoiserKey GetIndexKey(int index)
        {
            index = std::clamp(index + 1, 1, (int) DenoiserConfiguration::kCount);
            return static_cast<FfxApiConfigureDenoiserKey>(index);
        }

        float& GetMember(int index) { return AsArray[index]; }

        float& GetMember(FfxApiConfigureDenoiserKey key) { return AsArray[GetKeyIndex(key)]; }
    };

    ffxContext _pDenoiserCtx;
    ffxCreateContextDescDenoiser _denoiserCtxDesc;
    DenoiserConfiguration _denoiserSettings;

    static bool s_isHWDepth;
    static bool s_isRoughnessPacked;

    FSRDConvDesc _convDesc;
    DirectX::XMFLOAT3 _lastCamPos; // Last world space camera position

    // Matrices
    DirectX::XMMATRIX _invViewMatrix;   // Camera rotation and translation
    DirectX::XMMATRIX _viewMatrix;      // World to camera space
    DirectX::XMMATRIX _prevViewMatrix;  // Last world to camera space
    DirectX::XMMATRIX _projMatrix;      // Perspective projection matrix
    bool _isRightHanded;                // True if the camera matrix is right handed

    std::unique_ptr<FSRDPreprocessor_Dx12> FSRDConvShader;

    bool InitFFX(const NVSDK_NGX_Parameter* InParameters) override;

    bool CreateDenoiserContext();

    bool QueryDenoiserVersions();

    void DestroyDenoiserContext();

    void UpdateSize();

    /**
     * @brief Generates FFX denoiser configuration and input buffers from DLSS-RR inputs and NGX configurations.
     * Converts and repacks resources internally.
     *
     * Phase 3 note (transplant, 22 Sep): no longer templated over a Mode-1-vs-Mode-2 signal shape -
     * denoiser 1.2 has no combined-signal concept to template over (see the transplant plan §6e/6f).
     * Always populates both split signals, chained together into dispatchDesc.
     */
    bool PrepareDenoiserInput(ID3D12GraphicsCommandList* InCommandList, const NVSDK_NGX_Parameter& ngxParams,
                              ffxDispatchDescDenoiser& dispatchDesc,
                              ffxDispatchDescDenoiserIndirectDiffuse& indirectDiffuseSignal,
                              ffxDispatchDescDenoiserIndirectSpecular& indirectSpecularSignal);

    /**
     * @brief Retrieves DLSS-RR inputs to populate the inputs for the interop layer in order to generate
     FSR-RR compatible buffers.
     */
    bool PrepareDenoiseConvInput(const NVSDK_NGX_Parameter& inParams);

    /**
     * @brief Converts previously retrieved DLSS-RR resources into FSR-RR inputs.
     */
    bool ConvertDenoiserBuffers(ID3D12GraphicsCommandList* InCommandList);

    /**
     * @brief Dispatches FSR-RR denoiser converted inputs. Runs before upscaler.
     */
    bool DispatchDenoiser(ID3D12GraphicsCommandList* InCommandList, const ffxDispatchDescDenoiser& dispatchDesc);

    void SetDefaultConfiguration();

    ffxReturnCode_t SetDefaultConfiguration(FfxApiConfigureDenoiserKey key);

    ffxReturnCode_t ApplyConfiguration(FfxApiConfigureDenoiserKey key);
};