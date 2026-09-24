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
 * dropped: IFeature::Name() is non-virtual now and derives from GetUpscalerType(). (23 Sep
 * correction: that makes GetUpscalerType() the hook, not a dead end - FFXFeatureDx12 had it
 * `final`, which is now relaxed so this class can report Upscaler::FSRD, see below.) Evaluate() is renamed to
 * EvaluateInternal(): IFeature_Dx12::Evaluate() is now a fixed, non-overridable template-method entry point that runs
 * the shared RCAS/OutputScaling/Magnifier post-process pipeline and GPU timing around whatever
 * EvaluateInternal() does - see FSRDFeature_Dx12.cpp for how the old PrepareUpscalerInput /
 * DispatchUpscaler / PostProcess / SetConfigurableBarriers calls collapse into one call to
 * FFXFeatureDx12::EvaluateInternal().
 *
 * Phase 3 note (transplant, 22 Sep): denoiser 1.2 has no "Mode 1" (single combined signal) concept -
 * see the transplant plan §6e/6f. Mode 1 is removed outright (Chris's call, not preserved as a
 * preset); the shim now always dispatches two split signals, diffuse and specular. (23 Sep: which
 * 1.2 bucket each goes to - DIRECT or INDIRECT - is selectable, see DesiredSignalFlags().)
 */
class FSRDFeatureDx12 : public FFXFeatureDx12
{
  public:
    using FSRDConvDesc = FSRDPreprocessor_Dx12::ConversionDesc;

    FSRDFeatureDx12(uint32_t InHandleId, NVSDK_NGX_Parameter* InParameters);

    ~FSRDFeatureDx12();

    feature_version Version() override { return FFXFeatureDx12::Version(); }

    // Transplant fix, 23 Sep: without this FSR-RR inherits FFXFeatureDx12's Upscaler::FFX, so
    // IFeature::Name() reads "FSR" (shown in game as "FSR 1.2", the denoiser's version) and the
    // menu's usesFsrd gate never opens: no FSR-RR settings, no debug views. The fork got its
    // display name from overriding Name(), which master made non-virtual; the type is the hook now.
    Upscaler GetUpscalerType() const override { return Upscaler::FSRD; }

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
            return std::clamp((int) key - 1, 0, (int) DenoiserConfiguration::kCount - 1);
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
    DenoiserConfiguration _denoiserSettings; // values currently in force in the denoiser context

    // Troubleshooting (24 Sep, see shaders/fsrd_preprocess/FSRDDiagnostics.h)
    DenoiserConfiguration _sdkDefaults {}; // denoiser 1.2's own defaults, as queried at context creation
    std::array<int, DenoiserConfiguration::kCount> _sdkDefaultCodes {}; // ffxReturnCode_t of each query
    std::array<int, DenoiserConfiguration::kCount> _applyCodes {};      // last configure result, -1 = never
    bool _albedo16AtCreate = false;  // A/B finding 3 as it was when the converter was created
    bool _messageCallbackOk = false; // the denoiser DLL accepted the runtime message callback
    int _lastDispatchCode = -1;      // ffxReturnCode_t of this frame's dispatch, -1 = not dispatched
    uint32_t _lastDispatchFlags = 0; // dispatchDesc.flags of the last dispatch
    float _lastProjTerms[3] = {};    // projection A, B, W as GetViewPlanes read them
    bool _lastInfiniteFar = false;
    bool _lastRightHanded = false;
    bool _lastDeclaredStates = false;

    static bool DesiredAlbedo16();
    void PublishDiagnostics(const NVSDK_NGX_Parameter& inParams, bool denoiseBypassed, bool upscaleBypassed);

    static bool s_isHWDepth;
    static bool s_isRoughnessPacked;

    FSRDConvDesc _convDesc;
    bool _isInReset = false; // Was inherited from the fork's FSR31FeatureDx12; master's FFXFeatureDx12 has none
    bool _loggedCameraConvention = false; // One-time camera/projection convention log (23 Sep)
    bool _loggedDiffuseHitDist = false;   // One-time report of whether the game supplies DLSSD.DiffuseHitDistance
    bool _warnedFfxDebugWithoutFlag = false;

    // Create-time options derived from Config (23 Sep). Compared against _denoiserCtxDesc every frame so a
    // menu change recreates the context instead of needing a resolution change.
    static uint32_t DesiredSignalFlags();
    static uint32_t DesiredCreateFlags();

    DirectX::XMFLOAT3 _lastCamPos; // Last world space camera position

    // Denoiser frame index, advanced exactly once per EvaluateInternal() call. _frameCount can't be used:
    // FFXFeatureDx12::EvaluateInternal() advances it as well, so it moves by 2 per frame whenever the
    // upscaler runs, and 1.2 resets its history on every frame index jump.
    uint32_t _denoiserFrameIndex = 0;

    // Frame index continuity counters for the diagnostics panel (24 Sep), see FSRD::FrameInfo.
    bool _dispatchedOnContext = false; // cleared when a context is created
    uint32_t _lastDispatchedIndex = 0;
    uint64_t _dispatchCount = 0;
    uint64_t _indexGapCount = 0;
    uint32_t _lastGapFrames = 0;
    uint64_t _contextStartCount = 0;
    uint64_t _gameResetCount = 0;

    // Matrices
    DirectX::XMMATRIX _invViewMatrix;  // Camera rotation and translation
    DirectX::XMMATRIX _viewMatrix;     // World to camera space
    DirectX::XMMATRIX _prevViewMatrix; // Last world to camera space
    DirectX::XMMATRIX _projMatrix;     // Perspective projection matrix
    bool _isRightHanded;               // True if the camera matrix is right handed

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