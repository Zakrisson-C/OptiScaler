#pragma once
#include "FFXFeature.h"
#include <upscalers/IFeature_Dx12.h>

#include "dx12/ffx_api_dx12.h"
#include "proxies/FfxApi_Proxy.h"

class FFXFeatureDx12 : public FFXFeature, public IFeature_Dx12
{
  private:
    // 25 Sep: was uninitialised. The destructor SAFE_RELEASEs both and CreateBufferResourceWithSize reads
    // them, so a feature that never took the padded-colour path (every FSR-RR and most FSR 4 features)
    // released whatever the heap had left there on destruction: harmless when that happened to be zero, a
    // crash on switching upscalers otherwise.
    ID3D12Resource* smallerColor[2] = {};

    NVSDK_NGX_Parameter* SetParameters(NVSDK_NGX_Parameter* InParameters);

  protected:
    bool InitFFX(const NVSDK_NGX_Parameter* InParameters);

  public:
    FFXFeatureDx12(unsigned int InHandleId, NVSDK_NGX_Parameter* InParameters);

    bool InitInternal(ID3D12GraphicsCommandList* InCommandList, NVSDK_NGX_Parameter* InParameters) override;
    bool EvaluateInternal(ID3D12GraphicsCommandList* InCommandList, NVSDK_NGX_Parameter* InParameters) override;
    bool QueryProviders(ID3D12Device* device);

    feature_version Version() override { return FFXFeature::Version(); }
    // Not `final` (transplant fix, 23 Sep): FSRDFeatureDx12 derives from this class and must report
    // Upscaler::FSRD, or the menu, Name() and every "is FSR-RR active" check see plain FFX.
    Upscaler GetUpscalerType() const override { return Upscaler::FFX; }
    API Api() const override { return IFeature_Dx12::Api(); }
    bool CallsUpscalerEndByItself() override { return IFeature_Dx12::CallsUpscalerEndByItself(); }

    bool IsWithDx12() final { return false; }

    ~FFXFeatureDx12()
    {
        if (State::Instance().isShuttingDown)
            return;

        if (_context != nullptr)
            FfxApiProxy::D3D12_DestroyContext(&_context, NULL);

        SAFE_RELEASE(smallerColor[0]);
        SAFE_RELEASE(smallerColor[1]);
    }
};
