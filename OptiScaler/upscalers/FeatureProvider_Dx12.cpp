#include "pch.h"
#include "FeatureProvider_Dx12.h"

#include "Util.h"
#include "Config.h"

#include "NVNGX_Parameter.h"

#include "upscalers/dlss/DLSSFeature_Dx12.h"
#include "upscalers/dlssd/DLSSDFeature_Dx12.h"
#include "upscalers/fsr2/FSR2Feature_Dx12.h"
#include "upscalers/fsr2_212/FSR2Feature_Dx12_212.h"
#include "upscalers/ffx/FFXFeature_Dx12.h"
#include "upscalers/ffx/FSRDFeature_Dx12.h"
#include "upscalers/xess/XeSSFeature_Dx12.h"
#include "FeatureProvider_Dx11.h"
#include <misc/IdentifyGpu.h>
// Transplant, 22 Sep (plan §7/Phase 4c): gates FSRD construction on the denoiser module actually
// being loaded, mirroring how DLSSD gates on primaryGpu.dlssCapable/NVNGX_DLSSD_Path above.
#include "proxies/FfxApi_Proxy.h"

bool FeatureProvider_Dx12::GetFeature(Upscaler upscaler, UINT handleId, NVSDK_NGX_Parameter* parameters,
                                      std::unique_ptr<IFeature_Dx12>* feature)
{
    State& state = State::Instance();
    Config& cfg = *Config::Instance();
    auto primaryGpu = IdentifyGpu::getPrimaryGpu();
    ScopedSkipHeapCapture skipHeapCapture {};

    switch (upscaler)
    {
    case Upscaler::XeSS:
        *feature = std::make_unique<XeSSFeatureDx12>(handleId, parameters);
        break;

    case Upscaler::FSR21:
        *feature = std::make_unique<FSR2FeatureDx12_212>(handleId, parameters);
        break;

    case Upscaler::FSR22:
        *feature = std::make_unique<FSR2FeatureDx12>(handleId, parameters);
        break;

    case Upscaler::FFX:
        *feature = std::make_unique<FFXFeatureDx12>(handleId, parameters);
        break;

    case Upscaler::DLSS:
        if (primaryGpu.dlssCapable && state.NVNGX_DLSS_Path.has_value())
        {
            *feature = std::make_unique<DLSSFeatureDx12>(handleId, parameters);
            break;
        }
        else
        {
            *feature = std::make_unique<FSR2FeatureDx12_212>(handleId, parameters);
            upscaler = Upscaler::FSR21;
            break;
        }

    case Upscaler::DLSSD:
        if (primaryGpu.dlssCapable && state.NVNGX_DLSSD_Path.has_value())
        {
            *feature = std::make_unique<DLSSDFeatureDx12>(handleId, parameters);
            break;
        }
        else
        {
            *feature = std::make_unique<FSR2FeatureDx12_212>(handleId, parameters);
            upscaler = Upscaler::FSR21;
            break;
        }

    // Transplant, 22 Sep (plan §7/Phase 4c). The FSRD registration point: this is the only place
    // that constructs FSRDFeatureDx12. Gated on the denoiser module the way IsDenoiserReady() gates
    // FeatureProvider elsewhere in this class, mirroring DLSSD's own gate above. sendNotification is
    // false for the same reason as the NVNGX_Parameter.cpp probe: this can run every backend-select
    // pass, not just on an explicit user action.
    case Upscaler::FSRD:
        if (FfxApiProxy::IsDenoiserReady(false))
        {
            *feature = std::make_unique<FSRDFeatureDx12>(handleId, parameters);
            break;
        }
        else
        {
            *feature = std::make_unique<FSR2FeatureDx12_212>(handleId, parameters);
            upscaler = Upscaler::FSR21;
            break;
        }

    default:
        *feature = std::make_unique<FSR2FeatureDx12_212>(handleId, parameters);
        upscaler = Upscaler::FSR21;
        break;
    }

    bool loaded = (*feature)->ModuleLoaded();

    if (!loaded)
    {
        // Fail after the constructor
        ImGui::InsertNotification({ ImGuiToastType::Warning, 10000, "Falling back to FSR 2.1.2" });
        *feature = std::make_unique<FSR2FeatureDx12_212>(handleId, parameters);
        upscaler = Upscaler::FSR21;
        loaded = true; // Assuming the fallback always loads successfully
    }

    // DLSSD is stored in the config as DLSS
    if (upscaler == Upscaler::DLSSD)
        upscaler = Upscaler::DLSS;

    cfg.Dx12Upscaler = upscaler;

    return loaded;
}

bool FeatureProvider_Dx12::ChangeFeature(Upscaler upscaler, ID3D12Device* device, ID3D12GraphicsCommandList* cmdList,
                                         UINT handleId, NVSDK_NGX_Parameter* parameters,
                                         ContextData<IFeature_Dx12>* contextData)
{
    State& state = State::Instance();
    Config& cfg = *Config::Instance();

    if (!state.changeBackend[handleId])
        return false;

    const bool dlssOnNonCapable = !IdentifyGpu::getPrimaryGpu().dlssCapable && state.newBackend == Upscaler::DLSS;
    if (state.newBackend == Upscaler::Reset || dlssOnNonCapable)
        state.newBackend = cfg.Dx12Upscaler.value_or_default();

    contextData->changeBackendCounter++;

    LOG_INFO("changeBackend is true, counter: {0}", contextData->changeBackendCounter);

    // first release everything
    if (contextData->changeBackendCounter == 1)
    {
        if (state.currentFG != nullptr && state.activeFgInput == FGInput::Upscaler)
        {
            state.fgChanged = true;
            state.clearCapturedHudlesses = true;
        }

        if (contextData->feature != nullptr)
        {
            LOG_INFO("changing backend to {}", UpscalerDisplayName(state.newBackend));

            auto* dc = contextData->feature.get();
            // Use given params if using DLSS passthrough
            const bool isPassthrough = state.newBackend == Upscaler::DLSSD || state.newBackend == Upscaler::DLSS;

            contextData->createParams = isPassthrough ? parameters : GetNGXParameters(API::DX12, false);
            contextData->createParams->Set(NVSDK_NGX_Parameter_DLSS_Feature_Create_Flags, dc->GetFeatureFlags());
            contextData->createParams->Set(NVSDK_NGX_Parameter_Width, dc->RenderWidth());
            contextData->createParams->Set(NVSDK_NGX_Parameter_Height, dc->RenderHeight());
            contextData->createParams->Set(NVSDK_NGX_Parameter_OutWidth, dc->DisplayWidth());
            contextData->createParams->Set(NVSDK_NGX_Parameter_OutHeight, dc->DisplayHeight());
            contextData->createParams->Set(NVSDK_NGX_Parameter_PerfQualityValue, dc->PerfQualityValue());

            dc = nullptr;

            State::Instance().currentFeature = nullptr;

            Util::DelayedDestroy(std::move(contextData->feature));

            // LOG_DEBUG("sleeping before reset of current feature for 1000ms");
            // std::this_thread::sleep_for(std::chrono::milliseconds(1000));

            // contextData->feature.reset();
            // contextData->feature = nullptr;
        }
        else // Clean up state if no feature is set
        {
            LOG_ERROR("can't find handle {0} in Dx12Contexts!", handleId);

            state.newBackend = Upscaler::Reset;
            state.changeBackend[handleId] = false;

            if (contextData->createParams != nullptr)
            {
                TryDestroyNGXParameters(contextData->createParams, NVNGXProxy::D3D12_DestroyParameters());
                contextData->createParams = nullptr;
            }

            contextData->changeBackendCounter = 0;

            return false;
        }

        return true;
    }

    // create new feature
    if (contextData->changeBackendCounter == 2)
    {
        LOG_INFO("Creating new {} upscaler", UpscalerDisplayName(state.newBackend));
        contextData->feature.reset();

        if (!GetFeature(state.newBackend, handleId, contextData->createParams, &contextData->feature))
        {
            LOG_ERROR("Upscaler can't created");
            return false;
        }

        return true;
    }

    // init feature
    if (contextData->changeBackendCounter == 3)
    {
        auto initResult = contextData->feature->Init(device, cmdList, contextData->createParams);

        contextData->changeBackendCounter = 0;

        if (!initResult)
        {
            LOG_ERROR("init failed with {0} feature", UpscalerDisplayName(state.newBackend));

            // Transplant, 22 Sep (plan §7/Phase 4c): FSRD added to the "RR features retry instead of
            // silently becoming an unrelated SR backend" set below, alongside DLSSD. Missing this
            // would downgrade a failed FSRD init to FSR21 SR - a different feature than the game
            // thinks it has - instead of retrying like DLSSD already does.
            if (state.newBackend != Upscaler::DLSSD && state.newBackend != Upscaler::FSRD)
            {
                if (cfg.Dx12Upscaler == Upscaler::DLSS)
                {
                    state.newBackend = Upscaler::XeSS;
                    ImGui::InsertNotification({ ImGuiToastType::Warning, 10000, "Falling back to XeSS" });
                }
                else
                {
                    state.newBackend = Upscaler::FSR21;
                    ImGui::InsertNotification({ ImGuiToastType::Warning, 10000, "Falling back to FSR 2.1.2" });
                }
            }
            else
            {
                // Retry RR feature (DLSSD or FSRD) with the same backend. state.newBackend is
                // already DLSSD or FSRD here (that's the condition that reached this branch) -
                // original code reassigned it to the Upscaler::DLSSD literal, which was a no-op;
                // left as a no-op rather than hardcoding a value that would wrongly force FSRD
                // retries back onto DLSSD.
            }

            state.changeBackend[handleId] = true;

            return false;
        }
        else
        {
            LOG_INFO("init successful for {0}, upscaler changed", UpscalerDisplayName(state.newBackend));

            state.newBackend = Upscaler::Reset;
            state.changeBackend[handleId] = false;
        }

        // If this is an OptiScaler fake NVNGX param table, delete it
        int optiParam = 0;

        if (contextData->createParams->Get("OptiScaler", &optiParam) == NVSDK_NGX_Result_Success && optiParam == 1)
        {
            TryDestroyNGXParameters(contextData->createParams, NVNGXProxy::D3D12_DestroyParameters());
            contextData->createParams = nullptr;
        }
    }

    // if initial feature can't be inited
    state.currentFeature = contextData->feature.get();

    if (state.currentFG != nullptr && state.activeFgInput == FGInput::Upscaler)
        state.currentFG->UpdateTarget();

    return true;
}
