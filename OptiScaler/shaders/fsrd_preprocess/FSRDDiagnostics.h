#pragma once

// FSR-RR troubleshooting data (24 Sep 2026).
//
// Written by the shim on the render thread (FSRDFeatureDx12, FSRDPreprocessor_Dx12), read by the
// menu on the present thread. Everything in Diagnostics is guarded by Diagnostics::Mutex; readers
// copy what they need under the lock and render from the copy.
//
// Nothing in here changes what the shim does. The A/B switches that do live in Config.h
// (FfxDenoiserAb*), all off by default.

#include <array>
#include <cstdint>
#include <deque>
#include <dxgiformat.h>
#include <mutex>
#include <string>
#include <vector>

namespace FSRD
{
namespace Probe
{
// float4 per record. Must match PROBE_SLOT_COUNT in FSRDInputConv.hlsl.
constexpr uint32_t kSlotCount = 16;

// Largest half-width of the probe window: 9 x 9 pixels.
constexpr int kMaxRadius = 4;
constexpr uint32_t kMaxRecords = (2 * kMaxRadius + 1) * (2 * kMaxRadius + 1);

// Frames between recording a readback copy and mapping it. There is no fence here, so this is
// the margin for the GPU to have finished; a late frame only means one stale readout.
constexpr uint32_t kRingSize = 6;

// Record layout. Must match the PROBE_* defines in FSRDInputConv.hlsl.
enum Slot : uint32_t
{
    RawColor = 0,  // raw colour rgb, raw luminance
    RawDiffAlbedo, // input diffuse albedo rgb, input bias mask
    RawSpecAlbedo, // input specular albedo rgb, albedo sum (the emissive test's input)
    Roughness,     // input roughness, after exponent, emissive score, emissive applied
    HitDist,       // input hit distance, hit distance sent, gate, 1 = skip path
    GateTerms,     // gate roughness term, emissive term, bias term, roughness sent
    SpecUsed,      // specular albedo used (after override + clamp) rgb, bias weight
    DiffUsed,      // diffuse albedo used (after override + clamp) rgb, floor similarity
    FloorIn,       // floor from the a-trous filter rgb, its luminance
    FloorUsed,     // floor after isolation/guard/bias/soft min rgb, denoiser fraction
    DenoiserColor, // raw - floor rgb, demodulation gain
    DemodSpec,     // signal 1 rgb (demodulated specular), specular split fraction
    DemodDiff,     // signal 2 rgb (demodulated diffuse), signal 2 alpha
    SkipOut,       // skip signal rgb as written, alpha as written
    DepthMotion,   // linear depth, compressed depth, motion x, motion y (input units)
    Geometry,      // depth delta, input normal length, N.V, residual luminance
};
static_assert(Geometry + 1 == kSlotCount, "Probe slot layout out of sync");

// Windows copied straight out of the shim's own textures after composition.
enum Box : uint32_t
{
    DenoisedSpec = 0, // denoiser output 1 (demodulated specular)
    DenoisedDiff,     // denoiser output 2 (demodulated diffuse)
    Composited,       // composition output, the colour handed to the upscaler
    SpecAlbedoTex,    // specular albedo as stored for the denoiser and remodulation (8 or 16 bit)
    DiffAlbedoTex,    // diffuse albedo as stored for the denoiser and remodulation (8 or 16 bit)
    BoxCount
};

struct Stats
{
    float mean[4] = {};    // spatial mean over the window, this readout
    float min[4] = {};     // spatial min over the window
    float max[4] = {};     // spatial max over the window
    float emaMean[4] = {}; // temporal average of the spatial mean
    float emaStd[4] = {};  // temporal standard deviation of the spatial mean
};
} // namespace Probe

struct ProbeReadout
{
    bool valid = false;    // slots hold data
    bool hasBoxes = false; // boxes hold data (composition ran that frame)
    uint64_t frame = 0;    // conversion frame the data was recorded on
    int centerX = 0;
    int centerY = 0;
    int radius = 0;
    uint32_t renderWidth = 0;
    uint32_t renderHeight = 0;
    uint32_t convFlags = 0;  // conversion flags that frame (debug mode, A/B bits)
    uint32_t emaSamples = 0; // readouts folded into the temporal average so far
    uint32_t boxFormats[Probe::BoxCount] = {};
    Probe::Stats slots[Probe::kSlotCount];
    Probe::Stats boxes[Probe::BoxCount];
};

// One of the six runtime tuning values denoiser 1.2 exposes (FfxApiConfigureDenoiserKey 1..6).
struct TuningEntry
{
    const char* name = "";
    float sdkDefault = 0.0f; // what the SDK reports as its own default
    int queryCode = -1;      // ffxReturnCode_t of that query; -1 = not queried yet
    float applied = 0.0f;    // value currently in force (the default until something is configured)
    int applyCode = -1;      // ffxReturnCode_t of the last configure call; -1 = never configured
};

struct InputInfo
{
    std::string name;
    bool present = false;
    uint32_t format = 0; // DXGI_FORMAT
    uint64_t width = 0;
    uint32_t height = 0;
};

struct FrameInfo
{
    uint64_t frame = 0;
    uint32_t renderWidth = 0;
    uint32_t renderHeight = 0;
    uint32_t targetWidth = 0;
    uint32_t targetHeight = 0;

    // Camera, as the shim derives it from the game's matrices
    float nearPlane = 0.0f;
    float farPlane = 0.0f;
    bool infiniteFar = false;
    bool rightHanded = false; // from the projection's w row (GetViewPlanes)
    bool depthInverted = false;
    bool hwDepth = false;
    bool roughnessPacked = false;
    bool reset = false;
    float fovVerticalDeg = 0.0f;
    float projA = 0.0f; // projection z row / w row terms, as in the one-time "Camera:" log line
    float projB = 0.0f;
    float projW = 0.0f;
    float camPos[3] = {};
    float jitterPx[2] = {};
    float mvScale[2] = {};

    // Denoiser
    std::string denoiserVersion;
    uint32_t signalFlags = 0;
    uint32_t createFlags = 0;
    uint32_t dispatchFlags = 0;
    std::string signalText;    // which 1.2 bucket each channel goes to
    std::string createText;    // debugging / validation
    std::string dispatchText;  // non-gamma albedo / reset
    int lastDispatchCode = -1; // ffxReturnCode_t, -1 = not dispatched this frame (bypassed)
    bool denoiseBypassed = false;
    bool upscaleBypassed = false;
    bool declaredStatesFixed = false; // A/B finding 5 in force this frame
    bool albedo16 = false;            // albedo textures created as RGBA16F (A/B finding 3)
    bool messageCallback = false;     // runtime message callback accepted by the denoiser DLL

    // Frame index continuity as the shim sees it (24 Sep). 1.2 warns "Frame index jump detected" when a
    // dispatch's index isn't the previous one + 1; these count the cases the shim itself causes.
    uint64_t dispatches = 0;
    uint64_t indexGaps = 0;     // index moved by more than 1: frames on which the denoiser didn't run
    uint32_t lastGapFrames = 0; // frames skipped before the most recent gap
    uint64_t contextStarts = 0; // first dispatch on a newly created context
    uint64_t gameResets = 0;    // dispatches carrying the game's reset flag
};

struct RuntimeMessage
{
    std::string text;
    uint32_t type = 0; // FFX_API_MESSAGE_TYPE_*
    uint64_t count = 0;
};

class Diagnostics
{
  public:
    static Diagnostics& Instance()
    {
        static Diagnostics instance;
        return instance;
    }

    std::mutex Mutex; // guards everything below

    ProbeReadout Probe;
    std::array<TuningEntry, 6> Tuning;
    std::vector<InputInfo> Inputs;
    FrameInfo Frame;

    std::deque<RuntimeMessage> Messages; // unique messages, oldest first
    uint64_t MessagesTotal = 0;

    // Returns true the first time a message text is seen, so the caller logs each one once.
    bool AddMessage(const std::string& text, uint32_t type)
    {
        std::scoped_lock lock(Mutex);
        MessagesTotal++;

        for (auto& message : Messages)
        {
            if (message.text == text)
            {
                message.count++;
                return false;
            }
        }

        Messages.push_back({ text, type, 1 });

        while (Messages.size() > 48)
            Messages.pop_front();

        return true;
    }
};

inline const char* ReturnCodeName(int code)
{
    switch (code)
    {
    case -1:
        return "-";
    case 0:
        return "OK";
    case 1:
        return "ERROR";
    case 2:
        return "UNKNOWN_DESCTYPE";
    case 3:
        return "RUNTIME_ERROR";
    case 4:
        return "NO_PROVIDER";
    case 5:
        return "ERROR_MEMORY";
    case 6:
        return "ERROR_PARAMETER";
    default:
        return "?";
    }
}

inline const char* FormatName(uint32_t format)
{
    switch ((DXGI_FORMAT) format)
    {
    case DXGI_FORMAT_UNKNOWN:
        return "UNKNOWN";
    case DXGI_FORMAT_R32G32B32A32_TYPELESS:
        return "R32G32B32A32_TYPELESS";
    case DXGI_FORMAT_R32G32B32A32_FLOAT:
        return "R32G32B32A32_FLOAT";
    case DXGI_FORMAT_R32G32B32_FLOAT:
        return "R32G32B32_FLOAT";
    case DXGI_FORMAT_R16G16B16A16_TYPELESS:
        return "R16G16B16A16_TYPELESS";
    case DXGI_FORMAT_R16G16B16A16_FLOAT:
        return "R16G16B16A16_FLOAT";
    case DXGI_FORMAT_R16G16B16A16_UNORM:
        return "R16G16B16A16_UNORM";
    case DXGI_FORMAT_R16G16B16A16_SNORM:
        return "R16G16B16A16_SNORM";
    case DXGI_FORMAT_R32G32_TYPELESS:
        return "R32G32_TYPELESS";
    case DXGI_FORMAT_R32G32_FLOAT:
        return "R32G32_FLOAT";
    case DXGI_FORMAT_R32G8X24_TYPELESS:
        return "R32G8X24_TYPELESS";
    case DXGI_FORMAT_D32_FLOAT_S8X24_UINT:
        return "D32_FLOAT_S8X24_UINT";
    case DXGI_FORMAT_R32_FLOAT_X8X24_TYPELESS:
        return "R32_FLOAT_X8X24_TYPELESS";
    case DXGI_FORMAT_R10G10B10A2_TYPELESS:
        return "R10G10B10A2_TYPELESS";
    case DXGI_FORMAT_R10G10B10A2_UNORM:
        return "R10G10B10A2_UNORM";
    case DXGI_FORMAT_R11G11B10_FLOAT:
        return "R11G11B10_FLOAT";
    case DXGI_FORMAT_R8G8B8A8_TYPELESS:
        return "R8G8B8A8_TYPELESS";
    case DXGI_FORMAT_R8G8B8A8_UNORM:
        return "R8G8B8A8_UNORM";
    case DXGI_FORMAT_R8G8B8A8_UNORM_SRGB:
        return "R8G8B8A8_UNORM_SRGB";
    case DXGI_FORMAT_R8G8B8A8_SNORM:
        return "R8G8B8A8_SNORM";
    case DXGI_FORMAT_R16G16_TYPELESS:
        return "R16G16_TYPELESS";
    case DXGI_FORMAT_R16G16_FLOAT:
        return "R16G16_FLOAT";
    case DXGI_FORMAT_R16G16_UNORM:
        return "R16G16_UNORM";
    case DXGI_FORMAT_R16G16_SNORM:
        return "R16G16_SNORM";
    case DXGI_FORMAT_R32_TYPELESS:
        return "R32_TYPELESS";
    case DXGI_FORMAT_D32_FLOAT:
        return "D32_FLOAT";
    case DXGI_FORMAT_R32_FLOAT:
        return "R32_FLOAT";
    case DXGI_FORMAT_R32_UINT:
        return "R32_UINT";
    case DXGI_FORMAT_R24G8_TYPELESS:
        return "R24G8_TYPELESS";
    case DXGI_FORMAT_D24_UNORM_S8_UINT:
        return "D24_UNORM_S8_UINT";
    case DXGI_FORMAT_R24_UNORM_X8_TYPELESS:
        return "R24_UNORM_X8_TYPELESS";
    case DXGI_FORMAT_R8G8_UNORM:
        return "R8G8_UNORM";
    case DXGI_FORMAT_R16_TYPELESS:
        return "R16_TYPELESS";
    case DXGI_FORMAT_R16_FLOAT:
        return "R16_FLOAT";
    case DXGI_FORMAT_D16_UNORM:
        return "D16_UNORM";
    case DXGI_FORMAT_R16_UNORM:
        return "R16_UNORM";
    case DXGI_FORMAT_R8_TYPELESS:
        return "R8_TYPELESS";
    case DXGI_FORMAT_R8_UNORM:
        return "R8_UNORM";
    case DXGI_FORMAT_R8_UINT:
        return "R8_UINT";
    case DXGI_FORMAT_B8G8R8A8_UNORM:
        return "B8G8R8A8_UNORM";
    case DXGI_FORMAT_B8G8R8A8_TYPELESS:
        return "B8G8R8A8_TYPELESS";
    case DXGI_FORMAT_R9G9B9E5_SHAREDEXP:
        return "R9G9B9E5_SHAREDEXP";
    default:
        return nullptr; // caller prints the number
    }
}
} // namespace FSRD
