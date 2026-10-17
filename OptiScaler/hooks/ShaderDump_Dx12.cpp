#include "pch.h"
#include "ShaderDump_Dx12.h"

#include <Util.h>
#include <Config.h>

#include <detours/detours.h>
#include <d3dx/d3dx12.h>

#include "Hook_Utils.h"

#include <intrin.h>

#include <algorithm>
#include <atomic>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <string>
#include <unordered_set>

#pragma intrinsic(_ReturnAddress)

// Vtable slots, checked against DirectX-Headers' C interface (offsetof in ID3D12Device*Vtbl /
// ID3D12PipelineLibrary*Vtbl) on 17 Oct 2026: 11, 47, 10, 13.
using PFN_CreateComputePipelineState = rewrite_signature<decltype(&ID3D12Device::CreateComputePipelineState)>::type;
using PFN_CreatePipelineState = rewrite_signature<decltype(&ID3D12Device2::CreatePipelineState)>::type;
using PFN_LoadComputePipeline = rewrite_signature<decltype(&ID3D12PipelineLibrary::LoadComputePipeline)>::type;
using PFN_LoadPipeline = rewrite_signature<decltype(&ID3D12PipelineLibrary1::LoadPipeline)>::type;

static PFN_CreateComputePipelineState o_CreateComputePipelineState = nullptr;
static PFN_CreatePipelineState o_CreatePipelineState = nullptr;
static PFN_LoadComputePipeline o_LoadComputePipeline = nullptr;
static PFN_LoadPipeline o_LoadPipeline = nullptr;

static std::mutex hookMutex;   // HookDevice
static std::mutex recordMutex; // everything below
static std::atomic<bool> enabled = false;
static std::filesystem::path dumpDir;
static std::ofstream manifest;
static std::string session;
static std::unordered_set<uint64_t> seen;
static uint64_t calls = 0;
static uint64_t unique = 0;
static uint64_t written = 0;
static uint64_t writeFailures = 0;

// --- Blob naming --------------------------------------------------------------------------------------------------

static uint64_t Fnv1a64(const uint8_t* data, size_t size)
{
    uint64_t hash = 0xCBF29CE484222325ull;

    for (size_t i = 0; i < size; i++)
    {
        hash ^= data[i];
        hash *= 0x100000001B3ull;
    }

    return hash;
}

static void AppendHex(std::string& out, const uint8_t* data, size_t size)
{
    static const char digits[] = "0123456789abcdef";

    for (size_t i = 0; i < size; i++)
    {
        out += digits[data[i] >> 4];
        out += digits[data[i] & 0xF];
    }
}

struct BlobName
{
    std::string file; // <32 hex>.dxil | .dxbc | .bin
    uint64_t key = 0; // dedup key
    size_t size = 0;  // bytes to write
};

// A DXBC container: "DXBC", 16-byte digest, u16 major, u16 minor, u32 total size, u32 part count, u32 offsets[].
// The digest is what the validator (DXIL) or compiler (FXC) signs, and what a PSO hook reads at bytes 4..19.
static BlobName NameBlob(const uint8_t* data, size_t length)
{
    BlobName name;
    name.size = length;

    const char* ext = ".bin";
    bool isContainer = length >= 32 && std::memcmp(data, "DXBC", 4) == 0;

    if (isContainer)
    {
        uint32_t total = 0;
        uint32_t parts = 0;
        std::memcpy(&total, data + 24, 4);
        std::memcpy(&parts, data + 28, 4);

        if (total >= 32 && total <= length)
            name.size = total;

        for (uint32_t i = 0; i < parts && i < 64 && 32 + 4 * (i + 1) <= name.size; i++)
        {
            uint32_t offset = 0;
            std::memcpy(&offset, data + 32 + 4 * i, 4);

            if (static_cast<uint64_t>(offset) + 8 > name.size)
                continue;

            const char* fourcc = reinterpret_cast<const char*>(data + offset);

            if (std::memcmp(fourcc, "DXIL", 4) == 0 || std::memcmp(fourcc, "ILDB", 4) == 0)
                ext = ".dxil";
            else if ((std::memcmp(fourcc, "SHEX", 4) == 0 || std::memcmp(fourcc, "SHDR", 4) == 0) &&
                     std::strcmp(ext, ".dxil") != 0)
                ext = ".dxbc";
        }
    }

    const uint8_t* digest = data + 4;
    bool digestSet = false;

    if (isContainer)
    {
        for (int i = 0; i < 16; i++)
            digestSet |= digest[i] != 0;
    }

    if (digestSet)
    {
        uint64_t lo = 0;
        uint64_t hi = 0;
        std::memcpy(&lo, digest, 8);
        std::memcpy(&hi, digest + 8, 8);
        name.key = lo ^ (hi * 0x9E3779B97F4A7C15ull);
        AppendHex(name.file, digest, 16);
    }
    else
    {
        // Unsigned container or not a container at all: hash the bytes ourselves
        name.key = Fnv1a64(data, name.size);
        name.file = "fnv_";
        AppendHex(name.file, reinterpret_cast<const uint8_t*>(&name.key), 8);
    }

    name.file += ext;
    return name;
}

// --- Caller ---------------------------------------------------------------------------------------------------------

static bool IsPassThroughModule(const std::string& module)
{
    // Layers between the real caller and the D3D12 runtime: Streamline's proxy device and the runtime itself
    return _stricmp(module.c_str(), "sl.interposer.dll") == 0 || _strnicmp(module.c_str(), "d3d12", 5) == 0;
}

// The module that asked for the PSO. If the immediate caller is a pass-through layer (the game talking to the device
// through Streamline's proxy), walk up the stack to the first module that isn't, and report both.
static std::string CallerModule(void* returnAddress)
{
    std::string immediate = Util::WhoIsTheCaller(returnAddress);

    if (!IsPassThroughModule(immediate))
        return immediate.empty() ? "unknown" : immediate;

    void* frames[48] = {};
    const USHORT count = CaptureStackBackTrace(0, 48, frames, nullptr);
    bool pastImmediate = false;

    for (USHORT i = 0; i < count; i++)
    {
        if (!pastImmediate)
        {
            pastImmediate = frames[i] == returnAddress;
            continue;
        }

        std::string module = Util::WhoIsTheCaller(frames[i]);

        if (!module.empty() && !IsPassThroughModule(module))
            return module + " via " + immediate;
    }

    return immediate;
}

// --- Recording ------------------------------------------------------------------------------------------------------

static void Record(const D3D12_SHADER_BYTECODE& cs, const char* api, void* returnAddress)
{
    if (!enabled.load(std::memory_order_acquire) || cs.pShaderBytecode == nullptr || cs.BytecodeLength < 20)
        return;

    // A debug aid must never take PSO creation down with it
    try
    {
        const auto* data = static_cast<const uint8_t*>(cs.pShaderBytecode);
        const BlobName name = NameBlob(data, cs.BytecodeLength);
        uint64_t order = 0;

        {
            std::lock_guard<std::mutex> lock(recordMutex);
            calls++;

            if (!seen.insert(name.key).second)
                return;

            order = ++unique;
        }

        // Same file from an earlier session: keep it, but still list it in this session's manifest
        const auto path = dumpDir / name.file;
        std::error_code ec;
        const bool existed = std::filesystem::exists(path, ec);
        bool ok = true;

        if (!existed)
        {
            std::ofstream file(path, std::ios::binary | std::ios::trunc);
            file.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(name.size));
            ok = file.good();
        }

        std::string caller = CallerModule(returnAddress);
        std::replace(caller.begin(), caller.end(), ',', ';');

        std::lock_guard<std::mutex> lock(recordMutex);

        if (!existed && ok)
            written++;
        else if (!ok)
            writeFailures++;

        if (manifest.is_open())
        {
            manifest << session << ',' << order << ',' << name.file << ',' << name.size << ',' << api << ',' << caller
                     << '\n';
            manifest.flush();
        }

        if (!ok && writeFailures == 1)
            LOG_ERROR("Shader dump: could not write {}", wstring_to_string(path.wstring()));

        if (order == 1 || order % 250 == 0)
        {
            LOG_INFO("Shader dump: {} unique compute shaders so far ({} new files, {} PSO calls), latest from {}",
                     order, written, calls, caller);
        }
    }
    catch (...)
    {
    }
}

// Compute PSOs built from a pipeline-state stream carry the shader in a CS subobject
struct ComputeShaderFinder : ID3DX12PipelineParserCallbacks
{
    D3D12_SHADER_BYTECODE cs {};
    void CSCb(const D3D12_SHADER_BYTECODE& shader) override { cs = shader; }
};

static void RecordStream(const D3D12_PIPELINE_STATE_STREAM_DESC& desc, const char* api, void* returnAddress)
{
    if (!enabled.load(std::memory_order_acquire))
        return;

    try
    {
        ComputeShaderFinder finder;

        // The parser stops at subobject types it doesn't know (newer than this d3dx12.h); a CS seen before that
        // still counts, and graphics streams simply have none.
        D3DX12ParsePipelineStream(desc, &finder);

        if (finder.cs.pShaderBytecode != nullptr)
            Record(finder.cs, api, returnAddress);
    }
    catch (...)
    {
    }
}

// --- Hooks ----------------------------------------------------------------------------------------------------------

static HRESULT STDMETHODCALLTYPE hkCreateComputePipelineState(ID3D12Device* device,
                                                              const D3D12_COMPUTE_PIPELINE_STATE_DESC* pDesc,
                                                              REFIID riid, void** ppPipelineState)
{
    const HRESULT result = o_CreateComputePipelineState(device, pDesc, riid, ppPipelineState);

    if (SUCCEEDED(result) && pDesc != nullptr)
        Record(pDesc->CS, "CreateComputePipelineState", _ReturnAddress());

    return result;
}
VALIDATE_MEMBER_HOOK(hkCreateComputePipelineState, PFN_CreateComputePipelineState)

static HRESULT STDMETHODCALLTYPE hkCreatePipelineState(ID3D12Device2* device,
                                                       const D3D12_PIPELINE_STATE_STREAM_DESC* pDesc, REFIID riid,
                                                       void** ppPipelineState)
{
    const HRESULT result = o_CreatePipelineState(device, pDesc, riid, ppPipelineState);

    if (SUCCEEDED(result) && pDesc != nullptr)
        RecordStream(*pDesc, "CreatePipelineState", _ReturnAddress());

    return result;
}
VALIDATE_MEMBER_HOOK(hkCreatePipelineState, PFN_CreatePipelineState)

static HRESULT STDMETHODCALLTYPE hkLoadComputePipeline(ID3D12PipelineLibrary* library, LPCWSTR pName,
                                                       const D3D12_COMPUTE_PIPELINE_STATE_DESC* pDesc, REFIID riid,
                                                       void** ppPipelineState)
{
    const HRESULT result = o_LoadComputePipeline(library, pName, pDesc, riid, ppPipelineState);

    if (SUCCEEDED(result) && pDesc != nullptr)
        Record(pDesc->CS, "LoadComputePipeline", _ReturnAddress());

    return result;
}
VALIDATE_MEMBER_HOOK(hkLoadComputePipeline, PFN_LoadComputePipeline)

static HRESULT STDMETHODCALLTYPE hkLoadPipeline(ID3D12PipelineLibrary1* library, LPCWSTR pName,
                                                const D3D12_PIPELINE_STATE_STREAM_DESC* pDesc, REFIID riid,
                                                void** ppPipelineState)
{
    const HRESULT result = o_LoadPipeline(library, pName, pDesc, riid, ppPipelineState);

    if (SUCCEEDED(result) && pDesc != nullptr)
        RecordStream(*pDesc, "LoadPipeline", _ReturnAddress());

    return result;
}
VALIDATE_MEMBER_HOOK(hkLoadPipeline, PFN_LoadPipeline)

// --- Setup ----------------------------------------------------------------------------------------------------------

static bool OpenDumpFolder()
{
    std::error_code ec;
    dumpDir = Util::DllPath().parent_path() / "ShaderDump";
    std::filesystem::create_directories(dumpDir, ec);

    if (ec)
    {
        LOG_ERROR("Shader dump: can't create {}: {}", wstring_to_string(dumpDir.wstring()), ec.message());
        return false;
    }

    const auto manifestPath = dumpDir / "manifest.csv";
    const bool newManifest = !std::filesystem::exists(manifestPath, ec);
    manifest.open(manifestPath, std::ios::out | std::ios::app);

    if (!manifest.is_open())
    {
        LOG_ERROR("Shader dump: can't open {}", wstring_to_string(manifestPath.wstring()));
        return false;
    }

    if (newManifest)
        manifest << "session,order,file,bytes,api,caller\n";

    SYSTEMTIME now {};
    GetLocalTime(&now);
    char stamp[32] = {};
    snprintf(stamp, sizeof(stamp), "%04u%02u%02u-%02u%02u%02u", (unsigned) now.wYear, (unsigned) now.wMonth,
             (unsigned) now.wDay, (unsigned) now.wHour, (unsigned) now.wMinute, (unsigned) now.wSecond);
    session = stamp;

    return true;
}

// One transaction per hook: a failed attach aborts its whole transaction, and losing the library hooks
// shouldn't cost the main one
template <typename T> static bool Attach(T& original, T hook, const char* name)
{
    if (original == nullptr)
        return false;

    DetourTransactionBegin();
    DetourUpdateThread(GetCurrentThread());
    DetourAttach(&(PVOID&) original, reinterpret_cast<PVOID>(hook));
    const auto result = DetourTransactionCommit();

    if (result != NO_ERROR)
    {
        LOG_ERROR("Shader dump: failed to detour {}, error: {:X}", name, (UINT) result);
        original = nullptr;
        return false;
    }

    return true;
}

void ShaderDump_Dx12::HookDevice(ID3D12Device* device)
{
    if (device == nullptr || !Config::Instance()->DumpComputeShaders.value_or_default())
        return;

    std::lock_guard<std::mutex> lock(hookMutex);

    static bool attempted = false;

    if (attempted)
        return;

    attempted = true;

    // This runs inside the game's D3D12CreateDevice: nothing may escape from here
    try
    {
        if (!OpenDumpFolder())
            return;

        // Detour the real implementation: behind Streamline the game's device is a proxy that forwards to it
        ID3D12Device* realDevice = nullptr;

        if (!Util::CheckForRealObject(__FUNCTION__, device, (IUnknown**) &realDevice))
            realDevice = device;

        o_CreateComputePipelineState = (PFN_CreateComputePipelineState) (*(PVOID**) realDevice)[11];

        ID3D12Device2* device2 = nullptr;

        if (SUCCEEDED(realDevice->QueryInterface(IID_PPV_ARGS(&device2))) && device2 != nullptr)
        {
            o_CreatePipelineState = (PFN_CreatePipelineState) (*(PVOID**) device2)[47];
            device2->Release();
        }

        // The library methods are only reachable through a library object; an empty one is enough to find them
        ID3D12Device1* device1 = nullptr;

        if (SUCCEEDED(realDevice->QueryInterface(IID_PPV_ARGS(&device1))) && device1 != nullptr)
        {
            ID3D12PipelineLibrary* library = nullptr;

            if (SUCCEEDED(device1->CreatePipelineLibrary(nullptr, 0, IID_PPV_ARGS(&library))) && library != nullptr)
            {
                o_LoadComputePipeline = (PFN_LoadComputePipeline) (*(PVOID**) library)[10];

                ID3D12PipelineLibrary1* library1 = nullptr;

                if (SUCCEEDED(library->QueryInterface(IID_PPV_ARGS(&library1))) && library1 != nullptr)
                {
                    o_LoadPipeline = (PFN_LoadPipeline) (*(PVOID**) library1)[13];
                    library1->Release();
                }

                library->Release();
            }
            else
            {
                LOG_INFO("Shader dump: pipeline libraries not supported here, so the game can't use them either");
            }

            device1->Release();
        }
    }
    catch (...)
    {
        LOG_ERROR("Shader dump: setup failed, not hooking");
        return;
    }

    // Recording is armed before the hooks go live
    enabled.store(true, std::memory_order_release);

    const bool compute =
        Attach(o_CreateComputePipelineState, hkCreateComputePipelineState, "CreateComputePipelineState");
    const bool stream = Attach(o_CreatePipelineState, hkCreatePipelineState, "CreatePipelineState");
    const bool libraryCompute = Attach(o_LoadComputePipeline, hkLoadComputePipeline, "LoadComputePipeline");
    const bool libraryStream = Attach(o_LoadPipeline, hkLoadPipeline, "LoadPipeline");

    LOG_INFO("Shader dump: on (session {}), writing compute shaders to {}. Hooked: CreateComputePipelineState {}, "
             "CreatePipelineState {}, LoadComputePipeline {}, LoadPipeline {}",
             session, wstring_to_string(dumpDir.wstring()), compute, stream, libraryCompute, libraryStream);
}
