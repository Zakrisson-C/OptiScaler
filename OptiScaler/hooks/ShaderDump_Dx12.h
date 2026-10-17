#pragma once
#include <d3d12.h>

// Debug aid for the NRD route (17 Oct 2026, fsrd-debug-tools, default off).
//
// Writes every compute shader the process creates to a ShaderDump folder next to OptiScaler.dll, one file per
// unique shader, named by the container hash (bytes 4..19 of the blob, the same 16 bytes a PSO-creation hook sees),
// plus manifest.csv with the creating API and calling module. Feed the folder to shader_blob_scan.py to label
// NRD's passes by thread-group size and register layout.
//
// Covers ID3D12Device::CreateComputePipelineState, ID3D12Device2::CreatePipelineState (pipeline-state streams)
// and ID3D12PipelineLibrary(1)::LoadComputePipeline / LoadPipeline (PSOs restored from an on-disk cache still
// pass their shader bytecode). Pass-through only: the original is always called first and its result returned
// unchanged.
//
// Enabled by [FSR-RR] DumpComputeShaders=true, read once when the first D3D12 device is hooked. Off = no hooks.
class ShaderDump_Dx12
{
  public:
    static void HookDevice(ID3D12Device* device);
};
