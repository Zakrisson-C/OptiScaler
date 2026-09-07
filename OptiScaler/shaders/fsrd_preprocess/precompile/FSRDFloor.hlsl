#include "FSRDPreprocessCommon.hlsli"

#define MainRS \
    "RootFlags(0), " \
    "CBV(b0), " \
    "DescriptorTable(SRV(t0, numDescriptors = 3), visibility = SHADER_VISIBILITY_ALL), " \
    "DescriptorTable(UAV(u0, numDescriptors = 1), visibility = SHADER_VISIBILITY_ALL), " \
    "StaticSampler(s0, " \
        "filter = FILTER_MIN_MAG_MIP_LINEAR, " \
        "addressU = TEXTURE_ADDRESS_CLAMP, " \
        "addressV = TEXTURE_ADDRESS_CLAMP, " \
        "addressW = TEXTURE_ADDRESS_CLAMP, " \
        "visibility = SHADER_VISIBILITY_ALL)"

// Dispatch config
#define THREAD_GROUP_SIZE_X     8
#define THREAD_GROUP_SIZE_Y     8
#define NUM_THREADS             (THREAD_GROUP_SIZE_X * THREAD_GROUP_SIZE_Y)

static const uint2 s_ThreadGroupSize = uint2(THREAD_GROUP_SIZE_X, THREAD_GROUP_SIZE_Y);

// A-Trous kernel config
#define KERNEL_SIZE             3
#define KERNEL_RANGE_MIN        (-KERNEL_SIZE / 2)
#define KERNEL_RANGE_MAX        (KERNEL_SIZE / 2)

static const float s_Kernel1D[2] = { 0.44198f, 0.27901f };

Texture2D<half4> InColor : register(t0);
Texture2D<float> InLinearDepth : register(t1);

// RG: View space depth gradient, BA: Octahedrally encoded world normal
Texture2D<half4> InDepthGradient : register(t2);

RWTexture2D<half4> OutColor : register(u0);

SamplerState LinearSampler : register(s0);

cbuffer CB_Analysis : register(b0)
{
    float4 DstTexSize;

    float RcpCrossBlNorm;
    float RcpSelfBlNorm;

    int StepSize;
    uint FrameIndex;

    uint Flags;

    // Fraction of the Laplacian luminance residual re-injected into the floor.
    // 0 reproduces the previous behaviour exactly. Only non-zero on the final pass.
    float DetailBoost;

    // Exponent on the normal edge-stopping weight. Higher = harder stop at creases.
    float NormalSharpness;

    float _Padding;
}

bool IsSet(uint mask) { return (Flags & mask) == mask; }

// Ceiling on the plane extrapolation, as a fraction of the centre depth. The gradient
// is a single pixel central difference, so extrapolating it over a 16 pixel stride is
// only trustworthy up to a point - past this the tap is treated as off-plane.
static const float s_MaxPlaneOffset = 0.5f;

float GetSpatialWeight(int x, int y)
{
    return s_Kernel1D[abs(x)] * s_Kernel1D[abs(y)];
}

float GetRangeWeight(float delta, float scale)
{
    // W = ( 1 - ( (center - tap) * scale )^2 )^2
    // scale = 1 / norm
    return Square(max(1.0f - Square(delta * scale), 1e-2f));
}

[RootSignature(MainRS)]
[numthreads(THREAD_GROUP_SIZE_X, THREAD_GROUP_SIZE_Y, 1)]
void CSMain(uint3 groupID : SV_GroupID, uint3 gtID : SV_GroupThreadID)
{
    const int2 px = groupID.xy * s_ThreadGroupSize + gtID.xy;
    
    if (px.x >= DstTexSize.x || px.y >= DstTexSize.y)
    {
        OutColor[px] = half4(0, 0, 0, 0);
        return;
    }
    
    const float4 centerColor = InColor[px];    
    const float centerLum = GetLuminance(centerColor.rgb);
    const float rcpCenterLum = rcp(max(centerLum, 1e-1f));   
    
    const float centerDepth = InLinearDepth[px];
    const half4 centerGuide = InDepthGradient[px];
    const float2 centerDepthGrad = centerGuide.xy;
    const float3 centerNormal = OctahedralDecode(centerGuide.zw);
    const float rcpDepthScale = rcp((1.0f + centerDepth) * float(StepSize));
    const float maxPlaneOffset = s_MaxPlaneOffset * centerDepth;
    
    // As the scaling increases, bilateral weighting becomes stricter. As smoothness increases,
    // blur strength should decrease. Where smoothness remains low, the weights should allow
    // more blending.
    //
    // StepSize scaling keeps range strictness consistent as the stride increases.
    const float smoothness = saturate(1.0f - 2.0f * centerColor.a);
    const float adaptiveScale = float(StepSize) * (1.0f + 2.0f * smoothness);
      
    const float depthNormScale = adaptiveScale * (rcpDepthScale * RcpCrossBlNorm);
    const float selfNormScale = adaptiveScale * RcpSelfBlNorm;
    
    const int2 maxBounds = int2(DstTexSize.xy) - 1;
    float4 mean = 0;
    float totalWeight = 0;
    
    [unroll]
    for (int x = KERNEL_RANGE_MIN; x <= KERNEL_RANGE_MAX; x++)
    {
        [unroll]
        for (int y = KERNEL_RANGE_MIN; y <= KERNEL_RANGE_MAX; y++)
        {
            const bool isTap = (x != 0 || y != 0);
            const int2 tapOffset = StepSize * int2(x, y);
            const int2 tapPX = clamp(px + tapOffset, 0, maxBounds);
            const float4 color = isTap ? InColor[tapPX] : centerColor;
            const float lum = isTap ? GetLuminance(color.rgb) : centerLum;

            // Bilateral luma weight
            float lumDelta = (centerLum - lum) * rcpCenterLum;
            const float wLum = GetRangeWeight(lumDelta, selfNormScale);

            // Coplanarity weight
            //
            // The raw depth difference grows with the tap offset on any surface that is
            // not perpendicular to the view direction, so an oblique road or wall reads
            // as a depth discontinuity and the filter refuses to blur along it. Those are
            // exactly the surfaces where raster lighting has to be separated out, and
            // where leftover raster ends up imprinted in the denoiser signal.
            //
            // Extrapolating the centre depth along its own gradient makes the test
            // measure distance from the centre pixel's tangent plane instead, which is
            // orientation independent.
            const float depth = InLinearDepth[tapPX];
            const half4 tapGuide = InDepthGradient[tapPX];
            const float planeOffset = clamp(dot(centerDepthGrad, float2(tapOffset)),
                                            -maxPlaneOffset, maxPlaneOffset);
            const float depthDelta = (centerDepth + planeOffset) - depth;
            const float wDepth = GetRangeWeight(depthDelta, depthNormScale);

            // Orientation weight
            //
            // Stops the kernel at creases and silhouettes that the plane test cannot see,
            // which is what previously forced the depth and luma norms to stay tight.
            const float3 tapNormal = OctahedralDecode(tapGuide.zw);
            const float wNormal = isTap ? GetNormalWeight(centerNormal, tapNormal, NormalSharpness) : 1.0f;

            const float wSpatial = GetSpatialWeight(x, y);
            const float w = wSpatial * wDepth * wLum * wNormal;

            mean += w * color;
            totalWeight += w;
        }
    }

    mean *= rcp(max(totalWeight, 1e-2f));

    // Microcontrast restoration.
    //
    // The floor is subtracted from the raw colour to form the denoiser input, and the
    // remainder travels around the denoiser in the skip signal. Pushing part of the
    // high frequency residual back into the floor therefore routes texture detail
    // through the skip path untouched rather than through the denoiser, which is what
    // attenuates it. DetailBoost is 0 on every pass but the last.
    const float meanLum = GetLuminance(mean.rgb);
    const float residualLum = centerLum - meanLum;
    const float3 chroma = mean.rgb * rcp(max(meanLum, 1e-3f));

    float4 outColor = mean;
    outColor.rgb = max(mean.rgb + (DetailBoost * residualLum) * chroma, 0.0f);

    OutColor[px] = GetSafeFP16(outColor);
}