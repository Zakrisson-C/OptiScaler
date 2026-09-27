// FSR-RR additive light split (27 Sep)
//
// A pixel's colour is C = A_d * L_d + A_s * L_s + F, with F the light the surface doesn't reflect (fog, haze).
// The shim's split assumes one lighting value for both lobes and no F, so in effect it divides everything by
// A = A_d + A_s. Whatever isn't proportional to A - fog, and a reflection brighter than the diffuse lighting on
// a textured glossy surface - gets divided by the texture; the denoiser smooths that inverted pattern away and
// multiplying the albedo back in stamps the texture onto it (the extra contrast on hazy backgrounds, the
// liveries in the lighting).
//
// Over 7x7 on one surface the lighting and the fog barely change while the albedo varies, so a least-squares
// line through (A, C), per colour channel, gives the slope a (the lighting that follows the albedo) and the
// intercept b (the part that doesn't). Routing: the diffuse lobe gets A_d * a, the specular lobe the rest,
// A_s * a + b. The specular albedo is smooth across a surface (Fresnel; dielectrics carry no texture in it), so
// b / A_s holds no texture pattern, and the albedo division itself stays per pixel: no texture is lost. With
// b = 0 this is exactly the old reflectance-ratio split.
//
// The fit is pulled toward b = 0 where the albedo barely varies over the window (no information there, and
// nothing to imprint either), b is clamped to [0, mean C] and scaled by the strength slider. The resulting
// specular share is averaged over frames along the motion vectors: it only decides routing - every pixel's
// total stays its own colour - so a lag can't ghost the image, while a noisy fit per frame would make both
// lobes boil.
#include "FSRDPreprocessCommon.hlsli"

#define MainRS \
    "RootFlags(0), " \
    "CBV(b0), " \
    "DescriptorTable(SRV(t0, numDescriptors = 7), visibility = SHADER_VISIBILITY_ALL), " \
    "DescriptorTable(UAV(u0, numDescriptors = 2), visibility = SHADER_VISIBILITY_ALL), "

#define THREAD_GROUP_SIZE_X     8
#define THREAD_GROUP_SIZE_Y     8
#define NUM_THREADS             (THREAD_GROUP_SIZE_X * THREAD_GROUP_SIZE_Y)

static const uint2 s_ThreadGroupSize = uint2(THREAD_GROUP_SIZE_X, THREAD_GROUP_SIZE_Y);

#define FLAGS_RESET             (1 << 0) // history invalid this frame

#define FIT_KERNEL_SIZE         7
#define FIT_RADIUS              (FIT_KERNEL_SIZE / 2)
#define FIT_MIN_SAMPLES         12.0f

DEFINE_LDS_CONFIG(s_SM, FIT_KERNEL_SIZE);
DECLARE_LDS_ARRAY_2D(float3, g_Albedo, FIT_KERNEL_SIZE); // A_d + A_s, 0 = not usable (sky, emissive)
DECLARE_LDS_ARRAY_2D(float3, g_Color, FIT_KERNEL_SIZE);
DECLARE_LDS_ARRAY_2D(float3, g_Normal, FIT_KERNEL_SIZE);
DECLARE_LDS_ARRAY_2D(float, g_Depth, FIT_KERNEL_SIZE);

Texture2D<half3> InColor : register(t0);
Texture2D<half3> InDiffAlbedo : register(t1);
Texture2D<half3> InSpecAlbedo : register(t2);
Texture2D<float> InLinearDepth : register(t3);
Texture2D<float4> InNormals : register(t4);        // the game's world normals (xyz)
Texture2D<float3> InMotionVectors : register(t5);  // UV units, as the packing shader reads them
Texture2D<half4> InHistory : register(t6);         // last frame's specular share (rgb), linear depth (a, 0 = none)

RWTexture2D<half4> OutHistory : register(u0);      // this frame's specular share, read by the packing shader
RWTexture2D<half4> OutIntercept : register(u1);    // b, for the AdditiveLight view

cbuffer CB_SplitFit : register(b0)
{
    float4 DstTexSize;    // XY = size, ZW = 1 / size
    float Strength;       // scales b; 0 = the old split
    float HistoryAlpha;   // weight of the current frame in the average; 1 = none
    uint Flags;
    float _Padding;
}

bool IsSet(uint mask) { return (Flags & mask) == mask; }

bool IsUsableAlbedo(const float3 diff, const float3 spec)
{
    const float sum = dot(diff + spec, 1.0f);
    return sum > 1e-2f && sum < 5.0f; // skipped (sky, no albedo) and emissive-encoded pixels stay out
}

void PopulateSharedMemory(const uint2 groupID, const uint2 gtID)
{
    const uint flatID = gtID.x + gtID.y * s_ThreadGroupSize.x;
    const int2 origin = int2(groupID * s_ThreadGroupSize) - int2(s_SM_HaloOffset);
    const int2 maxPx = int2(DstTexSize.xy) - 1;

    [unroll]
    for (uint i = 0; i < s_SM_LoadsPerThread; i++)
    {
        const uint smFlatID = flatID + i * NUM_THREADS;

        if (smFlatID < s_SM_ElementCount)
        {
            const uint2 smID = uint2(smFlatID % s_SM_Size.x, smFlatID / s_SM_Size.x);
            const int2 p = clamp(origin + int2(smID), int2(0, 0), maxPx);

            const float3 diff = GetSafeFP16(InDiffAlbedo[p].rgb);
            const float3 spec = GetSafeFP16(InSpecAlbedo[p].rgb);

            g_Albedo[smID.x][smID.y] = IsUsableAlbedo(diff, spec) ? saturate(diff) + saturate(spec) : 0.0f;
            g_Color[smID.x][smID.y] = GetSafeFP16(InColor[p].rgb);
            g_Normal[smID.x][smID.y] = SafeNormalize(InNormals[p].xyz, float3(0.0f, 0.0f, 0.0f));
            g_Depth[smID.x][smID.y] = abs(InLinearDepth[p]);
        }
    }
}

// Specular share averaged over frames: bilinear reprojection by hand with a per-tap depth test, the same
// scheme as the SSS history.
float3 AccumulateShare(const int2 px, const float3 shareNow, const float depth)
{
    float3 result = shareNow;

    [branch]
    if (HistoryAlpha < 1.0f && !IsSet(FLAGS_RESET))
    {
        const float2 pixelUV = (float2(px) + 0.5f) * DstTexSize.zw;
        const float2 prevUV = pixelUV + InMotionVectors[px].rg;

        if (all(prevUV >= 0.0f) && all(prevUV < 1.0f))
        {
            const float2 prevPos = prevUV * DstTexSize.xy - 0.5f;
            const int2 base = int2(floor(prevPos));
            const float2 f = prevPos - float2(base);
            const int2 maxPx = int2(DstTexSize.xy) - 1;

            float3 sum = 0.0f;
            float weightSum = 0.0f;

            [unroll]
            for (int i = 0; i < 4; i++)
            {
                const int2 offset = int2(i & 1, i >> 1);
                const float4 history = InHistory[clamp(base + offset, int2(0, 0), maxPx)];
                const float bilinear = ((offset.x != 0) ? f.x : 1.0f - f.x) * ((offset.y != 0) ? f.y : 1.0f - f.y);
                const bool sameSurface = (history.a > 0.0f) && (abs(history.a - depth) < 0.1f * depth);
                const float w = sameSurface ? bilinear : 0.0f;

                sum += w * history.rgb;
                weightSum += w;
            }

            if (weightSum > 0.25f)
                result = lerp(sum / weightSum, shareNow, HistoryAlpha);
        }
    }

    return result;
}

[RootSignature(MainRS)]
[numthreads(THREAD_GROUP_SIZE_X, THREAD_GROUP_SIZE_Y, 1)]
void CSMain(uint3 groupID : SV_GroupID, uint3 gtID : SV_GroupThreadID)
{
    PopulateSharedMemory(groupID.xy, gtID.xy);
    GroupMemoryBarrierWithGroupSync();

    const int2 px = int2(groupID.xy * s_ThreadGroupSize + gtID.xy);

    if (px.x >= int(DstTexSize.x) || px.y >= int(DstTexSize.y))
        return;

    const int2 smCenter = int2(gtID.xy + s_SM_HaloOffset);
    const float3 centerAlbedo = g_Albedo[smCenter.x][smCenter.y];
    const float centerDepth = g_Depth[smCenter.x][smCenter.y];
    const float3 centerNormal = g_Normal[smCenter.x][smCenter.y];

    // No usable albedo here: the packing shader keeps its own split (history alpha 0).
    if (all(centerAlbedo <= 0.0f))
    {
        OutHistory[px] = half4(0.0f, 0.0f, 0.0f, 0.0f);
        OutIntercept[px] = half4(0.0f, 0.0f, 0.0f, 0.0f);
        return;
    }

    const bool hasNormal = dot(centerNormal, centerNormal) > 0.5f;

    float n = 0.0f;
    float3 sA = 0.0f;
    float3 sC = 0.0f;
    float3 sAA = 0.0f;
    float3 sAC = 0.0f;

    [unroll]
    for (int y = -FIT_RADIUS; y <= FIT_RADIUS; y++)
    {
        [unroll]
        for (int x = -FIT_RADIUS; x <= FIT_RADIUS; x++)
        {
            const int2 sm = smCenter + int2(x, y);
            const float3 albedo = g_Albedo[sm.x][sm.y];
            const float depth = g_Depth[sm.x][sm.y];
            const float3 normal = g_Normal[sm.x][sm.y];

            const bool sameSurface = any(albedo > 0.0f) && (abs(depth - centerDepth) < 0.02f * centerDepth) &&
                                     (!hasNormal || dot(normal, centerNormal) > 0.9f);

            if (!sameSurface)
                continue;

            const float3 color = g_Color[sm.x][sm.y];

            n += 1.0f;
            sA += albedo;
            sC += color;
            sAA += albedo * albedo;
            sAC += albedo * color;
        }
    }

    float3 intercept = 0.0f;
    float3 slope = 0.0f;
    bool fitted = false;

    if (n >= FIT_MIN_SAMPLES)
    {
        const float3 mA = sA / n;
        const float3 mC = sC / n;
        const float3 varA = max(sAA / n - mA * mA, 0.0f);
        const float3 covAC = sAC / n - mA * mC;
        const float3 safeMeanA = max(mA, 1e-3f);

        // Ridge toward the old split's model (C proportional to A, b = 0), weighted by the albedo's own
        // contrast: the free fit takes over once the albedo varies by more than ~15% over the window.
        const float3 ratioSlope = mC / safeMeanA;
        const float3 lambda = 0.02f * mA * mA + 1e-6f;
        const float3 fitSlope = (covAC + lambda * ratioSlope) / (varA + lambda);

        intercept = clamp(mC - fitSlope * mA, 0.0f, mC) * saturate(Strength);
        slope = (mC - intercept) / safeMeanA;
        fitted = true;
    }

    // Specular share at this pixel: (A_s * a + b) / ((A_d + A_s) * a + b). With b = 0: A_s / (A_d + A_s).
    const float3 diff = saturate(float3(GetSafeFP16(InDiffAlbedo[px].rgb)));
    const float3 spec = saturate(float3(GetSafeFP16(InSpecAlbedo[px].rgb)));
    const float3 ratioShare = spec / max(diff + spec, 1e-4f);

    float3 shareNow = ratioShare;

    if (fitted)
    {
        const float3 model = (diff + spec) * slope + intercept;
        shareNow = select(model > 1e-6f, saturate((spec * slope + intercept) / max(model, 1e-6f)), ratioShare);
    }

    const float3 share = AccumulateShare(px, shareNow, centerDepth);

    OutHistory[px] = half4(GetSafeFP16(share), centerDepth);
    OutIntercept[px] = half4(GetSafeFP16(intercept), fitted ? 1.0f : 0.0f);
}
