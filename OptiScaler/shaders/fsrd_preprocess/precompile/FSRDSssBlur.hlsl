// FSR-RR SSS re-blur (28 Sep)
//
// Cyberpunk hands Ray Reconstruction its colour after a screen-space subsurface scattering blur. With SSS
// Separation the shim takes that blur back out using the game's SSS guide (luminance of after - before), so the
// denoiser gets the colour before the blur. The part the blur added, Delta = B(X) - X with X the noisy colour
// before the blur, was then averaged over frames and added back after denoising. Delta is noise at the scale of
// the effect itself, and the path tracer's noise is correlated over many frames, so no history length removes
// the boiling without lagging.
//
// This pass does what the game's NRD path does instead: denoise first, then blur. It applies a Gaussian blur of
// a given world-space width to the denoised colour on the pixels the guide marks as SSS material, so the SSS
// contribution is computed from clean lighting and needs no history:
//
//   out = lerp(D, B(D), Strength),   D = the composed, denoised colour before the blur
//
// Two separable passes (horizontal, then vertical). Taps count only on SSS pixels at a similar depth, and the
// kernel width in pixels follows each pixel's depth: sigma_px = SigmaScale / depth, SigmaScale = sigma (m) *
// focal length (px). Per-channel widths (ChannelScale) let red scatter further than blue.
//
// Fit (FLAGS_FIT): the same kernel applied to the noisy colour before the blur, X = C - C * guide / lum(C),
// predicts the game's guide: d = lum(B(X)) - lum(X). The vertical pass reduces (g d, d d, g g, n) over each
// 8x8 group, relative to the pixel's brightness, into one texel per group; the CPU sums them: the least-squares
// strength is sum(g d) / sum(d d), and the residual sqrt(1 - R^2) says how well radius and profile match the
// game's own blur. The noise makes this sharp: blurred noise has a correlation length that only the right
// radius reproduces.
#include "FSRDPreprocessCommon.hlsli"

#define MainRS \
    "RootFlags(0), " \
    "CBV(b0), " \
    "DescriptorTable(SRV(t0, numDescriptors = 7), visibility = SHADER_VISIBILITY_ALL), " \
    "DescriptorTable(UAV(u0, numDescriptors = 2), visibility = SHADER_VISIBILITY_ALL), "

#define THREAD_GROUP_SIZE_X     8
#define THREAD_GROUP_SIZE_Y     8
#define NUM_THREADS             (THREAD_GROUP_SIZE_X * THREAD_GROUP_SIZE_Y)

#define FLAGS_VERTICAL          (1 << 0) // second pass; clear = horizontal
#define FLAGS_APPLY             (1 << 1) // blur the denoised colour and write it over the composition output
#define FLAGS_FIT               (1 << 2) // blur the noisy colour too and reduce the fit terms
#define FLAGS_VIEW              (1 << 3) // vertical pass: draw the fit residual over the whole frame

// Taps per side. Past 3 sigma = MAX_TAPS the taps spread out (stride > 1 px).
#define MAX_TAPS                32

// Horizontal: InBlurSrc = the denoised colour before the blur (written by the composition on SSS pixels).
// Vertical:   InBlurSrc = the horizontal pass's result, InFitSrc = its fit counterpart.
Texture2D<half4> InBlurSrc : register(t0);
Texture2D<half4> InFitSrc : register(t1);
Texture2D<float4> InColor : register(t2);       // the game's colour (after its SSS blur)
Texture2D<float> InSSSGuide : register(t3);     // lum(after SSS - before SSS), lum = (r + 2g + b) / 4; 0 = no SSS
Texture2D<float> InLinearDepth : register(t4);
Texture2D<half4> InColorBeforeParticles : register(t5); // premultiplied particles, as the composition applies them
Texture2D<half4> InSssSource : register(t6);    // vertical: the denoised colour before the blur, centre pixel

// Horizontal: OutBlur = blurred denoised colour, OutFit = blurred noisy colour.
// Vertical:   OutBlur = the composition output (SSS pixels only; every pixel with FLAGS_VIEW),
//             OutFit = fit terms, one texel per 8x8 group.
RWTexture2D<half4> OutBlur : register(u0);
RWTexture2D<float4> OutFit : register(u1);

cbuffer CB_SssBlur : register(b0)
{
    float4 DstTexSize;    // XY = size, ZW = 1 / size
    float3 ChannelScale;  // sigma per channel relative to the widest (red) one
    float SigmaScale;     // sigma (m) * focal length (px): sigma_px = SigmaScale / linear depth
    float Strength;       // 0 = no blur, 1 = full
    float DepthTolerance; // depth difference (m) at which a tap stops counting, before the 1% of depth added below
    uint Flags;
    float _Padding;
}

bool IsSet(uint mask) { return (Flags & mask) == mask; }

float Lum4(const float3 c) { return (c.r + 2.0f * c.g + c.b) * 0.25f; }

// The noisy colour before the game's SSS blur, as the packing shader reconstructs it (SSS Separation 1).
// valid = the guide's ratio is inside the range the packing shader accepts unclamped.
float3 GetPreBlurColor(const int2 p, out bool valid)
{
    const float3 color = max(InColor[p].rgb, 0.0f);
    const float ratio = InSSSGuide[p] / max(Lum4(color), 1e-4f);
    valid = (ratio >= -4.0f) && (ratio <= 1.0f);
    return color * (1.0f - clamp(ratio, -4.0f, 1.0f));
}

groupshared float4 g_FitTerms[NUM_THREADS];

[RootSignature(MainRS)]
[numthreads(THREAD_GROUP_SIZE_X, THREAD_GROUP_SIZE_Y, 1)]
void CSMain(uint3 groupID : SV_GroupID, uint3 gtID : SV_GroupThreadID, uint3 dtID : SV_DispatchThreadID)
{
    const int2 px = int2(dtID.xy);
    const int2 maxPx = int2(DstTexSize.xy) - 1;
    const bool inBounds = all(px <= maxPx);
    const bool vertical = IsSet(FLAGS_VERTICAL);
    const bool apply = IsSet(FLAGS_APPLY);
    const bool fit = IsSet(FLAGS_FIT);

    const float guide = inBounds ? InSSSGuide[px] : 0.0f;
    const bool isSss = inBounds && (guide != 0.0f);

    float3 blurred = 0.0f;
    float3 fitBlurred = 0.0f;

    [branch]
    if (isSss)
    {
        const float depth = abs(InLinearDepth[px]);
        const float sigmaPx = SigmaScale / max(depth, 1e-3f);
        const float3 sigma = max(sigmaPx * ChannelScale, 1e-3f);
        const float3 rcpTwoSigmaSq = 0.5f / (sigma * sigma);
        const float tolerance = DepthTolerance + 0.01f * depth;

        const float reach = 3.0f * sigmaPx;
        const int taps = int(min(ceil(reach), float(MAX_TAPS)));
        const float stride = max(reach / float(MAX_TAPS), 1.0f);
        const int2 dir = vertical ? int2(0, 1) : int2(1, 0);

        float3 sum = 0.0f;
        float3 fitSum = 0.0f;
        float3 weightSum = 0.0f;

        [loop]
        for (int i = -taps; i <= taps; i++)
        {
            const int offset = int(round(float(i) * stride));
            const int2 p = px + dir * offset;

            if (any(p < 0) || any(p > maxPx))
                continue;

            if (InSSSGuide[p] == 0.0f)
                continue;

            const float depthWeight = saturate(1.0f - abs(abs(InLinearDepth[p]) - depth) / tolerance);

            if (depthWeight <= 0.0f)
                continue;

            const float3 w = exp(-float(offset * offset) * rcpTwoSigmaSq) * depthWeight;
            weightSum += w;

            if (apply)
                sum += w * float3(InBlurSrc[p].rgb);

            [branch]
            if (fit)
            {
                float3 source;

                if (vertical)
                {
                    source = float3(InFitSrc[p].rgb);
                }
                else
                {
                    bool unusedValid;
                    source = GetPreBlurColor(p, unusedValid);
                }

                fitSum += w * source;
            }
        }

        // The centre always counts (weight 1), so the sum is never empty.
        const float3 rcpWeight = rcp(max(weightSum, 1e-6f));
        blurred = sum * rcpWeight;
        fitBlurred = fitSum * rcpWeight;
    }

    // Horizontal pass: intermediate results, SSS pixels only (the vertical pass only reads those).
    if (!vertical)
    {
        if (isSss)
        {
            if (apply)
                OutBlur[px] = half4(GetSafeFP16(blurred), 1.0f);

            if (fit)
                OutFit[px] = float4(fitBlurred, 1.0f);
        }

        return;
    }

    // Vertical pass --------------------------------------------------------------------------------------------

    // Fit terms relative to the pixel's brightness, so dark and bright skin count alike
    float4 terms = 0.0f;
    float residual = 0.0f;

    [branch]
    if (fit && isSss)
    {
        bool valid;
        const float3 preBlur = GetPreBlurColor(px, valid);
        const float scale = rcp(max(Lum4(max(InColor[px].rgb, 0.0f)), 1e-3f));
        const float d = (Lum4(fitBlurred) - Lum4(preBlur)) * scale;
        const float g = guide * scale;

        if (valid)
            terms = float4(g * d, d * d, g * g, 1.0f);

        residual = g - Strength * d;
    }

    // Apply: the blurred denoised colour replaces the composition output on SSS pixels, with the particle layer
    // put back on top the way the composition did.
    if (apply && isSss && !IsSet(FLAGS_VIEW))
    {
        const float3 center = float3(InSssSource[px].rgb);
        float3 color = lerp(center, blurred, Strength);

        half4 particles = GetSafeFP16(InColorBeforeParticles[px]);
        particles.a = saturate(particles.a);
        color = (1.0f - particles.a) * color + particles.rgb;

        OutBlur[px] = half4(GetSafeFP16(color), 1.0f);
    }

    // Fit residual view: green = the game's blur added more than this kernel predicts, red = less (full at 50% of
    // the pixel). At the right radius and strength it is fine, unstructured noise; blotches that follow the SSS
    // pattern = radius off, an overall tint = strength off. Dim grey = no SSS material.
    if (IsSet(FLAGS_VIEW) && inBounds)
    {
        const float3 color = max(InColor[px].rgb, 0.0f);
        const float3 view = isSss ? (VisualizeSignedDiff(residual, 2.0f) + 0.1f) : 0.25f * saturate(Lum4(color)).xxx;
        OutBlur[px] = half4(view, 1.0f);
    }

    // Group reduction of the fit terms. Every thread reaches the barriers: nothing above returns in this pass.
    if (fit)
    {
        const uint flatID = gtID.x + gtID.y * THREAD_GROUP_SIZE_X;
        g_FitTerms[flatID] = terms;
        GroupMemoryBarrierWithGroupSync();

        [unroll]
        for (uint half_ = NUM_THREADS / 2; half_ > 0; half_ >>= 1)
        {
            if (flatID < half_)
                g_FitTerms[flatID] += g_FitTerms[flatID + half_];

            GroupMemoryBarrierWithGroupSync();
        }

        if (flatID == 0)
            OutFit[groupID.xy] = g_FitTerms[0];
    }
}
