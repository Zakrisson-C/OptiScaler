// FSR-RR Conversion & Packing Shader
#include "FSRDPreprocessCommon.hlsli"

#define MainRS \
    "RootFlags(0), " \
    "CBV(b0), " \
    "DescriptorTable(SRV(t0, numDescriptors = 11), visibility = SHADER_VISIBILITY_ALL), " \
    "DescriptorTable(UAV(u0, numDescriptors = 8), visibility = SHADER_VISIBILITY_ALL), "

// Dispatch config
#define THREAD_GROUP_SIZE_X     8
#define THREAD_GROUP_SIZE_Y     8
#define NUM_THREADS             (THREAD_GROUP_SIZE_X * THREAD_GROUP_SIZE_Y)

static const uint2 s_ThreadGroupSize = uint2(THREAD_GROUP_SIZE_X, THREAD_GROUP_SIZE_Y);

// Flags
#define FLAGS_NON_GAMMA_ALBEDO          (1 << 0)

#define FLAGS_PACKED_ROUGHNESS          (1 << 2)
// FLAGS_MODE_2_SIGNAL (1 << 3) removed (transplant, 22 Sep): the split-signal path below is now
// unconditional - denoiser 1.2 has no combined-signal shape left to select away from.
#define FLAGS_HAS_BIAS_MASK             (1 << 4)

// Troubleshooting (24 Sep). All off by default. The FLAGS_AB_* bits are the A/B switches for the
// pipeline audit's findings: most try a candidate fix (a clear bit reproduces the previous behaviour
// exactly); FLAGS_AB_CAMERA_DEPTH_DELTA instead restores the behaviour a confirmed fix replaced.
#define FLAGS_PROBE                     (1 << 5)  // Write pixel probe records to OutProbe
#define FLAGS_AB_NO_EMISSIVE            (1 << 6)  // Never reinterpret a pixel as emissive
#define FLAGS_AB_GATE_NO_ROUGHNESS      (1 << 7)  // Hit distance gate ignores roughness (finding 7)
#define FLAGS_AB_GATE_NO_BIAS           (1 << 8)  // Hit distance gate ignores the bias mask
#define FLAGS_AB_SOFTMIN_NONNEG         (1 << 9)  // Clamp the soft-min floor at zero (finding 1)
#define FLAGS_AB_SKIP_ALPHA_FINAL       (1 << 10) // SkipSignal alpha from the final floor (finding 2)
#define FLAGS_AB_SKIPPED_INACTIVE       (1 << 11) // Skipped pixels sent as inactive, alpha -1 (finding 6)
#define FLAGS_AB_CAMERA_DEPTH_DELTA     (1 << 12) // Old camera-only depth delta (no object-motion delta)

// Debug Flags
#define FLAGS_DEBUG                     (1 << 16)
#define FLAGS_DEBUG_MODE_MASK           (0xFF << 16)

// Inputs
#define FLAGS_DEBUG_IN_SPEC_HIT_DIST    (1 << 17 | FLAGS_DEBUG)
#define FLAGS_DEBUG_IN_MOTION           (2 << 17 | FLAGS_DEBUG)
#define FLAGS_DEBUG_IN_NORMALS          (3 << 17 | FLAGS_DEBUG)
#define FLAGS_DEBUG_IN_ROUGHNESS        (4 << 17 | FLAGS_DEBUG)
#define FLAGS_DEBUG_IN_DIFF_ALBEDO      (5 << 17 | FLAGS_DEBUG)
#define FLAGS_DEBUG_IN_SPEC_ALBEDO      (6 << 17 | FLAGS_DEBUG)

// Outputs
// FLAGS_DEBUG_OUT_FUSED_ALBEDO (7 << 17 | FLAGS_DEBUG) removed (transplant, 22 Sep): Mode-1-only,
// fusedAlbedo no longer exists.
#define FLAGS_DEBUG_OUT_LINEAR_DEPTH    (8 << 17 | FLAGS_DEBUG)
#define FLAGS_DEBUG_OUT_MOTION          (9 << 17 | FLAGS_DEBUG)
#define FLAGS_DEBUG_OUT_NORMALS         (10 << 17 | FLAGS_DEBUG)
#define FLAGS_DEBUG_OUT_SPEC_ALBEDO     (11 << 17 | FLAGS_DEBUG)
#define FLAGS_DEBUG_OUT_DIFF_ALBEDO     (12 << 17 | FLAGS_DEBUG)

#define FLAGS_DEBUG_OUT_DEPTH_DELTA     (13 << 17 | FLAGS_DEBUG)
#define FLAGS_DEBUG_NORM_DEPTH          (14 << 17 | FLAGS_DEBUG)

#define FLAGS_DEBUG_ALBEDO_OVERSHOOT    (15 << 17 | FLAGS_DEBUG)

#define FLAGS_DEBUG_FLOOR_VARIANCE      (16 << 17 | FLAGS_DEBUG)
#define FLAGS_DEBUG_FLOOR_COLOR         (17 << 17 | FLAGS_DEBUG)

#define FLAGS_DEBUG_IN_BIAS_MASK        (18 << 17 | FLAGS_DEBUG)
#define FLAGS_DEBUG_DEMOD_GAIN          (19 << 17 | FLAGS_DEBUG)
#define FLAGS_DEBUG_HIT_DIST_GATE       (20 << 17 | FLAGS_DEBUG)
#define FLAGS_DEBUG_DENOISER_FRACTION   (21 << 17 | FLAGS_DEBUG)

#define FLAGS_DEBUG_SIGNAL_DELTA        (22 << 17 | FLAGS_DEBUG)
#define FLAGS_DEBUG_ROUGHNESS_PROBE     (23 << 17 | FLAGS_DEBUG)

// 24 Sep. Chosen to read unambiguously through the game's tonemapper and grading.
#define FLAGS_DEBUG_EMISSIVE_CHECK      (24 << 17 | FLAGS_DEBUG)
#define FLAGS_DEBUG_HIT_GATE_PARTS      (25 << 17 | FLAGS_DEBUG)
// 24 Sep. Game motion vectors vs the camera matrices: black = agree, see the case below.
#define FLAGS_DEBUG_MOTION_CONSISTENCY  (26 << 17 | FLAGS_DEBUG)
// 25 Sep. Pixels the firefly clamp scales down: red, brighter = more removed.
#define FLAGS_DEBUG_FIREFLY_CLAMP       (27 << 17 | FLAGS_DEBUG)

// DLSS-RR Inputs
Texture2D<half3> InColor : register(t0); // RGB - NVSDK_NGX_Parameter_Color
Texture2D<float> InDepth : register(t1); // R - NVSDK_NGX_Parameter_Depth - hardware or linear - inverted or not
Texture2D<float3> InMotionVectors : register(t2); // RG - NVSDK_NGX_Parameter_MotionVectors
Texture2D<float4> InNormals : register(t3); // RGB: Normals, A: Roughness (Optional) - NVSDK_NGX_Parameter_GBuffer_Normals
Texture2D<float> InRoughness : register(t4); // R - May be packed in normals. NVSDK_NGX_Parameter_GBuffer_Roughness
Texture2D<float> InSpecHitDist : register(t5); // R - NVSDK_NGX_Parameter_DLSSD_SpecularHitDistance
Texture2D<half3> InDiffAlbedo : register(t6); // RGB - NVSDK_NGX_Parameter_GBuffer_DiffuseAlbedo
Texture2D<half3> InSpecAlbedo : register(t7); // RGB - NVSDK_NGX_Parameter_GBuffer_SpecularAlbedo
Texture2D<half> InBiasMask : register(t8);

Texture2D<half4> InFloorColor : register(t9);

// Last frame's linear depth, for the object-motion depth delta (24 Sep). Read only with
// FLAGS_AB_CAMERA_DEPTH_DELTA clear (the default since 25 Sep).
Texture2D<float> InPrevLinearDepth : register(t10);

// FSR-RR - ffxDispatchDescDenoiserIndirectSpecular / ffxDispatchDescDenoiserIndirectDiffuse
// (transplant, 22 Sep: Mode 1's combined-signal shape removed, no successor in denoiser 1.2)
//
// RGB: Noisy specular lighting A: Specular Ray Length
RWTexture2D<half4> OutSignal1 : register(u0);

// RGB: Noisy diffuse lighting
RWTexture2D<half4> OutSignal2 : register(u1);

// ffxDispatchDescDenoiser
RWTexture2D<half4> OutMotion : register(u2); // RG: Standard TSR motion vectors, B: Linear Depth Delta (CurrentLinearDepth - PrevLinearDepth)
RWTexture2D<half4> OutNormals : register(u3); // RG: Octahedrally encoded normals, B: Linear Roughness, A: Material Type (Optional)
RWTexture2D<half4> OutSpecAlbedo : register(u4); // RGB: Specular Albedo, A: dot(Normal, ViewDir)
RWTexture2D<half4> OutDiffAlbedo : register(u5); // RGB: Diffuse Albedo, A: Metalness (not provided)

RWTexture2D<half4> OutSkipSignal : register(u6);

// Pixel probe (24 Sep). One row per pixel in the probe window, PROBE_SLOT_COUNT float4 per row.
// Read back on the CPU and shown in the menu, so values are exact rather than read off a
// colour map through the game's tonemapper.
RWTexture2D<float4> OutProbe : register(u7);

cbuffer CB_Packing : register(b0)
{
    float4x4 InvViewMatrix; // DLSSD WorldToView^-1
    float4x4 InvProjMatrix; // DLSSD ViewToClip^-1
    float4x4 PrevViewMatrix; // DLSSD WorldToView from last frame
    
    float4 DstTexSize; // Resolution of inputs
    
    float NearPlane;
    float FarPlane;   
    
    float FloorIsolation;
    uint Flags;

    // Fraction of the DLSS bias mask applied when routing pixels around the denoiser.
    // 0 reproduces the previous behaviour exactly.
    float BiasMaskStrength;

    // Smoothing radius on the floor/raw clamp. 0 reproduces the exact min().
    float FloorSoftMin;

    // Re-encodes roughness before anything consumes it. DLSS supplies the perceptual
    // parameter (its spec passes roughness^2 into EnvBRDFApprox2, so alpha = roughness^2)
    // while FSR-RR documents normals.B as linear roughness, and nothing converts between
    // them. 2.0 tests the squared-convention hypothesis; 1.0 is bit-identical to before.
    float RoughnessExponent;

    // Scales the specular ray length handed to FSR-RR. Units are unverified - a normalised
    // [0,1] hit distance arriving where world units are expected would collapse the virtual
    // image onto the surface. 1.0 is bit-identical.
    float HitDistScale;

    // Pulls the floor off near-mirror surfaces. 0 is bit-identical.
    float FloorSpecGuard;

    // Biases the Mode 2 split toward specular on smooth surfaces. 0 is bit-identical.
    float SplitPriorStrength;

    // Target value for the roughness null-probe debug view.
    float RoughnessProbe;

    float _Padding;

    // Pixel probe window: centre pixel and half-width. Only read when FLAGS_PROBE is set.
    int2 ProbeCenter;
    int ProbeRadius;
    uint _Padding2;

    // Distance fade for the specular guard (24 Sep), in linear depth units. Full strength nearer
    // than FadeStart, none beyond FadeEnd. FadeEnd <= FadeStart disables the fade (bit-identical).
    float FloorSpecGuardFadeStart;
    float FloorSpecGuardFadeEnd;

    // Firefly clamp (25 Sep): a pixel whose lighting exceeds this many times the brightest of its
    // 8 neighbours is scaled down to that limit. 0 = off (bit-identical).
    float FireflyClampK;
    float _Padding3;

    // Current (unjittered) view to clip, for the MotionConsistency view and probe row (24 Sep).
    float4x4 ProjMatrix;
};

bool IsSet(uint mask) { return (Flags & mask) == mask; }
uint GetDebugMode() { return (Flags & FLAGS_DEBUG_MODE_MASK); }

// Pixel probe record layout. Must match FSRD::Probe::Slot in FSRDDiagnostics.h.
#define PROBE_RAW_COLOR         0  // raw colour rgb, raw luminance
#define PROBE_RAW_DIFF_ALBEDO   1  // input diffuse albedo rgb, input bias mask
#define PROBE_RAW_SPEC_ALBEDO   2  // input specular albedo rgb, albedo sum (emissive test input)
#define PROBE_ROUGHNESS         3  // input roughness, after exponent, emissive score, emissive applied
#define PROBE_HIT_DIST          4  // input hit distance, hit distance sent, gate, 1 = skip path
#define PROBE_GATE_TERMS        5  // gate roughness term, emissive term, bias term, roughness sent
#define PROBE_SPEC_USED         6  // specular albedo used (after override + clamp) rgb, bias weight
#define PROBE_DIFF_USED         7  // diffuse albedo used (after override + clamp) rgb, floor similarity
#define PROBE_FLOOR_IN          8  // floor from the a-trous filter rgb, its luminance
#define PROBE_FLOOR_USED        9  // floor after isolation/guard/bias/soft min rgb, denoiser fraction
#define PROBE_DENOISER_COLOR    10 // raw - floor rgb, demodulation gain
#define PROBE_DEMOD_SPEC        11 // signal 1 rgb (demodulated specular), specular split fraction
#define PROBE_DEMOD_DIFF        12 // signal 2 rgb (demodulated diffuse), signal 2 alpha
#define PROBE_SKIP_OUT          13 // skip signal rgb as written, alpha as written
#define PROBE_DEPTH_MOTION      14 // linear depth, camera-model motion error (px), motion x, motion y (input units)
#define PROBE_GEOMETRY          15 // depth delta, input normal length, N.V, residual luminance
#define PROBE_SLOT_COUNT        16

// Evaluates value only inside the branch, so pixels outside the window pay for a compare.
#define PROBE_WRITE(slot, value)                                    \
    {                                                               \
        [branch] if (isProbe)                                       \
            OutProbe[uint2((slot), probeRecord)] = float4(value);   \
    }

// Lower bound on the albedo used as a demodulation divisor.
//
// The albedos are clamped to 1e-4 to stay non zero, which permits a gain of 1e4 on
// dark surfaces. Radiance noise scaled by that lands well outside the range FP16
// resolves usefully, and the denoiser sees a signal whose magnitude swings by orders
// of magnitude between frames. Capping the gain at ~1/kMinReflectance keeps the
// demodulated signal in a sane range; whatever energy that costs is recovered by the
// existing residual path, which folds it into the skip signal.
static const float kMinReflectance = 8e-3f;

// Firefly clamp (25 Sep).
//
// Lighting estimate: luminance divided by the summed input albedos, so a bright texel on a dark
// texture doesn't count as an outlier. Floored so near-black albedo can't blow it up.
float GetFireflyLighting(const int2 p)
{
    const float3 albedo = GetSafeFP16(InDiffAlbedo[p].rgb) + GetSafeFP16(InSpecAlbedo[p].rgb);
    return GetLuminance(GetSafeFP16(InColor[p].rgb)) / max(GetLuminance(albedo), 0.02f);
}

// Scale that brings a pixel down to FireflyClampK x the brightest of its 8 neighbours. An isolated
// outlier - one path that found a light its neighbours missed, which is what subsurface random walks
// and small bright sources produce - has no neighbour anywhere near it; any real feature two pixels
// wide or more has at least one, and is left alone. Biased (the removed energy is gone), which is the
// usual price of firefly suppression. Image borders see the centre as a neighbour and are never
// clamped. 1 = untouched.
float GetFireflyScale(const int2 px)
{
    const float center = GetFireflyLighting(px);
    float neighbourMax = 0.0f;

    [unroll]
    for (int y = -1; y <= 1; y++)
    {
        [unroll]
        for (int x = -1; x <= 1; x++)
        {
            if (x == 0 && y == 0)
                continue;

            const int2 p = clamp(px + int2(x, y), int2(0, 0), int2(DstTexSize.xy) - 1);
            neighbourMax = max(neighbourMax, GetFireflyLighting(p));
        }
    }

    const float limit = FireflyClampK * neighbourMax;
    return (center > limit && center > 0.0f) ? limit / center : 1.0f;
}

float3 GetViewSpacePos(const int2 px)
{
    const float inDepth = abs(InDepth[px]);
    const float2 uv = (float2(px) + 0.5) * DstTexSize.zw;
    float3 viewSpacePos = 0.0f;
    
    viewSpacePos = InvProjectPosition(float3(uv, 1.0f), InvProjMatrix);
    viewSpacePos.xy *= abs(inDepth / viewSpacePos.z);
    viewSpacePos.z = inDepth;

    return viewSpacePos;
}

// Main Kernel
//
[RootSignature(MainRS)]
[numthreads(THREAD_GROUP_SIZE_X, THREAD_GROUP_SIZE_Y, 1)]
void CSMain(uint3 groupID : SV_GroupID, uint3 gtID : SV_GroupThreadID)
{
    const uint2 px = groupID.xy * s_ThreadGroupSize + gtID.xy;
    
    if (px.x >= DstTexSize.x || px.y >= DstTexSize.y)
        return;

    // Pixel probe (24 Sep). Every pixel inside the window writes one record of intermediate
    // values (see PROBE_* above). probeRecord is only used when isProbe is true.
    const int2 probeOffset = int2(px) - ProbeCenter;
    const bool isProbe = IsSet(FLAGS_PROBE) && all(abs(probeOffset) <= ProbeRadius);
    const uint probeRecord =
        uint((probeOffset.y + ProbeRadius) * (2 * ProbeRadius + 1) + (probeOffset.x + ProbeRadius));

    // Albedo / reflectance
    //
    // Zeroed albedos are unusable sentinels and must be skipped.
    // Depth values at the far plane indicate a skybox or other skippable content.
    //
    // DLSS-RR specular albedo is hemispherical specular reflectance at (NoV, roughness).
    // Diffuse albedo is the diffuse component of reflectance.     
    float3 specReflectance = GetSafeFP16(InSpecAlbedo[px].rgb);
    float3 diffAlbedo = GetSafeFP16(InDiffAlbedo[px].rgb);
    
    const float totalAlbedo = dot(specReflectance.rgb + diffAlbedo.rgb, 1.0f);

    // Emissive reinterpretation, softened.
    //
    // As a hard step this flips per frame on any surface whose albedo sits near the
    // threshold - animated signage and video billboards in particular - and the two
    // sides of the branch demodulate very differently, so the classification itself
    // becomes a source of temporal instability. The transition band costs nothing.
    //
    // 24 Sep: emissiveScore is the classification itself; isEmissive is what gets applied.
    // FLAGS_AB_NO_EMISSIVE switches the override off (the Turbo-R paint question) while the
    // EmissiveCheck view and the probe still report what the test would have decided.
    const float emissiveScore = SoftAbove(totalAlbedo, 5.9f, 0.5f);
    const float isEmissive = IsSet(FLAGS_AB_NO_EMISSIVE) ? 0.0f : emissiveScore;
    diffAlbedo.rgb *= (1.0f - isEmissive);
    specReflectance.rgb = lerp(specReflectance.rgb, 0.1f, isEmissive);

    PROBE_WRITE(PROBE_RAW_SPEC_ALBEDO, float4(GetSafeFP16(InSpecAlbedo[px].rgb), totalAlbedo));

    // Clamp albedo
    const float3 albedoOvershoot = max((specReflectance.rgb + diffAlbedo.rgb) - 1.0f, 0.0f);
    specReflectance.rgb = saturate(specReflectance.rgb - albedoOvershoot);
    diffAlbedo.rgb -= max((specReflectance.rgb + diffAlbedo.rgb) - 1.0f, 0.0f);
    specReflectance.rgb = max(specReflectance.rgb, 1e-4f);
    diffAlbedo.rgb = max(diffAlbedo.rgb, 1e-4f);
    
    // Surface roughness, read here because the floor guard below needs it.
    //
    // Everything downstream consumes this single value, so the exponent is applied once.
    const float rawRoughness = IsSet(FLAGS_PACKED_ROUGHNESS) ? InNormals[px].a : InRoughness[px];
    const float surfaceRoughness = saturate(pow(max(rawRoughness, 1e-5f), RoughnessExponent));

    PROBE_WRITE(PROBE_ROUGHNESS, float4(rawRoughness, surfaceRoughness, emissiveScore, isEmissive));

    // Denoiser input color and floor residual
    //
    // 25 Sep: optional firefly clamp on the raw colour, before the floor split, so both the floor
    // and the denoiser see the clamped value. Pixels the emissive test claims are left alone: a
    // distant light is exactly an isolated bright pixel. Skip-path pixels (sky, near-black albedo)
    // are composited from the game's own colour and are unaffected either way.
    const float3 inputColor = GetSafeFP16(InColor[px].rgb);
    float3 rawColor = inputColor;
    float fireflyScale = 1.0f;

    [branch]
    if (FireflyClampK > 0.0f)
    {
        fireflyScale = lerp(GetFireflyScale(px), 1.0f, emissiveScore);
        rawColor *= fireflyScale;
    }
    float4 floorColor = InFloorColor[px];
    const float rawLuma = GetLuminance(rawColor);
    const float floorLuma = GetLuminance(floorColor.rgb);
    floorColor.a = floorLuma;

    // The probe reports the game's colour as delivered, before the firefly clamp.
    PROBE_WRITE(PROBE_RAW_COLOR, float4(inputColor, GetLuminance(inputColor)));
    PROBE_WRITE(PROBE_FLOOR_IN, float4(floorColor.rgb, floorLuma));

    // Floor color blending
    //
    // Diffuse dominant surfaces are relatively well behaved.
    const float avgSpecular = dot(specReflectance.rgb, 0.33f);
    const float diffuseDominance = smoothstep(0.08f, 0.0f, avgSpecular);
    const float similarityThreshold = lerp(0.5f, 0.2f, diffuseDominance);
    
    // Clamp floor to minimum and blend in raw values where similar to preserve microcontrast.
    const float floorSimilarity = GetRelativeSimilarity(floorLuma, rawLuma, similarityThreshold);
    floorColor.rgb = FloorIsolation * lerp(floorColor.rgb, rawColor, saturate(floorSimilarity));

    // Specular guard.
    //
    // The floor exists to capture stable raster lighting. A mirror is spatially smooth,
    // which is what the stability heuristic actually measures, but it is not view-stable at
    // all - so reflected content gets captured and returned through SkipSignal after five
    // a-trous passes at stride up to 16, permanently blurred and never denoised. Reflected
    // content is confirmed present in FloorColor on the Type-66. The usual cost of pulling
    // the floor off a surface - imprinting returning - is smallest precisely here, because a
    // polished panel's albedo is near-uniform.
    //
    // 24 Sep: faded out with distance. Far away, the floor also carries the fog and haze composited
    // over the surface; pulling it off a distant window sends that through the denoiser with the
    // reflection, and in game the window punches through the fog (the exact route through the
    // denoiser is not established; the probe can show it). InDepth here is the
    // floor seed's linear depth, the same value GetViewSpacePos uses.
    float specGuard = FloorSpecGuard;

    if (FloorSpecGuardFadeEnd > FloorSpecGuardFadeStart)
        specGuard *= 1.0f - smoothstep(FloorSpecGuardFadeStart, FloorSpecGuardFadeEnd, abs(InDepth[px]));

    floorColor.rgb *= lerp(1.0f, smoothstep(0.05f, 0.25f, surfaceRoughness), specGuard);

    // Transparency / bias mask routing
    //
    // InBiasMask was bound at t8 but never sampled. DLSS-RR marks here every pixel whose
    // colour should come from the current frame rather than from history: particles,
    // alpha layers, decals, and animated or video textures. Those are precisely the
    // surfaces reported as blurred, and the floor mechanism already provides the right
    // escape hatch - anything pushed into the floor is subtracted from the denoiser input
    // and re-added verbatim from the skip signal after denoising.
    //
    // Driving the floor to the raw colour therefore routes flagged content around the
    // denoiser entirely instead of trying to classify it inside the floor prefilter.
    // An absent mask reads as 0 through the null descriptor, so the flag is belt and
    // braces rather than load bearing.
    const float biasMask = IsSet(FLAGS_HAS_BIAS_MASK) ? saturate(float(InBiasMask[px])) : 0.0f;
    const float biasWeight = saturate(biasMask * BiasMaskStrength);
    floorColor.rgb = lerp(floorColor.rgb, rawColor, biasWeight);

    PROBE_WRITE(PROBE_RAW_DIFF_ALBEDO, float4(GetSafeFP16(InDiffAlbedo[px].rgb), biasMask));
    PROBE_WRITE(PROBE_SPEC_USED, float4(specReflectance.rgb, biasWeight));
    PROBE_WRITE(PROBE_DIFF_USED, float4(diffAlbedo.rgb, saturate(floorSimilarity)));

    // Soft clamp.
    //
    // An exact min() of two smooth fields creases along the curve where they cross, and
    // the two sides of that curve are routed differently: below it the split is normal,
    // on it denoiserColor collapses to zero and the pixel travels entirely through the
    // skip path. A discrete operator producing a two region split, in the same family as
    // the isEmissive and canUseHitDist cuts. The quadratic soft min only ever dips below
    // min(), so the floor can never exceed the raw colour.
    floorColor.rgb = SoftMin(rawColor, floorColor.rgb, FloorSoftMin);

    // Audit finding 1 (24 Sep): the soft min can go negative. The subtraction below uses the
    // negative floor, but the SkipSignal write clamps it to 0, so the pixel gains light.
    // Clamping here keeps floor <= min(raw, floor) and >= 0.
    if (IsSet(FLAGS_AB_SOFTMIN_NONNEG))
        floorColor.rgb = max(floorColor.rgb, 0.0f);

    const float3 denoiserColor = rawColor - floorColor.rgb;

    PROBE_WRITE(PROBE_FLOOR_USED,
                float4(floorColor.rgb, saturate(GetLuminance(denoiserColor) * rcp(max(rawLuma, 1e-3f)))));

    // Depth - full position needed for reprojected depth delta
    const float3 viewSpacePos = GetViewSpacePos(px);
    const float compressedDepth = log(viewSpacePos.z + 1.0f) / log(FarPlane + 1.0f);
    
    // Motion Vectors & Depth Delta
    //
    // Computed before the validity gate and written unconditionally. OutMotion is the only
    // UAV the skip path left untouched, so skipped pixels used to hand FSR-RR whatever the
    // texture happened to contain - never cleared, and for a pixel that has always been
    // skipped, never written at all. The gate rejects two large populations: everything at
    // the far plane (the whole sky outdoors) and everything with near-black albedo
    // (emissives, billboards). Both were feeding the denoiser undefined motion and an
    // undefined linear depth delta.
    //
    // The computation is unchanged and depends only on viewSpacePos, which is already
    // available here, so the valid path is bit-identical - this only extends the same
    // values to the pixels that previously got none.
    //
    // Find the current pixel in world space and calculate movement in view space
    const float3 worldSpacePos = mul(InvViewMatrix, float4(viewSpacePos, 1.0f)).xyz;
    float3 prevViewSpacePos = mul(PrevViewMatrix, float4(worldSpacePos, 1.0f)).xyz;
    prevViewSpacePos.z = abs(prevViewSpacePos.z);

    // FSR-RR requires Linear Depth Delta in Blue channel
    const float2 motionIn = InMotionVectors[px].rg; // RG: Pixel Movement

    // Camera-model motion (24 Sep): where this pixel lands in the previous frame according to the
    // matrices alone (previous view, current projection), against the game's motion vectors. For
    // static geometry the two agree to a fraction of a pixel if the matrices describe the game's
    // camera. The depth delta below and the denoiser's own reprojection of reflections (camera
    // matrices + cameraPositionDelta + hit distance) rest on the same camera model, so a mismatch
    // here on static scenery is a mismatch there. Feeds the probe, the debug view and the
    // object-motion A/B below.
    const float4 prevClip = mul(ProjMatrix, float4(mul(PrevViewMatrix, float4(worldSpacePos, 1.0f)).xyz, 1.0f));
    const float2 pixelUV = (float2(px) + 0.5f) * DstTexSize.zw;
    const float2 cameraMotion = NDCToUV(prevClip.xy / prevClip.w) - pixelUV;
    const float2 motionErrorPx = (cameraMotion - motionIn) * DstTexSize.xy;

    float depthDelta = (prevViewSpacePos.z - viewSpacePos.z);

    // Object-motion depth delta (24 Sep; default since 25 Sep, confirmed by Chris while driving - the
    // A/B switch now restores the camera-only delta). The delta above treats every pixel as static
    // geometry seen from a moving camera. For anything that moves by itself it is wrong by that
    // object's own displacement - in a chase camera the driven car sits still on screen while this
    // reports the camera's v*dt, and the denoiser's depth test then rejects the car's history.
    // Where the game's motion disagrees with the camera model by more than ~1 px, take the delta
    // from last frame's linear depth at the position the game's motion vector points to. Static
    // pixels keep the camera-model delta, so background revealed behind a moving object still
    // fails the depth test (no ghosting there); a self-moving pixel's own newly exposed side can
    // pass it.
    if (!IsSet(FLAGS_AB_CAMERA_DEPTH_DELTA))
    {
        const float2 prevUV = pixelUV + motionIn;

        if (all(prevUV >= 0.0f) && all(prevUV < 1.0f))
        {
            const float prevDepth = abs(InPrevLinearDepth[int2(prevUV * DstTexSize.xy)]);
            const float selfMotion = smoothstep(0.75f, 1.5f, length(motionErrorPx));
            depthDelta = lerp(depthDelta, prevDepth - viewSpacePos.z, selfMotion);
        }
    }
    const float3 motionOut = float3(motionIn, depthDelta);
    OutMotion[px] = half4(motionOut, 0.0f);

    PROBE_WRITE(PROBE_DEPTH_MOTION, float4(viewSpacePos.z, length(motionErrorPx), motionIn));

    if (((compressedDepth < 0.99f) && totalAlbedo > 1e-2f) || IsSet(FLAGS_DEBUG))
    {
        // Normals - FSR-RR requries world normals.
        //
        // [TODO!] DLSS-RR normals may be in view or world space. They will need to be transformed to account
        // for both configurations. Cyberpunk happens to use world normals, thankfully.
        float4 worldSurfaceNormal = InNormals[px];        
        const float2 octNormal = OctahedralEncode(worldSurfaceNormal.rgb);
        const float materialType = 0.0f;
    
        // DLSS-RR provides 3D normals
        // Linear roughness optionally included in the A channel, or in a separate single-channel 
        // buffer (InRoughness).
        float roughness = surfaceRoughness;
        roughness *= (1.0f - isEmissive);
        
        // Output: RG=OctNormal, B=Roughness, A=MaterialID
        OutNormals[px] = GetSafeFP16(float4(octNormal, roughness, materialType));
   
        half hitDist = 0.0f;
        float dbgHitGate = 0.0f;
        float3 dbgGateParts = 0.0f;
        half3 demodColor = 0.0f;
        float demodGain = 0.0f;
        float3 signalDelta = 0.0f;

        // Transplant, 22 Sep: unconditional now - this was the Mode 2 (split signal) branch.
        // Mode 1's combined-signal packing (the old else, fusedAlbedo and all) is removed
        // outright: denoiser 1.2 has no successor for it (transplant plan §6e/6f).
        {
            const float3 specWeight = saturate(specReflectance.rgb);
            const float3 diffWeight = saturate(diffAlbedo.rgb);

            // Split the composited radiance between the two signals by reflectance ratio.
            // Where both are effectively zero the ratio is meaningless, so fall back to
            // sending everything down the diffuse path - it carries no reprojection state
            // and cannot smear.
            const float3 totalWeight = diffWeight + specWeight;
            const float3 specFraction = specWeight * rcp(max(totalWeight, kMinReflectance));
            const float3 isSplitValid = smoothstep(0.5f * kMinReflectance, kMinReflectance, totalWeight);

            // Split prior - CONTINGENT, leave at 0 until the signal-delta view confirms.
            //
            // The reflectance-ratio split is exactly the assumption L_diffuse == L_specular:
            // C/(A_d + A_s) is a weighted average of the two lighting terms whose weights
            // track albedo, so wherever albedo varies AND the two lighting terms differ, the
            // albedo pattern survives the division. That is the vehicle-livery residue seen
            // in OutRadiance. Biasing the split toward specular on smooth surfaces is still a
            // guess - one equation, two unknowns - but a better motivated one. It never
            // pushes below what the reflectance ratio already says.
            const float3 priorFraction = max(specFraction, SoftBelow(roughness, 0.35f, 0.20f).xxx);
            const float3 biasedFraction = lerp(specFraction, priorFraction, SplitPriorStrength);

            const float3 specularColor = denoiserColor * (biasedFraction * isSplitValid);
            const float3 diffuseColor = denoiserColor - specularColor;

            // Demodulate against a floored divisor. The remodulation below still uses the
            // true albedo, so the gap between the two is caught by the residual and routed
            // into the skip signal rather than silently lost.
            const float3 specDenom = max(specReflectance.rgb, kMinReflectance);
            const float3 diffDenom = max(diffAlbedo.rgb, kMinReflectance);

            half3 demodSpecular = GetSafeFP16(specularColor / specDenom);
            half3 demodDiffuse = GetSafeFP16(diffuseColor / diffDenom);

            demodGain = rcp(min(GetLuminance(specDenom), GetLuminance(diffDenom)));
            signalDelta = abs(float3(demodSpecular) - float3(demodDiffuse));

            // Anything that can't survive modulation and clamping should be skipped
            const float3 remodColor = (demodSpecular * specReflectance.rgb) + (demodDiffuse * diffAlbedo.rgb);
            const float3 residual = max(0.0f, denoiserColor - remodColor);
            floorColor.rgb += residual;

            // Specular motion tracking handover.
            //
            // As a hard cut at roughness 0.2 this toggles per pixel and per frame wherever
            // roughness varies around the threshold - clearcoat, wet asphalt, painted metal
            // at grazing angles - and each toggle discontinuously changes how FSR-RR
            // reprojects that pixel. Ramping the hit distance instead is a continuous
            // handover with a defensible meaning: a shorter virtual hit distance places the
            // reflection nearer the surface, so the specular reprojects with the surface as
            // the weight goes to zero, which is the behaviour the hard cut was after.
            //
            // Bias-masked pixels are excluded: their colour is not a surface reflection and
            // the hit distance that comes with them is not meaningful.
            //
            // 24 Sep: the three factors are kept apart for the HitGateParts view and the probe,
            // with A/B switches on the roughness and bias terms. Same product in the same order,
            // so with both switches clear the result is bit-identical to before.
            const float gateRoughness = IsSet(FLAGS_AB_GATE_NO_ROUGHNESS) ? 1.0f : SoftBelow(roughness, 0.30f, 0.15f);
            const float gateEmissive = 1.0f - isEmissive;
            const float gateBias = IsSet(FLAGS_AB_GATE_NO_BIAS) ? 1.0f : (1.0f - biasWeight);
            const float canUseHitDist = gateRoughness * gateEmissive * gateBias;
            hitDist = GetSafeFP16(InSpecHitDist[px] * HitDistScale * canUseHitDist);
            dbgHitGate = canUseHitDist;
            dbgGateParts = float3(gateRoughness, gateEmissive, gateBias);

            PROBE_WRITE(PROBE_HIT_DIST, float4(InSpecHitDist[px], hitDist, canUseHitDist, 0.0f));
            PROBE_WRITE(PROBE_GATE_TERMS, float4(gateRoughness, gateEmissive, gateBias, roughness));
            PROBE_WRITE(PROBE_DENOISER_COLOR, float4(denoiserColor, demodGain));
            PROBE_WRITE(PROBE_DEMOD_SPEC, float4(demodSpecular, GetLuminance(biasedFraction * isSplitValid)));
            PROBE_WRITE(PROBE_DEMOD_DIFF, float4(demodDiffuse, 0.0f));
            PROBE_WRITE(PROBE_GEOMETRY,
                        float4(depthDelta, length(worldSurfaceNormal.rgb),
                               dot(normalize(worldSurfaceNormal.rgb),
                                   normalize(mul(InvViewMatrix, float4(0.0f, 0.0f, 0.0f, 1.0f)).xyz - worldSpacePos)),
                               GetLuminance(residual)));

            [branch]
            if (!IsSet(FLAGS_DEBUG))
            {
                OutSignal1[px] = half4(demodSpecular, hitDist);
                OutSignal2[px] = half4(demodDiffuse, 0.0f);
            }
            else
                demodColor = demodDiffuse + demodSpecular;
        }

        // May be for better perceptual encoding efficiency in some configurations
        [branch]
        if (!IsSet(FLAGS_NON_GAMMA_ALBEDO))
        {
            specReflectance = sqrt(specReflectance);
            diffAlbedo = sqrt(diffAlbedo);
        }
        
        OutSpecAlbedo[px] = half4(GetSafeFP16(specReflectance), 0.0f);
        OutDiffAlbedo[px] = half4(GetSafeFP16(diffAlbedo), 0.0f);

        // Audit finding 2 (24 Sep): floorColor.a still holds the luminance of the floor as it
        // came out of the a-trous filter, before isolation, spec guard, bias mask, min and the
        // residual. The composition uses it as the skip luminance in the SSIM reference that
        // drives the Correlation Bias raw blend.
        if (IsSet(FLAGS_AB_SKIP_ALPHA_FINAL))
            floorColor.a = GetLuminance(floorColor.rgb);

        OutSkipSignal[px] = half4(GetSafeFP16(floorColor));

        PROBE_WRITE(PROBE_SKIP_OUT, GetSafeFP16(floorColor));

        [branch]
        if (IsSet(FLAGS_DEBUG))
        {
            float3 debugColor = float3(0, 0, 0);
        
            switch (GetDebugMode())
            {
                // Inputs
                case FLAGS_DEBUG_IN_SPEC_HIT_DIST:
                {
                    // frac() wrapped every 10 units, so an absent buffer and a 10 unit
                    // buffer rendered identically - the reason this view could never
                    // settle whether hit distance reaches the panel at all. The log remap
                    // over [0, 127] is monotone across the whole range, and an exactly
                    // zero sample is flagged magenta so an absent input is distinct from
                    // a near hit.
                    const float rawHitDist = InSpecHitDist[px];
                    debugColor = (rawHitDist <= 0.0f)
                               ? float3(1.0f, 0.0f, 1.0f)
                               : TurboColormap(log2(1.0f + rawHitDist) * (1.0f / 7.0f));
                    break;
                }

                case FLAGS_DEBUG_HIT_DIST_GATE:
                    // The canUseHitDist ramp itself, separated from the buffer it scales.
                    // 24 Sep correction: this Turbo fit starts near black, not blue. Near
                    // black = closed (specular reprojects with the surface), blue/cyan/green =
                    // partly closed, dark red = open. The game's grading turns the dark red
                    // orange-tan. HitGateParts shows which factor closes it.
                    debugColor = TurboColormap(dbgHitGate);
                    break;

                // 24 Sep. R = roughness term, G = emissive term, B = bias mask term, each 1 when
                // it lets hit distance through. White = open. Cyan = closed by roughness,
                // magenta = closed by the emissive override, yellow = closed by the bias mask,
                // black = closed by more than one. Saturated primaries survive the grading.
                case FLAGS_DEBUG_HIT_GATE_PARTS:
                    debugColor = dbgGateParts;
                    break;

                // 24 Sep. Grey = raw specular + diffuse albedo summed over RGB, divided by 6
                // (physical materials stay well below 0.5). Magenta = inside the emissive band
                // (sum >= 5.4), where the override forces roughness 0, diffuse 0, spec 0.1 and
                // closes the hit distance gate. Reports the test, whether or not the override
                // is switched off.
                // 24 Sep. Difference between the game's motion vectors and the motion the camera
                // matrices predict for a static point. Black = agree. Brightness = disagreement,
                // full at 4 px; hue = its direction. Moving cars and people light up legitimately.
                // Static scenery lighting up while the camera moves or turns means the matrices
                // (or the previous view) don't describe the game's camera.
                case FLAGS_DEBUG_MOTION_CONSISTENCY:
                    debugColor = VisualizeMotionVec(motionErrorPx, 0.25f);
                    break;

                // 25 Sep. Red where the firefly clamp scales a pixel down, brighter = more of it
                // removed (full red at 75% or more), over a dim grey copy of the raw luminance.
                // Nothing red with Firefly Clamp at 0.
                case FLAGS_DEBUG_FIREFLY_CLAMP:
                    debugColor = lerp(0.25f * saturate(rawLuma).xxx, float3(1.0f, 0.0f, 0.0f),
                                      saturate((1.0f - fireflyScale) * (4.0f / 3.0f)));
                    break;

                case FLAGS_DEBUG_EMISSIVE_CHECK:
                    debugColor = (emissiveScore > 0.0f) ? float3(1.0f, 0.0f, 1.0f) : saturate(totalAlbedo / 6.0f).xxx;
                    break;

                case FLAGS_DEBUG_DENOISER_FRACTION:
                {
                    // Share of the pixel routed to the denoiser rather than around it via
                    // the skip signal. Blue = travelling around the denoiser blurred by
                    // the floor, red = being denoised. Reflections and shadows reading
                    // blue is the floor capturing lighting it should have passed through.
                    const float rawLum = GetLuminance(rawColor);
                    const float denLum = GetLuminance(denoiserColor);
                    debugColor = TurboColormap(saturate(denLum * rcp(max(rawLum, 1e-3f))));
                    break;
                }
                
                case FLAGS_DEBUG_NORM_DEPTH:
                    debugColor = TurboColormap(compressedDepth);
                    break;
                
                case FLAGS_DEBUG_IN_MOTION:
                    debugColor = VisualizeMotionVec(motionIn * DstTexSize.xy, 0.1f);
                    break;
                
                case FLAGS_DEBUG_IN_NORMALS:
                    debugColor = worldSurfaceNormal.rgb * 0.5 + 0.5;
                    break;
                
                case FLAGS_DEBUG_IN_ROUGHNESS:
                    debugColor = roughness;
                    break;
                
                case FLAGS_DEBUG_IN_DIFF_ALBEDO:
                    debugColor = InDiffAlbedo[px];
                    break;
                
                case FLAGS_DEBUG_IN_SPEC_ALBEDO:
                    debugColor = InSpecAlbedo[px];
                    break;
                // Outputs
                // FLAGS_DEBUG_OUT_FUSED_ALBEDO case removed (transplant, 22 Sep): Mode-1-only,
                // fusedAlbedo no longer exists.
                case FLAGS_DEBUG_OUT_LINEAR_DEPTH:
                    // Same wrap bug as the hit distance view; same monotone remap.
                    debugColor = TurboColormap(log2(1.0f + viewSpacePos.z) * (1.0f / 11.0f));
                    break;
                
                case FLAGS_DEBUG_OUT_MOTION:
                    debugColor = VisualizeMotionVec(motionOut.xy * DstTexSize.xy, 0.1f);
                    break;

                case FLAGS_DEBUG_OUT_DEPTH_DELTA:
                    debugColor = VisualizeSignedDiff(motionOut.z, 5.0f);
                    break;
                
                case FLAGS_DEBUG_OUT_NORMALS:
                    debugColor = OctahedralDecode(octNormal) * 0.5 + 0.5;
                    break;

                case FLAGS_DEBUG_OUT_SPEC_ALBEDO:
                    debugColor = specReflectance.rgb;
                    break;
                
                case FLAGS_DEBUG_OUT_DIFF_ALBEDO:
                    debugColor = diffAlbedo.rgb;
                    break;

                case FLAGS_DEBUG_FLOOR_VARIANCE:
                    debugColor = TurboColormap(InFloorColor[px].a);
                    break;
                
                case FLAGS_DEBUG_FLOOR_COLOR:
                    debugColor = InFloorColor[px].rgb;
                    break;

                case FLAGS_DEBUG_ALBEDO_OVERSHOOT:
                    debugColor = albedoOvershoot;
                    break;

                // Where the game is asking for the current frame to be trusted over
                // history. Should light up on billboards, particles and alpha layers.
                case FLAGS_DEBUG_IN_BIAS_MASK:
                    debugColor = TurboColormap(biasWeight);
                    break;

                // Demodulation amplification, log2 scaled over [1, 128].
                // Red areas are where radiance noise is being multiplied hardest.
                case FLAGS_DEBUG_DEMOD_GAIN:
                    debugColor = TurboColormap(saturate(log2(max(demodGain, 1.0f)) * (1.0f / 7.0f)));
                    break;
                
                // Mode 2 split degeneracy test. Working the algebra through gives
                // demodSpecular == demodDiffuse == C/(spec+diff), i.e. the split conveys
                // nothing. Uniform deep blue confirms. Faint rounding noise is expected from
                // the FP16 reciprocal/multiply/divide; structure that tracks scene content is
                // what refutes it.
                case FLAGS_DEBUG_SIGNAL_DELTA:
                    debugColor = TurboColormap(saturate(GetLuminance(signalDelta) * 100.0f));
                    break;

                // Roughness null probe: white where roughness matches RoughnessProbe. Reads
                // out real values through any display transfer, which a greyscale ramp cannot
                // - the squared-convention hypothesis is a power function and so is the
                // display transfer, so they are otherwise perfectly confounded.
                case FLAGS_DEBUG_ROUGHNESS_PROBE:
                    debugColor = 1.0f - saturate(abs(surfaceRoughness - RoughnessProbe) * 50.0f);
                    break;

                default:
                    debugColor = demodColor;
                    break;
            }
        
            OutSignal1[px] = half4(debugColor, 1.0f);
        }
    }
    else // Skip
    {
        // OutMotion is written above the gate for every pixel and is deliberately not
        // zeroed here: the game's own motion vectors are valid for skipped pixels too.
        OutNormals[px] = 0.0f;
        OutSpecAlbedo[px] = 0.0f;
        OutDiffAlbedo[px] = 0.0f;

        // Audit finding 6 (24 Sep): alpha 0 declares a skipped pixel an active, valid black
        // sample. Denoiser 1.2 wants a negative alpha for pixels without a signal.
        const float skippedAlpha = IsSet(FLAGS_AB_SKIPPED_INACTIVE) ? -1.0f : 0.0f;
        OutSignal1[px] = half4(0.0f, 0.0f, 0.0f, skippedAlpha);
        OutSignal2[px] = half4(0.0f, 0.0f, 0.0f, skippedAlpha);
        OutSkipSignal[px] = half4(rawColor, rawLuma);

        PROBE_WRITE(PROBE_HIT_DIST, float4(InSpecHitDist[px], 0.0f, 0.0f, 1.0f));
        PROBE_WRITE(PROBE_GATE_TERMS, float4(0.0f, 0.0f, 0.0f, 0.0f));
        PROBE_WRITE(PROBE_DENOISER_COLOR, float4(denoiserColor, 0.0f));
        PROBE_WRITE(PROBE_DEMOD_SPEC, float4(0.0f, 0.0f, 0.0f, 0.0f));
        PROBE_WRITE(PROBE_DEMOD_DIFF, float4(0.0f, 0.0f, 0.0f, skippedAlpha));
        PROBE_WRITE(PROBE_SKIP_OUT, float4(rawColor, rawLuma));
        PROBE_WRITE(PROBE_GEOMETRY, float4(depthDelta, 0.0f, 0.0f, 0.0f));
    }
}