// FSR-RR Conversion & Packing Shader
#include "FSRDPreprocessCommon.hlsli"

#define MainRS \
    "RootFlags(0), " \
    "CBV(b0), " \
    "DescriptorTable(SRV(t0, numDescriptors = 10), visibility = SHADER_VISIBILITY_ALL), " \
    "DescriptorTable(UAV(u0, numDescriptors = 7), visibility = SHADER_VISIBILITY_ALL), "

// Dispatch config
#define THREAD_GROUP_SIZE_X     8
#define THREAD_GROUP_SIZE_Y     8
#define NUM_THREADS             (THREAD_GROUP_SIZE_X * THREAD_GROUP_SIZE_Y)

static const uint2 s_ThreadGroupSize = uint2(THREAD_GROUP_SIZE_X, THREAD_GROUP_SIZE_Y);

// Flags
#define FLAGS_NON_GAMMA_ALBEDO          (1 << 0)

#define FLAGS_PACKED_ROUGHNESS          (1 << 2)
#define FLAGS_MODE_2_SIGNAL             (1 << 3)
#define FLAGS_HAS_BIAS_MASK             (1 << 4)

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
#define FLAGS_DEBUG_OUT_FUSED_ALBEDO    (7 << 17 | FLAGS_DEBUG)
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

// FSR-RR - ffxDispatchDescDenoiserInput1Signal or ffxDispatchDescDenoiserInput2Signals
//
// Mode 1: RGB: Noisy fused lighting
// Mode 2: RGB: Noisy specular lighting A: Specular Ray Length
RWTexture2D<half4> OutSignal1 : register(u0); 

// Mode 1: RGB Fused Albedo: max(specularAlbedo, diffuseAlbedo)
// Mode 2: RGB: Noisy diffuse lighting for Mode 2
RWTexture2D<half4> OutSignal2 : register(u1);

// ffxDispatchDescDenoiser
RWTexture2D<half4> OutMotion : register(u2); // RG: Standard TSR motion vectors, B: Linear Depth Delta (CurrentLinearDepth - PrevLinearDepth)
RWTexture2D<half4> OutNormals : register(u3); // RG: Octahedrally encoded normals, B: Linear Roughness, A: Material Type (Optional)
RWTexture2D<half4> OutSpecAlbedo : register(u4); // RGB: Specular Albedo, A: dot(Normal, ViewDir)
RWTexture2D<half4> OutDiffAlbedo : register(u5); // RGB: Diffuse Albedo, A: Metalness (not provided)

RWTexture2D<half4> OutSkipSignal : register(u6);

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
    float3 _Padding;
};

bool IsSet(uint mask) { return (Flags & mask) == mask; }
uint GetDebugMode() { return (Flags & FLAGS_DEBUG_MODE_MASK); }

// Lower bound on the albedo used as a demodulation divisor.
//
// The albedos are clamped to 1e-4 to stay non zero, which permits a gain of 1e4 on
// dark surfaces. Radiance noise scaled by that lands well outside the range FP16
// resolves usefully, and the denoiser sees a signal whose magnitude swings by orders
// of magnitude between frames. Capping the gain at ~1/kMinReflectance keeps the
// demodulated signal in a sane range; whatever energy that costs is recovered by the
// existing residual path, which folds it into the skip signal.
static const float kMinReflectance = 8e-3f;

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
    const float isEmissive = SoftAbove(totalAlbedo, 5.9f, 0.5f);
    diffAlbedo.rgb *= (1.0f - isEmissive);
    specReflectance.rgb = lerp(specReflectance.rgb, 0.1f, isEmissive);
    
    // Clamp albedo
    const float3 albedoOvershoot = max((specReflectance.rgb + diffAlbedo.rgb) - 1.0f, 0.0f);
    specReflectance.rgb = saturate(specReflectance.rgb - albedoOvershoot);
    diffAlbedo.rgb -= max((specReflectance.rgb + diffAlbedo.rgb) - 1.0f, 0.0f);
    specReflectance.rgb = max(specReflectance.rgb, 1e-4f);
    diffAlbedo.rgb = max(diffAlbedo.rgb, 1e-4f);
    
    // Denoiser input color and floor residual
    const float3 rawColor = GetSafeFP16(InColor[px].rgb);
    float4 floorColor = InFloorColor[px];  
    const float rawLuma = GetLuminance(rawColor);
    const float floorLuma = GetLuminance(floorColor.rgb);
    floorColor.a = floorLuma;

    // Floor color blending
    //
    // Diffuse dominant surfaces are relatively well behaved.
    const float avgSpecular = dot(specReflectance.rgb, 0.33f);
    const float diffuseDominance = smoothstep(0.08f, 0.0f, avgSpecular);
    const float similarityThreshold = lerp(0.5f, 0.2f, diffuseDominance);
    
    // Clamp floor to minimum and blend in raw values where similar to preserve microcontrast.
    const float floorSimilarity = GetRelativeSimilarity(floorLuma, rawLuma, similarityThreshold);
    floorColor.rgb = FloorIsolation * lerp(floorColor.rgb, rawColor, saturate(floorSimilarity));

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

    floorColor.rgb = min(rawColor, floorColor.rgb);
    const float3 denoiserColor = rawColor - floorColor.rgb;

    // Depth - full position needed for reprojected depth delta
    const float3 viewSpacePos = GetViewSpacePos(px);
    const float compressedDepth = log(viewSpacePos.z + 1.0f) / log(FarPlane + 1.0f);
    
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
        float roughness = IsSet(FLAGS_PACKED_ROUGHNESS) ? worldSurfaceNormal.a : InRoughness[px];
        roughness *= (1.0f - isEmissive);
        
        // Output: RG=OctNormal, B=Roughness, A=MaterialID
        OutNormals[px] = GetSafeFP16(float4(octNormal, roughness, materialType));
   
        // Motion Vectors & Depth Delta
        //
        // Find the current pixel in world space and calculate movement in view space
        const float3 worldSpacePos = mul(InvViewMatrix, float4(viewSpacePos, 1.0f)).xyz;
        float3 prevViewSpacePos = mul(PrevViewMatrix, float4(worldSpacePos, 1.0f)).xyz;
        prevViewSpacePos.z = abs(prevViewSpacePos.z);
            
        // FSR-RR requires Linear Depth Delta in Blue channel
        const float2 motionIn = InMotionVectors[px].rg; // RG: Pixel Movement
        const float depthDelta = (prevViewSpacePos.z - viewSpacePos.z);
        const float3 motionOut = float3(motionIn, depthDelta);
        OutMotion[px] = half4(motionOut, 0.0f);

        half hitDist = 0.0f;
        half3 demodColor = 0.0f;
        float3 fusedAlbedo = 0.0f;
        float demodGain = 0.0f;

        [branch]
        if (IsSet(FLAGS_MODE_2_SIGNAL)) // Primary radiance packing - Mode 2 Signal
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

            const float3 specularColor = denoiserColor * (specFraction * isSplitValid);
            const float3 diffuseColor = denoiserColor - specularColor;

            // Demodulate against a floored divisor. The remodulation below still uses the
            // true albedo, so the gap between the two is caught by the residual and routed
            // into the skip signal rather than silently lost.
            const float3 specDenom = max(specReflectance.rgb, kMinReflectance);
            const float3 diffDenom = max(diffAlbedo.rgb, kMinReflectance);

            half3 demodSpecular = GetSafeFP16(specularColor / specDenom);
            half3 demodDiffuse = GetSafeFP16(diffuseColor / diffDenom);

            demodGain = rcp(min(GetLuminance(specDenom), GetLuminance(diffDenom)));

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
            const float canUseHitDist = SoftBelow(roughness, 0.30f, 0.15f)
                                      * (1.0f - isEmissive)
                                      * (1.0f - biasWeight);
            hitDist = GetSafeFP16(InSpecHitDist[px] * canUseHitDist);
            
            [branch]
            if (!IsSet(FLAGS_DEBUG))
            {
                OutSignal1[px] = half4(demodSpecular, hitDist);
                OutSignal2[px] = half4(demodDiffuse, 0.0f);
            }
            else
                demodColor = demodDiffuse + demodSpecular;
        }
        else // Primary radiance packing - Mode 1 Signal
        {           
            fusedAlbedo = max(specReflectance.rgb, diffAlbedo.rgb);

            // Same floored divisor as Mode 2, for the same reason. Remodulation still uses
            // the true fused albedo so the residual path stays energy conserving.
            const float3 fusedDenom = max(fusedAlbedo.rgb, kMinReflectance);
            demodColor = GetSafeFP16(denoiserColor / fusedDenom);
            demodGain = rcp(GetLuminance(fusedDenom));

            const float3 residual = max(0.0f, denoiserColor - (demodColor * fusedAlbedo.rgb));
            floorColor.rgb += residual;
            
            [branch]
            if (!IsSet(FLAGS_NON_GAMMA_ALBEDO))
                fusedAlbedo = sqrt(fusedAlbedo);
            
            [branch]
            if (!IsSet(FLAGS_DEBUG))
            {
                OutSignal1[px] = half4(demodColor, hitDist);
                OutSignal2[px] = half4(GetSafeFP16(fusedAlbedo), 0.0f);
            }
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
        OutSkipSignal[px] = half4(GetSafeFP16(floorColor));
        
        [branch]
        if (IsSet(FLAGS_DEBUG))
        {
            float3 debugColor = float3(0, 0, 0);
        
            switch (GetDebugMode())
            {
                // Inputs
                case FLAGS_DEBUG_IN_SPEC_HIT_DIST:
                    debugColor = TurboColormap(frac(hitDist * 0.1f));
                    break;
                
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
                case FLAGS_DEBUG_OUT_FUSED_ALBEDO:
                    debugColor = fusedAlbedo.rgb;
                    break;
                
                case FLAGS_DEBUG_OUT_LINEAR_DEPTH:
                    debugColor = TurboColormap(frac(viewSpacePos.z * 0.1));
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
                
                default:
                    debugColor = demodColor;
                    break;
            }
        
            OutSignal1[px] = half4(debugColor, 1.0f);
        }
    }
    else // Skip
    {
        OutNormals[px] = 0.0f;
        OutSpecAlbedo[px] = 0.0f;
        OutDiffAlbedo[px] = 0.0f;
        OutSignal1[px] = 0.0f;
        OutSignal2[px] = 0.0f;
        OutSkipSignal[px] = half4(rawColor, rawLuma);
    }
}