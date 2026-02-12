// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2026

struct Params {
    // Peter de Jong coefficients: x' = sin(a*y) - cos(b*x), y' = sin(c*x) - cos(d*y).
    a: f32, b: f32, c: f32, d: f32,
    
    // Canvas size in device pixels.
    width: f32, height: f32,
    
    // Tone mapping controls (ACES + gamma/contrast in fragment).
    exposure: f32, gamma: f32, contrast: f32,
    
    // RNG seed (bitcast to u32).
    seed: f32,
    
    // Color mode: 0 density, 1–4 RGB shuffles, 5 iteration hue.
    colorMethod: u32,
    
    // Precomputed invTotal for density normalization.
    invTotal: f32,
    
    // View transform in fractal units.
    viewOffsetX: f32, viewOffsetY: f32, viewScale: f32,
    
    // Current fractal bounds (rebased).
    fractalMinX: f32, fractalMaxX: f32, fractalMinY: f32, fractalMaxY: f32,
    
    // Original bounds for random starts (immutable).
    originalFractalMinX: f32, originalFractalMaxX: f32, originalFractalMinY: f32, originalFractalMaxY: f32,
    
    // Compute dispatch metadata.
    rngMode: u32,        // 0–6 selects RNG
    dispatchDimX: u32,   // For 2D dispatch when workgroups > 65535
    workgroupCount: u32, // Total workgroups this frame
    frameOffsetLo: u32,  // Cumulative RNG index offset (low 32 bits)
    frameOffsetHi: u32,  // Cumulative RNG index offset (high 32 bits)
    frameId: u32,        // Monotonic frame counter (for per-frame variation)
    r2StartX: f32,       // CPU fract(frameOffset * R2_ALPHA) for precision mode
    r2StartY: f32,       // CPU fract(frameOffset * R2_BETA) for precision mode
    // CPU precomputed accumulation buffer center (floating origin anchor).
    viewBufferCenterX: f32,
    viewBufferCenterY: f32,

    // View center in attractor space [-2,2] to avoid cancellation on deep zoom.
    attractorCenterX: f32,
    attractorCenterY: f32,
    // Pixels per attractor unit (CPU precomputed).
    attractorPixelScale: f32,
    
    // Adaptive iteration count for deep zoom.
    iterationCount: u32,
    
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,  // 16-byte alignment (160 bytes total)
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> densityBuffer: array<atomic<u32>>;
// Color average buffers (float32 via bitcast) for running average accumulation.
@group(0) @binding(2) var<storage, read_write> colorBufferR: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> colorBufferG: array<atomic<u32>>;
@group(0) @binding(4) var<storage, read_write> colorBufferB: array<atomic<u32>>;
@group(0) @binding(5) var<storage, read> densityRead: array<u32>;
@group(0) @binding(6) var<storage, read> colorReadR: array<u32>;
@group(0) @binding(7) var<storage, read> colorReadG: array<u32>;
@group(0) @binding(8) var<storage, read> colorReadB: array<u32>;

// Color range and fixed-point scale for sum accumulation.
const RGB10_SCALE: f32 = 16.0;

// --- RNG helpers ---

fn hash12(p: vec2<f32>) -> f32 {
    var p3 = fract(vec3<f32>(p.xyx) * .1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

fn hash32(n: u32) -> u32 {
    var x = n;
    x ^= x >> 16u;
    x *= 0x85ebca6bu;
    x ^= x >> 13u;
    x *= 0xc2b2ae35u;
    x ^= x >> 16u;
    return x;
}

fn reverseBits(x: u32) -> u32 {
    var v = x;
    v = ((v & 0x55555555u) << 1u) | ((v & 0xAAAAAAAAu) >> 1u);
    v = ((v & 0x33333333u) << 2u) | ((v & 0xCCCCCCCCu) >> 2u);
    v = ((v & 0x0F0F0F0Fu) << 4u) | ((v & 0xF0F0F0F0u) >> 4u);
    v = ((v & 0x00FF00FFu) << 8u) | ((v & 0xFF00FF00u) >> 8u);
    v = ((v & 0x0000FFFFu) << 16u) | ((v & 0xFFFF0000u) >> 16u);
    return v;
}

fn lkPermute(x: u32, seed: u32) -> u32 {
    var v = x + seed;
    v ^= v * 0x6c50b47cu;
    v ^= v * 0xb82f1e52u;
    v ^= v * 0xc7afe638u;
    v ^= v * 0x8d22f6e6u;
    v ^= v >> 16u; // Mix high bits into LSBs for strict Owen + reverseBits.
    return v;
}

fn owenScramble(x: u32, seed: u32) -> u32 {
    return reverseBits(lkPermute(reverseBits(x), seed));
}

fn toUnit(u: u32) -> f32 {
    return f32(u >> 8u) * (1.0 / 16777216.0);
}

fn sobol02Bits(index: u32) -> vec2<u32> {
    let x = reverseBits(index);
    let y = x ^ (x >> 1u);
    return vec2<u32>(x, y);
}

fn sobol2dPure(index: u32, seed: u32) -> vec2<f32> {
    let offset = (seed & 0xFFFF0000u) | ((seed & 0xFFFFu) * 0x9e3779b9u);
    let offsetIndex = index + offset;
    let bits = sobol02Bits(offsetIndex);
    return vec2<f32>(toUnit(bits.x), toUnit(bits.y));
}

fn sobol2dScrambled(index: u32, seed: u32) -> vec2<f32> {
    let scrambledIndex = owenScramble(index, seed);
    let bits = sobol02Bits(scrambledIndex);
    let x = owenScramble(bits.x, seed ^ 0xa511e9b3u);
    let y = owenScramble(bits.y, seed ^ 0x63d83595u);
    return vec2<f32>(toUnit(x), toUnit(y));
}

// Practical Owen/Sobol 0,2 scramble for sampling (avoids fast visual repeats).
// Key change vs strict Owen: we DO NOT reverseBits() the final per-dimension permute
// before toUnit(), so weak low bits from lkPermute never become "big" float bits.
fn sobol2dOwenOptimized(index: u32, seed: u32) -> vec2<f32> {
    // v = lkPermute(reverseBits(index), seed)
    var v = reverseBits(index) + seed;
    v ^= v * 0x6c50b47cu;
    v ^= v * 0xb82f1e52u;
    v ^= v * 0xc7afe638u;
    v ^= v * 0x8d22f6e6u;

    // Sobol (0,2) in the same "van der Corput domain".
    let g = v ^ (v >> 1u);

    // Final per-dimension decorrelation (LK core), but feed toUnit() directly.
    // Using vec2<u32> encourages ILP / keeps code tight.
    var xy = vec2<u32>(
        v + (seed ^ 0xa511e9b3u),
        g + (seed ^ 0x63d83595u)
    );

    xy ^= xy * 0x6c50b47cu;
    xy ^= xy * 0xb82f1e52u;
    xy ^= xy * 0xc7afe638u;
    xy ^= xy * 0x8d22f6e6u;

    return vec2<f32>(toUnit(xy.x), toUnit(xy.y));
}

// R2 constants: 2D Golden Ratio (avoid 3D Plastic Number artifacts).

// Fixed-point R2 increments (floor(2^32 * 1/φ, 1/φ²)); odd => full period.
const R2_INC_X: u32 = 0x9E3779B9u;  // 2654435769 ≈ 2^32 / φ
const R2_INC_Y: u32 = 0x61C88647u;  // 1640531527 ≈ 2^32 / φ²

// Float R2 deltas (single-step only; fixed-point for index math).
const R2_ALPHA: f32 = 0.6180339887498949; // 1/φ (Golden Ratio, single-step delta, f32 exact)
const R2_BETA: f32 = 0.3819660112501052;  // 1/φ² (single-step delta, f32 exact)

fn r2Sequence(index: u32) -> vec2<f32> {
    // Fixed-point: (index * INC) mod 2^32.
    let x = index * R2_INC_X;
    let y = index * R2_INC_Y;
    return vec2<f32>(toUnit(x), toUnit(y));
}

fn cpOffset(seed: f32) -> vec2<f32> {
    let seedBits = bitcast<u32>(seed);
    let h1 = hash32(seedBits);
    let h2 = hash32(seedBits + 0x9e3779b9u);
    return vec2<f32>(toUnit(h1), toUnit(h2));
}

fn r2WithRotation(index: u32, seed: f32) -> vec2<f32> {
    return fract(r2Sequence(index) + cpOffset(seed));
}

// Integer hash RNG (avoids f32 precision loss at large indices).
fn rand2_u32(index: u32, seedHash: u32) -> vec2<f32> {
    let x = hash32(index ^ seedHash);
    let y = hash32((index + 0x9e3779b9u) ^ seedHash);
    return vec2<f32>(toUnit(x), toUnit(y));
}

// Combined LCG (2 classic LCGs, raw output).
fn combinedLCG(index: u32, seedHash: u32) -> vec2<f32> {
    // Init states from index + seed.
    var state1 = u32(index) ^ seedHash;
    var state2 = u32(index + 0x9e3779b9u) ^ seedHash;
    
    // LCG1: Numerical Recipes.
    state1 = 1664525u * state1 + 1013904223u;
    
    // LCG2: Borland.
    state2 = 134775813u * state2 + 1u;
    
    // Combine via XOR; shift to decorrelate X/Y.
    let combinedX = state1 ^ state2;
    let combinedY = (state1 << 16u) ^ (state2 >> 16u);
    
    // Normalize to [0,1) without redistribution.
    return vec2<f32>(
        f32(combinedX) / 4294967296.0,
        f32(combinedY) / 4294967296.0
    );
}

fn getRandom2D(index: u32, seed: f32, rngMode: u32) -> vec2<f32> {
    let seedBits = bitcast<u32>(seed);
    
    switch rngMode {
        case 6u: {
            // Combined LCG: raw random; mix frameId for per-frame variation.
            let frameMix = hash32(params.frameId ^ 0x9e3779b9u);
            let seedHash = hash32(seedBits ^ frameMix);
            return combinedLCG(index, seedHash);
        }
        case 5u: {
            // Owen-Sobol: keep scramble stable; use CP rotation per 2^32 wrap.
            let seedHash = hash32(seedBits);
            let wrapOffset = cpOffset(f32(params.frameOffsetHi));
            return fract(sobol2dOwenOptimized(index, seedHash) + wrapOffset);
        }
        case 4u: {
            // R2 precision: return thread start; caller advances by add recurrence.
            let threadBase = (index / 128u) * 128u;
            let baseBitsX = threadBase * R2_INC_X;
            let baseBitsY = threadBase * R2_INC_Y;
            return fract(vec2<f32>(
                params.r2StartX + toUnit(baseBitsX),
                params.r2StartY + toUnit(baseBitsY)
            ));
        }
        case 3u: {
            // Sobol pure; CP rotation per 2^32 wrap.
            let seedHash = hash32(seedBits);
            let wrapOffset = cpOffset(f32(params.frameOffsetHi));
            return fract(sobol2dPure(index, seedHash) + wrapOffset);
        }
        case 2u: {
            // R2 legacy: apply CP rotation per 2^32 wrap.
            let wrapOffset = cpOffset(f32(params.frameOffsetHi));
            return fract(r2Sequence(index) + wrapOffset);
        }
        case 1u: {
            // Sobol scrambled; CP rotation per 2^32 wrap.
            let seedHash = hash32(seedBits);
            let wrapOffset = cpOffset(f32(params.frameOffsetHi));
            return fract(sobol2dScrambled(index, seedHash) + wrapOffset);
        }
        default: {
            // Hash RNG: integer hash + frameId for variation.
            let frameMix = hash32(params.frameId ^ 0x9e3779b9u);
            let seedHash = hash32(seedBits ^ frameMix);
            return rand2_u32(index, seedHash);
        }
    }
}

// Compute how many color updates have occurred for a given lastCount (prevCount).
fn colorUpdateCount(lastCount: u32) -> f32 {
    let logCount = 31u - countLeadingZeros(lastCount | 1u);
    let tier = max(0i, i32(logCount) - 6);
    if (tier <= 0) {
        return f32(lastCount) + 1.0;
    }
    let throttleBits = min(u32(tier) + 3u, 10u);
    let stride = 1u << throttleBits;
    let tierStart = 1u << (u32(tier) + 6u);
    let priorTierUpdates = 8u * u32(max(0i, tier - 1));
    let updatesInTier = (lastCount - tierStart) / stride;
    return f32(128u + 1u + priorTierUpdates + updatesInTier);
}

// Maximum color updates before considering pixel "converged".
const MAX_COLOR_UPDATES: f32 = 100000.0;

// Atomic CAS-based float32 running average blend for RGB channels.
fn atomicBlendRGBIndex(
    index: u32,
    rgb: vec3<f32>,
    alpha: f32
) {
    var oldR = atomicLoad(&colorBufferR[index]);
    var oldG = atomicLoad(&colorBufferG[index]);
    var oldB = atomicLoad(&colorBufferB[index]);

    for (var k = 0u; k < 8u; k++) {
        let oldVec = vec3<f32>(
            bitcast<f32>(oldR),
            bitcast<f32>(oldG),
            bitcast<f32>(oldB)
        );
        let newVec = oldVec + (rgb - oldVec) * alpha;

        let resR = atomicCompareExchangeWeak(&colorBufferR[index], oldR, bitcast<u32>(newVec.r));
        let resG = atomicCompareExchangeWeak(&colorBufferG[index], oldG, bitcast<u32>(newVec.g));
        let resB = atomicCompareExchangeWeak(&colorBufferB[index], oldB, bitcast<u32>(newVec.b));

        if (resR.exchanged && resG.exchanged && resB.exchanged) { return; }

        if (!resR.exchanged) { oldR = resR.old_value; }
        if (!resG.exchanged) { oldG = resG.old_value; }
        if (!resB.exchanged) { oldB = resB.old_value; }
    }
}

// Hue -> RGB helper.
fn iterationHueToRGB(t: f32) -> vec3<f32> {
    let clamped = clamp(t, 0.0, 1.0);
    let a = vec3<f32>(0.5, 0.5, 0.5);
    let b = vec3<f32>(0.5, 0.5, 0.5);
    let c = vec3<f32>(1.0, 1.0, 1.0);
    let d = vec3<f32>(0.0, 0.333, 0.667);
    return a + b * cos(6.28318 * (c * clamped + d));
}

// --- COMPUTE SHADER ---

@compute @workgroup_size(64)
fn computeMainDensity(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
    let groupIndex = wg_id.x + wg_id.y * params.dispatchDimX;
    if (groupIndex >= params.workgroupCount) { return; }
    let threadIndex = groupIndex * 64u + local_id.x;
    let index = threadIndex + params.frameOffsetLo;

    // RNG setup (same as computeMain, but density-only never needs jitter).
    var r2State: vec2<f32> = vec2<f32>(0.0);
    let useR2Precision = params.rngMode == 4u;
    if (useR2Precision) {
        let needsJitter = false;
        let stride: u32 = select(1u, 129u, needsJitter);
        let threadSampleOffset = threadIndex * stride;
        let threadOffsetBitsX = threadSampleOffset * R2_INC_X;
        let threadOffsetBitsY = threadSampleOffset * R2_INC_Y;
        r2State = fract(vec2<f32>(
            params.r2StartX + toUnit(threadOffsetBitsX),
            params.r2StartY + toUnit(threadOffsetBitsY)
        ));
        if (params.frameOffsetHi != 0u) {
            let wrapOffset = cpOffset(f32(params.frameOffsetHi));
            r2State = fract(r2State + wrapOffset);
        }
    }

    // Get initial random values.
    var rnd: vec2<f32>;
    if (useR2Precision) {
        rnd = r2State;
        r2State = fract(r2State + vec2<f32>(R2_ALPHA, R2_BETA));
    } else {
        rnd = getRandom2D(index, params.seed, params.rngMode);
    }

    // Starting point: original bounds.
    let originalRangeX = params.originalFractalMaxX - params.originalFractalMinX;
    let originalRangeY = params.originalFractalMaxY - params.originalFractalMinY;
    let originalCenterX = params.originalFractalMinX + originalRangeX * 0.5;
    let originalCenterY = params.originalFractalMinY + originalRangeY * 0.5;

    var x = originalCenterX + (rnd.x - 0.5) * originalRangeX;
    var y = originalCenterY + (rnd.y - 0.5) * originalRangeY;

    // Warmup (settle onto attractor).
    for (var j = 0; j < 12; j++) {
        let nx = sin(params.a * y) - cos(params.b * x);
        let ny = sin(params.c * x) - cos(params.d * y);
        x = nx;
        y = ny;
    }

    if (x * x + y * y > 16.0) { return; }

    // Adaptive accumulation (density-only).
    let w = u32(params.width);
    let h = u32(params.height);
    let scale = params.attractorPixelScale;
    let halfW = params.width * 0.5;
    let halfH = params.height * 0.5;

    var sinAY = sin(params.a * y);
    var cosBX = cos(params.b * x);
    var sinCX = sin(params.c * x);
    var cosDY = cos(params.d * y);

    var hitCount = 0u;
    var itersSinceHit = 0u;

    for (var i = 0u; i < params.iterationCount; i++) {
        let nx = sinAY - cosBX;
        let ny = sinCX - cosDY;

        if (nx * nx + ny * ny > 16.0) { break; }

        let next_sinAY = sin(params.a * ny);
        let next_cosBX = cos(params.b * nx);
        let next_sinCX = sin(params.c * nx);
        let next_cosDY = cos(params.d * ny);

        let dx = nx - params.attractorCenterX;
        let dy = ny - params.attractorCenterY;

        let px = fma(dx, scale, halfW);
        let py = fma(dy, scale, halfH);
        // Floor for negative values without calling floor().
        let sx = i32(px * 65536.0) >> 16;
        let sy = i32(py * 65536.0) >> 16;

        if (u32(sx) < w && u32(sy) < h) {
            let pixelIndex = u32(sy) * w + u32(sx);
            atomicAdd(&densityBuffer[pixelIndex], 1u);
            hitCount += 1u;
            itersSinceHit = 0u;
        } else {
            itersSinceHit += 1u;

            // EARLY EXIT: Never found viewport.
            let neverHitLimit = select(512u, 256u, scale > 5000.0);
            if (hitCount == 0u && i >= neverHitLimit) { break; }

            // EARLY EXIT: Was productive but wandered off.
            let missLimit = select(256u, 64u, scale > 5000.0);
            if (hitCount > 0u && itersSinceHit > missLimit) { break; }
        }

        x = nx;
        y = ny;
        sinAY = next_sinAY;
        cosBX = next_cosBX;
        sinCX = next_sinCX;
        cosDY = next_cosDY;
    }
}

@compute @workgroup_size(64)
fn computeMain(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
    let groupIndex = wg_id.x + wg_id.y * params.dispatchDimX;
    if (groupIndex >= params.workgroupCount) { return; }
    let threadIndex = groupIndex * 64u + local_id.x;
    let index = threadIndex + params.frameOffsetLo;
    
    // ═══════════════════════════════════════════════════════════════════════
    // RNG SETUP (existing logic preserved)
    // ═══════════════════════════════════════════════════════════════════════
    
    var r2State: vec2<f32> = vec2<f32>(0.0);
    let useR2Precision = params.rngMode == 4u;
    if (useR2Precision) {
        let needsJitter = params.colorMethod != 0u;
        let stride: u32 = select(1u, 129u, needsJitter);
        let threadSampleOffset = threadIndex * stride;
        let threadOffsetBitsX = threadSampleOffset * R2_INC_X;
        let threadOffsetBitsY = threadSampleOffset * R2_INC_Y;
        r2State = fract(vec2<f32>(
            params.r2StartX + toUnit(threadOffsetBitsX),
            params.r2StartY + toUnit(threadOffsetBitsY)
        ));
        if (params.frameOffsetHi != 0u) {
            let wrapOffset = cpOffset(f32(params.frameOffsetHi));
            r2State = fract(r2State + wrapOffset);
        }
    }
    
    // Get initial random values
    var rnd: vec2<f32>;
    if (useR2Precision) {
        rnd = r2State;
        r2State = fract(r2State + vec2<f32>(R2_ALPHA, R2_BETA));
    } else {
        rnd = getRandom2D(index, params.seed, params.rngMode);
    }
    
    // ═══════════════════════════════════════════════════════════════════════
    // STARTING POINT: Always use original bounds (standard sampling)
    // ═══════════════════════════════════════════════════════════════════════
    
    let originalRangeX = params.originalFractalMaxX - params.originalFractalMinX;
    let originalRangeY = params.originalFractalMaxY - params.originalFractalMinY;
    let originalCenterX = params.originalFractalMinX + originalRangeX * 0.5;
    let originalCenterY = params.originalFractalMinY + originalRangeY * 0.5;
    
    var x = originalCenterX + (rnd.x - 0.5) * originalRangeX;
    var y = originalCenterY + (rnd.y - 0.5) * originalRangeY;
    
    // ═══════════════════════════════════════════════════════════════════════
    // WARMUP (settle onto attractor)
    // ═══════════════════════════════════════════════════════════════════════
    
    for (var i = 0; i < 12; i++) {
        let nx = sin(params.a * y) - cos(params.b * x);
        let ny = sin(params.c * x) - cos(params.d * y);
        x = nx;
        y = ny;
    }
    
    if (x * x + y * y > 16.0) { return; }
    
    // ═══════════════════════════════════════════════════════════════════════
    // ADAPTIVE ACCUMULATION (extended iterations + early exit)
    // ═══════════════════════════════════════════════════════════════════════
    
    let w = u32(params.width);
    let h = u32(params.height);
    let scale = params.attractorPixelScale;
    let halfW = params.width * 0.5;
    let halfH = params.height * 0.5;
    
    var sinAY = sin(params.a * y);
    var cosBX = cos(params.b * x);
    var sinCX = sin(params.c * x);
    var cosDY = cos(params.d * y);
    
    var jitterState: u32 = 0u;
    let needsJitter = params.colorMethod != 0u;
    if (needsJitter) {
        let seedBits = bitcast<u32>(params.seed);
        let frameMix = hash32(params.frameId ^ 0x9e3779b9u);
        jitterState = (index ^ hash32(seedBits ^ frameMix)) * 747796405u + 2891336453u;
    }
    
    var prevX = x;
    var prevY = y;
    var hitCount = 0u;
    var itersSinceHit = 0u;
    
    for (var i = 0u; i < params.iterationCount; i++) {
        let nx = sinAY - cosBX;
        let ny = sinCX - cosDY;
        
        if (nx * nx + ny * ny > 16.0) { break; }
        
        let next_sinAY = sin(params.a * ny);
        let next_cosBX = cos(params.b * nx);
        let next_sinCX = sin(params.c * nx);
        let next_cosDY = cos(params.d * ny);
        
        let dx = nx - params.attractorCenterX;
        let dy = ny - params.attractorCenterY;
        
        var jitter = vec2<f32>(0.0);
        if (needsJitter) {
            jitterState = jitterState * 747796405u + 2891336453u;
            jitter = vec2<f32>(
                f32(jitterState & 0xFFFFu) / 65536.0 - 0.5,
                f32(jitterState >> 16u) / 65536.0 - 0.5
            );
        }
        
        // fma() keeps a single rounding step; some drivers already fuse mul+add,
        // so this may or may not improve precision (performance is typically neutral).
        let px = fma(dx, scale, halfW);
        let py = fma(dy, scale, halfH);
        let sx = i32((px + jitter.x) * 65536.0) >> 16;
        let sy = i32((py + jitter.y) * 65536.0) >> 16;
        
        if (u32(sx) < w && u32(sy) < h) {
            let pixelIndex = u32(sy) * w + u32(sx);
            
            if (params.colorMethod == 0u) {
                atomicAdd(&densityBuffer[pixelIndex], 1u);
            } else {
                let prevCount = atomicAdd(&densityBuffer[pixelIndex], 1u);
                
                let logCount = 31u - countLeadingZeros(prevCount | 1u);
                let tier = max(0i, i32(logCount) - 6);
                let throttleBits = min(u32(max(0i, tier)) + 3u, 10u);
                let mask = select(0u, (1u << throttleBits) - 1u, tier > 0);
                let shouldUpdateColor = (prevCount & mask) == 0u;
                
                if (shouldUpdateColor) {
                    let colorUpdates = colorUpdateCount(prevCount);
                    
                    if (colorUpdates <= MAX_COLOR_UPDATES) {
                        var rgb: vec3<f32>;
                        
                        if (params.colorMethod == 5u) {
                            const MAX_ITER_LOG: f32 = 12.0;
                            let countF = f32(prevCount);
                            let logCountF = log2(max(countF, 1.0));
                            let normalized = clamp(logCountF / MAX_ITER_LOG, 0.0, 1.0);
                            rgb = iterationHueToRGB(normalized) * RGB10_SCALE;
                        } else {
                            let base = abs(vec3<f32>(prevY - ny, prevX - ny, prevX - nx)) * 2.5;
                            switch params.colorMethod {
                                case 1u: { rgb = vec3<f32>(base.x, base.y, base.z); }
                                case 2u: { rgb = vec3<f32>(base.y, base.z, base.x); }
                                case 3u: { rgb = vec3<f32>(base.z, base.x, base.y); }
                                case 4u: { rgb = vec3<f32>(base.z, base.y, base.x); }
                                default: { rgb = base; }
                            }
                        }
                        
                        rgb = clamp(rgb, vec3<f32>(0.0), vec3<f32>(RGB10_SCALE));
                        
                        if (prevCount == 0u) {
                            atomicStore(&colorBufferR[pixelIndex], bitcast<u32>(rgb.r));
                            atomicStore(&colorBufferG[pixelIndex], bitcast<u32>(rgb.g));
                            atomicStore(&colorBufferB[pixelIndex], bitcast<u32>(rgb.b));
                        } else {
                            let alpha = 1.0 / colorUpdates;
                            atomicBlendRGBIndex(pixelIndex, rgb, alpha);
                        }
                    }
                }
            }
            
            hitCount += 1u;
            itersSinceHit = 0u;
        } else {
            itersSinceHit += 1u;
            
            // EARLY EXIT: Never found viewport - trajectory isn't passing through visible region
            let neverHitLimit = select(512u, 256u, scale > 5000.0);
            if (hitCount == 0u && i >= neverHitLimit) { break; }
            
            // EARLY EXIT: Was productive but wandered off - won't likely return at deep zoom
            // Threshold is tighter when zoomed in (scale > 5000 means >~25× zoom)
            let missLimit = select(256u, 64u, scale > 5000.0);
            if (hitCount > 0u && itersSinceHit > missLimit) { break; }
        }
        
        prevX = x;
        prevY = y;
        x = nx;
        y = ny;
        sinAY = next_sinAY;
        cosBX = next_cosBX;
        sinCX = next_sinCX;
        cosDY = next_cosDY;
    }
}

// --- RENDER SHADER ---

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vertexMain(@builtin(vertex_index) vertexIndex: u32) -> VertexOutput {
    // Keep array inside function to avoid older backend issues.
    let pos = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(-1.0, 1.0),
        vec2<f32>(-1.0, 1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0)
    );
    var output: VertexOutput;
    output.position = vec4<f32>(pos[vertexIndex], 0.0, 1.0);
    output.uv = pos[vertexIndex] * 0.5 + 0.5;
    output.uv.y = 1.0 - output.uv.y;
    return output;
}

fn getGradient(t: f32) -> vec3<f32> {
    let col1 = vec3<f32>(0.0, 0.0, 0.0);
    let col2 = vec3<f32>(0.1, 0.0, 0.4);
    let col3 = vec3<f32>(0.8, 0.1, 0.3);
    let col4 = vec3<f32>(1.0, 0.9, 0.5);

    if (t < 0.33) { return mix(col1, col2, t * 3.0); }
    if (t < 0.66) { return mix(col2, col3, (t - 0.33) * 3.0); }
    return mix(col3, col4, (t - 0.66) * 3.0);
}

// ACES filmic tone mapping.
fn toneMapACES(v: vec3<f32>, exposure: f32) -> vec3<f32> {
    let x = v * exposure * 0.6;
    let a = 2.51; let b = 0.03;
    let c = 2.43; let d = 0.59; let e = 0.14;
    return saturate((x * (a * x + b)) / (x * (c * x + d) + e));
}

// Tone mapping helper: ACES + optional saturation + gamma/contrast.
fn applyToneMapping(rawR: f32, rawG: f32, rawB: f32, boostSaturation: bool) -> vec3<f32> {
    // ACES outputs [0,1] via saturate().
    var color = toneMapACES(vec3<f32>(rawR, rawG, rawB), params.exposure);

    if (boostSaturation) {
        let maxVal = max(max(color.r, color.g), color.b);
        let minVal = min(min(color.r, color.g), color.b);
        let delta = maxVal - minVal;
        if (delta > 0.001 && maxVal > 0.001) {
            let saturation = delta / maxVal;
            let newSaturation = min(1.0, saturation * 1.5);
            let scale = newSaturation / saturation;
            color = minVal + (color - minVal) * scale;
        }
    }

    // Vectorized gamma.
    color = pow(color, vec3<f32>(params.gamma));

    // Vectorized contrast with final clamp.
    color = saturate((color - 0.5) * params.contrast + 0.5);

    return color;
}

@fragment
fn fragmentMainDensity(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {
    // Floating origin: center + relative offset keeps intermediates small.
    let relX = (uv.x - 0.5) * params.width  / params.viewScale;
    let relY = (uv.y - 0.5) * params.height / params.viewScale;

    // Add relative offset to CPU-computed center.
    let bufferX = params.viewBufferCenterX + relX;
    let bufferY = params.viewBufferCenterY + relY;

    if (bufferX < 0.0 || bufferX >= params.width || bufferY < 0.0 || bufferY >= params.height) {
        return vec4<f32>(0.0, 0.0, 0.0, 1.0);
    }

    let x = u32(clamp(bufferX, 0.0, params.width - 1.0));
    let y = u32(clamp(bufferY, 0.0, params.height - 1.0));
    let index = y * u32(params.width) + x;

    let count = densityRead[index];
    if (count == 0u) {
        return vec4<f32>(0.0, 0.0, 0.0, 1.0);
    }

    let invTotal: f32 = params.invTotal;
    let densityBrightnessScale = 250.0;
    let val = f32(count) * invTotal * densityBrightnessScale;

    // Density-only mode (ACES for consistency).
    let mapped = toneMapACES(vec3<f32>(val), params.exposure).r;
    let gray = pow(mapped, params.gamma);
    let contrasted = saturate((gray - 0.5) * params.contrast + 0.5);
    let col = getGradient(contrasted);
    return vec4<f32>(col, 1.0);
}

@fragment
fn fragmentMain(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {
    // Floating origin: center + relative offset keeps intermediates small.
    let relX = (uv.x - 0.5) * params.width  / params.viewScale;
    let relY = (uv.y - 0.5) * params.height / params.viewScale;

    // Add relative offset to CPU-computed center.
    let bufferX = params.viewBufferCenterX + relX;
    let bufferY = params.viewBufferCenterY + relY;

    if (bufferX < 0.0 || bufferX >= params.width || bufferY < 0.0 || bufferY >= params.height) {
        return vec4<f32>(0.0, 0.0, 0.0, 1.0);
    }
    
    let x = u32(clamp(bufferX, 0.0, params.width - 1.0));
    let y = u32(clamp(bufferY, 0.0, params.height - 1.0));
    let index = y * u32(params.width) + x;
    
    let count = densityRead[index];
    
    // Early-out for empty pixels.
    if (count == 0u) {
        return vec4<f32>(0.0, 0.0, 0.0, 1.0);
    }
    
    // invTotal precomputed on CPU.
    let invTotal: f32 = params.invTotal;
    let densityBrightnessScale = 250.0;
    
    var col: vec3<f32>;
    
    if (params.colorMethod >= 1u && count > 0u) {
        let colorAvg = vec3<f32>(
            bitcast<f32>(colorReadR[index]),
            bitcast<f32>(colorReadG[index]),
            bitcast<f32>(colorReadB[index])
        );

        let density = f32(count) * invTotal;
        const BRIGHTNESS_SCALE: f32 = 250.0;

        let boostSat = params.colorMethod != 5u;

        col = applyToneMapping(
            colorAvg.r * density * BRIGHTNESS_SCALE,
            colorAvg.g * density * BRIGHTNESS_SCALE,
            colorAvg.b * density * BRIGHTNESS_SCALE,
            boostSat
        );
    } else {
        // Density-only mode (ACES for consistency).
        let val = f32(count) * invTotal * densityBrightnessScale;
        let mapped = toneMapACES(vec3<f32>(val), params.exposure).r;
        let gray = pow(mapped, params.gamma);
        let contrasted = saturate((gray - 0.5) * params.contrast + 0.5);
        col = getGradient(contrasted);
    }

    return vec4<f32>(col, 1.0);
}
