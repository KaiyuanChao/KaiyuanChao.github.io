struct Params {
    a: f32,
    b: f32,
    c: f32,
    d: f32,
    width: f32,
    height: f32,
    exposure: f32,
    gamma: f32,
    contrast: f32,
    seed: f32,
    colorMethod: u32,
    totalIterations: f32,
    viewOffsetX: f32,
    viewOffsetY: f32,
    viewScale: f32,
    fractalMinX: f32,
    fractalMaxX: f32,
    fractalMinY: f32,
    fractalMaxY: f32,
    originalFractalMinX: f32,
    originalFractalMaxX: f32,
    originalFractalMinY: f32,
    originalFractalMaxY: f32,
    rngMode: u32,
    dispatchDimX: u32,
    workgroupCount: u32,
    frameOffset: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> densityBuffer: array<atomic<u32>>;
@group(0) @binding(2) var<storage, read_write> colorBufferR: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> colorBufferG: array<atomic<u32>>;
@group(0) @binding(4) var<storage, read_write> colorBufferB: array<atomic<u32>>;
@group(0) @binding(5) var<storage, read> densityRead: array<u32>;
@group(0) @binding(6) var<storage, read> colorReadR: array<u32>;
@group(0) @binding(7) var<storage, read> colorReadG: array<u32>;
@group(0) @binding(8) var<storage, read> colorReadB: array<u32>;

// --- RNG Helper Functions ---

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

// R2 Constants
const R2_ALPHA: f32 = 0.7548776662466927;
const R2_BETA: f32 = 0.5698402909980532;
const PHI_INV: f32 = 0.6180339887498949;

fn r2Sequence(index: u32) -> vec2<f32> {
    let n = f32(index);
    return fract(vec2<f32>(n * R2_ALPHA, n * R2_BETA));
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

fn xorshift32(state: u32) -> u32 {
    var s = state;
    s ^= s << 13u;
    s ^= s >> 17u;
    s ^= s << 5u;
    return s;
}

fn getRandom2D(index: u32, seed: f32, rngMode: u32) -> vec2<f32> {
    switch rngMode {
        case 3u: {
            let seedHash = hash32(bitcast<u32>(seed));
            return sobol2dPure(index, seedHash);
        }
        case 2u: {
            return r2WithRotation(index, seed);
        }
        case 1u: {
            let seedHash = hash32(bitcast<u32>(seed));
            return sobol2dScrambled(index, seedHash);
        }
        default: {
            let hashX = hash12(vec2<f32>(f32(index), seed));
            let hashY = hash12(vec2<f32>(f32(index) + 13.0, seed));
            return vec2<f32>(hashX, hashY);
        }
    }
}

// Helper for Hue to RGB
fn iterationHueToRGB(t: f32) -> vec3<f32> {
    let clamped = clamp(t, 0.0, 1.0);
    let a = vec3<f32>(0.5, 0.5, 0.5);
    let b = vec3<f32>(0.5, 0.5, 0.5);
    let c = vec3<f32>(1.0, 1.0, 1.0);
    let d = vec3<f32>(0.0, 0.333, 0.667);
    return a + b * cos(6.28318 * (c * clamped + d));
}

// --- COMPUTE SHADER ---

@compute @workgroup_size(256)
fn clearMain(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    let w = u32(params.width);
    let h = u32(params.height);
    let bufferSize = w * h;
    
    if (index < bufferSize) {
        atomicExchange(&densityBuffer[index], 0u);
        atomicExchange(&colorBufferR[index], 0u);
        atomicExchange(&colorBufferG[index], 0u);
        atomicExchange(&colorBufferB[index], 0u);
    }
}

@compute @workgroup_size(64)
fn computeMain(@builtin(global_invocation_id) global_id: vec3<u32>, @builtin(workgroup_id) wg_id: vec3<u32>, @builtin(local_invocation_id) local_id: vec3<u32>) {
    let groupIndex = wg_id.x + wg_id.y * params.dispatchDimX;
    if (groupIndex >= params.workgroupCount) { return; }
    let threadIndex = groupIndex * 64u + local_id.x;
    let index = threadIndex + params.frameOffset;
    
    let originalRangeX = params.originalFractalMaxX - params.originalFractalMinX;
    let originalRangeY = params.originalFractalMaxY - params.originalFractalMinY;
    let rnd = getRandom2D(index, params.seed, params.rngMode);
    var x = params.originalFractalMinX + rnd.x * originalRangeX;
    var y = params.originalFractalMinY + rnd.y * originalRangeY;

    let w = u32(params.width);
    let h = u32(params.height);
    let fractalRangeX = params.fractalMaxX - params.fractalMinX;
    let fractalRangeY = params.fractalMaxY - params.fractalMinY;
    
    let baseScale = min(params.width, params.height) * 0.2;
    let baseRange = 4.0;
    let maxRange = max(fractalRangeX, fractalRangeY);
    let scale = baseScale * (baseRange / maxRange) * 0.95;
    
    let fractalCenterX = (params.fractalMinX + params.fractalMaxX) * 0.5;
    let fractalCenterY = (params.fractalMinY + params.fractalMaxY) * 0.5;
    let centerX = params.width * 0.5 - fractalCenterX * scale;
    let centerY = params.height * 0.5 - fractalCenterY * scale;

    for(var i = 0; i < 12; i++) {
        let nx = sin(params.a * y) - cos(params.b * x);
        let ny = sin(params.c * x) - cos(params.d * y);
        x = nx;
        y = ny;
    }

    var jitterState: u32 = 0u;
    let needsJitter = params.colorMethod != 0u;
    if (needsJitter) {
        jitterState = hash32(index ^ bitcast<u32>(params.seed));
    }

    var prevX = x;
    var prevY = y;
    var sinAY = sin(params.a * y);
    var cosBX = cos(params.b * x);
    var sinCX = sin(params.c * x);
    var cosDY = cos(params.d * y);
    
    // Accumulation loop: 128 iterations to match JavaScript calculations
    // Each iteration computes one point and plots it to the density/color buffers
    for (var i = 0; i < 128; i++) {
        let nx = sinAY - cosBX;
        let ny = sinCX - cosDY;
        
        prevX = x;
        prevY = y;
        x = nx;
        y = ny;
        
        sinAY = sin(params.a * y);
        cosBX = cos(params.b * x);
        sinCX = sin(params.c * x);
        cosDY = cos(params.d * y);

        var jitter = vec2<f32>(0.0, 0.0);
        if (needsJitter) {
            jitterState = xorshift32(jitterState);
            let h = jitterState;
            jitter = vec2<f32>(
                f32(h & 0xFFFFu) / 65536.0 - 0.5,
                f32(h >> 16u) / 65536.0 - 0.5
            );
        }

        let px: f32 = x * scale + centerX;
        let py: f32 = y * scale + centerY;
        let sx: i32 = i32(floor(px + jitter.x));
        let sy: i32 = i32(floor(py + jitter.y));

        if (sx >= 0 && sx < i32(w) && sy >= 0 && sy < i32(h)) {
            let pixelIndex = u32(sy) * w + u32(sx);
            
            if (params.colorMethod == 0u) {
                atomicAdd(&densityBuffer[pixelIndex], 1u);
                continue;
            }
            
            let prevCount: u32 = atomicAdd(&densityBuffer[pixelIndex], 1u);
            let mask = select(0u, 7u, prevCount > 256u);
            let shouldUpdateColor = (prevCount & mask) == 0u;
            
            // --- Determine Color ---
            var r: f32 = 0.0;
            var g: f32 = 0.0;
            var b: f32 = 0.0;
            var alpha: f32 = 0.0;

            if (params.colorMethod == 5u && shouldUpdateColor) {
                const MAX_ITER_LOG: f32 = 12.0;
                let countF = f32(prevCount);
                let logCount = log2(max(countF, 1.0));
                let normalized = clamp(logCount / MAX_ITER_LOG, 0.0, 1.0);
                let hueRGB = iterationHueToRGB(normalized);
                r = hueRGB.r; g = hueRGB.g; b = hueRGB.b;
                
                let effCount: u32 = min(prevCount, 65535u);
                let weightScalar = select(1.0, 8.0, prevCount > 256u);
                alpha = (1.0 / f32(effCount + 1u)) * weightScalar;
            } else if (shouldUpdateColor) {
                let baseR = abs(prevY - ny);
                let baseG = abs(prevX - ny);
                let baseB = abs(prevX - nx);
                let scaledR = baseR * 2.5;
                let scaledG = baseG * 2.5;
                let scaledB = baseB * 2.5;
                
                switch params.colorMethod {
                    case 1u: { r = scaledR; g = scaledG; b = scaledB; }
                    case 2u: { r = scaledG; g = scaledB; b = scaledR; }
                    case 3u: { r = scaledB; g = scaledR; b = scaledG; }
                    case 4u: { r = scaledB; g = scaledG; b = scaledR; }
                    default: { r = scaledR; g = scaledG; b = scaledB; }
                }
                
                let effCount: u32 = min(prevCount, 65535u);
                let weightScalar = select(1.0, 8.0, prevCount > 256u);
                alpha = (1.0 / f32(effCount + 1u)) * weightScalar;
            }

            if (shouldUpdateColor) {
                // Atomic color channel update (inlined for WGSL compatibility)
                // Update RED
                {
                    var oldBits = atomicLoad(&colorBufferR[pixelIndex]);
                    for (var k = 0u; k < 3u; k++) {
                        let oldVal = bitcast<f32>(oldBits);
                        let newVal = oldVal + (r - oldVal) * alpha;
                        let newBits = bitcast<u32>(newVal);
                        let res = atomicCompareExchangeWeak(&colorBufferR[pixelIndex], oldBits, newBits);
                        if (res.exchanged) { break; }
                        oldBits = res.old_value;
                    }
                }
                // Update GREEN
                {
                    var oldBits = atomicLoad(&colorBufferG[pixelIndex]);
                    for (var k = 0u; k < 3u; k++) {
                        let oldVal = bitcast<f32>(oldBits);
                        let newVal = oldVal + (g - oldVal) * alpha;
                        let newBits = bitcast<u32>(newVal);
                        let res = atomicCompareExchangeWeak(&colorBufferG[pixelIndex], oldBits, newBits);
                        if (res.exchanged) { break; }
                        oldBits = res.old_value;
                    }
                }
                // Update BLUE
                {
                    var oldBits = atomicLoad(&colorBufferB[pixelIndex]);
                    for (var k = 0u; k < 3u; k++) {
                        let oldVal = bitcast<f32>(oldBits);
                        let newVal = oldVal + (b - oldVal) * alpha;
                        let newBits = bitcast<u32>(newVal);
                        let res = atomicCompareExchangeWeak(&colorBufferB[pixelIndex], oldBits, newBits);
                        if (res.exchanged) { break; }
                        oldBits = res.old_value;
                    }
                }
            }
        }
    }
}

// --- RENDER SHADER ---

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

const pos: array<vec2<f32>, 6> = array<vec2<f32>, 6>(
    vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(-1.0, 1.0),
    vec2<f32>(-1.0, 1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0)
);

@vertex
fn vertexMain(@builtin(vertex_index) vertexIndex: u32) -> VertexOutput {
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

@fragment
fn fragmentMain(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {
    let baseScale = min(params.width, params.height) * 0.2;
    let screenX = uv.x * params.width;
    let screenY = uv.y * params.height;
    
    let bufferX = (screenX - params.width * 0.5) / params.viewScale - params.viewOffsetX * baseScale + params.width * 0.5;
    let bufferY = (screenY - params.height * 0.5) / params.viewScale - params.viewOffsetY * baseScale + params.height * 0.5;
    
    if (bufferX < 0.0 || bufferX >= params.width || bufferY < 0.0 || bufferY >= params.height) {
        return vec4<f32>(0.0, 0.0, 0.0, 1.0);
    }
    
    let x = u32(clamp(bufferX, 0.0, params.width - 1.0));
    let y = u32(clamp(bufferY, 0.0, params.height - 1.0));
    let index = y * u32(params.width) + x;
    
    let count = densityRead[index];
    let log2Total: f32 = params.totalIterations;
    let safeLog2Total: f32 = max(log2Total, f32(-50.0));
    let negLog2Total: f32 = -safeLog2Total;
    let invTotal: f32 = pow(f32(2.0), negLog2Total);
    let densityBrightnessScale = 250.0;
    
    var col: vec3<f32>;
    
    if (params.colorMethod == 5u && count > 0u) {
        let rAvg = bitcast<f32>(colorReadR[index]);
        let gAvg = bitcast<f32>(colorReadG[index]);
        let bAvg = bitcast<f32>(colorReadB[index]);
        let density: f32 = f32(count) * invTotal;
        const BRIGHTNESS_SCALE: f32 = 250.0;
        var rVal = rAvg * density * BRIGHTNESS_SCALE;
        var gVal = gAvg * density * BRIGHTNESS_SCALE;
        var bVal = bAvg * density * BRIGHTNESS_SCALE;
        
        rVal = (rVal * params.exposure) / (1.0 + rVal * params.exposure);
        gVal = (gVal * params.exposure) / (1.0 + gVal * params.exposure);
        bVal = (bVal * params.exposure) / (1.0 + bVal * params.exposure);
        
        rVal = pow(clamp(rVal, 0.0, 1.0), params.gamma);
        gVal = pow(clamp(gVal, 0.0, 1.0), params.gamma);
        bVal = pow(clamp(bVal, 0.0, 1.0), params.gamma);
        
        let contrastFactor: f32 = params.contrast;
        rVal = clamp((rVal - 0.5) * contrastFactor + 0.5, 0.0, 1.0);
        gVal = clamp((gVal - 0.5) * contrastFactor + 0.5, 0.0, 1.0);
        bVal = clamp((bVal - 0.5) * contrastFactor + 0.5, 0.0, 1.0);
        
        col = vec3<f32>(rVal, gVal, bVal);
    } else if (params.colorMethod >= 1u && count > 0u) {
        let rAvg = bitcast<f32>(colorReadR[index]);
        let gAvg = bitcast<f32>(colorReadG[index]);
        let bAvg = bitcast<f32>(colorReadB[index]);
        let density: f32 = f32(count) * invTotal;
        const BRIGHTNESS_SCALE: f32 = 250.0;
        var rVal = rAvg * density * BRIGHTNESS_SCALE;
        var gVal = gAvg * density * BRIGHTNESS_SCALE;
        var bVal = bAvg * density * BRIGHTNESS_SCALE;
        
        rVal = (rVal * params.exposure) / (1.0 + rVal * params.exposure);
        gVal = (gVal * params.exposure) / (1.0 + gVal * params.exposure);
        bVal = (bVal * params.exposure) / (1.0 + bVal * params.exposure);
        
        let maxVal = max(max(rVal, gVal), bVal);
        let minVal = min(min(rVal, gVal), bVal);
        let delta = maxVal - minVal;
        if (delta > 0.001 && maxVal > 0.001) {
            let saturation = delta / maxVal;
            let newSaturation = min(1.0, saturation * 1.5);
            let scale = newSaturation / saturation;
            rVal = minVal + (rVal - minVal) * scale;
            gVal = minVal + (gVal - minVal) * scale;
            bVal = minVal + (bVal - minVal) * scale;
        }
        
        rVal = pow(clamp(rVal, 0.0, 1.0), params.gamma);
        gVal = pow(clamp(gVal, 0.0, 1.0), params.gamma);
        bVal = pow(clamp(bVal, 0.0, 1.0), params.gamma);
        
        let contrastFactor: f32 = params.contrast;
        rVal = clamp((rVal - 0.5) * contrastFactor + 0.5, 0.0, 1.0);
        gVal = clamp((gVal - 0.5) * contrastFactor + 0.5, 0.0, 1.0);
        bVal = clamp((bVal - 0.5) * contrastFactor + 0.5, 0.0, 1.0);
        
        col = vec3<f32>(rVal, gVal, bVal);
    } else {
        var val = f32(count) * invTotal * densityBrightnessScale;
        val = (val * params.exposure) / (1.0 + val * params.exposure);
        val = clamp(val, 0.0, 1.0);
        val = pow(val, params.gamma);
        val = clamp((val - 0.5) * params.contrast + 0.5, 0.0, 1.0);
        col = getGradient(val);
    }

    return vec4<f32>(col, 1.0);
}
