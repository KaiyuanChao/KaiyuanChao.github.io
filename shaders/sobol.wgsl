// ╔══════════════════════════════════════════════════════════════════════╗
// ║  MULTI-MODE RNG COMPUTE SHADER                                       ║
// ║  Generates 2D sample points for visualization and analysis           ║
// ╠══════════════════════════════════════════════════════════════════════╣
// ║  MODES: 0=Sobol, 1=Scrambled, 2=Exact, 3=LCG, 4=XorShift, 5=PCG, 6=R2║
// ║  OUTPUT: vec2<f32> array, each component in [0, 1)                   ║
// ║  INVOCATION: One thread per sample, workgroup size 64                ║
// ╠══════════════════════════════════════════════════════════════════════╣
// ║  BUFFER LAYOUT (binding order matches main.js createBindGroup):      ║
// ║    @binding(0) output        - vec2<f32>[] write destination         ║
// ║    @binding(1) directions    - u32[64] Sobol direction vectors       ║
// ║    @binding(2) scrambleSeeds - u32[2] for Cranley-Patterson rotation ║
// ║    @binding(3) pixelPermutation - u32[W*H] for Exact Uniform mode    ║
// ║    @binding(4) params        - Uniform struct (40 bytes)             ║
// ╚══════════════════════════════════════════════════════════════════════╝

struct Params {
    numSamples: u32,      // [1, 8192] — number of vec2<f32> outputs to generate
    numDimensions: u32,   // Always 2 (reserved for future N-dimensional support)
    offset: u32,          // [0, 2³²) — starting index in sequence
    useScrambling: u32,   // {0, 1} — boolean: apply Cranley-Patterson rotation
    mode: u32,            // {0..6} — see RNG_SHADER_MODE enum in config.js
                          //   0=Sobol, 1=Scrambled(uses flag), 2=Exact, 
                          //   3=LCG, 4=XorShift, 5=PCG, 6=R2
    canvasWidth: u32,     // [400, 2048] — canvas width for Exact mode permutation
    canvasHeight: u32,   // [400, 2048] — canvas height for Exact mode permutation
    _pad: u32,            // Padding to maintain 16-byte alignment
    r2StartX: f32,        // [0, 1) — R2 starting X, computed on CPU with f64 precision
    r2StartY: f32,        // [0, 1) — R2 starting Y, computed on CPU with f64 precision
};

@group(0) @binding(0) var<storage, read_write> output: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read> directions: array<u32>;
@group(0) @binding(2) var<storage, read> scrambleSeeds: array<u32>;
@group(0) @binding(3) var<storage, read> pixelPermutation: array<u32>;
@group(0) @binding(4) var<uniform> params: Params;

// Reusable hash function (MurmurHash-style integer hashing)
// Returns normalized float in [0, 1) range
fn hash(seed: u32) -> f32 {
    var h = seed;
    h ^= h >> 16u;
    h *= 0x85ebca6bu;  // MurmurHash3 constant
    h ^= h >> 13u;
    h *= 0xc2b2ae35u;  // MurmurHash3 constant
    h ^= h >> 16u;
    return f32(h) * (1.0 / 4294967296.0);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.numSamples) {
        return;
    }
    
    // Compute global index (used by all modes)
    let index = gid.x + params.offset;
    
    // Initialize result to prevent uninitialized variable issues
    var result: vec2<f32> = vec2<f32>(0.0, 0.0);
    
    // Route to appropriate generator based on mode
    // FIX: Ensure mode is checked correctly - R2 is mode 6
    if (params.mode == 6u) {
        // R2 sequence (plastic constant) - optimal 2D low-discrepancy sequence
        // Formula: x_n = fract(n * α₁), y_n = fract(n * α₂)
        // where α₁ = 1/φ₂, α₂ = 1/φ₂², and φ₂ is the plastic constant
        // NOTE: These values MUST match R2_CONSTANTS.ALPHA1/ALPHA2 in config.js
        // The shader uses f32, so precision is limited to ~7 decimal digits here.
        const alpha1: f32 = 0.7548776662466927;  // 1/φ₂ where φ₂ = 1.32471795724474602596
        const alpha2: f32 = 0.5698402909980532;  // 1/φ₂²
        
        // PRECISION-PRESERVING COMPUTATION using additive recurrence:
        // Starting point (r2StartX, r2StartY) is computed on CPU with f64 precision,
        // then passed as uniforms. This avoids f32 precision loss from large offset * alpha.
        // GPU only needs to add gid.x * alpha to the starting point.
        // Since gid.x is small (< 8192), this preserves precision.
        let gidF32: f32 = f32(gid.x);
        let deltaX: f32 = gidF32 * alpha1;
        let deltaY: f32 = gidF32 * alpha2;
        
        // Final result: fract(start + delta) - all values stay in [0,1) range
        result = vec2<f32>(
            fract(params.r2StartX + deltaX),
            fract(params.r2StartY + deltaY)
        );
    } else if (params.mode == 2u) {
        // Exact uniform mode: use pre-generated permutation buffer
        let totalPixels = params.canvasWidth * params.canvasHeight;
        if (totalPixels > 0u) {
            let wrappedIndex = index % totalPixels;
            // Look up pixel from permutation buffer (contains linear index: y*width + x)
            let pixelLinearIndex = pixelPermutation[wrappedIndex];
            let px = pixelLinearIndex % params.canvasWidth;
            let py = pixelLinearIndex / params.canvasWidth;
            
            // Add small jitter for visual randomness (0.3 * pixel size)
            // Use hash for deterministic jitter (decorrelated X/Y seeds)
            let jitterX = hash(index) * 0.3;
            let jitterY = hash(index + 0x9e3779b9u) * 0.3;
            
            // FIX: Add 0.5 offset to center the sample within the pixel
            // This prevents floating point precision errors (e.g. 4.999 -> 4) 
            // from causing "black pixels" and overly hit neighbors.
            result = vec2<f32>(
                (f32(px) + 0.5 + jitterX) / f32(params.canvasWidth),
                (f32(py) + 0.5 + jitterY) / f32(params.canvasHeight)
            );
            result.x = clamp(result.x, 0.0, 0.999999);
            result.y = clamp(result.y, 0.0, 0.999999);
        } else {
            result = vec2<f32>(0.0, 0.0);
        }
    } else if (params.mode == 3u) {
        // LCG (Linear Congruential Generator)
        // FIX: Use SINGLE state, advance TWICE for proper independent X/Y
        var state = index * 0x9E3779B9u + scrambleSeeds[0u];  // better initial mixing
        
        // First LCG advance → X
        state = 1664525u * state + 1013904223u;
        result.x = f32(state) * (1.0 / 4294967296.0);
        
        // Second LCG advance → Y  
        state = 1664525u * state + 1013904223u;
        result.y = f32(state) * (1.0 / 4294967296.0);
    } else if (params.mode == 4u) {
        // XOR-shift RNG (Marsaglia's xorshift32)
        // FIX: Use SINGLE state, advance TWICE for proper independent X/Y
        var state = index * 0x9E3779B9u + scrambleSeeds[0u];
        if (state == 0u) { state = 1u; }  // xorshift requires nonzero
        
        // First xorshift advance → X
        state ^= state << 13u;
        state ^= state >> 17u;
        state ^= state << 5u;
        result.x = f32(state) * (1.0 / 4294967296.0);
        
        // Second xorshift advance → Y
        state ^= state << 13u;
        state ^= state >> 17u;
        state ^= state << 5u;
        result.y = f32(state) * (1.0 / 4294967296.0);
    } else if (params.mode == 5u) {
        // PCG (Permuted Congruential Generator)
        // FIX: Correct PCG-XSH-RS implementation (32-bit state → 32-bit output)
        // Original was wrong: XSH-RR formula produced only 5 bits of output
        var state = index * 0x9E3779B9u + scrambleSeeds[0u];
        
        // Generate X: save state, advance, permute
        var oldState = state;
        state = state * 747796405u + 2891336453u;  // PCG LCG multiplier and increment constants
        var word = ((oldState >> ((oldState >> 28u) + 4u)) ^ oldState) * 277803737u;  // PCG permutation constant
        result.x = f32((word >> 22u) ^ word) * (1.0 / 4294967296.0);
        
        // Generate Y: save state, advance, permute
        oldState = state;
        state = state * 747796405u + 2891336453u;  // PCG LCG multiplier and increment constants
        word = ((oldState >> ((oldState >> 28u) + 4u)) ^ oldState) * 277803737u;  // PCG permutation constant
        result.y = f32((word >> 22u) ^ word) * (1.0 / 4294967296.0);
    } else {
        // Sobol sequence generation (mode 0 or 1)
        // Data type flow: u32 integers → f32 floats [0,1) → JavaScript maps to integer pixels
        // This preserves full 32-bit precision throughout the pipeline
        
        var gray = index ^ (index >> 1u);
        
        // Generate Sobol point for dimension 0 (X)
        // Computes as u32 integer using bitwise XOR with direction numbers
        var x: u32 = 0u;
        for (var bit: u32 = 0u; bit < 32u; bit = bit + 1u) {
            if ((gray & (1u << bit)) != 0u) {
                x ^= directions[0u * 32u + bit];
            }
        }
        
        // Generate Sobol point for dimension 1 (Y)
        // Computes as u32 integer using bitwise XOR with direction numbers
        var y: u32 = 0u;
        for (var bit: u32 = 0u; bit < 32u; bit = bit + 1u) {
            if ((gray & (1u << bit)) != 0u) {
                y ^= directions[1u * 32u + bit];
            }
        }
        
        // Normalize u32 integers to f32 floats in [0, 1) range
        // JavaScript will convert these back to integer pixel coordinates
        result = vec2<f32>(
            f32(x) * (1.0 / 4294967296.0),
            f32(y) * (1.0 / 4294967296.0)
        );
        
        // For standard Sobol (without scrambling), add small deterministic jitter
        // to improve pixel coverage when mapped to discrete grids
        // This helps fill gaps that occur due to Sobol's structured nature
        if (params.useScrambling == 0u) {
            // Use hash function to generate deterministic jitter (decorrelated X/Y seeds)
            // Scale jitter by 1/width and 1/height to make it relative to PIXEL size (0.5px jitter)
            let jitterX = hash(index) * 0.5 / f32(params.canvasWidth);
            let jitterY = hash(index + 0x9e3779b9u) * 0.5 / f32(params.canvasHeight);
            
            // Add jitter and wrap to [0, 1)
            result.x = fract(result.x + jitterX);
            result.y = fract(result.y + jitterY);
        }
        
        // FIX: Correct Owen Scrambling - must incorporate index for per-sample variation
        // Matches CPU implementation: XOR seeds with index, then LCG, then add to result
        if (params.useScrambling != 0u) {
            // XOR seeds with index to get per-sample variation (matches CPU)
            var seedX = scrambleSeeds[0u] ^ index;
            var seedY = scrambleSeeds[1u] ^ index;
            
            // LCG mixing for seeds (matches CPU exactly)
            seedX = (1664525u * seedX + 1013904223u);
            seedY = (1664525u * seedY + 1013904223u);
            
            // Apply random shift (Toroidal shift / Cranley-Patterson)
            result.x = fract(result.x + f32(seedX) * (1.0 / 4294967296.0));
            result.y = fract(result.y + f32(seedY) * (1.0 / 4294967296.0));
        }
    }
    
    output[gid.x] = result;
}
