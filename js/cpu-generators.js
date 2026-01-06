/**
 * @fileoverview CPU-based RNG implementations (fallback when WebGPU unavailable)
 * 
 * ARCHITECTURE ROLE: Computation layer - pure functions for sequence generation
 * ┌─────────────────────────────────────────────────────────────────┐
 * │  cpu-generators.js                                              │
 * │    ← config.js: Constants (SOBOL_DIRECTIONS, R2_CONSTANTS)    │
 * │    ← state.js: Global state (exactPermutation cache)          │
 * │    → main.js: Called when useCPU=true or WebGPU unavailable    │
 * └─────────────────────────────────────────────────────────────────┘
 * 
 * EXPORTS:
 * - generateSobolSamples_CPU: Sobol sequence (with optional scrambling)
 * - generateLCG_CPU: Linear Congruential Generator
 * - generateXorShift_CPU: XorShift32 PRNG
 * - generatePCG_CPU: PCG-XSH-RS PRNG
 * - generateR2_CPU: R2 sequence (plastic constant)
 * - generateExactUniformSamples_CPU: Perfect pixel coverage
 * - generatePixelPermutation: Fisher-Yates shuffle for Exact mode
 * - generateLinearPermutation: Linear index permutation for GPU
 * 
 * INVARIANTS:
 * - All functions return Array<[number, number]> with values in [0, 1)
 * - CPU implementations must match GPU shader logic exactly
 * - R2 uses f64 precision (JavaScript default) to avoid precision loss
 */

import { SOBOL_DIRECTIONS, UINT32_MAX_PLUS_ONE, GOLDEN_RATIO_HASH, R2_CONSTANTS, toU32 } from './config.js';
import { state } from './state.js';

// Seeded RNG for deterministic random permutation (Fisher-Yates shuffle)
function seededRandom(seed) {
    let value = seed;
    return function() {
        value = (value * 1664525 + 1013904223) % UINT32_MAX_PLUS_ONE;
        return value / UINT32_MAX_PLUS_ONE;
    };
}

// Generate random permutation of pixel coordinates using Fisher-Yates shuffle
// This creates a deterministic but random-looking order of all pixels
export function generatePixelPermutation(width, height, seed = 12345) {
    const pixels = [];
    // Create array of all pixel coordinates
    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
            pixels.push({ x, y });
        }
    }
    
    // Fisher-Yates shuffle with seeded RNG for determinism
    const rng = seededRandom(seed);
    for (let i = pixels.length - 1; i > 0; i--) {
        const j = Math.floor(rng() * (i + 1));
        [pixels[i], pixels[j]] = [pixels[j], pixels[i]];
    }
    
    return pixels;
}

/**
 * Generate Sobol sequence samples on CPU (fallback when WebGPU unavailable)
 * @param {number} numSamples - Number of 2D points to generate
 * @param {number} offset - Starting index in Sobol sequence (for progressive generation)
 * @param {boolean} [scrambled=false] - Apply Cranley-Patterson rotation for randomization
 * @param {number} [width=800] - Canvas width for jitter scaling
 * @param {number} [height=600] - Canvas height for jitter scaling
 * @returns {Array<[number, number]>} Array of [x, y] pairs in [0, 1) range
 */
export function generateSobolSamples_CPU(numSamples, offset, scrambled = false, width = 800, height = 600) {
    const samples = [];
    for (let i = 0; i < numSamples; i++) {
        const index = offset + i;
        let gray = index ^ (index >>> 1);
        
        let x = 0, y = 0;
        for (let bit = 0; bit < 32; bit++) {
            if (gray & (1 << bit)) {
                x ^= SOBOL_DIRECTIONS[0][bit];
                y ^= SOBOL_DIRECTIONS[1][bit];
            }
        }
        
        // Force unsigned interpretation using toU32 before division
        // JS bitwise operators result in signed 32-bit ints; 0x80000000 becomes -2147483648
        // Without this, half the points become negative and are clamped to 0
        let fx = toU32(x) / UINT32_MAX_PLUS_ONE;
        let fy = toU32(y) / UINT32_MAX_PLUS_ONE;
        
        // Add deterministic jitter for non-scrambled Sobol to break up lattice artifacts
        // Matches GPU implementation - jitter scaled to 0.5 pixel width/height
        if (!scrambled) {
            // Simple hash function for deterministic jitter (matches GPU hash)
            let jitterState = index;
            jitterState ^= jitterState >>> 16;
            jitterState = toU32(Math.imul(jitterState, 0x85ebca6b));
            jitterState ^= jitterState >>> 13;
            jitterState = toU32(Math.imul(jitterState, 0xc2b2ae35));
            jitterState ^= jitterState >>> 16;
            
            // Scale jitter to 0.5 pixel width dynamically (matches GPU)
            const jitterX = (jitterState / UINT32_MAX_PLUS_ONE) * 0.5 * (1.0 / width);
            fx = (fx + jitterX) % 1;
            
            jitterState = toU32(index + GOLDEN_RATIO_HASH);
            jitterState ^= jitterState >>> 16;
            jitterState = toU32(Math.imul(jitterState, 0x85ebca6b));
            jitterState ^= jitterState >>> 13;
            jitterState = toU32(Math.imul(jitterState, 0xc2b2ae35));
            jitterState ^= jitterState >>> 16;
            
            // FIX: Scale jitter to 0.5 pixel height dynamically (matches GPU)
            const jitterY = (jitterState / UINT32_MAX_PLUS_ONE) * 0.5 * (1.0 / height);
            fy = (fy + jitterY) % 1;
        }
        
        if (scrambled) {
            // Cranley-Patterson rotation with fixed seeds
            // Use Math.imul and toU32 for correct 32-bit integer wrapping
            let seedX = toU32((0x12345678 ^ index) * 1664525 + 1013904223);
            let seedY = toU32((0x9ABCDEF0 ^ index) * 1664525 + 1013904223);
            fx = (fx + seedX / UINT32_MAX_PLUS_ONE) % 1;
            fy = (fy + seedY / UINT32_MAX_PLUS_ONE) % 1;
        }
        
        samples.push([fx, fy]);
    }
    return samples;
}

/**
 * Generate LCG samples on CPU
 * Matches GPU implementation exactly
 * @param {number} numSamples - Number of 2D points to generate
 * @param {number} offset - Starting index in sequence
 * @param {number} [seed=0xDEADBEEF] - Initial seed value
 * @returns {Array<[number, number]>} Array of [x, y] pairs in [0, 1) range
 */
export function generateLCG_CPU(numSamples, offset, seed = 0xDEADBEEF) {
    const samples = [];
    for (let i = 0; i < numSamples; i++) {
        const index = offset + i;
        // Same mixing as GPU: index * GOLDEN_RATIO_HASH + seed
        let state = toU32(Math.imul(index, GOLDEN_RATIO_HASH) + seed);
        
        // First LCG advance → X
        state = toU32(Math.imul(1664525, state) + 1013904223);
        const x = state / UINT32_MAX_PLUS_ONE;
        
        // Second LCG advance → Y
        state = toU32(Math.imul(1664525, state) + 1013904223);
        const y = state / UINT32_MAX_PLUS_ONE;
        
        samples.push([x, y]);
    }
    return samples;
}

/**
 * Generate XorShift samples on CPU
 * Matches GPU implementation exactly
 * @param {number} numSamples - Number of 2D points to generate
 * @param {number} offset - Starting index in sequence
 * @param {number} [seed=0xDEADBEEF] - Initial seed value
 * @returns {Array<[number, number]>} Array of [x, y] pairs in [0, 1) range
 */
export function generateXorShift_CPU(numSamples, offset, seed = 0xDEADBEEF) {
    const samples = [];
    for (let i = 0; i < numSamples; i++) {
        const index = offset + i;
        let state = toU32(Math.imul(index, GOLDEN_RATIO_HASH) + seed);
        if (state === 0) state = 1;  // XorShift requires nonzero
        
        // First xorshift advance → X
        state ^= state << 13;
        state ^= state >>> 17;
        state ^= state << 5;
        state = toU32(state);
        const x = state / UINT32_MAX_PLUS_ONE;
        
        // Second xorshift advance → Y
        state ^= state << 13;
        state ^= state >>> 17;
        state ^= state << 5;
        state = toU32(state);
        const y = state / UINT32_MAX_PLUS_ONE;
        
        samples.push([x, y]);
    }
    return samples;
}

/**
 * Generate PCG samples on CPU
 * Matches GPU implementation exactly (PCG-XSH-RS variant)
 * @param {number} numSamples - Number of 2D points to generate
 * @param {number} offset - Starting index in sequence
 * @param {number} [seed=0xDEADBEEF] - Initial seed value
 * @returns {Array<[number, number]>} Array of [x, y] pairs in [0, 1) range
 */
export function generatePCG_CPU(numSamples, offset, seed = 0xDEADBEEF) {
    const samples = [];
    for (let i = 0; i < numSamples; i++) {
        const index = offset + i;
        let state = toU32(Math.imul(index, GOLDEN_RATIO_HASH) + seed);
        
        // Generate X
        let oldState = state;
        state = toU32(Math.imul(state, 747796405) + 2891336453);
        let shift = (oldState >>> 28) + 4;
        let word = toU32(Math.imul(toU32((oldState >>> shift) ^ oldState), 277803737));
        const x = toU32((word >>> 22) ^ word) / UINT32_MAX_PLUS_ONE;
        
        // Generate Y
        oldState = state;
        state = toU32(Math.imul(state, 747796405) + 2891336453);
        shift = (oldState >>> 28) + 4;
        word = toU32(Math.imul(toU32((oldState >>> shift) ^ oldState), 277803737));
        const y = toU32((word >>> 22) ^ word) / UINT32_MAX_PLUS_ONE;
        
        samples.push([x, y]);
    }
    return samples;
}

/**
 * Generate R2 sequence samples on CPU
 * Matches GPU implementation exactly
 * @param {number} numSamples - Number of 2D points to generate
 * @param {number} offset - Starting index in sequence
 * @returns {Array<[number, number]>} Array of [x, y] pairs in [0, 1) range
 */
export function generateR2_CPU(numSamples, offset) {
    const { ALPHA1: alpha1, ALPHA2: alpha2 } = R2_CONSTANTS;
    
    // Use additive recurrence to avoid precision loss from large index * alpha
    // Mathematically equivalent: x_n = fract(n * alpha) = fract(x_0 + n * alpha)
    // where x_0 = fract(offset * alpha). Then x_{n+1} = fract(x_n + alpha)
    // This keeps all values in [0,1) and avoids precision loss from large integer parts
    
    // Compute starting point from offset (handle large offset with modulo to preserve precision)
    // For very large offsets, we can use: fract(offset * alpha) = fract((offset mod period) * alpha)
    // But additive recurrence works for any offset without needing period wrapping
    let x = (offset * alpha1) % 1;
    let y = (offset * alpha2) % 1;
    // Ensure positive modulo result (JavaScript % can return negative)
    if (x < 0) x += 1;
    if (y < 0) y += 1;
    
    const samples = [];
    for (let i = 0; i < numSamples; i++) {
        samples.push([x, y]);
        // Additive recurrence: next value = (current + alpha) mod 1
        // All values stay in [0,1), so no precision loss from large integers
        x = (x + alpha1) % 1;
        y = (y + alpha2) % 1;
        if (x < 0) x += 1;
        if (y < 0) y += 1;
    }
    return samples;
}

/**
 * Generate a linear permutation of pixel indices (0 to N-1) as Uint32Array.
 * Specifically designed for the GPU Exact Uniform buffer.
 */
export function generateLinearPermutation(width, height, seed = 12345) {
    const totalPixels = width * height;
    const permutation = new Uint32Array(totalPixels);
    
    // Initialize with linear indices 0..N-1
    for (let i = 0; i < totalPixels; i++) {
        permutation[i] = i;
    }
    
    // Fisher-Yates shuffle
    let state = seed;
    for (let i = totalPixels - 1; i > 0; i--) {
        // Generate deterministic random number
        state = toU32(state * 1664525 + 1013904223);
        
        // Map state to [0, i]
        const j = Math.floor((state / UINT32_MAX_PLUS_ONE) * (i + 1));
        
        // Swap
        const temp = permutation[i];
        permutation[i] = permutation[j];
        permutation[j] = temp;
    }
    
    return permutation;
}

// Hash function matching GPU implementation (MurmurHash-style)
function hashU32(x) {
    x = toU32(x);
    x ^= x >>> 16;
    x = toU32(Math.imul(x, 0x85ebca6b));
    x ^= x >>> 13;
    x = toU32(Math.imul(x, 0xc2b2ae35));
    x ^= x >>> 16;
    return toU32(x);
}

/**
 * Generate exact uniform samples from pixel permutation (CPU-based)
 * This method guarantees every pixel is hit exactly once per cycle
 * Now uses linear permutation like GPU for CPU/GPU equivalence
 * @param {number} numSamples - Number of 2D points to generate
 * @param {number} width - Canvas width in pixels
 * @param {number} height - Canvas height in pixels
 * @param {number} [offset=0] - Starting index in permutation
 * @returns {Array<[number, number]>} Array of [x, y] pairs in [0, 1) range
 */
export function generateExactUniformSamples_CPU(numSamples, width, height, offset = 0) {
    // Initialize linear permutation if needed or if canvas size changed
    // Use linear permutation (Uint32Array) to match GPU implementation
    if (!state.exactPermutation || 
        !(state.exactPermutation instanceof Uint32Array) ||
        state.exactPermutation.length !== width * height) {
        // GENERATE A RANDOM SEED for fresh permutations on reset
        const seed = Math.floor(Math.random() * UINT32_MAX_PLUS_ONE);
        state.exactPermutation = generateLinearPermutation(width, height, seed);
    }
    
    const samples = [];
    const totalPixels = width * height;
    const perm = state.exactPermutation; // Uint32Array of linear indices
    
    for (let i = 0; i < numSamples; i++) {
        // Calculate global index (matches GPU: index = gid.x + params.offset)
        const idx = toU32(offset + i);
        const wrapped = idx % totalPixels;
        
        // Get linear pixel index from permutation (matches GPU)
        const pixelLinear = perm[wrapped];
        
        // Convert linear index to x,y coordinates (matches GPU)
        const px = pixelLinear % width;
        const py = (pixelLinear / width) | 0;
        
        // Hash jitter (matches GPU exactly)
        const jitterX = hashU32(idx) / UINT32_MAX_PLUS_ONE * 0.3;
        const jitterY = hashU32(toU32(idx + GOLDEN_RATIO_HASH)) / UINT32_MAX_PLUS_ONE * 0.3;
        
        // Match GPU: 0.5 center offset, same jitter scaling, clamp to < 1.0
        const x = (px + 0.5 + jitterX) / width;
        const y = (py + 0.5 + jitterY) / height;
        
        samples.push([
            Math.max(0, Math.min(0.999999, x)),
            Math.max(0, Math.min(0.999999, y))
        ]);
    }
    
    return samples;
}
