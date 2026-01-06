/**
 * @fileoverview GPU buffer creation and management for RandomNumberLab
 * 
 * ARCHITECTURE ROLE: GPU resource layer - WebGPU buffer allocation
 * ┌─────────────────────────────────────────────────────────────────┐
 * │  gpu.js                                                         │
 * │    ← config.js: Constants (SOBOL_DIRECTIONS)                   │
 * │    ← state.js: Global state (gpuDevice)                        │
 * │    ← cpu-generators.js: generateLinearPermutation              │
 * │    → main.js: Called during initialization and buffer creation  │
 * └─────────────────────────────────────────────────────────────────┘
 * 
 * EXPORTS:
 * - createDirectionsBuffer: Sobol direction vectors (static, 64 u32s)
 * - createScrambleSeedsBuffer: Scrambling seeds (2 u32s, updated per frame)
 * - createPixelPermutationBuffer: Exact Uniform permutation (W*H u32s)
 * 
 * LIFECYCLE:
 * - Static buffers created once in init()
 * - Permutation buffer recreated on canvas resize
 * - All buffers registered in state.gpuResources for cleanup
 */

import { SOBOL_DIRECTIONS } from './config.js';
import { state } from './state.js';
import { generateLinearPermutation } from './cpu-generators.js';

/**
 * Create directions buffer (static, created once)
 * @returns {GPUBuffer} Buffer containing Sobol direction vectors
 * @throws {Error} If state.gpuDevice is null or invalid
 */
export function createDirectionsBuffer() {
    if (!state.gpuDevice?.createBuffer) {
        throw new Error('createDirectionsBuffer: invalid GPUDevice');
    }
    const directionsArray = new Uint32Array(2 * 32);
    for (let dim = 0; dim < 2; dim++) {
        for (let bit = 0; bit < 32; bit++) {
            directionsArray[dim * 32 + bit] = SOBOL_DIRECTIONS[dim][bit];
        }
    }
    const buf = state.gpuDevice.createBuffer({
        mappedAtCreation: true,
        size: directionsArray.byteLength,
        usage: GPUBufferUsage.STORAGE
    });
    new Uint32Array(buf.getMappedRange()).set(directionsArray);
    buf.unmap();
    return buf;
}

/**
 * Create scramble seeds buffer (static, created once)
 * @returns {GPUBuffer} Buffer for scramble seeds (2 u32s)
 * @throws {Error} If state.gpuDevice is null or invalid
 */
export function createScrambleSeedsBuffer() {
    if (!state.gpuDevice?.createBuffer) {
        throw new Error('createScrambleSeedsBuffer: invalid GPUDevice');
    }
    const buf = state.gpuDevice.createBuffer({
        size: 8,  // 2 u32s
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
    });
    return buf;
}

/**
 * Create pixel permutation buffer (static, created once per resize or reset)
 * Contains shuffled linear indices 0..(width*height)-1
 * @param {number} width - Canvas width
 * @param {number} height - Canvas height
 * @param {number} [seed] - Optional seed for deterministic shuffle. If not provided, uses default seed (12345).
 * @returns {GPUBuffer} Buffer containing the permutation
 * @throws {Error} If state.gpuDevice is null or invalid
 */
export function createPixelPermutationBuffer(width, height, seed) {
    if (!state.gpuDevice?.createBuffer) {
        throw new Error('createPixelPermutationBuffer: invalid GPUDevice');
    }
    // Pass the seed to the generator (or undefined to use default)
    const permutationData = generateLinearPermutation(width, height, seed);
    
    const buf = state.gpuDevice.createBuffer({
        mappedAtCreation: true,
        size: permutationData.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
    });
    
    new Uint32Array(buf.getMappedRange()).set(permutationData);
    buf.unmap();
    return buf;
}
