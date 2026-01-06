/**
 * @fileoverview Global application state management for RandomNumberLab
 * 
 * ARCHITECTURE ROLE: State layer - centralized application state
 * ┌─────────────────────────────────────────────────────────────────┐
 * │  state.js (this file)                                           │
 * │    ← config.js: Constants (CONFIG, SEQUENCE_TYPES)             │
 * │    → All modules: Single source of truth for application state  │
 * └─────────────────────────────────────────────────────────────────┘
 * 
 * STATE CATEGORIES:
 * - RNG configuration: sequenceType, samples, offset
 * - GPU resources: gpuDevice, computePipeline, buffers
 * - Visualization: accumulationImageData, zoom, frameCount
 * - UI state: continuousMode, useCPU, webgpuAvailable
 * 
 * INVARIANTS:
 * - State is mutable (not frozen) to allow updates
 * - GPU resources are tracked in gpuResources Set for cleanup
 * - Buffer pools are reused to minimize allocations
 */

import { CONFIG, SEQUENCE_TYPES } from './config.js';

/**
 * @typedef {Object} GPUBufferPool
 * @property {number} samples
 * @property {number} width
 * @property {number} height
 * @property {GPUBuffer} output
 * @property {GPUBuffer} params
 * @property {GPUBuffer} readback
 */

/**
 * @typedef {Object} ApplicationState
 * @property {string} sequenceType - Current RNG method (from SEQUENCE_TYPES)
 * @property {number} dimensions
 * @property {number} samples
 * @property {Array<[number, number]>|null} data - Generated points
 * @property {GPUDevice|null} gpuDevice
 * @property {GPUComputePipeline|null} computePipeline
 * @property {number} lastGenerationTime
 * @property {boolean} continuousMode
 * @property {boolean} continuousRunning
 * @property {GPUBuffer|null} accumulationBuffer
 * @property {ImageData|null} accumulationImageData
 * @property {ImageData|null} baseAccumulationImageData
 * @property {number} frameCount
 * @property {number} offset
 * @property {number} zoom
 * @property {Uint32Array|null} exactPermutation - CPU fallback permutation
 * @property {GPUBuffer|null} pixelPermutationBuffer
 * @property {[number, number]|null} pixelPermutationDims
 * @property {boolean} useCPU
 * @property {boolean} webgpuAvailable
 * @property {GPUBuffer|null} directionsBuffer
 * @property {GPUBuffer|null} scrambleSeedsBuffer
 * @property {number|null} resizeTimeout
 * @property {ResizeObserver|null} resizeObserver
 * @property {Set<Object>} gpuResources
 * @property {number|null} statisticsCacheInterval
 * @property {GPUBufferPool|null} gpuBufferPool
 * @property {boolean} verbose
 * @property {GPUBindGroup|null} cachedBindGroup
 * @property {string|null} cachedBindGroupVersion
 * @property {Float32Array|null} xDataBuffer
 * @property {Float32Array|null} yDataBuffer
 * @property {boolean} forcePermutationReset - Signal to regenerate GPU permutation
 * @property {number} permutationVersion - Version counter for bind group caching
 */

/** @type {ApplicationState} */
export const state = {
    sequenceType: SEQUENCE_TYPES.SOBOL, // Use constant
    dimensions: 2,
    samples: CONFIG.DEFAULT_SAMPLES,
    data: null,
    gpuDevice: null,
    computePipeline: null,
    lastGenerationTime: 0,
    continuousMode: false,
    continuousRunning: false,
    accumulationBuffer: null,
    accumulationImageData: null,
    baseAccumulationImageData: null, // Cached at base resolution (1x)
    frameCount: 0,
    offset: 0,
    zoom: 1,
    exactPermutation: null, // Used for CPU implementation (Uint32Array of linear indices)
    pixelPermutationBuffer: null, // GPU buffer for exact uniform permutation (linear indices)
    pixelPermutationDims: null, // [width, height] of the permutation buffer for dimension-aware checks
    useCPU: false, // Toggle between GPU and CPU for methods that support both
    webgpuAvailable: false, // Track WebGPU availability for CPU fallback
    directionsBuffer: null, // Static buffer - hoisted to state
    scrambleSeedsBuffer: null, // Static buffer - hoisted to state
    resizeTimeout: null, // For debouncing resize events
    resizeObserver: null, // ResizeObserver instance
    gpuResources: new Set(), // Track all GPU resources for cleanup
    statisticsCacheInterval: null, // Interval ID for periodic cache reset
    gpuBufferPool: null, // Cached buffers for reuse (output, params, readback)
    verbose: false, // Verbose logging mode (off by default)
    cachedBindGroup: null, // Cached bind group to prevent leaks
    cachedBindGroupVersion: null, // Version string to detect buffer changes
    xDataBuffer: null, // Pre-allocated buffer for X coordinates
    yDataBuffer: null, // Pre-allocated buffer for Y coordinates
    performanceTimings: null,  // Set to {} to enable detailed timing, null to disable
    forcePermutationReset: false, // Signal to recreate Exact Uniform buffer on GPU
    permutationVersion: 0  // Track buffer version to invalidate Bind Groups
};
