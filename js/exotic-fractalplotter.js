// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2026

/**
 * Exotic Fractal Plotter - WebGPU Peter de Jong attractor renderer (CPU fallback).
 *
 * Architecture:
 * - init() chooses WebGPU (preferred) or CPU fallback.
 * - UI mutates `state`; updateParams() writes a packed uniform buffer.
 * - compute shader accumulates into storage buffers; render shader tonemaps to the canvas.
 *
 * Glossary (terms used consistently in this file):
 * - point: one simulated de Jong step that may hit a pixel.
 * - iteration: same as point (one step of the attractor map).
 * - thread: one shader invocation that runs WARMUP + ACCUMULATE iterations.
 * - workgroup: a group of threads dispatched together by WebGPU.
 *
 * Modes:
 * - RNG (state.rngMode): 0 hash12, 1 Sobol scrambled, 2 R2 legacy, 3 Sobol, 4 R2 precision, 5 Owen-Sobol, 6 LCG.
 * - Color (state.colorMethod): 0 density, 1-4 trajectory RGB, 5 iteration hue.
 */

// --- WGSL Shaders ---

// Load WGSL via project-root-relative URL (cache-busted).
async function loadShader(path) {
    const assetBase = new URL('../', import.meta.url);
    const url = new URL(path, assetBase);
    url.searchParams.set('v', String(Date.now()));
    const response = await fetch(url);
    if (!response.ok) {
        throw new Error(`Failed to load shader: ${path}`);
    }
    const shaderCode = await response.text();
    // Sanity check for stale shader caches.
    if (path.includes('fractal-accumulation.wgsl') && !shaderCode.includes('computeMain')) {
        console.error('WARNING: computeMain not found in loaded shader code!');
        console.error('First 500 chars of shader:', shaderCode.substring(0, 500));
    }
    return shaderCode;
}

// PNG metadata: store fractal state in PNG tEXt chunks.
const PNG_SIGNATURE = new Uint8Array([0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A]);

// CRC32 lookup table (cached once).
const CRC32_TABLE = (() => {
    const table = new Uint32Array(256);
    for (let i = 0; i < 256; i++) {
        let c = i;
        for (let j = 0; j < 8; j++) {
            c = (c & 1) ? (0xEDB88320 ^ (c >>> 1)) : (c >>> 1);
        }
        table[i] = c;
    }
    return table;
})();

function crc32(data) {
    let crc = 0xFFFFFFFF;
    for (let i = 0; i < data.length; i++) {
        crc = CRC32_TABLE[(crc ^ data[i]) & 0xFF] ^ (crc >>> 8);
    }
    return (crc ^ 0xFFFFFFFF) >>> 0;
}

function writeUint32BE(buffer, offset, value) {
    buffer[offset] = (value >>> 24) & 0xFF;
    buffer[offset + 1] = (value >>> 16) & 0xFF;
    buffer[offset + 2] = (value >>> 8) & 0xFF;
    buffer[offset + 3] = value & 0xFF;
}

function readUint32BE(buffer, offset) {
    return (((buffer[offset] << 24) | (buffer[offset + 1] << 16) | (buffer[offset + 2] << 8) | buffer[offset + 3]) >>> 0);
}

// Embed JSON metadata into a PNG (Blob or URL).
async function embedPNGMetadata(pngSource, metadata) {
    let arrayBuffer;
    if (pngSource instanceof Blob) {
        arrayBuffer = await pngSource.arrayBuffer();
    } else {
        const response = await fetch(pngSource);
        arrayBuffer = await response.arrayBuffer();
    }
    const pngData = new Uint8Array(arrayBuffer);
    
    // Validate 8-byte PNG signature.
    for (let i = 0; i < 8; i++) {
        if (pngData[i] !== PNG_SIGNATURE[i]) {
            throw new Error('Invalid PNG file');
        }
    }
    
    // Find IEND chunk to insert tEXt just before it.
    let offset = 8;
    let iendOffset = -1;
    
    // Walk chunks: length(4) + type(4) + data + crc(4).
    while (offset < pngData.length - 8) {
        if (offset + 8 > pngData.length) {
            break;
        }
        
        const length = readUint32BE(pngData, offset);
        const typeOffset = offset + 4;
        
        if (typeOffset + 4 > pngData.length) {
            break;
        }
        
        const type = String.fromCharCode(
            pngData[typeOffset],
            pngData[typeOffset + 1],
            pngData[typeOffset + 2],
            pngData[typeOffset + 3]
        );
        
        if (type === 'IEND') {
            iendOffset = offset;
            break;
        }
        
        const chunkSize = 12 + length;
        if (offset + chunkSize > pngData.length) {
            console.warn('Chunk extends beyond PNG data, stopping search');
            break;
        }
        
        offset += chunkSize;
    }
    
    if (iendOffset === -1) {
        // Fallback: treat last 12 bytes as IEND if signature matches.
        const lastChunkOffset = pngData.length - 12;
        if (lastChunkOffset >= 8) {
            const lastType = String.fromCharCode(
                pngData[lastChunkOffset + 4],
                pngData[lastChunkOffset + 5],
                pngData[lastChunkOffset + 6],
                pngData[lastChunkOffset + 7]
            );
            if (lastType === 'IEND') {
                iendOffset = lastChunkOffset;
            }
        }
        
        if (iendOffset === -1) {
            console.error('PNG file size:', pngData.length);
            console.error('Last 20 bytes:', Array.from(pngData.slice(Math.max(0, pngData.length - 20))).map(b => '0x' + b.toString(16).padStart(2, '0')).join(' '));
            throw new Error('Could not find IEND chunk. PNG may be corrupted or in an unexpected format.');
        }
    }
    
    // Build tEXt payload: "keyword\0json" (keyword identifies our metadata).
    const metadataJson = JSON.stringify(metadata);
    const keyword = 'ExoticFractalPlotter';
    const textData = keyword + '\0' + metadataJson;
    const textBytes = new TextEncoder().encode(textData);
    const chunkLength = textBytes.length;
    
    // New PNG = original up to IEND + tEXt chunk + IEND.
    const newPngSize = pngData.length + 12 + chunkLength; // existing data + tEXt chunk
    const newPng = new Uint8Array(newPngSize);
    newPng.set(pngData.slice(0, iendOffset), 0);
    
    // Write tEXt chunk.
    let writeOffset = iendOffset;
    writeUint32BE(newPng, writeOffset, chunkLength);
    writeOffset += 4;
    newPng.set(new TextEncoder().encode('tEXt'), writeOffset);
    writeOffset += 4;
    newPng.set(textBytes, writeOffset);
    writeOffset += chunkLength;
    
    // CRC32 is computed over [type||data].
    const crcData = new Uint8Array(4 + chunkLength);
    crcData.set(new TextEncoder().encode('tEXt'), 0);
    crcData.set(textBytes, 4);
    const crc = crc32(crcData);
    writeUint32BE(newPng, writeOffset, crc);
    writeOffset += 4;
    
    // Append IEND chunk.
    newPng.set(pngData.slice(iendOffset), writeOffset);
    
    return new Blob([newPng], { type: 'image/png' });
}

// Extract metadata JSON from a PNG tEXt chunk.
async function extractPNGMetadata(file) {
    const arrayBuffer = await file.arrayBuffer();
    const pngData = new Uint8Array(arrayBuffer);
    
    // Validate 8-byte PNG signature.
    for (let i = 0; i < 8; i++) {
        if (pngData[i] !== PNG_SIGNATURE[i]) {
            throw new Error('Invalid PNG file');
        }
    }
    
    // Scan tEXt chunks for metadata payload.
    let offset = 8;
    
    while (offset < pngData.length - 12) {
        const length = readUint32BE(pngData, offset);
        const typeOffset = offset + 4;
        const type = String.fromCharCode(...pngData.slice(typeOffset, typeOffset + 4));
        
        if (type === 'tEXt') {
            const dataOffset = typeOffset + 4;
            const textData = new TextDecoder().decode(pngData.slice(dataOffset, dataOffset + length));
            const nullIndex = textData.indexOf('\0');
            
            if (nullIndex !== -1) {
                const value = textData.substring(nullIndex + 1);

                // Avoid coupling parsing to any legacy keyword string; validate the decoded JSON shape instead.
                try {
                    const obj = JSON.parse(value);
                    const isObj = obj && typeof obj === 'object';
                    const hasParams = !!(obj?.parameters && typeof obj.parameters === 'object');
                    const hasABCD =
                        typeof obj?.parameters?.a === 'number' &&
                        typeof obj?.parameters?.b === 'number' &&
                        typeof obj?.parameters?.c === 'number' &&
                        typeof obj?.parameters?.d === 'number';
                    const hasView = !!(obj?.view && typeof obj.view === 'object');
                    const hasRendering = !!(obj?.rendering && typeof obj.rendering === 'object');

                    if (isObj && hasParams && hasABCD && hasView && hasRendering) {
                        return obj;
                    }
                } catch {
                    // Ignore unrelated tEXt chunks.
                }
            }
        }
        
        if (type === 'IEND') {
            break;
        }
        
        offset += 12 + length;
    }
    
    return null; // No metadata found
}

// --- Application Logic ---

const FACTORY_EXPOSURE = 30.0;
const FACTORY_GAMMA = 0.21;
const FACTORY_CONTRAST = 1.5;

// Global state; declared before init() to avoid TDZ in closures.
const state = {
    a: -2.24,
    b: 0.43,
    c: -0.65,
    d: -2.43,
    exposure: FACTORY_EXPOSURE, // Default exposure value (slider will be positioned accordingly)
    gamma: FACTORY_GAMMA,
    contrast: FACTORY_CONTRAST,    // Contrast (1.0 = neutral, >1.0 = more contrast, <1.0 = less contrast) - default 1.5
    seed: Math.random() * 100.0, // RNG seed (Ctrl+S to reseed)
    colorMethod: 1, // 0 = default, 1 = Classic, 2 = Exotic, 3 = Vibrant, 4 = Surprise, 5 = Iteration Hues
    totalIterations: 0,
    viewOffsetX: 0.0, // Pan offset X
    viewOffsetY: 0.0, // Pan offset Y
    viewScale: 1.0,   // Zoom scale (1.0 = default)
    fractalMinX: -2.0, // Fractal space bounds (for rebase, changes on rebase)
    fractalMaxX: 2.0,
    fractalMinY: -2.0,
    fractalMaxY: 2.0,
    originalFractalMinX: -2.0, // Original fractal bounds (never change, used for random starting positions)
    originalFractalMaxX: 2.0,
    originalFractalMinY: -2.0,
    originalFractalMaxY: 2.0,
    exposureMultiplier: 1.0, // Internal multiplier, compounds on rebase
    rngMode: 5, // 0 = hash12 (PCG), 1 = Sobol scrambled (QMC), 2 = R2 (Legacy), 3 = Sobol pure (QMC), 4 = R2 Precision, 5 = Owen-Sobol Optimized (default), 6 = Combined LCG (Random)
    // Smooth interpolation targets (only used when randomize is pressed)
    targetA: -2.24,
    targetB: 0.43,
    targetC: -0.65,
    targetD: -2.43,
    defaultLerpSpeed: 0.05, // Base lerp speed for non-randomize transitions
    randomizeLerpSpeed: 0.0286, // 1.4x longer than default for Randomize
    paramLerpSpeed: 0.05, // Active lerp speed (set per transition)
    isLerping: false, // Flag to enable/disable interpolation
    lerpStartA: 0, // Initial values when lerping starts (for progress calculation)
    lerpStartB: 0,
    lerpStartC: 0,
    lerpStartD: 0,
    lerpProgress: 0, // Progress from 0 to 1 during interpolation
};

// Global WebGPU availability flag
let webgpuAvailable = false;
let device = null;
let context = null;
let gpuMaxBufferSize = null; // maxStorageBufferBindingSize (bytes) for current device; null in CPU-only mode

// GPU-only helper function hooks (assigned during WebGPU init)
// Keep these at module scope so CPU mode (and code outside the WebGPU init block) can safely call them.
let recreateGpuBuffers = async () => {}; // Recreated in WebGPU init; no-op in CPU mode

// CPU-only rendering state (when WebGPU unavailable)
let cpuDensityBuffer = null;
let cpuColorBufferR = null;
let cpuColorBufferG = null;
let cpuColorBufferB = null;
let cpuCanvas = null;
let cpuCtx = null;
let useCpuRng = false; // Internal flag: automatically set to true when WebGPU unavailable (fallback mode)

// Persistent JS-side buffers for CPU-math → GPU-render fast path (avoids mapAsync stalls)
let jsDensityBuffer = null;
let jsColorBufferR = null;
let jsColorBufferG = null;
let jsColorBufferB = null;

function ensureJsBuffers(size) {
    if (!jsDensityBuffer || jsDensityBuffer.length !== size) {
        jsDensityBuffer = new Uint32Array(size);
        jsColorBufferR = new Float32Array(size);
        jsColorBufferG = new Float32Array(size);
        jsColorBufferB = new Float32Array(size);
    }
}

const COLOR_RANGE = 16.0;

// Animation state (shared between GPU and CPU modes)
let isRunning = true; // Start in running mode
let animationFrameId = null;

// CPU fallback state (for adaptive performance)
let cpuWorkgroupCount = 1; // Start with absolute minimum - just 1 workgroup
let cpuLastFrameTime = 0;
let cpuFrameCounter = 0; // Frame counter for logging (temporary, CPU mode only)
const CPU_DEBUG_LOGGING = false; // Set to true to enable verbose CPU mode logging
const GPU_TIMING_DEBUG_LOGGING = false; // Set to true to log GPU timestamp query deltas

// CPU-only presentation helpers (optimization to avoid per-frame allocations)
let cpuImageData = null; // Reused ImageData buffer (CPU mode only)
let cpuMaxDensity = 0; // Tracked incrementally (CPU mode only)
let cpuLastPresentMs = 0; // Throttle presentation rate (CPU mode only)
const CPU_PRESENT_INTERVAL_MS = 100; // Presentation cap for CPU mode (10 fps) — compute is time-budgeted, so UI stays responsive
const CPU_MAX_PIXELS = 4_000_000; // Safety cap to avoid extreme allocations on high-DPI displays in CPU-only mode

// CPU-only compute tuning (used by startCpuRenderLoop)
let cpuRngIndex = 0; // Monotonic RNG index for CPU-only mode (separate from totalIterations display)
const CPU_WARMUP_ITERS = 12;
const CPU_ACCUM_ITERS = 32;
const CPU_BATCH_SIZE = 500; // Particles per batch (good cache locality; small enough for time slicing)
const CPU_COMPUTE_BUDGET_MS = 12; // Compute time slice per tick (keeps UI responsive on the main thread)
const CPU_TICK_INTERVAL_MS = 33; // ~30 fps scheduling target
const CPU_DENSITY_VISIBILITY_BOOST = 5.0; // Boost density-mode brightness so low-sample CPU renders are visible

// GPU buffer references (initialized in GPU block if WebGPU is available)
let paramsBufferInactive = null; // Inactive buffer (JS writes to this, only set if WebGPU available)
let computeBindGroupActive = null; // Active compute bind group (GPU)
let computeBindGroupInactive = null; // Inactive compute bind group (GPU)
let paramsBufferActive = null; // Active params buffer (GPU reads from this)
let computePipeline = null; // Compute pipeline (GPU)
let renderPipeline = null; // Render pipeline (GPU)
let renderBindGroupActive = null; // Active render bind group (GPU)
let renderBindGroupInactive = null; // Inactive render bind group (GPU)
let densityBuffer = null; // Accumulation density buffer (GPU)
let colorBufferR = null; // Accumulation color buffer R sum (GPU)
let colorBufferG = null; // Accumulation color buffer G sum (GPU)
let colorBufferB = null; // Accumulation color buffer B sum (GPU)

// Performance measurement (timestamp queries) - used by frame loop
let enableTimestamps = false;
let timestampQuerySet = null;
let timestampBuffer = null;
let timestampReadbackBuffer = null;
let lastComputeTimeNs = null; // Store compute time from previous frame for adaptive scaling
let lastTotalTimeNs = null; // Compute + render total time (timestamps), used for scheduling
let lastSubmittedGpuMs = null; // Fallback timing using queue.onSubmittedWorkDone (includes queue/backlog)

// Adaptive scheduler (frame loop): EMA timing + caps.
const TARGET_FRAME_TIME_MS = 23.4; // Target ~43fps
const WORKGROUP_DEADBAND_RATIO = 0.03; // Ignore tiny budget shifts to avoid jitter
const WORKGROUP_UP_ALPHA = 0.08; // Slow ramp-up to prevent oscillation
const WORKGROUP_DOWN_ALPHA = 0.2; // Faster ramp-down for responsiveness
let avgTimePerWorkgroup = 0.02; // ms/workgroup (EMA)
let inFlightFrames = 0; // Number of frames currently in GPU queue
const MAX_IN_FLIGHT_FRAMES = 3; // Maximum frames allowed in flight
let workgroupCount = 5000; // Adaptive scheduler adjusts this each frame (used by GPU + CPU-math fast path)

// --- Resilient GPU Renderer ---
/**
 * ResilientGPURenderer - Error recovery for WebGPU rendering
 * Handles device loss, validation errors, and transient failures with automatic recovery.
 */
class ResilientGPURenderer {
    constructor(options = {}) {
        this.recoveryAttempts = 0;
        this.maxRecoveryAttempts = options.maxRecoveryAttempts ?? 3;
        this.onFatalError = options.onFatalError ?? null;
        this.fallbackToCPU = options.fallbackToCPU ?? false;
        this.lastSuccessfulFrame = null;
        this.deviceLostHandler = null;
        this.isRecovering = false;
        this.consecutiveFailures = 0; // Track consecutive failures for smarter backoff
        this.lastErrorTime = null; // Track when last error occurred
        this.errorCooldown = 2000; // 2 second cooldown after max attempts before allowing retry
        this.lastFatalErrorTime = null;
    }
    
    /**
     * Run a render pass with retry/backoff and recovery.
     * @param {Function} renderFn - Async render body.
     */
        async render(renderFn) {
            if (this.isRecovering) return;
            
            try {
            await renderFn();
            // Successful frame resets failure state.
            this.recoveryAttempts = 0;
            this.consecutiveFailures = 0;
            this.lastSuccessfulFrame = performance.now();
        } catch (error) {
            console.error('Rendering error:', error);
            
            // Validation errors are code bugs, not transient.
            if (this.isUnrecoverableError(error)) {
                this.handleFatalError(error);
                return;
            }
            
            // Cooldown after fatal failure.
            if (this.lastFatalErrorTime && 
                (performance.now() - this.lastFatalErrorTime) < this.errorCooldown) {
                return;
            }
            
            this.consecutiveFailures++;
            this.lastErrorTime = performance.now();
            
            if (this.recoveryAttempts < this.maxRecoveryAttempts) {
                this.recoveryAttempts++;
                
                // Exponential backoff with ±20% jitter.
                const baseDelay = Math.min(50 * Math.pow(2, this.recoveryAttempts - 1), 1000);
                const jitter = baseDelay * 0.2 * (Math.random() * 2 - 1); // ±20%
                    const backoff = Math.max(50, baseDelay + jitter); // Minimum 50ms
                    
                    // Device-lost needs longer backoff.
                    if (this.isDeviceLostError(error)) {
                        const deviceLostBackoff = Math.min(500 * this.recoveryAttempts, 2000);
                        await new Promise(r => setTimeout(r, deviceLostBackoff));
                    } else {
                        await new Promise(r => setTimeout(r, backoff));
                    }
                
                await this.recoverFromError(error);
            } else {
                // Enter cooldown after max attempts.
                this.lastFatalErrorTime = performance.now();
                this.handleFatalError(error);
            }
        }
    }
    
        /**
         * Treat validation errors as unrecoverable.
         */
        isUnrecoverableError(error) {
            // Guard against browsers that don't surface GPUValidationError as a global class.
            const ctor = globalThis.GPUValidationError;
            return (
                (typeof ctor === 'function' && error instanceof ctor) ||
                error?.name === 'GPUValidationError' ||
                (error?.message && error.message.includes('validation'))
            );
        }

        isDeviceLostError(error) {
            // WebGPU device-loss signaling varies by browser/version (and sometimes surfaces as a
            // DOMException). Prefer name/reason checks; use constructor when present.
            const ctor = globalThis.GPUDeviceLostError;
            return (
                (typeof ctor === 'function' && error instanceof ctor) ||
                error?.name === 'GPUDeviceLostError' ||
                error?.constructor?.name === 'GPUDeviceLostError' ||
                (error?.reason && (error.reason === 'destroyed' || error.reason === 'unknown'))
            );
        }
        
        /**
         * Recover by reinit, quality drop, or state clear.
         */
    async recoverFromError(error) {
        this.isRecovering = true;
        
            try {
                // Device lost: reinit path.
                if (this.isDeviceLostError(error)) {
                    console.warn('Device lost detected, attempting full reinitialization...');
                    await this.reinitializeDevice();
                    return;
                }
            
            // Memory/buffer errors: drop quality.
            if (error.message && (
                error.message.includes('memory') || 
                error.message.includes('buffer') ||
                error.message.includes('out of memory') ||
                error.message.includes('too large'))) {
                console.warn('Memory error detected, reducing quality...');
                await this.reduceQuality();
                return;
            }
            
            // Otherwise: clear transient state.
            console.warn('Clearing corrupted state...');
            await this.clearCorruptedState();
        } finally {
            this.isRecovering = false;
        }
    }
    
    /**
     * Escalate after repeated failures (CPU fallback or user message).
     */
    handleFatalError(error) {
        console.error('GPU renderer failed after multiple recovery attempts');
        
        if (this.fallbackToCPU) {
            console.warn('Falling back to CPU mode');
            if (typeof window !== 'undefined') {
                webgpuAvailable = false;
                useCpuRng = true;
            }
        } else if (this.onFatalError) {
            this.onFatalError(error);
        } else {
            const statusEl = document.getElementById('status-temp');
            if (statusEl) {
                statusEl.textContent = 'GPU rendering failed - refresh page to retry';
                statusEl.style.color = '#f55';
            }
        }
    }
    
    /**
     * Kick reinit flow for device loss.
     */
    async reinitializeDevice() {
        console.log('Reinitializing WebGPU device...');
        
        // Clear local state; init() handles full rebuild.
        this.clearCorruptedState();
        
        device = null;
        context = null;
        
        const statusEl = document.getElementById('status-temp');
        if (statusEl) {
            statusEl.textContent = 'GPU device lost - attempting recovery...';
            statusEl.style.color = '#fa5';
        }
        
        // device.lost handler triggers init() retry after a short delay.
    }
    
    /**
     * Reduce quality to recover from memory pressure.
     */
    async reduceQuality() {
        console.log('Reducing rendering quality...');

        // Also trim workgroups.
        if (typeof workgroupCount !== 'undefined') {
            workgroupCount = Math.max(100, Math.floor(workgroupCount * 0.5));
            console.log(`Reduced workgroup count to ${workgroupCount}`);
        }
    }
    
    /**
     * Reset local GPU state tracking.
     */
    async clearCorruptedState() {
        console.log('Clearing corrupted GPU state...');
        
        if (typeof inFlightFrames !== 'undefined') {
            inFlightFrames = 0;
        }
        
            if (typeof lastComputeTimeNs !== 'undefined') {
                lastComputeTimeNs = null;
            }
            if (typeof lastTotalTimeNs !== 'undefined') {
                lastTotalTimeNs = null;
            }
            if (typeof lastSubmittedGpuMs !== 'undefined') {
                lastSubmittedGpuMs = null;
            }
        
        // Buffers/pipelines are left intact; device.lost handles full teardown.
    }
    
    /**
     * Setup device lost handler
     */
    setupDeviceLostHandler(gpuDevice, onDeviceLost) {
        if (!gpuDevice) return;

        this.onDeviceLost = typeof onDeviceLost === 'function' ? onDeviceLost : null;
        
        this.deviceLostHandler = gpuDevice.lost.then((info) => {
            console.error('WebGPU device lost:', info.message);
            this.recoveryAttempts = 0; // Reset for device loss recovery
            this.handleDeviceLost(info);
        });
    }
    
    /**
     * Handle device lost event
     */
    async handleDeviceLost(info) {
        console.error('Device lost - reason:', info.reason, 'message:', info.message);
        
        // Clear all GPU resources
        await this.clearCorruptedState();
        
        // Show user message
        const statusEl = document.getElementById('status-temp');
        if (statusEl) {
            statusEl.textContent = 'GPU device lost - attempting recovery...';
            statusEl.style.color = '#fa5';
        }
        
        // Note: The existing device.lost handler in init() will handle the actual recovery
        // This method is called by the device.lost promise

        // Surface the caller-provided device-lost recovery UI (if any).
        try {
            if (typeof this.onDeviceLost === 'function') {
                this.onDeviceLost(info);
            }
        } catch (e) {
            console.error('Device-lost callback failed:', e);
        }
    }
    
    /**
     * Reset recovery state (call after successful device reinitialization)
     */
    reset() {
        this.recoveryAttempts = 0;
        this.consecutiveFailures = 0;
        this.isRecovering = false;
        this.lastSuccessfulFrame = null;
        this.lastErrorTime = null;
        this.lastFatalErrorTime = null;
    }
}

// Global instance of resilient renderer
let resilientRenderer = null;

// Attract mode state (must be outside GPU block for CPU mode)
let attractModeActive = false;
let attractModeSpeed = 0; // 0=normal, 1=slow (Shift), 2=ultra-slow (Ctrl) - persists across stop/resume
let attractModeStartTime = 0;
let attractModeLastUpdateTime = 0; // Track last update time for delta calculation
let attractModeFirstStart = true; // Track if this is the first attract mode start since app load
let autoRebaseEnabled = true;
let autoRebaseTimeout = null;
const AUTO_REBASE_DELAY_MS = 120;
// Attract mode parameters: each variable has its own phase offset and frequency
// Velocity-based bouncing system for continuous animation
// Each parameter has: value (current position), velocity (speed with direction), 
// baseSpeed (target speed), and speedVariation (random speed changes)
const attractModeParams = {
    a: { value: 0, velocity: 0, baseSpeed: 0, speedVariation: 0, lastSpeedChange: 0, nextSpeedChangeAt: 0 },
    b: { value: 0, velocity: 0, baseSpeed: 0, speedVariation: 0, lastSpeedChange: 0, nextSpeedChangeAt: 0 },
    c: { value: 0, velocity: 0, baseSpeed: 0, speedVariation: 0, lastSpeedChange: 0, nextSpeedChangeAt: 0 },
    d: { value: 0, velocity: 0, baseSpeed: 0, speedVariation: 0, lastSpeedChange: 0, nextSpeedChangeAt: 0 }
};

// Calibration/benchmark mode state (Shift+C / Alt+C to trigger)
let calibrationActive = false;
let calibrationStartTime = 0;
let calibrationFrameCount = 0;
let calibrationSavedState = null;  // Saved state to restore after calibration
let calibrationSeedUsed = null;
let calibrationSeedLabel = '';
let calibrationParamSeedUsed = null;
let calibrationParamSeedLabel = '';
let calibrationPopupOffset = 0;
let calibrationCanvasState = null;
const CALIBRATION_DURATION_MS = 5000;  // Legacy (time-based) calibration duration; unused (fixed-frame calibration)
const CALIBRATION_FRAME_LIMIT = 375;  // Fixed-work benchmark: identical math across devices (~50% longer)
const CALIBRATION_WORKGROUPS = 50000;  // Fixed workgroup count for repeatability
const CALIBRATION_RENDER_WIDTH = 1920;
const CALIBRATION_RENDER_HEIGHT = 1080;
// Fixed parameters for calibration (consistent for benchmark comparisons)
const CALIBRATION_PARAMS = { a: -2.24, b: 0.43, c: -0.65, d: -2.43, seed: 137.5 };
let calibrationTimings = [];  // Array of per-frame timing data

// Application state flags (must be in global scope to avoid temporal dead zone)
let needsClear = true; // Flag to clear accumulation buffers (set by UI interactions)
let needsTimingReset = false; // Reset perf estimator when compute cost changes (params), not on clear/rebase
let isInteracting = false; // True when user is dragging/zooming or using UI (must be declared early for CPU mode)
let isMouseDown = false; // Track if any mouse button is down for dragging (must be declared early for CPU mode)
let interactionTimeout = null; // Timeout to reset interaction flag (must be global for CPU mode)

// Canvas dimensions (must be global for CPU mode compatibility)
let canvasWidth = 0;
let canvasHeight = 0;

// Helper function to mark UI interaction (triggers full-speed rendering)
// Must be global so CPU mode can access it
function markInteraction() {
    isInteracting = true;
    // Clear existing timeout
    if (interactionTimeout) {
        clearTimeout(interactionTimeout);
    }
    // Reset interaction flag after 200ms of inactivity
    interactionTimeout = setTimeout(() => {
        isInteracting = false;
    }, 200);
}

async function init() {
    // Check for forceCPU URL parameter (for testing CPU fallback)
    const urlParams = new URLSearchParams(window.location.search);
    const forceCPU = urlParams.get('forceCPU') === 'true';
    
    // Check WebGPU availability (unless forced to CPU mode)
    webgpuAvailable = !forceCPU && !!navigator.gpu;

    // Shared canvas reference (used by both CPU and GPU paths)
    const canvas = document.getElementById('canvas');
    if (!canvas) {
        throw new Error('Canvas element #canvas not found');
    }

        // Help overlay wiring (F1 toggles).
    const helpOverlay = document.getElementById('help-overlay');
    const helpCloseBtn = document.getElementById('help-close');
    const setHelpVisible = (visible) => {
        if (!helpOverlay) return;
        helpOverlay.style.display = visible ? 'flex' : 'none';
        helpOverlay.setAttribute('aria-hidden', visible ? 'false' : 'true');
    };
    if (helpOverlay) {
        helpOverlay.addEventListener('click', (e) => {
            if (e.target === helpOverlay) {
                setHelpVisible(false);
            }
        });
    }
    if (helpCloseBtn) {
        helpCloseBtn.addEventListener('click', () => setHelpVisible(false));
    }

    function computeScaledDims(targetW, targetH, maxPixels) {
        const desiredPixels = targetW * targetH;
        if (!maxPixels || desiredPixels <= maxPixels) {
            return { width: targetW, height: targetH, scale: 1.0 };
        }
        const scale = Math.min(1.0, Math.sqrt(maxPixels / desiredPixels));
        return {
            width: Math.max(1, Math.floor(targetW * scale)),
            height: Math.max(1, Math.floor(targetH * scale)),
            scale
        };
    }

    function applyCanvasSize(canvasEl, cssW, cssH, requestedDpr, maxPixels, reasonLabel = '') {
        const desiredW = Math.max(1, Math.ceil(cssW * requestedDpr));
        const desiredH = Math.max(1, Math.ceil(cssH * requestedDpr));
        const { width, height, scale } = computeScaledDims(desiredW, desiredH, maxPixels);

        canvasWidth = width;
        canvasHeight = height;
        canvasEl.width = canvasWidth;
        canvasEl.height = canvasHeight;
        canvasEl.style.width = cssW + 'px';
        canvasEl.style.height = cssH + 'px';

        if (scale < 0.999) {
            const statusTempEl = document.getElementById('status-temp');
            if (statusTempEl) {
                const pct = (scale * 100).toFixed(1);
                statusTempEl.textContent = `${reasonLabel ? reasonLabel + ': ' : ''}Internal res scaled to ${pct}%`;
                statusTempEl.style.color = '#fa5';
                setTimeout(() => {
                    if (statusTempEl.textContent.includes('Internal res scaled')) {
                        statusTempEl.textContent = '';
                        statusTempEl.style.color = '';
                    }
                }, 2500);
            }
        }

        return { desiredW, desiredH, actualW: width, actualH: height, scale };
    }

    function getCanvasPixelRatio(canvasEl) {
        const rect = canvasEl.getBoundingClientRect();
        if (!rect.width || !rect.height) return { x: 1, y: 1 };
        return { x: canvasEl.width / rect.width, y: canvasEl.height / rect.height };
    }

    if (forceCPU) {
        console.log('[Diagnostics] WebGPU disabled via ?forceCPU=true');
    }
    
    if (!webgpuAvailable) {
        // Show message but keep UI functional for CPU-only mode
        const statusTempEl = document.getElementById('status-temp');
        if (statusTempEl) {
            statusTempEl.textContent = 'WebGPU not available - using CPU-only mode';
            statusTempEl.style.color = '#fa5';
        }
        
        // Initialize CPU-only rendering (buffers only, don't start loop yet)
        await initCpuOnlyMode();
        // DON'T return here - continue to UI wiring so markInteraction() etc. are set up
        // The render loop will be started by startFrame() after UI is wired
        // Skip all GPU initialization below - go straight to UI wiring
    }
    
    // GPU initialization (only if WebGPU is available)
    if (webgpuAvailable) {
        // 1. Force High Performance GPU (Crucial for Desktops/Laptops with iGPU)
        // COMPAT: Some Chrome builds warn about powerPreference; the request still works.
        const adapter = await navigator.gpu.requestAdapter({
        powerPreference: "high-performance"
    });
    
    if (!adapter) {
        // Fallback to CPU-only mode
        const statusTempEl = document.getElementById('status-temp');
        if (statusTempEl) {
            statusTempEl.textContent = 'No WebGPU adapter - using CPU-only mode';
            statusTempEl.style.color = '#fa5';
        }
        await initCpuOnlyMode();
        return;
    }

    // 2. INSPECT HARDWARE LIMITS
    const adapterLimits = adapter.limits;
    
    // 3. REQUEST HIGHEST POSSIBLE LIMITS
    // We need to explicitly ask for the max values, otherwise we get 128MB defaults.
    const requiredLimits = {};
    
    // Helper to safely copy limits
    const copyLimit = (name) => {
        if (adapterLimits[name]) requiredLimits[name] = adapterLimits[name];
    };
    
    copyLimit('maxStorageBufferBindingSize'); // <--- CRITICAL for 4K/8K Buffers
    copyLimit('maxBufferSize');
    copyLimit('maxComputeWorkgroupStorageSize');
    
    // Request optional GPU features when available (not requesting -> device.features.has() will be false)
    const requiredFeatures = [];
    if (adapter.features && adapter.features.has('timestamp-query')) {
        requiredFeatures.push('timestamp-query');
    }
    
    device = await adapter.requestDevice({
        requiredLimits: requiredLimits,
        requiredFeatures,
    });
    
    // Initialize resilient renderer
    resilientRenderer = new ResilientGPURenderer({
        maxRecoveryAttempts: 3,
        fallbackToCPU: true, // Enable CPU fallback on fatal errors
        onFatalError: (error) => {
            console.error('Fatal GPU error after recovery attempts:', error);
            const statusEl = document.getElementById('status-temp');
            if (statusEl) {
                statusEl.textContent = 'GPU rendering failed - using CPU fallback';
                statusEl.style.color = '#fa5';
            }
        }
    });
    
    // Setup device lost handler with recovery callback
    resilientRenderer.setupDeviceLostHandler(device, () => {
        // Attempt to reinitialize WebGPU
        console.log('Attempting GPU recovery...');
        // Note: Full reinitialization requires calling init() again, but that's complex
        // For now, we'll show a message and let the user refresh if needed
        const statusEl = document.getElementById('status-temp');
        if (statusEl) {
            statusEl.textContent = 'GPU recovery failed - please refresh the page';
            statusEl.style.color = '#f55';
        }
    });
    
    // Handle device lost errors (GPU crashes/hangs).
    // COMPAT: some browsers surface device loss as a DOMException-like object; prefer name/reason checks.
    device.addEventListener('uncapturederror', (event) => {
        console.error('GPU uncaptured error:', event.error);
        // Some browsers may not expose GPUDeviceLostError as a global constructor.
        const error = event.error;
        if (error && (error.constructor.name === 'GPUDeviceLostError' || 
                      error.name === 'GPUDeviceLostError' ||
                      (error.reason && (error.reason === 'destroyed' || error.reason === 'unknown')))) {
            console.error('GPU device lost! Reason:', error.reason || 'unknown');
            // Don't show alert - let resilient renderer handle recovery
            // The device.lost promise will trigger recovery
        }
    });
    
        // Get device limits for buffer size checking (now using actual requested limits)
        const maxBufferSize = device.limits.maxStorageBufferBindingSize || 134217728; // Fallback to 128 MB if not available
        gpuMaxBufferSize = maxBufferSize;
        
        // Adaptive performance scaling handled in frame loop
        
            context = canvas.getContext('webgpu');
        
        // Handle High DPI with a safety valve: downscale internal resolution if a single
        // accumulation buffer would exceed maxStorageBufferBindingSize.
        const cssW = window.innerWidth;
        const cssH = window.innerHeight;
        const requestedDpr = window.devicePixelRatio || 1;
        const maxPixels = Math.max(1, Math.floor((maxBufferSize - 4) / 4));
        applyCanvasSize(canvas, cssW, cssH, requestedDpr, maxPixels, 'WebGPU');

        const presentationFormat = navigator.gpu.getPreferredCanvasFormat();
        context.configure({
            device,
        format: presentationFormat,
        alphaMode: 'premultiplied',
    });

    // --- Buffers ---

    // 1. Uniform Params Buffer (Double-buffered to prevent slider interaction blocking)
    // Extended to 160 bytes (adaptive iteration + padding) for 16-byte alignment
    const paramsBufferSize = 40 * 4; // 160 bytes (padded to 16-byte alignment)
    const paramsBufferA = device.createBuffer({
        size: paramsBufferSize,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    const paramsBufferB = device.createBuffer({
        size: paramsBufferSize,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    paramsBufferActive = paramsBufferA; // Currently active buffer (GPU reads from this) - set global variable
    paramsBufferInactive = paramsBufferB; // Inactive buffer (JS writes to this) - set global variable

    // getRenderDimensions() moved outside GPU block for updateParams()
    
    // Declare buffer variables first (use globals so other code paths can access them)
    densityBuffer = null;
    colorBufferR = null;
    colorBufferG = null;
    colorBufferB = null;
    
        const isColorEnabled = () => (state && state.colorMethod !== 0);
        
        // Function to create/recreate buffers based on current render dimensions.
        // In density-only mode, we skip allocating the three color buffers to save memory/bandwidth.
        function createBuffers(colorEnabled = isColorEnabled()) {
            const renderDims = getRenderDimensions();
            const bufferSize = renderDims.width * renderDims.height * 4 + 4; // +4 bytes avoids perf cliff on nvidia/chrome (must be multiple of 4)
            
            // Double-check buffer size doesn't exceed limit
            if (bufferSize > maxBufferSize) {
                throw new Error(`Buffer size ${bufferSize} exceeds limit ${maxBufferSize}`);
            }
            
            // Destroy old buffers if they exist
            if (densityBuffer) densityBuffer.destroy();
            if (colorBufferR) colorBufferR.destroy();
            if (colorBufferG) colorBufferG.destroy();
            if (colorBufferB) colorBufferB.destroy();
            
            const density = device.createBuffer({
                size: bufferSize,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
            });

            if (!colorEnabled) {
                return { density, colorR: null, colorG: null, colorB: null };
            }

            // Color buffers (atomic u32 holding bitcast f32 running averages).
            return {
                density,
                colorR: device.createBuffer({
                    size: bufferSize,
                    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
                }),
                colorG: device.createBuffer({
                    size: bufferSize,
                    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
                }),
                colorB: device.createBuffer({
                    size: bufferSize,
                    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
                }),
            };
        }
        
        // 2. Density Storage Buffer (The "Deep Canvas")
        let buffers = createBuffers(isColorEnabled());
        densityBuffer = buffers.density;
        colorBufferR = buffers.colorR;
        colorBufferG = buffers.colorG;
        colorBufferB = buffers.colorB;

    // Clear pipeline (to reset density buffer)
    // Actually, WebGPU queue.writeBuffer is fast enough to zero it out on change.

    // --- Pipelines ---

        // Load main shader
        const shaderCode = await loadShader('shaders/fractal-accumulation.wgsl');
        // Firefox: add label to avoid empty label error
        const module = device.createShaderModule({
            label: 'mainShader',
            code: shaderCode
        });

        // Optional diagnostics: compilation info is asynchronous and supported in some browsers.
        if (module.getCompilationInfo) {
            try {
                const compilationInfo = await module.getCompilationInfo();
                if (compilationInfo?.messages?.length) {
                    for (const msg of compilationInfo.messages) {
                        if (msg.type === 'error') {
                            console.error(`WGSL error at ${msg.lineNum}:${msg.linePos}: ${msg.message}`);
                        }
                    }
                }
            } catch (e) {
                // Non-fatal: some browsers throw or omit this API.
                console.warn('Could not read shader compilation info:', e);
            }
        }

        // Bind group layouts + pipelines: keep separate density-only and color modes.
        // This lets density mode avoid allocating/binding three extra color buffers.
        const densityComputeBindGroupLayout = device.createBindGroupLayout({
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }, // densityBuffer
            ]
        });
        const colorComputeBindGroupLayout = device.createBindGroupLayout({
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }, // densityBuffer
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }, // colorBufferR
                { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }, // colorBufferG
                { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }, // colorBufferB
            ]
        });

        const densityRenderBindGroupLayout = device.createBindGroupLayout({
            entries: [
                { binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
                { binding: 5, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } }, // density (non-atomic read)
            ]
        });
        const colorRenderBindGroupLayout = device.createBindGroupLayout({
            entries: [
                { binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
                { binding: 5, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } }, // density
                { binding: 6, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } }, // colorR
                { binding: 7, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } }, // colorG
                { binding: 8, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } }, // colorB
            ]
        });

        let computePipelineDensity = null;
        let computePipelineColor = null;
        let renderPipelineDensity = null;
        let renderPipelineColor = null;

        try {
            const statusTempEl = document.getElementById('status-temp');
            if (statusTempEl) {
                statusTempEl.textContent = 'Compiling GPU shaders…';
                statusTempEl.style.color = '#4a9eff';
            }

            // Compile both entry points concurrently (often helps on cold caches / driver compilers).
            [computePipelineDensity, computePipelineColor] = await Promise.all([
                device.createComputePipelineAsync({
                    layout: device.createPipelineLayout({ bindGroupLayouts: [densityComputeBindGroupLayout] }),
                    compute: { module, entryPoint: 'computeMainDensity' },
                }),
                device.createComputePipelineAsync({
                    layout: device.createPipelineLayout({ bindGroupLayouts: [colorComputeBindGroupLayout] }),
                    compute: { module, entryPoint: 'computeMain' },
                }),
            ]);

            if (statusTempEl && statusTempEl.textContent === 'Compiling GPU shaders…') {
                statusTempEl.textContent = '';
                statusTempEl.style.color = '';
            }
        } catch (e) {
            console.error('Compute pipeline creation failed:', e);
            // On failure, pull compilation diagnostics (this can be slow, so keep it out of the hot path).
            if (module && module.getCompilationInfo) {
                try {
                    const compilationInfo = await module.getCompilationInfo();
                    console.error('WebGPU compilation info for shader module', compilationInfo);
                    if (compilationInfo && compilationInfo.messages) {
                        for (const msg of compilationInfo.messages) {
                            console.error(`  ${msg.type} at ${msg.lineNum}:${msg.linePos}: ${msg.message}`);
                        }
                    }
                } catch (infoError) {
                    console.error('Could not get compilation info:', infoError);
                }
            }
            throw e;
        }

        // Render pipelines (density-only + color).
        renderPipelineDensity = device.createRenderPipeline({
            layout: device.createPipelineLayout({ bindGroupLayouts: [densityRenderBindGroupLayout] }),
            vertex: { module, entryPoint: 'vertexMain' },
            fragment: { module, entryPoint: 'fragmentMainDensity', targets: [{ format: presentationFormat }] },
            primitive: { topology: 'triangle-list' },
        });
        renderPipelineColor = device.createRenderPipeline({
            layout: device.createPipelineLayout({ bindGroupLayouts: [colorRenderBindGroupLayout] }),
            vertex: { module, entryPoint: 'vertexMain' },
            fragment: { module, entryPoint: 'fragmentMain', targets: [{ format: presentationFormat }] },
            primitive: { topology: 'triangle-list' },
        });

        // Bind groups (rebuilt when accumulation buffers are recreated).
        let densityComputeBindGroupA = null;
        let densityComputeBindGroupB = null;
        let densityRenderBindGroupA = null;
        let densityRenderBindGroupB = null;
        let colorComputeBindGroupA = null;
        let colorComputeBindGroupB = null;
        let colorRenderBindGroupA = null;
        let colorRenderBindGroupB = null;

        function rebuildBindGroups() {
            densityComputeBindGroupA = device.createBindGroup({
                layout: densityComputeBindGroupLayout,
                entries: [
                    { binding: 0, resource: { buffer: paramsBufferA } },
                    { binding: 1, resource: { buffer: densityBuffer } },
                ]
            });
            densityComputeBindGroupB = device.createBindGroup({
                layout: densityComputeBindGroupLayout,
                entries: [
                    { binding: 0, resource: { buffer: paramsBufferB } },
                    { binding: 1, resource: { buffer: densityBuffer } },
                ]
            });
            densityRenderBindGroupA = device.createBindGroup({
                layout: densityRenderBindGroupLayout,
                entries: [
                    { binding: 0, resource: { buffer: paramsBufferA } },
                    { binding: 5, resource: { buffer: densityBuffer } },
                ]
            });
            densityRenderBindGroupB = device.createBindGroup({
                layout: densityRenderBindGroupLayout,
                entries: [
                    { binding: 0, resource: { buffer: paramsBufferB } },
                    { binding: 5, resource: { buffer: densityBuffer } },
                ]
            });

            if (colorBufferR && colorBufferG && colorBufferB) {
                colorComputeBindGroupA = device.createBindGroup({
                    layout: colorComputeBindGroupLayout,
                    entries: [
                        { binding: 0, resource: { buffer: paramsBufferA } },
                        { binding: 1, resource: { buffer: densityBuffer } },
                        { binding: 2, resource: { buffer: colorBufferR } },
                        { binding: 3, resource: { buffer: colorBufferG } },
                        { binding: 4, resource: { buffer: colorBufferB } },
                    ]
                });
                colorComputeBindGroupB = device.createBindGroup({
                    layout: colorComputeBindGroupLayout,
                    entries: [
                        { binding: 0, resource: { buffer: paramsBufferB } },
                        { binding: 1, resource: { buffer: densityBuffer } },
                        { binding: 2, resource: { buffer: colorBufferR } },
                        { binding: 3, resource: { buffer: colorBufferG } },
                        { binding: 4, resource: { buffer: colorBufferB } },
                    ]
                });
                colorRenderBindGroupA = device.createBindGroup({
                    layout: colorRenderBindGroupLayout,
                    entries: [
                        { binding: 0, resource: { buffer: paramsBufferA } },
                        { binding: 5, resource: { buffer: densityBuffer } },
                        { binding: 6, resource: { buffer: colorBufferR } },
                        { binding: 7, resource: { buffer: colorBufferG } },
                        { binding: 8, resource: { buffer: colorBufferB } },
                    ]
                });
                colorRenderBindGroupB = device.createBindGroup({
                    layout: colorRenderBindGroupLayout,
                    entries: [
                        { binding: 0, resource: { buffer: paramsBufferB } },
                        { binding: 5, resource: { buffer: densityBuffer } },
                        { binding: 6, resource: { buffer: colorBufferR } },
                        { binding: 7, resource: { buffer: colorBufferG } },
                        { binding: 8, resource: { buffer: colorBufferB } },
                    ]
                });
            } else {
                colorComputeBindGroupA = null;
                colorComputeBindGroupB = null;
                colorRenderBindGroupA = null;
                colorRenderBindGroupB = null;
            }
        }

        function selectActiveMode() {
            const useColor = isColorEnabled() && !!(colorBufferR && colorBufferG && colorBufferB);

            computePipeline = useColor ? computePipelineColor : computePipelineDensity;
            renderPipeline = useColor ? renderPipelineColor : renderPipelineDensity;

            if (useColor) {
                computeBindGroupActive = (paramsBufferActive === paramsBufferA) ? colorComputeBindGroupA : colorComputeBindGroupB;
                computeBindGroupInactive = (paramsBufferActive === paramsBufferA) ? colorComputeBindGroupB : colorComputeBindGroupA;
                renderBindGroupActive = (paramsBufferActive === paramsBufferA) ? colorRenderBindGroupA : colorRenderBindGroupB;
                renderBindGroupInactive = (paramsBufferActive === paramsBufferA) ? colorRenderBindGroupB : colorRenderBindGroupA;
            } else {
                computeBindGroupActive = (paramsBufferActive === paramsBufferA) ? densityComputeBindGroupA : densityComputeBindGroupB;
                computeBindGroupInactive = (paramsBufferActive === paramsBufferA) ? densityComputeBindGroupB : densityComputeBindGroupA;
                renderBindGroupActive = (paramsBufferActive === paramsBufferA) ? densityRenderBindGroupA : densityRenderBindGroupB;
                renderBindGroupInactive = (paramsBufferActive === paramsBufferA) ? densityRenderBindGroupB : densityRenderBindGroupA;
            }
        }

        rebuildBindGroups();
        selectActiveMode();

        // Recreate accumulation buffers and bind groups when canvas backing-store size changes
        // (or when switching density-only <-> color mode).
        recreateGpuBuffers = async function recreateGpuBuffersForCurrentCanvas() {
            if (!webgpuAvailable || !device) return;
            buffers = createBuffers(isColorEnabled());
            densityBuffer = buffers.density;
            colorBufferR = buffers.colorR;
            colorBufferG = buffers.colorG;
            colorBufferB = buffers.colorB;
            rebuildBindGroups();
            selectActiveMode();
            needsClear = true;
            needsTimingReset = true;
            updateParams();
        };

        // --- Performance Measurement (Timestamp Queries) ---
        // Use globals so frame loop can read these values
        enableTimestamps = false; // Will be set to true if supported
        timestampQuerySet = null;
        timestampBuffer = null;
        timestampReadbackBuffer = null;
        lastComputeTimeNs = null; // Store compute time from previous frame for adaptive scaling
        lastTotalTimeNs = null;
        lastSubmittedGpuMs = null;
        
        // Check if timestamp queries are supported
    if (device.features.has('timestamp-query')) {
        try {
            timestampQuerySet = device.createQuerySet({
                type: 'timestamp',
                count: 4, // Before compute, after compute, after render, end
            });
            timestampBuffer = device.createBuffer({
                size: 4 * 8, // 4 timestamps × 8 bytes each
                usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
            });
            timestampReadbackBuffer = device.createBuffer({
                size: 4 * 8,
                usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
            });
            enableTimestamps = true;
            console.log('Timestamp queries enabled for performance measurement');
        } catch (e) {
            console.warn('Could not create timestamp queries:', e);
        }
    } else {
        console.log('Timestamp queries not supported on this device');
    }
    

    // --- State & UI ---
    // Note: state is now declared in global scope (above init() function) to avoid temporal dead zone issues

    // needsClear is now in global scope (above init() function) to avoid temporal dead zone
    
    // Attract mode state variables moved to global scope for CPU mode compatibility
    
    // Adaptive performance scheduler + frame pacing (globals used by frame loop)
    avgTimePerWorkgroup = 0.02; // Reset estimator on init
    inFlightFrames = 0; // Reset queue depth on init
    
    // Note: updateParams(), paramsArrayBuffer, paramsDataView, getRenderDimensions(), clearDensity() moved outside GPU block
    // Note: restoreStateFromHistory(), randomizeParameters(), resetViewTransform() are defined outside GPU block for CPU mode compatibility
    
        // Supersampling (SSAA) support was removed; internal resolution is handled via DPR and
        // the existing max-buffer safety scaling in applyCanvasSize().
        
        // Non-linear mapping functions for better precision at low values
        // Keyboard shortcuts are registered later with shared UI wiring (so CPU mode gets them too).
        
        // Reset resilient renderer after successful GPU initialization
        if (resilientRenderer) {
            resilientRenderer.reset();
        }
    
    } // End of GPU initialization block (if webgpuAvailable)

    // -----------------------------------------------------------------------------
    // Randomize / history / cycle helpers (must be outside WebGPU init block)
    // -----------------------------------------------------------------------------
    // These are used by UI wiring + attract mode in BOTH GPU and CPU fallback paths.
    const PARAM_HISTORY_SIZE = 64;
    const paramHistory = [];
    let paramHistoryIndex = -1;  // Current position in history (-1 = no history)
    let paramHistoryCount = 0;   // Number of entries in history (0 to PARAM_HISTORY_SIZE)
    
    // Cycle state for SHIFT+Randomize (8 subtle permutations)
    let cycleState = {
        initialValues: null,  // {a, b, c, d} - saved when cycle starts
        currentIndex: 0       // 0-7, cycles through 8 permutations
    };
    
    function clearCycleState() {
        cycleState.initialValues = null;
        cycleState.currentIndex = 0;
    }
    
    function getCyclePermutation(index, baseValues, variationScale = 1) {
        // Scale variation inversely with zoom level: higher zoom = smaller variations
        // At 2x zoom, variations are 2x smaller; at 0.5x zoom, variations are 2x larger
        // This keeps features from shifting out of view when zoomed in
        const baseVariation = 0.08;
        // Use current viewScale to scale variation - higher zoom = smaller variation
        const currentZoom = Math.max(state.viewScale || 1.0, 0.1); // Ensure we have a valid zoom value
        const variation = (baseVariation * variationScale) / currentZoom;
        const patterns = [
            { a: 0,   b: 0,   c: 0,   d: 0   },
            { a: 1,   b: 1,   c: -0.5,d: -0.5},
            { a: -0.5,b: -0.5,c: 1,   d: 1   },
            { a: 1,   b: -1,  c: 1,   d: -1  },
            { a: -1,  b: -1,  c: 0.5, d: 0.5 },
            { a: 0.5, b: 0.5, c: -1,  d: -1  },
            { a: -1,  b: 1,   c: -1,  d: 1   },
            { a: 0.7, b: -0.7,c: -0.7,d: 0.7 }
        ];
        
        const pattern = patterns[index % 8];
        return {
            a: baseValues.a + pattern.a * variation,
            b: baseValues.b + pattern.b * variation,
            c: baseValues.c + pattern.c * variation,
            d: baseValues.d + pattern.d * variation
        };
    }
    
    function applyCyclePermutation() {
        if (!cycleState.initialValues) {
            cycleState.initialValues = { a: state.a, b: state.b, c: state.c, d: state.d };
            cycleState.currentIndex = 1; // skip pattern 0 on first press
        }
        
        const perm = getCyclePermutation(cycleState.currentIndex, cycleState.initialValues);
        state.isLerping = false; // Stop lerping when cycling parameters
        state.a = perm.a;
        state.b = perm.b;
        state.c = perm.c;
        state.d = perm.d;
        // Update targets to match cycled values
        state.targetA = perm.a;
        state.targetB = perm.b;
        state.targetC = perm.c;
        state.targetD = perm.d;
        
        syncParamSlidersToState(3);
        
        cycleState.currentIndex = (cycleState.currentIndex + 1) % 8;
        needsClear = true;
        updateParams();
    }

    function applyMicroCyclePermutation() {
        if (!cycleState.initialValues) {
            cycleState.initialValues = { a: state.a, b: state.b, c: state.c, d: state.d };
            cycleState.currentIndex = 1; // skip pattern 0 on first press
        }
        
        const perm = getCyclePermutation(cycleState.currentIndex, cycleState.initialValues, 1 / 256);
        state.isLerping = false; // Stop lerping when cycling parameters
        state.a = perm.a;
        state.b = perm.b;
        state.c = perm.c;
        state.d = perm.d;
        state.targetA = perm.a;
        state.targetB = perm.b;
        state.targetC = perm.c;
        state.targetD = perm.d;
        syncParamSlidersToState(3);
        
        cycleState.currentIndex = (cycleState.currentIndex + 1) % 8;
        needsClear = true;
        updateParams();
    }
    
    function saveStateToHistory() {
        const savedState = {
            a: state.a,
            b: state.b,
            c: state.c,
            d: state.d,
            exposure: state.exposure,
            gamma: state.gamma,
            contrast: state.contrast,
            seed: state.seed,
            colorMethod: state.colorMethod,
            rngMode: state.rngMode,
            viewOffsetX: state.viewOffsetX,
            viewOffsetY: state.viewOffsetY,
            viewScale: state.viewScale,
            fractalMinX: state.fractalMinX,
            fractalMaxX: state.fractalMaxX,
            fractalMinY: state.fractalMinY,
            fractalMaxY: state.fractalMaxY,
            originalFractalMinX: state.originalFractalMinX,
            originalFractalMaxX: state.originalFractalMaxX,
            originalFractalMinY: state.originalFractalMinY,
            originalFractalMaxY: state.originalFractalMaxY,
            exposureMultiplier: state.exposureMultiplier
        };
        
        paramHistoryIndex = (paramHistoryIndex + 1) % PARAM_HISTORY_SIZE;
        paramHistory[paramHistoryIndex] = savedState;
        if (paramHistoryCount < PARAM_HISTORY_SIZE) paramHistoryCount++;
    }
    
    function restoreStateFromHistory() {
        if (paramHistoryCount === 0) {
            const statusTempEl = document.getElementById('status-temp');
            if (statusTempEl) {
                statusTempEl.textContent = 'No history available';
                statusTempEl.style.color = '#f55';
                setTimeout(() => { statusTempEl.textContent = ''; }, 2000);
            }
            return false;
        }
        
        // Go back one entry (wrap)
        let restoreIndex = paramHistoryIndex - 1;
        if (restoreIndex < 0) restoreIndex = paramHistoryCount - 1;
        const savedState = paramHistory[restoreIndex];
        if (!savedState) return false;
        
        // If attract mode is active, instantly change parameters (no lerp)
        // Otherwise, use smooth interpolation with ease-in-out curve
        if (attractModeActive) {
            // Instant change for attract mode
            Object.assign(state, savedState);
            
            // Update UI immediately
            const pa = document.getElementById('param-a');
            const pb = document.getElementById('param-b');
            const pc = document.getElementById('param-c');
            const pd = document.getElementById('param-d');
            const va = document.getElementById('val-a');
            const vb = document.getElementById('val-b');
            const vc = document.getElementById('val-c');
            const vd = document.getElementById('val-d');
            if (pa && va) { pa.value = state.a; va.innerText = state.a.toFixed(3); }
            if (pb && vb) { pb.value = state.b; vb.innerText = state.b.toFixed(3); }
            if (pc && vc) { pc.value = state.c; vc.innerText = state.c.toFixed(3); }
            if (pd && vd) { pd.value = state.d; vd.innerText = state.d.toFixed(3); }
            
            // Update non-linear sliders if present
            const exp = document.getElementById('param-exposure');
            const vexp = document.getElementById('val-exp');
            const gam = document.getElementById('param-gamma');
            const vgam = document.getElementById('val-gamma');
            const con = document.getElementById('param-contrast');
            const vcon = document.getElementById('val-contrast');
            const selColor = document.getElementById('select-color-mode');
            if (exp && typeof mapExposureToSlider === 'function') exp.value = mapExposureToSlider(state.exposure);
            if (vexp) vexp.innerText = state.exposure.toFixed(3);
            if (gam && typeof mapGammaToSlider === 'function') gam.value = mapGammaToSlider(state.gamma);
            if (vgam) vgam.innerText = state.gamma.toFixed(3);
            if (con && typeof mapContrastToSlider === 'function') con.value = mapContrastToSlider(state.contrast);
            if (vcon) vcon.innerText = state.contrast.toFixed(3);
            if (selColor) selColor.value = state.colorMethod;
            
            if (typeof previousSliderValues !== 'undefined') {
                previousSliderValues.set('a', state.a);
                previousSliderValues.set('b', state.b);
                previousSliderValues.set('c', state.c);
                previousSliderValues.set('d', state.d);
            }
            
            needsClear = true;
            updateParams();
        } else {
            // Restore instantly (lerp only applies to Randomize)
            state.a = savedState.a;
            state.b = savedState.b;
            state.c = savedState.c;
            state.d = savedState.d;
            state.targetA = savedState.a;
            state.targetB = savedState.b;
            state.targetC = savedState.c;
            state.targetD = savedState.d;
            state.isLerping = false;
            
            // Also restore other state properties immediately (exposure, gamma, etc.)
            state.exposure = savedState.exposure;
            state.gamma = savedState.gamma;
            state.contrast = savedState.contrast;
            state.colorMethod = savedState.colorMethod;
            
            // Update non-linear sliders immediately
            const exp = document.getElementById('param-exposure');
            const vexp = document.getElementById('val-exp');
            const gam = document.getElementById('param-gamma');
            const vgam = document.getElementById('val-gamma');
            const con = document.getElementById('param-contrast');
            const vcon = document.getElementById('val-contrast');
            const selColor = document.getElementById('select-color-mode');
            if (exp && typeof mapExposureToSlider === 'function') exp.value = mapExposureToSlider(state.exposure);
            if (vexp) vexp.innerText = state.exposure.toFixed(3);
            if (gam && typeof mapGammaToSlider === 'function') gam.value = mapGammaToSlider(state.gamma);
            if (vgam) vgam.innerText = state.gamma.toFixed(3);
            if (con && typeof mapContrastToSlider === 'function') con.value = mapContrastToSlider(state.contrast);
            if (vcon) vcon.innerText = state.contrast.toFixed(3);
            if (selColor) selColor.value = state.colorMethod;
            
            // Update UI for a/b/c/d immediately
            const pa = document.getElementById('param-a');
            const pb = document.getElementById('param-b');
            const pc = document.getElementById('param-c');
            const pd = document.getElementById('param-d');
            const va = document.getElementById('val-a');
            const vb = document.getElementById('val-b');
            const vc = document.getElementById('val-c');
            const vd = document.getElementById('val-d');
            if (pa && va) { pa.value = state.a; va.innerText = state.a.toFixed(3); }
            if (pb && vb) { pb.value = state.b; vb.innerText = state.b.toFixed(3); }
            if (pc && vc) { pc.value = state.c; vc.innerText = state.c.toFixed(3); }
            if (pd && vd) { pd.value = state.d; vd.innerText = state.d.toFixed(3); }
        }
        
        paramHistoryIndex = restoreIndex;
        clearCycleState();
        updateStatusDisplay();
        
        const statusTempEl = document.getElementById('status-temp');
        if (statusTempEl) {
            const remaining = Math.max(0, paramHistoryCount - 1);
            statusTempEl.textContent = `Restored (${remaining} more in history)`;
            statusTempEl.style.color = '#5a5';
            setTimeout(() => { statusTempEl.textContent = ''; }, 2000);
        }
        
        needsClear = true;
        needsTimingReset = true;
        // Don't call updateParams() here if lerping - frame loop will handle it during interpolation
        if (!state.isLerping) {
            updateParams();
        }
        return true;
    }
    
    function randomizeParameters(options = {}) {
        saveStateToHistory();
        const { instant = false } = options;
        
        const fBase = 4.0;
        const fRange = 8.0;
        const newA = -fBase + fRange * (Math.floor(Math.random() * 10239) / 10238.0);
        const newB = -fBase + fRange * (Math.floor(Math.random() * 10241) / 10240.0);
        const newC = -fBase + fRange * (Math.floor(Math.random() * 10242) / 10241.0);
        const newD = -fBase + fRange * (Math.floor(Math.random() * 10148) / 10147.0);
        
        // If attract mode is active or instant mode requested, instantly change parameters (no lerp)
        // Otherwise, use smooth interpolation
        if (attractModeActive || instant) {
            // Instant change for attract mode
            state.a = newA;
            state.b = newB;
            state.c = newC;
            state.d = newD;
            state.targetA = newA;
            state.targetB = newB;
            state.targetC = newC;
            state.targetD = newD;
            state.isLerping = false;
            state.lerpProgress = 1;
            
            // Update UI immediately
            const pa = document.getElementById('param-a');
            const pb = document.getElementById('param-b');
            const pc = document.getElementById('param-c');
            const pd = document.getElementById('param-d');
            const va = document.getElementById('val-a');
            const vb = document.getElementById('val-b');
            const vc = document.getElementById('val-c');
            const vd = document.getElementById('val-d');
            if (pa && va) { pa.value = state.a; va.innerText = state.a.toFixed(3); }
            if (pb && vb) { pb.value = state.b; vb.innerText = state.b.toFixed(3); }
            if (pc && vc) { pc.value = state.c; vc.innerText = state.c.toFixed(3); }
            if (pd && vd) { pd.value = state.d; vd.innerText = state.d.toFixed(3); }
            
            if (typeof previousSliderValues !== 'undefined') {
                previousSliderValues.set('a', state.a);
                previousSliderValues.set('b', state.b);
                previousSliderValues.set('c', state.c);
                previousSliderValues.set('d', state.d);
            }
            
            clearCycleState();
            needsClear = true;
            needsTimingReset = true;
            updateParams();
        } else {
            // Smooth interpolation when attract mode is off
            // Store initial values for progress-based ease-in-out interpolation
            state.paramLerpSpeed = state.randomizeLerpSpeed;
            state.lerpStartA = state.a;
            state.lerpStartB = state.b;
            state.lerpStartC = state.c;
            state.lerpStartD = state.d;
            state.targetA = newA;
            state.targetB = newB;
            state.targetC = newC;
            state.targetD = newD;
            state.lerpProgress = 0; // Reset progress to 0
            state.isLerping = true;
            
            // Note: UI will be updated in frame loop as values interpolate
            // Don't update sliders here - let them follow the interpolated values
            
            clearCycleState();
            needsClear = true;
            needsTimingReset = true;
            // Don't call updateParams() here - frame loop will handle it during interpolation
        }
        
        return true;
    }
    
    // updateIterCount() function (must be outside GPU block for CPU mode)
    function updateIterCount() {
        const iterText = document.getElementById('iter-count');
        if (iterText) {
            iterText.textContent = `${Math.floor(state.totalIterations).toLocaleString()} iterations`;
        }
        updateStatusDisplay();
    }
    
    // clearDensity() function (must be outside GPU block for CPU mode)
    function clearDensity() {
        state.seed = Math.random() * 100.0;
        state.totalIterations = 0;
        frameCounter = 0;
        frameOffsetCounter = 0;
        frameId = 0;  // Reset frame ID on clear
        cumulativeThreadCount = 0;  // Reset on clear
        updateIterCount();
        
        if (!webgpuAvailable) {
            // Clear CPU buffers
            if (cpuDensityBuffer) cpuDensityBuffer.fill(0);
            if (cpuColorBufferR) cpuColorBufferR.fill(0);
            if (cpuColorBufferG) cpuColorBufferG.fill(0);
            if (cpuColorBufferB) cpuColorBufferB.fill(0);
            // Reset max density tracker (CPU mode optimization)
            cpuMaxDensity = 0;
            cpuLastPresentMs = 0;
            cpuRngIndex = 0;
            return;
        }

        // Clear buffers using native clearBuffer() (fast and safe for any size)
        // FIX: Native buffer clearing replaces custom compute shader clearing pipeline.
        // Dispatching a clear shader requires calculating workgroup counts. On 4K/8K screens,
        // the buffer size requires more workgroups than the hardware limit (65,535), causing
        // the WebGPU device to crash/reset. Native clearBuffer is faster, simpler, and has no size limits.
        // Clear buffers.
        if (device && densityBuffer) {
            const clearEncoder = device.createCommandEncoder();
            clearEncoder.clearBuffer(densityBuffer);
            if (colorBufferR) clearEncoder.clearBuffer(colorBufferR);
            if (colorBufferG) clearEncoder.clearBuffer(colorBufferG);
            if (colorBufferB) clearEncoder.clearBuffer(colorBufferB);
            device.queue.submit([clearEncoder.finish()]);
        }
    }
    
    // updateStatusDisplay() function (must be outside GPU block for CPU mode)
    function updateStatusDisplay() {
        const statusRunningEl = document.getElementById('status-running');
        const statusRngEl = document.getElementById('status-rng');
        
        // Update running/paused status
        // PERF: Show simple performance metric
        const workload = Math.round((workgroupCount || 0) / 1000);
        const maxRange = Math.max(
            state.fractalMaxX - state.fractalMinX,
            state.fractalMaxY - state.fractalMinY
        );
        const zoomFactor = maxRange > 0 ? (4.0 / maxRange) * state.viewScale : 1.0;
        const zoomText = `Zoom: ${zoomFactor.toFixed(2)}x`;
        const iterationCount = getAdaptiveIterationCount(maxRange);
        const iterText = `Iter: ${iterationCount}`;
        const runText = isRunning ? `Running (${workload}k grp)` : 'Paused';
        statusRunningEl.textContent = `${zoomText}  ${iterText}  ${runText}`;
        
        // Update RNG status
        const generatorNames = ['hash12 (PCG)', 'Sobol scrambled (QMC)', 'R2 (Legacy)', 'Sobol pure (QMC)', 'R2 Precision', 'Owen-Sobol (Optimized)', 'Combined LCG (Random)'];
        const generatorName = generatorNames[state.rngMode] || generatorNames[0];
        statusRngEl.textContent = `RNG: ${generatorName}`;
    }
    
    // Function to update toggle button appearance (must be outside GPU block for CPU mode)
    function updateToggleButton() {
        const btn = document.getElementById('btn-toggle');
        if (!btn) {
            updateStatusDisplay();
            return;
        }
        if (isRunning) {
            btn.textContent = 'RUNNING';
            btn.style.background = '#2a5';
        } else {
            btn.textContent = 'PAUSED';
            btn.style.background = '#a52';
        }
        updateStatusDisplay(); // Update permanent status display
    }
    
    // R2 sequence constants (must be outside GPU block for CPU mode)
    // FIX: Use 2D Golden Ratio constants instead of 3D Plastic Number constants.
    // The Plastic Number (0.7548..., 0.5698...) is for 3D sequences and causes artifacts in 2D.
    const R2_ALPHA = 0.6180339887498949;  // 1/φ (Golden Ratio)
    const R2_BETA = 0.3819660112501052;    // 1/φ²
    
    // R2 Fixed-Point Constants (must match GPU constants for perfect alignment)
    // These are floor(2^32 * alpha) where alpha = 1/φ and 1/φ²
    const R2_INC_X = 0x9E3779B9;  // 2654435769 ≈ 2^32 / φ
    const R2_INC_Y = 0x61C88647;  // 1640531527 ≈ 2^32 / φ²
    
        // GPU performance tracking variables (must be outside GPU block for updateParams)
        let cumulativeThreadCount = 0;  // Total threads dispatched across all frames (for R2 precision mode)
        let frameId = 0; // Frame identifier, incremented every frame, mixed into RNG seeds
    // RNG sequence offset configuration (legacy, kept for compatibility)
    const FRAME_OFFSET_INTERVAL = 30; // Advance RNG sequence offset every N frames (legacy, kept for frameCounter)
    let frameCounter = 0; // Frame counter for RNG sequence offset (legacy)
    let frameOffsetCounter = 0; // Counter to track when to advance offset (legacy)
    
    // -----------------------------------------------------------------------------
    // JAVASCRIPT RNG IMPLEMENTATIONS (CPU fallback/verification)
    // -----------------------------------------------------------------------------
    // Needed for CPU fallback AND the optional CPU-math → GPU-render fast path.
    function hash32(n) {
        let x = n >>> 0;
        x ^= x >>> 16;
        x = Math.imul(x, 0x85ebca6b) >>> 0;
        x ^= x >>> 13;
        x = Math.imul(x, 0xc2b2ae35) >>> 0;
        x ^= x >>> 16;
        return x >>> 0;
    }
    
    function hash12(p) {
        const p3 = [
            (p[0] % 1 + 1) % 1 * 0.1031,
            (p[1] % 1 + 1) % 1 * 0.1031,
            (p[0] % 1 + 1) % 1 * 0.1031
        ];
        const dot1 = p3[0] * (p3[1] + 33.33) + p3[1] * (p3[2] + 33.33) + p3[2] * (p3[0] + 33.33);
        const p3New = [p3[0] + dot1, p3[1] + dot1, p3[2] + dot1];
        const dot2 = p3New[0] * (p3New[1] + 33.33) + p3New[1] * (p3New[2] + 33.33) + p3New[2] * (p3New[0] + 33.33);
        return ((p3New[0] + p3New[1]) * p3New[2] + dot2) % 1;
    }
    
    function reverseBits(x) {
        x = x >>> 0;
        x = ((x & 0x55555555) << 1) | ((x & 0xAAAAAAAA) >>> 1);
        x = ((x & 0x33333333) << 2) | ((x & 0xCCCCCCCC) >>> 2);
        x = ((x & 0x0F0F0F0F) << 4) | ((x & 0xF0F0F0F0) >>> 4);
        x = ((x & 0x00FF00FF) << 8) | ((x & 0xFF00FF00) >>> 8);
        x = ((x & 0x0000FFFF) << 16) | ((x & 0xFFFF0000) >>> 16);
        return x >>> 0;
    }
    
    function lkPermute(x, seed) {
        let v = (x + seed) >>> 0;
        v ^= Math.imul(v, 0x6c50b47c) >>> 0;
        v ^= Math.imul(v, 0xb82f1e52) >>> 0;
        v ^= Math.imul(v, 0xc7afe638) >>> 0;
        v ^= Math.imul(v, 0x8d22f6e6) >>> 0;
        return v >>> 0;
    }
    
    function owenScramble(x, seed) {
        return reverseBits(lkPermute(reverseBits(x), seed));
    }
    
    function toUnit(u) {
        return (u >>> 8) / 16777216.0;
    }
    
    function sobol02Bits(index) {
        const x = reverseBits(index);
        const y = (x ^ (x >>> 1)) >>> 0;
        return [x, y];
    }
    
    function sobol2dPure(index, seed) {
        const offset = ((seed & 0xFFFF0000) | ((seed & 0xFFFF) * 0x9e3779b9)) >>> 0;
        const offsetIndex = (index + offset) >>> 0;
        const bits = sobol02Bits(offsetIndex);
        return [toUnit(bits[0]), toUnit(bits[1])];
    }
    
    function sobol2dScrambled(index, seed) {
        const scrambledIndex = owenScramble(index, seed);
        const bits = sobol02Bits(scrambledIndex);
        const x = owenScramble(bits[0], seed ^ 0xa511e9b3);
        const y = owenScramble(bits[1], seed ^ 0x63d83595);
        return [toUnit(x), toUnit(y)];
    }
    
    function r2Sequence(index) {
        const n = index;
        return [(n * R2_ALPHA) % 1, (n * R2_BETA) % 1];
    }
    
    function sobol2dOwenOptimized(index, seed) {
        // Free speed: eliminate redundant reverseBits calls
        // sobol02Bits(owenScramble(index, seed))[0] == lkPermute(reverseBits(index), seed)
        // This removes two full reverse-bit passes
        const v = lkPermute(reverseBits(index), seed);
        
        function owenPermuteFast(x, seed) {
            let v = (x + seed) >>> 0;
            v = (v ^ Math.imul(v, 0x6c50b47c)) >>> 0;
            v = (v ^ Math.imul(v, 0xb82f1e52)) >>> 0;
            v = (v ^ Math.imul(v, 0xc7afe638)) >>> 0;
            v = (v ^ Math.imul(v, 0x8d22f6e6)) >>> 0;
            return v >>> 0;
        }
        
        const x = owenPermuteFast(v,                seed ^ 0xa511e9b3);
        const y = owenPermuteFast((v ^ (v >>> 1)) >>> 0, seed ^ 0x63d83595);
        return [toUnit(x), toUnit(y)];
    }
    
    // Combined LCG (Linear Congruential Generator) - textbook purely random
    // Combines 2 classic LCGs for better quality without uniform-ization
    function combinedLCG_JS(index, seed) {
        const seedBits = new Uint32Array(new Float32Array([seed]).buffer)[0];
        
        // Initialize state from index and seed
        let state1 = (index ^ seedBits) >>> 0;
        let state2 = ((index + 0x9e3779b9) ^ seedBits) >>> 0;
        
        // LCG1: Numerical Recipes (a=1664525, c=1013904223, m=2^32)
        // Full period 2^32, well-tested in numerical computing
        state1 = (1664525 * state1 + 1013904223) >>> 0;
        
        // LCG2: Borland (a=134775813, c=1, m=2^32)
        // Full period 2^32, classic implementation
        state2 = (134775813 * state2 + 1) >>> 0;
        
        // Combine outputs via XOR for decorrelation
        // Using different bit shifts for X and Y ensures independence
        const combinedX = (state1 ^ state2) >>> 0;
        const combinedY = ((state1 << 16) ^ (state2 >>> 16)) >>> 0;
        
        // Normalize to [0,1) - raw output, no uniform-ization
        return [
            combinedX / 4294967296.0,
            combinedY / 4294967296.0
        ];
    }
    
    function getRandom2D_JS(index, seed, rngMode) {
        switch (rngMode) {
            case 6: { // Combined LCG: Textbook purely random (2 classic LCGs combined)
                return combinedLCG_JS(index, seed);
            }
            case 5: { // Owen-Sobol Optimized
                const seedHash = hash32(new Uint32Array(new Float32Array([seed]).buffer)[0]);
                return sobol2dOwenOptimized(index, seedHash);
            }
            case 4: { // R2 Precision
                // In CPU mode we treat `index` as a monotonic sample index.
                let r2StartX = (index * R2_ALPHA) % 1.0;
                let r2StartY = (index * R2_BETA) % 1.0;
                if (r2StartX < 0) r2StartX += 1.0;
                if (r2StartY < 0) r2StartY += 1.0;
                return [r2StartX, r2StartY];
            }
            case 3: { // Sobol pure
                const seedHash = hash32(new Uint32Array(new Float32Array([seed]).buffer)[0]);
                return sobol2dPure(index, seedHash);
            }
            case 2: // R2 Legacy
                return r2Sequence(index);
            case 1: { // Sobol scrambled
                const seedHash = hash32(new Uint32Array(new Float32Array([seed]).buffer)[0]);
                return sobol2dScrambled(index, seedHash);
            }
            default: { // hash12 (PCG-ish)
                return [hash12([index, seed]), hash12([index + 13, seed])];
            }
        }
    }
    
    // getRenderDimensions() function (must be outside GPU block for CPU mode)
    function getRenderDimensions() {
        if (!webgpuAvailable) {
            // CPU mode: use canvas dimensions
            const canvas = document.getElementById('canvas');
            if (canvas) {
                return { width: canvas.width, height: canvas.height };
            }
            return { width: 800, height: 800 }; // Fallback
        }
        // WebGPU mode: internal resolution is the canvas backing store size.
        return { width: canvasWidth || 800, height: canvasHeight || 800 };
    }
    
    // getViewBaseScale() function: single source of truth for baseScale computation
    // Ensures consistency between pan/zoom/quantize/rebase/updateParams.
function getViewBaseScale() {
    const { width, height } = getRenderDimensions();
    return Math.min(width, height) * 0.2;
}

function getAdaptiveIterationCount(maxRange) {
    const zoomFactor = 4.0 / maxRange;
    if (zoomFactor <= 240) {
        return 128;
    }
    return Math.min(4096, Math.floor(128 * Math.pow(zoomFactor / 240, 0.2125)));
}
    
    // Reusable ArrayBuffer and DataView for updateParams() (must be outside GPU block for CPU mode)
    // These are used by updateParams() which is called from UI code in both CPU and GPU modes
    // Extended to 160 bytes (adaptive iteration + padding) for 16-byte alignment
    const paramsBufferSize = 40 * 4; // 160 bytes (padded to 16-byte alignment)
    const paramsArrayBuffer = new ArrayBuffer(paramsBufferSize);
    const paramsDataView = new DataView(paramsArrayBuffer);
    
    // updateParams() function (must be outside GPU block for CPU mode compatibility)
    // Updates parameter buffer and optionally writes to GPU buffer if WebGPU is available
    function updateParams() {
        // Reuse existing buffer instead of allocating new one every frame
        const renderDims = getRenderDimensions();
        
        let offset = 0;
        paramsDataView.setFloat32(offset, state.a, true); offset += 4;
        paramsDataView.setFloat32(offset, state.b, true); offset += 4;
        paramsDataView.setFloat32(offset, state.c, true); offset += 4;
        paramsDataView.setFloat32(offset, state.d, true); offset += 4;
        paramsDataView.setFloat32(offset, renderDims.width, true); offset += 4;
        paramsDataView.setFloat32(offset, renderDims.height, true); offset += 4;
        paramsDataView.setFloat32(offset, state.exposure * state.exposureMultiplier, true); offset += 4;
        paramsDataView.setFloat32(offset, state.gamma, true); offset += 4;
        paramsDataView.setFloat32(offset, state.contrast, true); offset += 4;
        paramsDataView.setFloat32(offset, state.seed, true); offset += 4;
        paramsDataView.setUint32(offset, state.colorMethod, true); offset += 4;
        // Precompute 1.0 / totalIterations for fragment shader normalization (avoids pow() per pixel)
        // SPARK 1: Upload invTotal directly instead of log2, eliminating fragment shader pow(2, -log2) computation
        const invTotal = state.totalIterations > 1 ? (1.0 / state.totalIterations) : 1.0;
        paramsDataView.setFloat32(offset, invTotal, true); offset += 4;
        paramsDataView.setFloat32(offset, state.viewOffsetX, true); offset += 4;
        paramsDataView.setFloat32(offset, state.viewOffsetY, true); offset += 4;
        paramsDataView.setFloat32(offset, state.viewScale, true); offset += 4;
        paramsDataView.setFloat32(offset, state.fractalMinX, true); offset += 4;
        paramsDataView.setFloat32(offset, state.fractalMaxX, true); offset += 4;
        paramsDataView.setFloat32(offset, state.fractalMinY, true); offset += 4;
        paramsDataView.setFloat32(offset, state.fractalMaxY, true); offset += 4;
        paramsDataView.setFloat32(offset, state.originalFractalMinX, true); offset += 4;
        paramsDataView.setFloat32(offset, state.originalFractalMaxX, true); offset += 4;
        paramsDataView.setFloat32(offset, state.originalFractalMinY, true); offset += 4;
        paramsDataView.setFloat32(offset, state.originalFractalMaxY, true); offset += 4;
        paramsDataView.setUint32(offset, state.rngMode, true); offset += 4;
        // Calculate 2D dispatch dimensions for >65535 workgroups
        const dispatchDimX = Math.min(workgroupCount || 2000, 65535);
        paramsDataView.setUint32(offset, dispatchDimX, true); offset += 4;
        paramsDataView.setUint32(offset, workgroupCount || 2000, true); offset += 4;
        // Frame offset ensures each frame uses different RNG sequence positions
        // Use cumulative thread count for all modes to handle adaptive workgroupCount
        // (workgroupCount changes based on performance, so we can't assume it's constant)
        const frameOffset = cumulativeThreadCount;
        const bigOffset = BigInt(frameOffset);
        const frameOffsetLo = Number(bigOffset & BigInt(0xFFFFFFFF));
        const frameOffsetHi = Number((bigOffset >> BigInt(32)) & BigInt(0xFFFFFFFF));
        paramsDataView.setUint32(offset, frameOffsetLo, true); offset += 4;
        paramsDataView.setUint32(offset, frameOffsetHi, true); offset += 4;
        
        // Frame ID: incremented every frame, mixed into RNG seeds to ensure unique samples per frame
        paramsDataView.setUint32(offset, frameId, true); offset += 4;
        
        // R2 precision: compute starting point using fixed-point arithmetic for perfect GPU alignment
        // frameOffset is total iterations dispatched so far (before this frame)
        // Use integer math to match GPU's fixed-point computation exactly
        const bigIncX = BigInt(R2_INC_X);
        const bigIncY = BigInt(R2_INC_Y);
        const r2BitsX = Number(bigOffset * bigIncX & BigInt(0xFFFFFFFF));
        const r2BitsY = Number(bigOffset * bigIncY & BigInt(0xFFFFFFFF));
        // Convert to [0,1) using same toUnit() logic as GPU: (bits >> 8) / 16777216.0
        let r2StartX = (r2BitsX >>> 8) / 16777216.0;
        let r2StartY = (r2BitsY >>> 8) / 16777216.0;
        paramsDataView.setFloat32(offset, r2StartX, true); offset += 4;
        paramsDataView.setFloat32(offset, r2StartY, true); offset += 4;

        // Floating-origin anchor: CPU-precomputed accumulation-buffer center (f64 precision -> f32)
        // This is the accumulation-buffer coordinate that maps to the *screen center*.
        // Computed as: width/2 - viewOffsetX*baseScale (avoids precision loss from subtracting
        // two large numbers in the fragment shader). Fragment shader uses: center + relativeOffset.
        // Stored as dedicated fields in the params buffer (see padding below).
        const baseScale = getViewBaseScale();
        const viewBufferCenterX = renderDims.width * 0.5 - state.viewOffsetX * baseScale;
        const viewBufferCenterY = renderDims.height * 0.5 - state.viewOffsetY * baseScale;
        paramsDataView.setFloat32(offset, viewBufferCenterX, true); offset += 4;
        paramsDataView.setFloat32(offset, viewBufferCenterY, true); offset += 4;

        // PRECISION FIX: View center in attractor-native coordinates
        // Computed here with JavaScript's f64 precision, then passed as f32.
        // Since the attractor lives in [-2, 2], these values are small and lose no precision.
        // The shader uses these for (x - attractorCenterX) which avoids catastrophic cancellation.
        const fractalRangeX = state.fractalMaxX - state.fractalMinX;
        const fractalRangeY = state.fractalMaxY - state.fractalMinY;
        const attractorCenterX = (state.fractalMinX + state.fractalMaxX) * 0.5;  // f64 computation
        const attractorCenterY = (state.fractalMinY + state.fractalMaxY) * 0.5;  // f64 computation
        paramsDataView.setFloat32(offset, attractorCenterX, true); offset += 4;
        paramsDataView.setFloat32(offset, attractorCenterY, true); offset += 4;

        // Precomputed pixel scale: how many pixels per attractor unit
        // Formula matches old shader computation: min(w,h) * 0.2 * 4.0 * 0.95 / max(rangeX, rangeY)
        const maxRange = Math.max(fractalRangeX, fractalRangeY);
        const attractorPixelScale = (baseScale * 4.0 * 0.95) / maxRange;
        paramsDataView.setFloat32(offset, attractorPixelScale, true); offset += 4;
        
        // Adaptive iteration fields
        const iterationCount = getAdaptiveIterationCount(maxRange);
        paramsDataView.setUint32(offset, iterationCount, true); offset += 4;
        paramsDataView.setFloat32(offset, 0.0, true); offset += 4;
        paramsDataView.setFloat32(offset, 0.0, true); offset += 4;
        paramsDataView.setFloat32(offset, 0.0, true); offset += 4;  // Padding for 16-byte alignment

        // Write to inactive buffer for next frame (only if WebGPU is available)
        if (webgpuAvailable && device && paramsBufferInactive) {
            device.queue.writeBuffer(paramsBufferInactive, 0, paramsArrayBuffer);
        }
    }
    
    // Slider mapping functions (must be outside GPU block for CPU mode compatibility)
    // Power function: maps 0-100 slider to actual value with exponent > 1
    // This gives more precision in the first 1/3 of the slider range
    function mapSliderToExposure(sliderValue) {
        // Map 0-100 to 0.001-40.0 using power function (x^2.5)
        // This compresses high end, expands low end for precision
        const normalized = sliderValue / 100.0;
        const power = Math.pow(normalized, 2.5);
        return 0.001 + power * (40.0 - 0.001);
    }

    function mapExposureToSlider(actualValue) {
        const clamped = Math.max(0.001, Math.min(40.0, actualValue));
        const normalized = (clamped - 0.001) / (40.0 - 0.001);
        const sliderPos = Math.pow(normalized, 1.0 / 2.5);
        return Math.round(sliderPos * 100);
    }
    
    function mapSliderToGamma(sliderValue) {
        // Map 0-100 to 0.025-1.0 using power function (x^2.2)
        const normalized = sliderValue / 100.0;
        const power = Math.pow(normalized, 2.2);
        return 0.025 + power * (1.0 - 0.025);
    }
    
    function mapGammaToSlider(actualValue) {
        const clamped = Math.max(0.025, Math.min(1.0, actualValue));
        const normalized = (clamped - 0.025) / (1.0 - 0.025);
        const sliderPos = Math.pow(normalized, 1.0 / 2.2);
        return Math.round(sliderPos * 100);
    }
    
    function mapSliderToContrast(sliderValue) {
        // Map 0-100 to 0.0-3.0 (1.0 = neutral)
        // Use power function for better precision at lower values
        const normalized = sliderValue / 100.0;
        const power = Math.pow(normalized, 1.5);
        return 0.0 + power * 3.0;
    }

    function mapContrastToSlider(actualValue) {
        // Reverse mapping: actual value to slider position
        // Clamp to valid range
        const clamped = Math.max(0.0, Math.min(3.0, actualValue));
        const normalized = clamped / 3.0;
        const sliderPos = Math.pow(normalized, 1.0 / 1.5);
        return Math.round(sliderPos * 100);
    }
    
    // Track SHIFT key state for fine-grained parameter control
    // Must be outside GPU block so CPU mode can access it
    let isShiftPressed = false;
    
    // Update step attribute for parameter sliders based on SHIFT state
    // This function is used by both CPU and GPU modes, so it's outside the GPU block
        function updateParameterSliderSteps() {
            const paramSliders = ['param-a', 'param-b', 'param-c', 'param-d'];
            const paramKeys = ['a', 'b', 'c', 'd'];
            const normalStep = 0.001;
            const fineStep = normalStep / 16; // 16x finer control
        
        paramSliders.forEach((id, index) => {
            const slider = document.getElementById(id);
            if (slider) {
                slider.step = isShiftPressed ? fineStep.toString() : normalStep.toString();
                // Update display to show appropriate precision
                const display = document.getElementById(`val-${paramKeys[index]}`);
                if (display) {
                    const value = parseFloat(slider.value);
                    const decimals = isShiftPressed ? 6 : 3;
                    display.innerText = value.toFixed(decimals);
                }
            }
            });
        }

        // Keyboard shortcuts (work in both GPU and CPU fallback modes).
        window.addEventListener('keydown', (e) => {
            if (e.key === 'F1') {
                e.preventDefault();
                const isOpen = helpOverlay && helpOverlay.style.display !== 'none' && helpOverlay.getAttribute('aria-hidden') !== 'true';
                setHelpVisible(!isOpen);
                return;
            }
            if (e.key === 'Escape') {
                if (helpOverlay && helpOverlay.style.display !== 'none') {
                    e.preventDefault();
                    setHelpVisible(false);
                    return;
                }
            }
            if (e.code === 'Space') {
                if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA' || e.target.tagName === 'SELECT' || e.target.isContentEditable) {
                    return;
                }
                e.preventDefault();
                e.stopPropagation();
                if (attractModeActive) {
                    stopAttractMode();
                } else {
                    // Resume with same speed mode as before (attractModeSpeed persists after stop)
                    startAttractMode(attractModeSpeed);
                }
            }
            if (e.key === 'Shift' && !e.repeat) {
                isShiftPressed = true;
                updateParameterSliderSteps();
            }
            // G: Cycle random generator (plain R reserved for attract reversal)
            if ((e.key === 'g' || e.key === 'G') && !e.shiftKey && !isShiftPressed) {
                if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA' || e.target.tagName === 'SELECT') {
                    return;
                }
                e.preventDefault();
                state.rngMode = (state.rngMode + 1) % 7;
                needsClear = true; // Clear accumulation buffer when switching RNG
                needsTimingReset = true;
                updateParams();
                updateStatusDisplay(); // Update permanent RNG display
                
                // Display RNG change confirmation
                const statusTempEl = document.getElementById('status-temp');
                const generatorNames = ['hash12 (PCG)', 'Sobol scrambled (QMC)', 'R2 (Legacy)', 'Sobol pure (QMC)', 'R2 Precision', 'Owen-Sobol (Optimized)', 'Combined LCG (Random)'];
                const generatorName = generatorNames[state.rngMode] || generatorNames[0];
                statusTempEl.textContent = `Switched to ${generatorName}`;
                statusTempEl.style.color = '#4a9eff';
                setTimeout(() => {
                    statusTempEl.textContent = '';
                }, 2000);
            }
            // R (without Shift): Reverse attract direction (no RNG cycling here)
            if ((e.key === 'r' || e.key === 'R') && !e.shiftKey && !isShiftPressed && !e.ctrlKey && !e.metaKey && !e.altKey) {
                e.preventDefault();
                if (attractModeActive) {
                    reverseAttractModeDirection();
                }
            }
            // Shift+R: Reset view transform to 1:1 without rebasing (preserves accumulated pixels)
            if ((e.key === 'r' || e.key === 'R') && (e.shiftKey || isShiftPressed)) {
                e.preventDefault();
                resetViewTransform();
            }
            // B: Rebase (take current view and make it the new full viewport)
            if ((e.key === 'b' || e.key === 'B') && !e.shiftKey && !isShiftPressed) {
                // Only trigger if not typing in an input field
                if (e.target.tagName !== 'INPUT' && e.target.tagName !== 'TEXTAREA' && e.target.tagName !== 'SELECT') {
                    e.preventDefault();
                    performRebase(false);
                }
            }
            // Shift+B: Reset view transform to 1:1 without rebasing (same as Shift+Rebase button)
            if ((e.key === 'b' || e.key === 'B') && (e.shiftKey || isShiftPressed)) {
                // Only trigger if not typing in an input field
                if (e.target.tagName !== 'INPUT' && e.target.tagName !== 'TEXTAREA' && e.target.tagName !== 'SELECT') {
                    e.preventDefault();
                    performRebase(true);
                }
            }
            // Ctrl+S: Reseed RNG (useful for checking seed-dependent graininess)
            // Note: Does NOT clear accumulation - allows continued pixel accumulation for more accuracy
            if ((e.key === 's' || e.key === 'S') && (e.ctrlKey || e.metaKey)) {
                e.preventDefault();
                state.seed = Math.random() * 100.0;
                // Don't clear - keep accumulating pixels for more accuracy
                updateParams();

                // Display seed change confirmation
                const statusTempEl = document.getElementById('status-temp');
                statusTempEl.textContent = `Seed changed (accumulation continues)`;
                statusTempEl.style.color = '#888';
                setTimeout(() => {
                    statusTempEl.textContent = '';
                }, 1000);
            }

            // Alt+C: Start calibration benchmark with alternate reproducible seed
            if ((e.key === 'c' || e.key === 'C') && e.altKey && !e.ctrlKey && !e.metaKey) {
                if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA' || e.target.tagName === 'SELECT') {
                    return;
                }
                e.preventDefault();
                if (!calibrationActive && webgpuAvailable) {
                    const calParams = CALIBRATION_PARAMS;
                    const altSeed = deriveCalibrationSeed(calParams.seed, 0);
                    startCalibration({
                        seed: altSeed,
                        seedLabel: 'Alt seed',
                        paramSeed: altSeed,
                        paramSeedLabel: 'Alt param seed'
                    });
                }
            }

            // Shift+C: Start calibration benchmark (fixed-work, reproducible)
            if ((e.key === 'c' || e.key === 'C') && (e.shiftKey || isShiftPressed) && !e.ctrlKey && !e.metaKey) {
                e.preventDefault();
                if (!calibrationActive && webgpuAvailable) {
                    const calParams = CALIBRATION_PARAMS;
                    startCalibration({
                        seed: calParams.seed,
                        seedLabel: 'Base seed',
                        paramSeed: calParams.seed,
                        paramSeedLabel: 'Base param seed'
                    });
                }
            }
        });
        
        window.addEventListener('keyup', (e) => {
            if (e.key === 'Shift') {
                isShiftPressed = false;
                updateParameterSliderSteps();
                // Clear cycle state when Shift is released
                // This allows the next Shift+Random to generate a NEW set of 8 variations
                // instead of continuing the previous cycle
                clearCycleState();
            }
        });

        // Track previous slider values for delta calculation (for parameter sliders only)
        const previousSliderValues = new Map();

        function syncParamSlidersToState(decimals = 3) {
            for (const key of ['a', 'b', 'c', 'd']) {
                const slider = document.getElementById(`param-${key}`);
                const display = document.getElementById(`val-${key}`);
                if (slider) slider.value = state[key];
                if (display) display.innerText = state[key].toFixed(decimals);
                previousSliderValues.set(key, state[key]);
            }
        }

        function syncToneSlidersToState(decimals = 3) {
            const expSlider = document.getElementById('param-exposure');
            const gammaSlider = document.getElementById('param-gamma');
            const contrastSlider = document.getElementById('param-contrast');
            const expDisplay = document.getElementById('val-exp');
            const gammaDisplay = document.getElementById('val-gamma');
            const contrastDisplay = document.getElementById('val-contrast');

            if (expSlider) expSlider.value = mapExposureToSlider(state.exposure);
            if (gammaSlider) gammaSlider.value = mapGammaToSlider(state.gamma);
            if (contrastSlider) contrastSlider.value = mapContrastToSlider(state.contrast);

            if (expDisplay) expDisplay.innerText = state.exposure.toFixed(decimals);
            if (gammaDisplay) gammaDisplay.innerText = state.gamma.toFixed(decimals);
            if (contrastDisplay) contrastDisplay.innerText = state.contrast.toFixed(decimals);
        }
        
        // Throttle slider input events to 30Hz (33ms) to prevent GPU queue saturation
    const sliderThrottleTimeouts = new Map();
    
    // UI Wiring (runs for both CPU and GPU modes)
    const setupSlider = (id, key, displayId, mapper = null, reverseMapper = null) => {
        const el = document.getElementById(id);
        const display = document.getElementById(displayId || `val-${key}`);
        if (!el || !display) {
            console.error(`Missing element: ${id} or ${displayId || `val-${key}`}`);
            return;
        }
        
        // Set initial slider value (use mapper if provided)
        let initialValue;
        if (mapper && reverseMapper) {
            initialValue = reverseMapper(state[key]);
        } else {
            initialValue = state[key];
        }
        el.value = initialValue;
        display.innerText = state[key].toFixed(3);
        
        // Store initial value for parameter sliders (for delta calculation)
        if (['a','b','c','d'].includes(key)) {
            previousSliderValues.set(key, parseFloat(initialValue));
        }
        
        el.addEventListener('input', (e) => {
            // Stop attract mode only for parameter sliders (a,b,c,d), not for exposure/gamma/contrast
            // NOTE: Exposure/gamma/contrast sliders work during attract mode
            if (attractModeActive && ['a','b','c','d'].includes(key)) {
                stopAttractMode();
            }
            
            markInteraction(); // Trigger full-speed rendering during slider interaction
            
            let newValue = parseFloat(e.target.value);
            
            // For parameter sliders with Shift pressed, apply 10x slower movement
            if (['a','b','c','d'].includes(key) && isShiftPressed) {
                const prevValue = previousSliderValues.get(key) || parseFloat(initialValue);
                const delta = newValue - prevValue;
                // Apply only 1/10th of the delta
                newValue = prevValue + delta * 0.1;
                // Update the slider element to reflect the actual value (causes slight snap-back)
                el.value = newValue.toString();
            }
            
            // Clear cycle state when user manually drags parameters
            if (['a','b','c','d'].includes(key)) {
                clearCycleState();
            }
            
            // Update previous value for next delta calculation
            if (['a','b','c','d'].includes(key)) {
                previousSliderValues.set(key, newValue);
            }
            
            // Map slider value to actual value if mapper provided
            if (mapper) {
                state[key] = mapper(newValue);
            } else {
                state[key] = newValue;
            }
            
            // Stop lerping when user manually adjusts sliders
            // Also update target to match current value so lerp doesn't jump
            if (['a','b','c','d'].includes(key)) {
                state.isLerping = false;
                if (key === 'a') state.targetA = state.a;
                if (key === 'b') state.targetB = state.b;
                if (key === 'c') state.targetC = state.c;
                if (key === 'd') state.targetD = state.d;
            }
            
            // Show more decimal places when SHIFT is held (fine mode)
            const decimals = (isShiftPressed && ['a','b','c','d'].includes(key)) ? 6 : 3;
            display.innerText = state[key].toFixed(decimals);
            
            // For parameter changes (a,b,c,d), throttle clear flag to 30Hz
            // FIX: Change from debounce (clearing timeout) to throttle (blocking duplicates).
            // The previous debounce logic reset the timer on every mouse movement, causing the
            // simulation to freeze completely until the user stops dragging the slider.
            // Throttling ensures updates occur at a steady 30fps during the drag, providing immediate visual feedback.
            // For other parameters (exposure/gamma/contrast), also throttle to 30Hz
            if (['a','b','c','d'].includes(key)) {
                // FIX: Check !has(key) to THROTTLE instead of clearing to DEBOUNCE
                if (!sliderThrottleTimeouts.has(key)) {
                    sliderThrottleTimeouts.set(key, setTimeout(() => {
                        sliderThrottleTimeouts.delete(key);
                        needsClear = true;
                        needsTimingReset = true;
                        // Don't call updateParams() here - let frame loop handle it after clear
                        // This ensures clear happens before new accumulation starts
                    }, 33)); // Max 30fps
                }
            } else {
                // Throttle lightweight updates (exposure/gamma/contrast) to 30Hz
                if (!sliderThrottleTimeouts.has(key)) {
                    sliderThrottleTimeouts.set(key, setTimeout(() => {
                        sliderThrottleTimeouts.delete(key);
                        updateParams();
                    }, 33)); // Max 30fps
                }
            }
        });
    };

    // Initialize parameter slider steps
    updateParameterSliderSteps();

    setupSlider('param-a', 'a');
    setupSlider('param-b', 'b');
    setupSlider('param-c', 'c');
    setupSlider('param-d', 'd');
    setupSlider('param-exposure', 'exposure', 'val-exp', mapSliderToExposure, mapExposureToSlider);
    setupSlider('param-gamma', 'gamma', null, mapSliderToGamma, mapGammaToSlider);
    setupSlider('param-contrast', 'contrast', 'val-contrast', mapSliderToContrast, mapContrastToSlider);

    // Presets
    const setPreset = (a, b, c, d) => {
        // Stop lerping when preset is applied
        state.isLerping = false;
        state.a = a; state.b = b; state.c = c; state.d = d;
        // Update targets to match preset values
        state.targetA = a; state.targetB = b; state.targetC = c; state.targetD = d;
        document.getElementById('param-a').value = a; document.getElementById('val-a').innerText = a.toFixed(3);
        document.getElementById('param-b').value = b; document.getElementById('val-b').innerText = b.toFixed(3);
        document.getElementById('param-c').value = c; document.getElementById('val-c').innerText = c.toFixed(3);
        document.getElementById('param-d').value = d; document.getElementById('val-d').innerText = d.toFixed(3);
        
        // Clear cycle state when preset is applied (major change)
        clearCycleState();
        
        // Update previous slider values for delta calculation
        previousSliderValues.set('a', a);
        previousSliderValues.set('b', b);
        previousSliderValues.set('c', c);
        previousSliderValues.set('d', d);
        
        // Update exposure and gamma sliders using reverse mapping
        document.getElementById('param-exposure').value = mapExposureToSlider(state.exposure);
        document.getElementById('val-exp').innerText = state.exposure.toFixed(3);
        document.getElementById('param-gamma').value = mapGammaToSlider(state.gamma);
        document.getElementById('val-gamma').innerText = state.gamma.toFixed(3);
        
        needsClear = true;
        needsTimingReset = true;
        updateParams();
    };

    // Shift+Rebase helper: reset view transform to 1:1 without rebasing fractal bounds.
    // Must live OUTSIDE the WebGPU init block so CPU-only mode can use it too.
    function resetViewTransform() {
        // Early return if no transform is applied (nothing to reset)
        if (state.viewScale === 1.0 && state.viewOffsetX === 0.0 && state.viewOffsetY === 0.0) {
            return; // No change needed
        }
        
        // Only reset view transform - do NOT change fractal bounds or clear buffers
        state.viewOffsetX = 0;
        state.viewOffsetY = 0;
        state.viewScale = 1.0;
        // Do NOT set needsClear = true - keep accumulated pixels!
        // Do NOT change fractal bounds - keep them as-is!
        // Do NOT update exposureMultiplier - it's tied to fractal bounds
        updateParams();
        
        // Show status
        const statusTempEl = document.getElementById('status-temp');
        if (statusTempEl) {
            statusTempEl.textContent = 'View reset to 1:1 (pixels preserved)';
            statusTempEl.style.color = '#5a5';
            setTimeout(() => {
                statusTempEl.textContent = '';
                statusTempEl.style.color = '';
            }, 3000);
        }
    }
    
    // Rebase: Take current view and make it the new full viewport
    // Eliminates the "jump" by solving for bounds that preserve exact pixel-to-fractal mapping
    function performRebase(shiftKey = false) {
        // NOTE: Rebase works during attract mode - does NOT stop it
        markInteraction(); // Trigger full-speed rendering during rapid button clicks
        // Check for shift key - if pressed, do shift+rebase (reset view only)
        if (shiftKey) {
            resetViewTransform();
            return;
        }
        
        // Early return if no transform is applied (nothing to rebase)
        if (state.viewScale === 1.0 && state.viewOffsetX === 0.0 && state.viewOffsetY === 0.0) {
            return; // No change needed
        }
        
        const baseScale = getViewBaseScale();
        const currentMaxRange = Math.max(state.fractalMaxX - state.fractalMinX, state.fractalMaxY - state.fractalMinY);
        const scale = baseScale * (4.0 / currentMaxRange) * 0.95; // Scale used in compute shader
        
        const fractalCenterX = (state.fractalMinX + state.fractalMaxX) * 0.5;
        const fractalCenterY = (state.fractalMinY + state.fractalMaxY) * 0.5;

        // Calculate exact fractal coordinates at screen corners to determine visible region
        // Fragment shader: bufferX = (screenX - width/2) / viewScale - viewOffsetX * baseScale + width/2
        // Compute shader: bufferX = fx * scale + (width/2 - fractalCenterX * scale)
        // So: fx = (bufferX - width/2 + fractalCenterX * scale) / scale
        
        // Helper function to convert screen coordinate to fractal coordinate
        const screenToFractal = (screenX, screenY) => {
            const bufferX = (screenX - canvasWidth * 0.5) / state.viewScale - state.viewOffsetX * baseScale + canvasWidth * 0.5;
            const bufferY = (screenY - canvasHeight * 0.5) / state.viewScale - state.viewOffsetY * baseScale + canvasHeight * 0.5;
            const fx = (bufferX - canvasWidth * 0.5 + fractalCenterX * scale) / scale;
            const fy = (bufferY - canvasHeight * 0.5 + fractalCenterY * scale) / scale;
            return { fx, fy };
        };
        
        // Get fractal coordinates at screen corners
        const topLeft = screenToFractal(0, 0);
        const topRight = screenToFractal(canvasWidth, 0);
        const bottomLeft = screenToFractal(0, canvasHeight);
        const bottomRight = screenToFractal(canvasWidth, canvasHeight);
        
        // Calculate exact visible fractal bounds from corners
        const visibleMinX = Math.min(topLeft.fx, topRight.fx, bottomLeft.fx, bottomRight.fx);
        const visibleMaxX = Math.max(topLeft.fx, topRight.fx, bottomLeft.fx, bottomRight.fx);
        const visibleMinY = Math.min(topLeft.fy, topRight.fy, bottomLeft.fy, bottomRight.fy);
        const visibleMaxY = Math.max(topLeft.fy, topRight.fy, bottomLeft.fy, bottomRight.fy);
        
        // Calculate visible center
        const visibleCenterX = (visibleMinX + visibleMaxX) * 0.5;
        const visibleCenterY = (visibleMinY + visibleMaxY) * 0.5;
        
        // Calculate visible ranges
        const visibleRangeX = visibleMaxX - visibleMinX;
        const visibleRangeY = visibleMaxY - visibleMinY;
        
        // Key insight: After rebase, the compute shader will use:
        //   newScale = baseScale * (4.0 / newMaxRange) * 0.95
        // We want this to equal the effective scale that was being used:
        //   effectiveScale = scale * viewScale
        // So: baseScale * (4.0 / newMaxRange) * 0.95 = scale * viewScale
        // Solving: newMaxRange = (baseScale * 4.0 * 0.95) / (scale * viewScale)
        const effectiveScale = scale * state.viewScale;
        const newMaxRange = (baseScale * 4.0 * 0.95) / effectiveScale;
        const newHalfRange = newMaxRange * 0.5;
        
        // Set new bounds centered on visible region, with range that preserves exact scale
        state.fractalMinX = visibleCenterX - newHalfRange;
        state.fractalMaxX = visibleCenterX + newHalfRange;
        state.fractalMinY = visibleCenterY - newHalfRange;
        state.fractalMaxY = visibleCenterY + newHalfRange;

        state.exposureMultiplier *= (state.viewScale * state.viewScale);
        state.viewOffsetX = 0; 
        state.viewOffsetY = 0; 
        state.viewScale = 1.0;
        needsClear = true;
        updateParams();
        
        // Show exposureMultiplier in status after rebase
        const statusTempEl = document.getElementById('status-temp');
        statusTempEl.textContent = `Rebased (exp ×${state.exposureMultiplier.toFixed(2)})`;
        statusTempEl.style.color = '#5a5';
        setTimeout(() => {
            statusTempEl.textContent = '';
            statusTempEl.style.color = '';
        }, 3000);
    }
    

    function updateRebaseButton() {
        const btn = document.getElementById('btn-rebase');
        if (!btn) return;
        if (autoRebaseEnabled) {
            btn.textContent = 'Rebase ON';
            btn.style.background = '#555';
        } else {
            btn.textContent = 'Rebase';
            btn.style.background = '#5a5';
        }
    }

    function scheduleAutoRebase() {
        if (!autoRebaseEnabled) return;
        if (autoRebaseTimeout) {
            clearTimeout(autoRebaseTimeout);
        }
        autoRebaseTimeout = setTimeout(() => {
            autoRebaseTimeout = null;
            performRebase(false);
        }, AUTO_REBASE_DELAY_MS);
    }

    // Wire up rebase button
    document.getElementById('btn-rebase').onclick = (e) => {
        if (e.ctrlKey || e.metaKey) {
            autoRebaseEnabled = !autoRebaseEnabled;
            updateRebaseButton();
            const statusTempEl = document.getElementById('status-temp');
            if (statusTempEl) {
                statusTempEl.textContent = autoRebaseEnabled ? 'Auto rebase enabled' : 'Auto rebase disabled';
                statusTempEl.style.color = '#5a5';
                setTimeout(() => {
                    statusTempEl.textContent = '';
                    statusTempEl.style.color = '';
                }, 1500);
            }
            return;
        }
        performRebase(e.shiftKey || isShiftPressed);
    };

    updateRebaseButton();
    
    // =============================================================================
    // ATTRACT MODE - Sinusoidal parameter variation
    // =============================================================================
    
    function startAttractMode(speedMode = 0) {
        if (attractModeActive) return;
        // Accept boolean for backward compatibility (true = slow mode = 1)
        attractModeSpeed = (speedMode === true) ? 1 : (speedMode || 0);

        // On fresh app start/load, randomize parameters for truly random initial combo
        if (attractModeFirstStart) {
            randomizeParameters({ instant: true });
            attractModeFirstStart = false; // Only randomize on first start
        }

        // Initialize velocity-based bouncing system
        // Each parameter starts at its current value with a random velocity
        // Velocities are out of phase (different directions) to ensure continuous variation
        const min = -4.0;
        const max = 4.0;
        // Speed modes: 0=normal (1.0x), 1=slow (0.2x), 2=ultra-slow (0.067x = 3x slower than slow)
        const speedMultipliers = [1.0, 0.2, 0.067];
        const speedMultiplier = speedMultipliers[attractModeSpeed] || 1.0;
        const baseSpeed = 0.5 * 0.6 * 0.25 * 0.8 * speedMultiplier; // Base speed with mode multiplier

        // Initialize each parameter with current value and random initial velocity
        // Use different velocity directions to keep them out of phase
        const params = ['a', 'b', 'c', 'd'];
        const initialVelocities = [
            baseSpeed * (0.5 + Math.random() * 0.5),      // a: positive direction
            -baseSpeed * (0.5 + Math.random() * 0.5),     // b: negative direction
            baseSpeed * (0.7 + Math.random() * 0.3),       // c: positive direction (different speed)
            -baseSpeed * (0.7 + Math.random() * 0.3)      // d: negative direction (different speed)
        ];

        params.forEach((param, index) => {
            const config = attractModeParams[param];
            config.value = Math.max(min, Math.min(max, state[param])); // Clamp initial value
            config.velocity = initialVelocities[index];
            config.baseSpeed = Math.abs(initialVelocities[index]);
            config.speedVariation = 0;
            config.lastSpeedChange = 0;
            config.nextSpeedChangeAt = 2000 + Math.random() * 3000;
        });

        attractModeActive = true;
        attractModeStartTime = performance.now();
        attractModeLastUpdateTime = attractModeStartTime;

        // Update button text to show speed mode
        const btn = document.getElementById('btn-attract');
        const modeLabels = ['ON', 'SLOW', 'ULTRA'];
        if (btn) btn.textContent = `Animate (${modeLabels[attractModeSpeed]})`;
        
        // Attract updates are driven by the main frame loop.
    }
    
    function reverseAttractModeDirection() {
        if (!attractModeActive) return;
        ['a', 'b', 'c', 'd'].forEach((param) => {
            const config = attractModeParams[param];
            config.velocity = -config.velocity;
        });
        const statusTempEl = document.getElementById('status-temp');
        if (statusTempEl) {
            statusTempEl.textContent = 'Attract direction reversed';
            statusTempEl.style.color = '#7a5a6a';
            setTimeout(() => {
                statusTempEl.textContent = '';
                statusTempEl.style.color = '';
            }, 1000);
        }
    }

    function stopAttractMode() {
        if (!attractModeActive) return;

        attractModeActive = false;
        // Update button text
        const btn = document.getElementById('btn-attract');
        if (btn) btn.textContent = 'Animate';
    }

    // =============================================================================
    // CALIBRATION MODE - Precise GPU benchmark (Shift+C / Alt+C)
    // =============================================================================

    function deriveCalibrationSeed(baseSeed, index) {
        const seedBits = new Uint32Array(new Float32Array([baseSeed]).buffer)[0];
        let x = (seedBits ^ Math.imul(index + 1, 0x9e3779b9)) >>> 0;
        x = hash32(x);
        return (x / 4294967295.0) * 1000.0;
    }

    function deriveCalibrationParams(seed) {
        const seedBits = new Uint32Array(new Float32Array([seed]).buffer)[0];
        let s = hash32(seedBits ^ 0x6d2b79f5);
        const fBase = 4.0;
        const fRange = 8.0;
        const nextValue = (mod, denom) => {
            s = (Math.imul(s, 1664525) + 1013904223) >>> 0;
            const r = s % mod;
            return -fBase + fRange * (r / denom);
        };
        return {
            a: nextValue(10239, 10238),
            b: nextValue(10241, 10240),
            c: nextValue(10242, 10241),
            d: nextValue(10148, 10147)
        };
    }

    function startCalibration(options = {}) {
        if (calibrationActive) return;

        // Stop attract mode if active
        if (attractModeActive) stopAttractMode();

        if (!calibrationCanvasState) {
            calibrationCanvasState = {
                width: canvasWidth,
                height: canvasHeight,
                styleWidth: canvas.style.width,
                styleHeight: canvas.style.height
            };
        }
        canvasWidth = CALIBRATION_RENDER_WIDTH;
        canvasHeight = CALIBRATION_RENDER_HEIGHT;
        canvas.width = canvasWidth;
        canvas.height = canvasHeight;
        // Fire-and-forget; callers don't await calibration start.
        recreateGpuBuffers().catch((e) => console.error('Failed to recreate GPU buffers for calibration start:', e));

        // Save current state to restore later
        calibrationSavedState = {
            a: state.a, b: state.b, c: state.c, d: state.d,
            fractalMinX: state.fractalMinX, fractalMaxX: state.fractalMaxX,
            fractalMinY: state.fractalMinY, fractalMaxY: state.fractalMaxY,
            viewOffsetX: state.viewOffsetX, viewOffsetY: state.viewOffsetY,
            viewScale: state.viewScale, seed: state.seed,
            exposure: state.exposure, gamma: state.gamma, contrast: state.contrast,
            totalIterations: state.totalIterations
        };

        // Use fixed calibration parameters (can be overridden by config.json)
        const calParams = CALIBRATION_PARAMS;
        const seed = options.seed ?? calParams.seed;
        const seedLabel = options.seedLabel ?? '';
        const paramSeed = options.paramSeed ?? null;
        const paramSeedLabel = options.paramSeedLabel ?? '';
        const seededParams = paramSeed !== null ? deriveCalibrationParams(paramSeed) : null;
        state.a = seededParams ? seededParams.a : calParams.a;
        state.b = seededParams ? seededParams.b : calParams.b;
        state.c = seededParams ? seededParams.c : calParams.c;
        state.d = seededParams ? seededParams.d : calParams.d;
        state.fractalMinX = -2.0;
        state.fractalMaxX = 2.0;
        state.fractalMinY = -2.0;
        state.fractalMaxY = 2.0;
        state.viewOffsetX = 0.0;
        state.viewOffsetY = 0.0;
        state.viewScale = 1.0;
        state.seed = seed;
        calibrationSeedUsed = seed;
        calibrationSeedLabel = seedLabel;
        calibrationParamSeedUsed = paramSeed;
        calibrationParamSeedLabel = paramSeedLabel;

        // Reset calibration data
        calibrationTimings = [];
        calibrationFrameCount = 0;
        calibrationStartTime = performance.now();
        calibrationActive = true;
        needsClear = true;

        // Show status
        const statusTempEl = document.getElementById('status-temp');
        if (statusTempEl) {
            const seedSuffix = seedLabel ? `, ${seedLabel}` : '';
            const paramSuffix = paramSeedLabel ? `, ${paramSeedLabel}` : '';
            statusTempEl.textContent = `Calibration running... (${CALIBRATION_FRAME_LIMIT} frames${seedSuffix}${paramSuffix})`;
            statusTempEl.style.color = '#f80';
        }

        // Update sliders to show calibration params
        ['a', 'b', 'c', 'd'].forEach(p => {
            const slider = document.getElementById(`param-${p}`);
            const display = document.getElementById(`val-${p}`);
            if (slider) slider.value = state[p];
            if (display) display.innerText = state[p].toFixed(3);
        });

        updateParams();
    }

    function finishCalibration() {
        if (!calibrationActive) return;
        calibrationActive = false;

        const totalTime = performance.now() - calibrationStartTime;
        const frameCount = calibrationTimings.length;

        // Calculate statistics
        const frameTimes = calibrationTimings.map(t => t.frameTime);
        const gpuTimes = calibrationTimings.filter(t => t.gpuTime !== null).map(t => t.gpuTime);

        const avgFrameTime = frameTimes.reduce((a, b) => a + b, 0) / frameTimes.length;
        const minFrameTime = Math.min(...frameTimes);
        const maxFrameTime = Math.max(...frameTimes);

        // Sort for percentiles
        const sortedFrameTimes = [...frameTimes].sort((a, b) => a - b);
        const p50 = sortedFrameTimes[Math.floor(sortedFrameTimes.length * 0.50)];
        const p95 = sortedFrameTimes[Math.floor(sortedFrameTimes.length * 0.95)];
        const p99 = sortedFrameTimes[Math.floor(sortedFrameTimes.length * 0.99)];

        // GPU timing stats (if available)
        let gpuStats = 'N/A (timestamps not supported)';
        if (gpuTimes.length > 0) {
            const avgGpu = gpuTimes.reduce((a, b) => a + b, 0) / gpuTimes.length;
            const minGpu = Math.min(...gpuTimes);
            const maxGpu = Math.max(...gpuTimes);
            gpuStats = `avg: ${avgGpu.toFixed(2)}ms, min: ${minGpu.toFixed(2)}ms, max: ${maxGpu.toFixed(2)}ms`;
        }

        // Calculate throughput
        const totalIterations = frameCount * CALIBRATION_WORKGROUPS * 64 * 128;
        const iterPerSec = totalIterations / (totalTime / 1000);
        const pointsPerSec = iterPerSec;

        // RNG mode name
        const rngModeNames = ['hash12', 'Sobol Scrambled', 'R2 Legacy', 'Sobol Pure', 'R2 Precision', 'Owen-Sobol', 'Combined LCG'];
        const rngName = rngModeNames[state.rngMode] || `Mode ${state.rngMode}`;

        // Build results HTML
        const seedLabel = calibrationSeedLabel ? ` (${calibrationSeedLabel})` : '';
        const seedDisplay = calibrationSeedUsed !== null ? `${calibrationSeedUsed.toFixed(3)}${seedLabel}` : 'N/A';
        const paramSeedLabel = calibrationParamSeedLabel ? ` (${calibrationParamSeedLabel})` : '';
        const paramSeedDisplay = calibrationParamSeedUsed !== null ? `${calibrationParamSeedUsed.toFixed(3)}${paramSeedLabel}` : 'N/A';
        const resultsHtml = `
            <div style="font-family: monospace; font-size: 13px; line-height: 1.6; color: #eee; background: #222; padding: 0; border-radius: 8px; max-width: 500px;">
                <div class="calibration-popup-header" style="padding: 10px 20px; background: #1a1a1a; border-radius: 8px 8px 0 0; cursor: move; user-select: none; display: flex; align-items: center; justify-content: space-between; gap: 10px;">
                    <h3 style="margin: 0; color: #4af;">GPU Calibration Results</h3>
                    <button class="calibration-popup-close" style="background: transparent; color: #bbb; border: none; font-size: 18px; line-height: 1; cursor: pointer; padding: 0 4px;">×</button>
                </div>
                <div style="padding: 15px 20px;">
                <table style="width: 100%; border-collapse: collapse;">
                    <tr><td style="color: #888;">RNG Mode:</td><td style="text-align: right;">${rngName}</td></tr>
                    <tr><td style="color: #888;">Seed:</td><td style="text-align: right;">${seedDisplay}</td></tr>
                    <tr><td style="color: #888;">Param seed:</td><td style="text-align: right;">${paramSeedDisplay}</td></tr>
                    <tr><td style="color: #888;">Duration:</td><td style="text-align: right;">${(totalTime/1000).toFixed(2)}s</td></tr>
                    <tr><td style="color: #888;">Frames:</td><td style="text-align: right;">${frameCount.toFixed(2)}</td></tr>
                    <tr><td style="color: #888;">Workgroups/frame:</td><td style="text-align: right;">${CALIBRATION_WORKGROUPS.toFixed(2)}</td></tr>
                    <tr><td colspan="2" style="border-top: 1px solid #444; padding-top: 10px;"></td></tr>
                    <tr><td style="color: #888;">Avg FPS:</td><td style="text-align: right; color: #4f4;">${(1000/avgFrameTime).toFixed(2)}</td></tr>
                    <tr><td style="color: #888;">Frame time avg:</td><td style="text-align: right;">${avgFrameTime.toFixed(2)}ms</td></tr>
                    <tr><td style="color: #888;">Frame time min:</td><td style="text-align: right;">${minFrameTime.toFixed(2)}ms</td></tr>
                    <tr><td style="color: #888;">Frame time max:</td><td style="text-align: right;">${maxFrameTime.toFixed(2)}ms</td></tr>
                    <tr><td style="color: #888;">Frame time p50:</td><td style="text-align: right;">${p50.toFixed(2)}ms</td></tr>
                    <tr><td style="color: #888;">Frame time p95:</td><td style="text-align: right;">${p95.toFixed(2)}ms</td></tr>
                    <tr><td style="color: #888;">Frame time p99:</td><td style="text-align: right;">${p99.toFixed(2)}ms</td></tr>
                    <tr><td colspan="2" style="border-top: 1px solid #444; padding-top: 10px;"></td></tr>
                    <tr><td style="color: #888;">GPU compute:</td><td style="text-align: right;">${gpuStats}</td></tr>
                    <tr><td colspan="2" style="border-top: 1px solid #444; padding-top: 10px;"></td></tr>
                    <tr><td style="color: #888;">Points/sec:</td><td style="text-align: right; color: #fa0;">${(pointsPerSec/1e9).toFixed(4)}B</td></tr>
                    <tr><td style="color: #888;">Total points:</td><td style="text-align: right;">${(totalIterations/1e9).toFixed(4)}B</td></tr>
                </table>
                <p style="margin: 15px 0 0 0; color: #666; font-size: 11px;">
                    Params: a=${state.a.toFixed(3)}, b=${state.b.toFixed(3)}, c=${state.c.toFixed(3)}, d=${state.d.toFixed(3)}
                </p>
                </div>
            </div>
        `;

        // Show popup
        const popup = document.createElement('div');
        const popupOffset = calibrationPopupOffset;
        calibrationPopupOffset += 24;
        popup.style.cssText = `position: fixed; top: calc(50% + ${popupOffset}px); left: calc(50% + ${popupOffset}px); transform: translate(-50%, -50%); z-index: 10000; box-shadow: 0 10px 40px rgba(0,0,0,0.5);`;
        popup.innerHTML = resultsHtml;
        document.body.appendChild(popup);
        const closeBtn = popup.querySelector('.calibration-popup-close');
        if (closeBtn) {
            closeBtn.addEventListener('click', (e) => {
                e.stopPropagation();
                popup.remove();
            });
        }
        
        // Minimal draggable popup (drag header only)
        let dragStartX = 0;
        let dragStartY = 0;
        let dragStartLeft = 0;
        let dragStartTop = 0;
        const onPointerMove = (e) => {
            popup.style.left = `${dragStartLeft + (e.clientX - dragStartX)}px`;
            popup.style.top = `${dragStartTop + (e.clientY - dragStartY)}px`;
        };
        const onPointerUp = () => {
            window.removeEventListener('pointermove', onPointerMove);
            window.removeEventListener('pointerup', onPointerUp);
        };
        const onPointerDown = (e) => {
            if (e.button !== 0) return;
            e.preventDefault();
            const rect = popup.getBoundingClientRect();
            dragStartX = e.clientX;
            dragStartY = e.clientY;
            dragStartLeft = rect.left;
            dragStartTop = rect.top;
            popup.style.transform = 'none';
            window.addEventListener('pointermove', onPointerMove);
            window.addEventListener('pointerup', onPointerUp);
        };
        const header = popup.querySelector('.calibration-popup-header');
        if (header) {
            header.addEventListener('pointerdown', onPointerDown);
        }

        // Keep calibration params active after finishing (no restore)
        calibrationSavedState = null;
        
        // Restore canvas resolution after calibration
        if (calibrationCanvasState) {
            canvasWidth = calibrationCanvasState.width;
            canvasHeight = calibrationCanvasState.height;
            canvas.width = canvasWidth;
            canvas.height = canvasHeight;
            canvas.style.width = calibrationCanvasState.styleWidth;
            canvas.style.height = calibrationCanvasState.styleHeight;
            calibrationCanvasState = null;
            recreateGpuBuffers().catch((e) => console.error('Failed to recreate GPU buffers after calibration:', e));
        }
        
        // Ensure sliders reflect current calibration params
        ['a', 'b', 'c', 'd'].forEach(p => {
            const slider = document.getElementById(`param-${p}`);
            const display = document.getElementById(`val-${p}`);
            if (slider) slider.value = state[p];
            if (display) display.innerText = state[p].toFixed(3);
        });

        // Clear status
        const statusTempEl = document.getElementById('status-temp');
        if (statusTempEl) {
            statusTempEl.textContent = '';
            statusTempEl.style.color = '';
        }

        needsClear = true;
        updateParams();
    }
    
    function updateAttractMode(currentTime) {
        if (!attractModeActive) return;
        
        const elapsed = currentTime - attractModeStartTime;
        const deltaTime = currentTime - (attractModeLastUpdateTime || attractModeStartTime);
        attractModeLastUpdateTime = currentTime;
        
        // Convert deltaTime from milliseconds to seconds for smooth movement
        const dt = Math.min(deltaTime / 1000.0, 0.1); // Cap at 100ms to prevent large jumps
        
        const min = -4.0;
        const max = 4.0;
        const params = ['a', 'b', 'c', 'd'];
        let anyChanged = false;
        
        params.forEach(param => {
            const config = attractModeParams[param];
            
            // Periodically introduce subtle random speed variations (every 2-5 seconds per parameter)
            if (elapsed > config.nextSpeedChangeAt) {
                // Random speed variation: ±30% of base speed for more noticeable but subtle changes
                const variationFactor = 0.7 + Math.random() * 0.6; // 0.7 to 1.3
                config.speedVariation = (variationFactor - 1.0) * config.baseSpeed;
                config.lastSpeedChange = elapsed;
                config.nextSpeedChangeAt = elapsed + 2000 + Math.random() * 3000;
            }
            
            // Calculate current speed with variation (always positive magnitude)
            // FIX: Reduced minimum speed proportionally (0.006 * 0.25 = 0.0015) to match 75% speed reduction
            const currentSpeed = Math.max(0.01 * 0.6 * 0.25, config.baseSpeed + config.speedVariation); // Ensure positive, minimum 0.0015
            // Update velocity to match current speed with correct direction
            const currentDirection = config.velocity >= 0 ? 1 : -1;
            config.velocity = currentDirection * currentSpeed;
            
            // Update position based on velocity
            let newValue = config.value + config.velocity * dt;
            
            // Bounce at boundaries: reverse velocity direction when hitting limits
            // Preserve speed magnitude but flip direction
            if (newValue >= max) {
                newValue = max;
                config.velocity = -currentSpeed; // Reverse to negative, preserve speed
            } else if (newValue <= min) {
                newValue = min;
                config.velocity = currentSpeed; // Reverse to positive, preserve speed
            }
            
            // Update stored value
            config.value = newValue;
            
            // Only update if value changed significantly (avoid unnecessary updates)
            // Slower modes use finer thresholds: normal=0.001, slow=0.0002, ultra=0.00005
            const delta = Math.abs(state[param] - newValue);
            const updateThresholds = [0.001, 0.0002, 0.00005];
            const displayPrecisions = [3, 6, 7];
            const updateThreshold = updateThresholds[attractModeSpeed] || 0.001;
            if (delta > updateThreshold) {
                state[param] = newValue;

                // Update UI
                const slider = document.getElementById(`param-${param}`);
                const display = document.getElementById(`val-${param}`);
                if (slider) slider.value = newValue;
                if (display) display.innerText = newValue.toFixed(displayPrecisions[attractModeSpeed] || 3);
                
                // Update previous slider values
                previousSliderValues.set(param, newValue);
                
                anyChanged = true;
            }
        });
        
        // If any parameter changed, trigger update
        if (anyChanged) {
            needsClear = true;
            needsTimingReset = true;
            // Don't call updateParams() here - let the main frame loop handle it
            // This ensures proper synchronization with the rendering pipeline
        }
        
        return anyChanged;
    }
    
    // Wire up attract mode button
    // Click = normal, Shift+click = slow (5x slower), Ctrl+click = ultra-slow (15x slower)
    document.getElementById('btn-attract').onclick = (e) => {
        if (attractModeActive) {
            stopAttractMode();
        } else {
            // Determine speed mode: Ctrl = 2 (ultra), Shift = 1 (slow), normal = 0
            const speedMode = (e.ctrlKey || e.metaKey) ? 2 : (e.shiftKey || isShiftPressed) ? 1 : 0;
            startAttractMode(speedMode);
        }
    };
    
    // Stop attract mode when any button is pressed (except the attract button itself)
    // NOTE: Mouse wheel and drag do NOT stop attract mode - they work together
    const stopAttractOnButtonClick = (e) => {
        if (e.target.id !== 'btn-attract' && attractModeActive) {
            stopAttractMode();
        }
    };
    
    // 'a' key toggles attract mode
    document.addEventListener('keydown', (e) => {
        // Only trigger if not typing in an input field
        if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA' || e.target.tagName === 'SELECT') {
            return;
        }
        if (e.key === 'a' || e.key === 'A') {
            e.preventDefault();
            if (attractModeActive) {
                stopAttractMode();
            } else {
                startAttractMode();
            }
        }
    });
    
    // Reset: Reset view transform and fractal bounds to default
    // Shift+Reset: Also reset brightness controls (exposure, gamma, contrast) to defaults
    document.getElementById('btn-reset').onclick = (e) => {
        if (attractModeActive) stopAttractMode(); // Stop attract mode on button press
        markInteraction(); // Trigger full-speed rendering during rapid button clicks
        state.viewOffsetX = 0.0;
        state.viewOffsetY = 0.0;
        state.viewScale = 1.0;
        state.fractalMinX = -2.0;
        state.fractalMaxX = 2.0;
        state.fractalMinY = -2.0;
        state.fractalMaxY = 2.0;
        state.exposureMultiplier = 1.0;

        const resetToneControls = () => {
            state.exposure = FACTORY_EXPOSURE;
            state.gamma = FACTORY_GAMMA;
            state.contrast = FACTORY_CONTRAST;
            // Update sliders to reflect new values
            document.getElementById('param-exposure').value = mapExposureToSlider(state.exposure);
            document.getElementById('val-exp').innerText = state.exposure.toFixed(3);
            document.getElementById('param-gamma').value = mapGammaToSlider(state.gamma);
            document.getElementById('val-gamma').innerText = state.gamma.toFixed(3);
            document.getElementById('param-contrast').value = mapContrastToSlider(state.contrast);
            document.getElementById('val-contrast').innerText = state.contrast.toFixed(3);
        };

        // Shift+Reset: Also reset brightness defaults
        if (e.shiftKey || isShiftPressed) {
            resetToneControls();
        }

        needsClear = true;
        updateParams();
    };

    document.getElementById('btn-randomize').onclick = (e) => {
        // FIX: Continue attract mode animation when randomizing instead of stopping it
        // This allows users to randomize parameters while the animation continues
        markInteraction(); // Trigger full-speed rendering during rapid button clicks
        
        // Check if CTRL is held (for history navigation)
        // Prioritize event's modifier keys over tracked state for button clicks
        if (e.ctrlKey || e.metaKey) {
            // CTRL+Randomize: Go backwards through parameter history
            restoreStateFromHistory();
            // Sync attract mode params if active
            if (attractModeActive) {
                ['a', 'b', 'c', 'd'].forEach(param => {
                    const config = attractModeParams[param];
                    config.value = Math.max(-4.0, Math.min(4.0, state[param]));
                });
            }
        } else if (e.altKey) {
            // ALT+Randomize: Cycle through 8 micro-permutations (256x smaller)
            applyMicroCyclePermutation();
            if (attractModeActive) {
                ['a', 'b', 'c', 'd'].forEach(param => {
                    const config = attractModeParams[param];
                    config.value = Math.max(-4.0, Math.min(4.0, state[param]));
                });
            }
        } else if (e.shiftKey) {
            // SHIFT+Randomize: Cycle through 8 subtle permutations
            // (Only check e.shiftKey for button clicks - more reliable than isShiftPressed)
            applyCyclePermutation();
            // Sync attract mode params if active
            if (attractModeActive) {
                ['a', 'b', 'c', 'd'].forEach(param => {
                    const config = attractModeParams[param];
                    config.value = Math.max(-4.0, Math.min(4.0, state[param]));
                });
            }
        } else {
            // Normal randomize: Full random values
            randomizeParameters();
            // Sync attract mode params if active - continue animation from new random values
            if (attractModeActive) {
                ['a', 'b', 'c', 'd'].forEach(param => {
                    const config = attractModeParams[param];
                    config.value = Math.max(-4.0, Math.min(4.0, state[param]));
                });
            }
        }
    };

    document.getElementById('btn-clear').onclick = () => {
        if (attractModeActive) stopAttractMode(); // Stop attract mode on button press
        markInteraction(); // Trigger full-speed rendering during rapid button clicks
        clearDensity();
        clearCycleState(); // Clear cycle state on clear
        // Status will be updated by updateToggleButton if needed
        // Display clear confirmation
        const statusTempEl = document.getElementById('status-temp');
        statusTempEl.textContent = 'Cleared';
        statusTempEl.style.color = '#5a5';
        setTimeout(() => {
            statusTempEl.textContent = '';
            statusTempEl.style.color = '';
        }, 1000);
    };


        document.getElementById('select-color-mode').addEventListener('change', async (e) => {
            markInteraction(); // Trigger full-speed rendering during UI interaction
            const prevColorEnabled = state.colorMethod !== 0;
            const nextMethod = parseInt(e.target.value);
            const nextColorEnabled = nextMethod !== 0;
            state.colorMethod = nextMethod;

            needsClear = true; // Clear when changing color mode
            needsTimingReset = true;

            // Only rebuild GPU buffers/pipelines when crossing density-only <-> color boundary.
            if (webgpuAvailable && prevColorEnabled !== nextColorEnabled) {
                try {
                    await recreateGpuBuffers();
                } catch (err) {
                    console.error('Failed to switch color mode resources:', err);
                }
                return;
            }

            updateParams();
            // Note: On Intel Arc (HP OmniStudio), Color Mode 0 (Default) provides best performance
            // as it avoids writing to 3 extra color buffers, reducing memory bandwidth pressure.
        });


    // Defensive validation helpers for loading metadata
    function safeNumber(value, defaultValue, min = -Infinity, max = Infinity) {
        if (typeof value !== 'number' || !isFinite(value)) return defaultValue;
        return Math.max(min, Math.min(max, value));
    }
    
    function safeInteger(value, defaultValue, min = -Infinity, max = Infinity) {
        const num = safeNumber(value, defaultValue, min, max);
        return Math.round(num);
    }

    // Load Fractal from PNG - restores state from embedded metadata with defensive validation
    document.getElementById('btn-load').onclick = () => {
        if (attractModeActive) stopAttractMode(); // Stop attract mode on button press
        markInteraction(); // Trigger full-speed rendering during rapid button clicks
        document.getElementById('file-input').click();
    };

    document.getElementById('file-input').addEventListener('change', async (e) => {
        const file = e.target.files[0];
        if (!file) return;

        try {
            const metadata = await extractPNGMetadata(file);
            
            // If no metadata at all, can't proceed
            if (!metadata) {
                alert('This PNG file does not contain fractal metadata. It may have been saved with an older version or edited.');
                return;
            }

            const warnings = [];

            // Restore parameters with validation (these are critical - need at least some valid values)
            if (metadata.parameters) {
                const origA = state.a, origB = state.b, origC = state.c, origD = state.d;
                const paramKeys = [
                    { key: 'a', target: 'targetA', original: origA },
                    { key: 'b', target: 'targetB', original: origB },
                    { key: 'c', target: 'targetC', original: origC },
                    { key: 'd', target: 'targetD', original: origD }
                ];

                state.isLerping = false; // Stop lerping when loading from PNG

                let validParamCount = 0;
                let invalidParamCount = 0;

                for (const { key, target, original } of paramKeys) {
                    const rawValue = metadata.parameters[key];
                    if (typeof rawValue === 'number' && Number.isFinite(rawValue)) {
                        state[key] = rawValue;
                        validParamCount++;
                    } else {
                        state[key] = original;
                        invalidParamCount++;
                    }
                    state[target] = state[key];
                }

                if (validParamCount === 0) {
                    warnings.push('Fractal parameters (a,b,c,d) were missing or invalid - using defaults');
                } else if (invalidParamCount > 0) {
                    warnings.push('Some fractal parameters were invalid - using defaults for those');
                }
            } else {
                warnings.push('Fractal parameters section missing - using defaults');
            }

            // Restore view state with validation
            if (metadata.view) {
                state.viewOffsetX = safeNumber(metadata.view.offsetX, 0.0, -1000, 1000);
                state.viewOffsetY = safeNumber(metadata.view.offsetY, 0.0, -1000, 1000);
                state.viewScale = safeNumber(metadata.view.scale, 1.0, 0.01, 100.0);
                // FIX: Quantize viewOffset after loading to ensure whole-pixel buffer alignment
                quantizeViewOffset();
            } else {
                // Use defaults if view section missing
                state.viewOffsetX = 0.0;
                state.viewOffsetY = 0.0;
                state.viewScale = 1.0;
            }

            // Restore fractal bounds with validation (ensure min < max)
            if (metadata.fractalBounds) {
                const minX = safeNumber(metadata.fractalBounds.minX, -2.0);
                const maxX = safeNumber(metadata.fractalBounds.maxX, 2.0);
                const minY = safeNumber(metadata.fractalBounds.minY, -2.0);
                const maxY = safeNumber(metadata.fractalBounds.maxY, 2.0);
                
                // Ensure valid bounds (min < max)
                if (minX < maxX && minY < maxY) {
                    state.fractalMinX = minX;
                    state.fractalMaxX = maxX;
                    state.fractalMinY = minY;
                    state.fractalMaxY = maxY;
                } else {
                    warnings.push('Fractal bounds were invalid (min >= max) - using defaults');
                    state.fractalMinX = -2.0;
                    state.fractalMaxX = 2.0;
                    state.fractalMinY = -2.0;
                    state.fractalMaxY = 2.0;
                }
            } else {
                warnings.push('Fractal bounds missing - using defaults');
                state.fractalMinX = -2.0;
                state.fractalMaxX = 2.0;
                state.fractalMinY = -2.0;
                state.fractalMaxY = 2.0;
            }

            // Restore rendering settings with validation and safe defaults
            if (metadata.rendering) {
                // Exposure: 0.01 to 40.0
                if (metadata.rendering.exposure !== undefined) {
                    state.exposure = safeNumber(metadata.rendering.exposure, 25.0, 0.01, 40.0);
                }
                
                // Gamma: 0.1 to 3.0
                if (metadata.rendering.gamma !== undefined) {
                    state.gamma = safeNumber(metadata.rendering.gamma, 0.255, 0.1, 3.0);
                }
                
                // Contrast: 0.0 to 2.0
                if (metadata.rendering.contrast !== undefined) {
                    state.contrast = safeNumber(metadata.rendering.contrast, 1.0, 0.0, 2.0);
                }
                
                // Color method: 0-5 (integer)
                if (metadata.rendering.colorMethod !== undefined) {
                    state.colorMethod = safeInteger(metadata.rendering.colorMethod, 0, 0, 5);
                }
                
                // Exposure multiplier: must be positive
                if (metadata.rendering.exposureMultiplier !== undefined) {
                    state.exposureMultiplier = safeNumber(metadata.rendering.exposureMultiplier, 1.0, 0.001, 1000.0);
                } else {
                    state.exposureMultiplier = 1.0; // Default for old saves
                }
                
                // FIX: Don't restore RNG mode from PNG metadata - keep user's current selection
                // RNG mode stays consistent in a session until deliberately changed by user
                // Metadata is still saved for future use, but not restored to avoid confusion
                // (metadata.rendering.rngMode is preserved in metadata but not applied to state)
            } else {
                warnings.push('Rendering settings missing - using defaults');
                // RNG mode remains at current session value (default: Owen-Sobol Optimized)
            }

            // Sync UI to loaded state (do not re-register slider listeners).
            syncParamSlidersToState(3);
            syncToneSlidersToState(3);
            const colorSelect = document.getElementById('select-color-mode');
            if (colorSelect) colorSelect.value = String(state.colorMethod);

            // Validate all parameters before updating GPU state
            // Check for NaN or Infinity that could cause GPU hangs
            const paramsToCheck = ['a', 'b', 'c', 'd', 'exposure', 'gamma', 'contrast', 'viewOffsetX', 'viewOffsetY', 'viewScale'];
            let hasInvalidParams = false;
            for (const key of paramsToCheck) {
                const value = state[key];
                if (!isFinite(value) || isNaN(value)) {
                    console.error(`Invalid parameter ${key}: ${value}`);
                    hasInvalidParams = true;
                }
            }
            
            if (hasInvalidParams) {
                alert('Loaded PNG contains invalid parameter values (NaN or Infinity). Using safe defaults.');
                // Reset to safe defaults
                state.isLerping = false; // Stop lerping when resetting to defaults
                state.a = -2.24;
                state.b = 0.43;
                state.c = -0.65;
                state.d = -2.43;
                // Update targets to match default values
                state.targetA = -2.24;
                state.targetB = 0.43;
                state.targetC = -0.65;
                state.targetD = -2.43;
                state.exposure = 25.0;
                state.gamma = 0.255;
                state.contrast = 1.0;
                state.viewOffsetX = 0.0;
                state.viewOffsetY = 0.0;
                state.viewScale = 1.0;
            }

            // Recreate buffers for current canvas size / mode (GPU only)
            if (webgpuAvailable) {
                try {
                    await recreateGpuBuffers();
                } catch (error) {
                    console.error('Error recreating GPU buffers during PNG load:', error);
                    alert('Error recreating GPU buffers. The GPU may have lost connection. Please refresh the page.');
                    return; // Don't proceed if buffer creation fails
                }
            }

            // Clear and restart accumulation with restored state
            needsClear = true;
            needsTimingReset = true;
            updateParams();

            // Clear cycle state on load (major change)
            clearCycleState();
            
            // Update status display to show correct RNG mode (important for old PNGs)
            updateStatusDisplay();
            
            // Show success message with warnings if any
            const statusTempEl = document.getElementById('status-temp');
            
            if (warnings.length > 0) {
                // Show warnings in console for debugging
                console.warn('PNG load warnings:', warnings);
                statusTempEl.textContent = `Loaded (${warnings.length} warning${warnings.length > 1 ? 's' : ''})`;
                statusTempEl.style.color = '#fa5';
            } else {
                statusTempEl.textContent = 'Fractal state loaded!';
                statusTempEl.style.color = '#2a5';
            }
            
            setTimeout(() => {
                statusTempEl.textContent = '';
                statusTempEl.style.color = '';
            }, 3000);

        } catch (error) {
            console.error('Error loading fractal:', error);
            alert('Failed to load fractal metadata: ' + error.message);
        }

        // Reset file input
        e.target.value = '';
    });

    const toggleBtn = document.getElementById('btn-toggle');
    if (toggleBtn) {
        toggleBtn.onclick = () => {
            if (attractModeActive) stopAttractMode(); // Stop attract mode on button press
            markInteraction(); // Trigger full-speed rendering during rapid button clicks
            if (isRunning) {
                // Pause
                isRunning = false;
                if (animationFrameId !== null) {
                    // CPU mode uses setTimeout, GPU mode uses requestAnimationFrame
                    if (useCpuRng) {
                        clearTimeout(animationFrameId);
                    } else {
                        cancelAnimationFrame(animationFrameId);
                    }
                    animationFrameId = null;
                }
            } else {
                // Resume
                isRunning = true;
                startFrame();
            }
            updateToggleButton();
        };
    }

    // Save Screen - saves what's currently visible with metadata
    // Using toBlob() for better performance (async, non-blocking, more memory efficient)
    document.getElementById('btn-save-screen').onclick = async () => {
        if (attractModeActive) stopAttractMode(); // Stop attract mode on button press
        if (!webgpuAvailable) {
            // CPU mode: save current canvas
            const canvas = document.getElementById('canvas');
            canvas.toBlob((blob) => {
                if (blob) {
                    const url = URL.createObjectURL(blob);
                    const a = document.createElement('a');
                    a.href = url;
                    a.download = 'fractal-screen.png';
                    a.click();
                    URL.revokeObjectURL(url);
                }
            }, 'image/png');
            return;
        }
        markInteraction(); // Trigger full-speed rendering during rapid button clicks
        // Show saving status
        const statusTempEl = document.getElementById('status-temp');
        statusTempEl.textContent = 'Saving...';
        statusTempEl.style.color = '#4a9eff';
        
        try {
            // Use toBlob() instead of toDataURL() for better performance
            const blob = await new Promise((resolve, reject) => {
                canvas.toBlob((blob) => {
                    if (blob) resolve(blob);
                    else reject(new Error('Failed to create PNG blob'));
                }, 'image/png');
            });
            
            // Create metadata object with current state
            const metadata = {
                version: '1.0',
                timestamp: Date.now(),
                parameters: {
                    a: state.a,
                    b: state.b,
                    c: state.c,
                    d: state.d
                },
                view: {
                    offsetX: state.viewOffsetX,
                    offsetY: state.viewOffsetY,
                    scale: state.viewScale
                },
                fractalBounds: {
                    minX: state.fractalMinX,
                    maxX: state.fractalMaxX,
                    minY: state.fractalMinY,
                    maxY: state.fractalMaxY
                },
                rendering: {
                    exposure: state.exposure,
                    gamma: state.gamma,
                    contrast: state.contrast,
                    colorMethod: state.colorMethod,
                    totalIterations: state.totalIterations,
                    exposureMultiplier: state.exposureMultiplier,
                    rngMode: state.rngMode,
                    rngMethod: ['hash12 (PCG)', 'Sobol scrambled (QMC)', 'R2 (Legacy)', 'Sobol pure (QMC)', 'R2 Precision', 'Owen-Sobol (Optimized)', 'Combined LCG (Random)'][state.rngMode] || 'hash12 (PCG)'
                },
                canvas: {
                    width: canvasWidth,
                    height: canvasHeight
                }
            };
            
            try {
                // Embed metadata into PNG
                const blobWithMetadata = await embedPNGMetadata(blob, metadata);
                const url = URL.createObjectURL(blobWithMetadata);
        const link = document.createElement('a');
                link.download = `exotic_fractal_screen_${Date.now()}.png`;
                link.href = url;
                link.click();
                URL.revokeObjectURL(url);
                statusTempEl.textContent = 'Saved!';
                statusTempEl.style.color = '#2a5';
            } catch (error) {
                console.error('Failed to embed metadata, saving without it:', error);
                // Fallback: save without metadata
                const url = URL.createObjectURL(blob);
                const link = document.createElement('a');
                link.download = `exotic_fractal_screen_${Date.now()}.png`;
                link.href = url;
                link.click();
                URL.revokeObjectURL(url);
                statusTempEl.textContent = 'Saved (no metadata)';
                statusTempEl.style.color = '#fa5';
            }
        } catch (error) {
            console.error('Failed to save:', error);
            statusTempEl.textContent = 'Save failed!';
            statusTempEl.style.color = '#f55';
        }
        
        setTimeout(() => {
            statusTempEl.textContent = '';
            statusTempEl.style.color = '';
        }, 2000);
    };


    // Initialize color mode dropdown
    document.getElementById('select-color-mode').value = state.colorMethod.toString();
    
    
    // Iteration Hues: Convert normalized iteration count (0-1) to RGB
    // Matches shader logic exactly - smooth transitions through spectrum ending at white
    // Iteration Hues: Convert normalized iteration count (0-1) to RGB
    // Uses branchless cosine palette for reduced register pressure and execution divergence
    function iterationHueToRGB(t) {
        // Clamp t to [0, 1]
        const clamped = Math.max(0, Math.min(1, t));
        
        // "Spectral" palette coefficients (a + b * cos(6.28318 * (c * t + d)))
        // This creates a smooth rainbow: Red -> Yellow -> Green -> Cyan -> Blue -> Purple
        const a = [0.5, 0.5, 0.5];
        const b = [0.5, 0.5, 0.5];
        const c = [1.0, 1.0, 1.0];
        const d = [0.0, 0.333, 0.667]; // Phase offsets for R, G, B
        
        const twoPi = 6.28318;
        return [
            a[0] + b[0] * Math.cos(twoPi * (c[0] * clamped + d[0])),
            a[1] + b[1] * Math.cos(twoPi * (c[1] * clamped + d[1])),
            a[2] + b[2] * Math.cos(twoPi * (c[2] * clamped + d[2]))
        ];
    }
    
    // Initialize CPU-only mode (when WebGPU unavailable)
        async function initCpuOnlyMode() {
            const canvas = document.getElementById('canvas');
            
            // CPU mode: use the SAME 1:1 pixel rules as GPU mode (full DPR).
            // This keeps rebase + pan/zoom mapping consistent between CPU and GPU.
            const cssW = window.innerWidth;
            const cssH = window.innerHeight;
            const requestedDpr = window.devicePixelRatio || 1;
            
            // CPU safety: cap total pixels to avoid very large allocations (density + RGB + ImageData).
            applyCanvasSize(canvas, cssW, cssH, requestedDpr, CPU_MAX_PIXELS, 'CPU');
            
            // Use Canvas 2D for CPU rendering (disable alpha for performance)
            cpuCanvas = canvas;
        cpuCtx = canvas.getContext('2d', { alpha: false });
        cpuCtx.imageSmoothingEnabled = false;
        
        // Initialize CPU buffers
        const bufferSize = canvasWidth * canvasHeight;
        cpuDensityBuffer = new Uint32Array(bufferSize);
        cpuColorBufferR = new Float32Array(bufferSize);
        cpuColorBufferG = new Float32Array(bufferSize);
        cpuColorBufferB = new Float32Array(bufferSize);
        
        // Pre-allocate and reuse ImageData (avoids per-frame allocation)
        cpuImageData = cpuCtx.createImageData(canvasWidth, canvasHeight);
        cpuMaxDensity = 0;
        cpuLastPresentMs = 0;
        cpuRngIndex = 0;
        
        // Force CPU RNG mode (automatic fallback when WebGPU unavailable)
        useCpuRng = true;
        
        // DON'T start the loop here - startFrame() will handle it
        // This ensures only one loop is running (CPU or GPU, not both)
    }
    
    // CPU-only compute: tight batch kernel (used by time-budgeted CPU loop).
    // We intentionally keep it synchronous for throughput; the caller time-slices it.
    function runCpuBatch(batchSize, startIndex) {
        if (!cpuDensityBuffer || !cpuCanvas) return;
        
        const w = cpuCanvas.width;
        const h = cpuCanvas.height;
        
        const halfW = w * 0.5;
        const halfH = h * 0.5;

        // Cache state locally (hot loop)
        const a = state.a, b = state.b, c = state.c, d = state.d;
        const seed = state.seed;
        const rngMode = state.rngMode;
        const colorMode = state.colorMethod | 0;
        const viewScale = state.viewScale;
        
        // Calculate fractal bounds once per batch (not per particle)
        const originalRangeX = state.originalFractalMaxX - state.originalFractalMinX;
        const originalRangeY = state.originalFractalMaxY - state.originalFractalMinY;
        const fractalRangeX = state.fractalMaxX - state.fractalMinX;
        const fractalRangeY = state.fractalMaxY - state.fractalMinY;
        const baseScale = getViewBaseScale();
        const baseRange = 4.0;
        const maxRange = Math.max(fractalRangeX, fractalRangeY) || 1e-9;
        const scale = baseScale * (baseRange / maxRange) * 0.95;
        const fractalCenterX = (state.fractalMinX + state.fractalMaxX) * 0.5;
        const fractalCenterY = (state.fractalMinY + state.fractalMaxY) * 0.5;
        
        // Apply the SAME view transform as the GPU fragment shader (but during accumulation).
        // GPU does: bufferX = (screenX-halfW)/viewScale - viewOffsetX*baseScale + halfW
        // So: screenX = halfW + viewScale*(bufferX - halfW + viewOffsetX*baseScale)
        const viewOffsetXPix = state.viewOffsetX * baseScale;
        const viewOffsetYPix = state.viewOffsetY * baseScale;
        
        const originalMinX = state.originalFractalMinX;
        const originalMinY = state.originalFractalMinY;
        
        for (let k = 0; k < batchSize; k++) {
            const rnd = getRandom2D_JS(startIndex + k, seed, rngMode);
            let x = originalMinX + rnd[0] * originalRangeX;
            let y = originalMinY + rnd[1] * originalRangeY;
            
            // Warmup
            for (let i = 0; i < CPU_WARMUP_ITERS; i++) {
                const nx = Math.sin(a * y) - Math.cos(b * x);
                const ny = Math.sin(c * x) - Math.cos(d * y);
                x = nx;
                y = ny;
            }
            
            // Accumulate
            let prevX = x, prevY = y;
            for (let i = 0; i < CPU_ACCUM_ITERS; i++) {
                const nx = Math.sin(a * y) - Math.cos(b * x);
                const ny = Math.sin(c * x) - Math.cos(d * y);
                
                // First map fractal -> buffer coords (same as compute shader), then buffer -> screen coords (same as fragment shader)
                // Fix numerical cancellation: subtract first, then scale
                // This avoids losing precision when scale is very large
                const dx = x - fractalCenterX;
                const dy = y - fractalCenterY;
                const bufferX = dx * scale + halfW;
                const bufferY = dy * scale + halfH;
                const screenX = halfW + viewScale * (bufferX - halfW + viewOffsetXPix);
                const screenY = halfH + viewScale * (bufferY - halfH + viewOffsetYPix);
                
                const sx = Math.floor(screenX);
                const sy = Math.floor(screenY);
                
                if (sx >= 0 && sx < w && sy >= 0 && sy < h) {
                    const pixelIndex = sy * w + sx;
                    const prevCount = cpuDensityBuffer[pixelIndex];
                    const newCount = ++cpuDensityBuffer[pixelIndex];
                    
                    // Track max density incrementally for normalization (CPU mode)
                    if (newCount > cpuMaxDensity) cpuMaxDensity = newCount;
                    
                    // Simple color accumulation (if enabled)
                    if (colorMode >= 1 && cpuColorBufferR && prevCount < 10000) {
                        let r, g, bb;
                        
                        // Iteration Hues mode (approximate: use density as a proxy)
                        if (colorMode === 5) {
                            const MAX_ITER_LOG = 12.0; // log2(4096)
                            const countF = Math.max(prevCount, 1);
                            const logCount = Math.log2(countF);
                            const normalized = Math.max(0, Math.min(1, logCount / MAX_ITER_LOG));
                            const hueRGB = iterationHueToRGB(normalized);
                            r = hueRGB[0];
                            g = hueRGB[1];
                            bb = hueRGB[2];
                        } else {
                            const baseR = Math.abs(prevY - ny);
                            const baseG = Math.abs(prevX - ny);
                            const baseB = Math.abs(prevX - nx);
                            
                            const scaledR = baseR * 2.5;
                            const scaledG = baseG * 2.5;
                            const scaledB = baseB * 2.5;
                            
                            r = scaledR; g = scaledG; bb = scaledB;
                            switch (colorMode) {
                                case 2: r = scaledG; g = scaledB; bb = scaledR; break;
                                case 3: r = scaledB; g = scaledR; bb = scaledG; break;
                                case 4: r = scaledB; g = scaledG; bb = scaledR; break;
                            }
                        }
                        
                        const effCount = Math.min(newCount, 65535);
                        const alpha = 1.0 / (effCount + 1);
                        cpuColorBufferR[pixelIndex] = cpuColorBufferR[pixelIndex] + (r - cpuColorBufferR[pixelIndex]) * alpha;
                        cpuColorBufferG[pixelIndex] = cpuColorBufferG[pixelIndex] + (g - cpuColorBufferG[pixelIndex]) * alpha;
                        cpuColorBufferB[pixelIndex] = cpuColorBufferB[pixelIndex] + (bb - cpuColorBufferB[pixelIndex]) * alpha;
                    }
                }
                
                prevX = x;
                prevY = y;
                x = nx;
                y = ny;
            }
        }
    }
    
    // CPU-only rendering loop (when WebGPU unavailable)
    // Time-budgeted compute: run as many batches as fit in ~12ms, then render and yield.
    function startCpuRenderLoop() {
        if (animationFrameId !== null) {
            clearTimeout(animationFrameId);
        }
        
        // Local throttle for iteration-count DOM updates (CPU loop runs ~30fps)
        let lastIterCountUpdateMs = 0;
        const ITER_COUNT_THROTTLE_MS_CPU = 250; // 4× per second
        
        async function cpuTick() {
            // 1) Yield first to let UI events process
            await new Promise(resolve => setTimeout(resolve, 0));
            
            if (!isRunning || document.hidden) {
                animationFrameId = setTimeout(cpuTick, 200);
                return;
            }
            
            // 2) Pause compute entirely during interaction for maximum responsiveness
            if (isInteracting || isMouseDown) {
                animationFrameId = setTimeout(cpuTick, 50);
                return;
            }

            const now = performance.now();
            if (attractModeActive) {
                updateAttractMode(now);
            }
            
            // 3) Handle clear immediately, but DO NOT skip compute.
            // In CPU attract mode, parameters change frequently; if we clear-and-return, we can end up
            // rendering an empty buffer every tick (black screen). Clearing then continuing fixes that.
            let didClearThisTick = false;
            if (needsClear) {
                needsClear = false;
                state.totalIterations = 0;
                cpuRngIndex = 0;
                // Clear CPU buffers
                if (cpuDensityBuffer) {
                    cpuDensityBuffer.fill(0);
                    cpuColorBufferR.fill(0);
                    cpuColorBufferG.fill(0);
                    cpuColorBufferB.fill(0);
                }
                cpuMaxDensity = 0;
                cpuLastPresentMs = 0; // Force next present to go through throttling gate
                didClearThisTick = true;
            }
            
            // 4) TIME-BUDGETED COMPUTE
            // Process as many particles as possible in CPU_COMPUTE_BUDGET_MS
            const start = performance.now();
            let itersThisTick = 0;
            while (performance.now() - start < CPU_COMPUTE_BUDGET_MS) {
                runCpuBatch(CPU_BATCH_SIZE, cpuRngIndex);
                cpuRngIndex += CPU_BATCH_SIZE;
                itersThisTick += CPU_BATCH_SIZE * CPU_ACCUM_ITERS;
            }
            state.totalIterations += itersThisTick;
            
            // Throttle DOM updates
            const nowMs = performance.now();
            if (didClearThisTick || (nowMs - lastIterCountUpdateMs > ITER_COUNT_THROTTLE_MS_CPU)) {
                updateIterCount();
                lastIterCountUpdateMs = nowMs;
            }
            
            // 5) Render result (presentation is throttled inside renderCpuToCanvas)
            renderCpuToCanvas();
            
            // 6) Schedule next tick (target ~30fps)
            animationFrameId = setTimeout(cpuTick, CPU_TICK_INTERVAL_MS);
        }
        
        animationFrameId = setTimeout(cpuTick, 100);
    }
    // Pure CPU compute (no WebGPU dependencies) - kept for compatibility
    function runCpuComputePure(iterationsPerFrame) {
        if (!cpuDensityBuffer) return;
        
        const canvas = document.getElementById('canvas');
        const w = canvas.width;
        const h = canvas.height;
        
        // Calculate fractal bounds
        const originalRangeX = state.originalFractalMaxX - state.originalFractalMinX;
        const originalRangeY = state.originalFractalMaxY - state.originalFractalMinY;
        const fractalRangeX = state.fractalMaxX - state.fractalMinX;
        const fractalRangeY = state.fractalMaxY - state.fractalMinY;
        const baseScale = getViewBaseScale();
        const baseRange = 4.0;
        const maxRange = Math.max(fractalRangeX, fractalRangeY);
        const scale = baseScale * (baseRange / maxRange) * 0.95;
        const fractalCenterX = (state.fractalMinX + state.fractalMaxX) * 0.5;
        const fractalCenterY = (state.fractalMinY + state.fractalMaxY) * 0.5;
        
        // Run simulation
        const numThreads = Math.min(iterationsPerFrame / 128, 1000); // Limit for CPU
        for (let threadIndex = 0; threadIndex < numThreads; threadIndex++) {
            const rnd = getRandom2D_JS(threadIndex, state.seed, state.rngMode);
            let x = state.originalFractalMinX + rnd[0] * originalRangeX;
            let y = state.originalFractalMinY + rnd[1] * originalRangeY;
            
            // Warmup
            for (let i = 0; i < 12; i++) {
                const nx = Math.sin(state.a * y) - Math.cos(state.b * x);
                const ny = Math.sin(state.c * x) - Math.cos(state.d * y);
                x = nx;
                y = ny;
            }
            
            // Accumulation (reduced to 32 iterations for CPU responsiveness)
            let prevX = x, prevY = y;
            for (let i = 0; i < 32; i++) {
                const nx = Math.sin(state.a * y) - Math.cos(state.b * x);
                const ny = Math.sin(state.c * x) - Math.cos(state.d * y);
                
                // Fix numerical cancellation: subtract first, then scale
                // This avoids losing precision when scale is very large
                const dx = x - fractalCenterX;
                const dy = y - fractalCenterY;
                const px = dx * scale + w * 0.5;
                const py = dy * scale + h * 0.5;
                const sx = Math.floor(px);
                const sy = Math.floor(py);
                
                if (sx >= 0 && sx < w && sy >= 0 && sy < h) {
                    const pixelIndex = sy * w + sx;
                    const prevCount = cpuDensityBuffer[pixelIndex];
                    const newCount = ++cpuDensityBuffer[pixelIndex];
                    // Track max density incrementally (CPU mode optimization - avoids full buffer scan)
                    if (newCount > cpuMaxDensity) cpuMaxDensity = newCount;
                    
                    if (state.colorMethod >= 1) {
                        let r, g, b;
                        
                        // Iteration Hues mode: separate code path
                        if (state.colorMethod === 5) {
                            // Normalize iteration count using logarithmic scale (matching shader)
                            const MAX_ITER_LOG = 12.0; // log2(4096)
                            const countF = Math.max(prevCount, 1);
                            const logCount = Math.log2(countF);
                            const normalized = Math.max(0, Math.min(1, logCount / MAX_ITER_LOG));
                            
                            // Convert to RGB hue
                            const hueRGB = iterationHueToRGB(normalized);
                            r = hueRGB[0];
                            g = hueRGB[1];
                            b = hueRGB[2];
                        } else {
                            // Other color modes
                            const baseR = Math.abs(prevY - ny);
                            const baseG = Math.abs(prevX - ny);
                            const baseB = Math.abs(prevX - nx);
                            const scaledR = baseR * 2.5;
                            const scaledG = baseG * 2.5;
                            const scaledB = baseB * 2.5;
                            
                            r = scaledR; g = scaledG; b = scaledB;
                            switch (state.colorMethod) {
                                case 2: r = scaledG; g = scaledB; b = scaledR; break;
                                case 3: r = scaledB; g = scaledR; b = scaledG; break;
                                case 4: r = scaledB; g = scaledG; b = scaledR; break;
                            }
                        }
                        
                        const count = cpuDensityBuffer[pixelIndex];
                        const effCount = Math.min(count, 65535);
                        const alpha = 1.0 / (effCount + 1);
                        cpuColorBufferR[pixelIndex] = cpuColorBufferR[pixelIndex] + (r - cpuColorBufferR[pixelIndex]) * alpha;
                        cpuColorBufferG[pixelIndex] = cpuColorBufferG[pixelIndex] + (g - cpuColorBufferG[pixelIndex]) * alpha;
                        cpuColorBufferB[pixelIndex] = cpuColorBufferB[pixelIndex] + (b - cpuColorBufferB[pixelIndex]) * alpha;
                    }
                }
                
                prevX = x; prevY = y;
                x = nx; y = ny;
            }
        }
    }
    
    // Render CPU buffers to Canvas 2D (CPU mode only - optimized to avoid per-frame allocations)
    function renderCpuToCanvas() {
        if (!cpuCtx || !cpuDensityBuffer || !cpuImageData) return;
        
        // Throttle presentation rate to keep UI responsive (CPU mode only)
        const now = performance.now();
        if (now - cpuLastPresentMs < CPU_PRESENT_INTERVAL_MS) return;
        cpuLastPresentMs = now;
        
        const w = cpuCanvas.width;
        const h = cpuCanvas.height;
        
        // Reuse pre-allocated ImageData (avoids per-frame allocation + GC pressure)
        const imageData = cpuImageData;
        const data = imageData.data;
        const len = w * h;
        
        // Pre-calculate constants to avoid doing math inside the loop
        const exposure = state.exposure * state.exposureMultiplier;
        const gamma = state.gamma;
        const contrast = state.contrast;
        
        // Match GPU shader’s normalization model: density = count / totalIterations
        // (CPU loop increments totalIterations in units of deposited samples.)
        const invTotal = state.totalIterations > 0 ? 1.0 / state.totalIterations : 0.0;
        const BRIGHTNESS_SCALE = 250.0;
        
        // Fallback: tracked max density (kept for density-mode legacy rendering)
        const invMax = cpuMaxDensity > 0 ? 1.0 / cpuMaxDensity : 0;
        
        for (let i = 0; i < len; i++) {
            const d = cpuDensityBuffer[i];
            const idx = i * 4;
            
            if (d === 0) {
                // Pixels are initialized to 0, but we must set Alpha to 255
                data[idx] = 0;
                data[idx + 1] = 0;
                data[idx + 2] = 0;
                data[idx + 3] = 255;
                continue;
            }

            if (state.colorMethod >= 1) {
                // Match GPU fragment shader: density-weighted color + reinhard + gamma + contrast
                const density = d * invTotal;
                
                let rVal = cpuColorBufferR[i] * density * BRIGHTNESS_SCALE;
                let gVal = cpuColorBufferG[i] * density * BRIGHTNESS_SCALE;
                let bVal = cpuColorBufferB[i] * density * BRIGHTNESS_SCALE;
                
                // Exposure (reinhard-like)
                rVal = (rVal * exposure) / (1.0 + rVal * exposure);
                gVal = (gVal * exposure) / (1.0 + gVal * exposure);
                bVal = (bVal * exposure) / (1.0 + bVal * exposure);
                
                // Saturation boost (skip for Iteration Hues to match shader behavior)
                if (state.colorMethod !== 5) {
                    const maxVal = Math.max(rVal, gVal, bVal);
                    const minVal = Math.min(rVal, gVal, bVal);
                    const delta = maxVal - minVal;
                    if (delta > 0.001 && maxVal > 0.001) {
                        const saturation = delta / maxVal;
                        const newSaturation = Math.min(1.0, saturation * 1.5);
                        const scale = newSaturation / saturation;
                        rVal = minVal + (rVal - minVal) * scale;
                        gVal = minVal + (gVal - minVal) * scale;
                        bVal = minVal + (bVal - minVal) * scale;
                    }
                }
                
                // Gamma
                rVal = Math.pow(Math.max(0, Math.min(1, rVal)), gamma);
                gVal = Math.pow(Math.max(0, Math.min(1, gVal)), gamma);
                bVal = Math.pow(Math.max(0, Math.min(1, bVal)), gamma);
                
                // Contrast
                rVal = Math.max(0, Math.min(1, (rVal - 0.5) * contrast + 0.5));
                gVal = Math.max(0, Math.min(1, (gVal - 0.5) * contrast + 0.5));
                bVal = Math.max(0, Math.min(1, (bVal - 0.5) * contrast + 0.5));
                
                data[idx]     = (rVal * 255) | 0;
                data[idx + 1] = (gVal * 255) | 0;
                data[idx + 2] = (bVal * 255) | 0;
            } else {
                // Default density mode - use tracked max density
                // CPU mode has far fewer samples than GPU; boost visibility so it doesn't look black.
                const val = Math.min(255, d * invMax * 255 * CPU_DENSITY_VISIBILITY_BOOST);
                data[idx]     = val;
                data[idx + 1] = val * 0.5;
                data[idx + 2] = val * 0.2;
            }
            data[idx + 3] = 255;
        }
        
        cpuCtx.putImageData(imageData, 0, 0);
    }
    
        // Handle window resize to maintain aspect ratio
        window.addEventListener('resize', async () => {
            if (calibrationActive) {
                return;
            }
            // CPU mode: reinitialize at full DPR (keeps 1:1 mapping + rebase rules consistent)
            if (!webgpuAvailable || useCpuRng) {
                await initCpuOnlyMode();
                return;
            }
            
        // GPU mode: full DPR unless buffer limits force downscale.
            const prevW = canvas.width;
            const prevH = canvas.height;
            const cssW = window.innerWidth;
            const cssH = window.innerHeight;
            const requestedDpr = window.devicePixelRatio || 1;
            const maxPixels = gpuMaxBufferSize ? Math.max(1, Math.floor((gpuMaxBufferSize - 4) / 4)) : null;
            applyCanvasSize(canvas, cssW, cssH, requestedDpr, maxPixels, 'WebGPU');
            
            if (canvas.width !== prevW || canvas.height !== prevH) {
                
                // FIX: Re-quantize viewOffset after resize to ensure whole-pixel alignment with new baseScale
                quantizeViewOffset();
                
                // Recreate buffers for new canvas backing-store size
                await recreateGpuBuffers();
            }
        });
    
    // Mouse controls for pan and zoom
    // Note: isMouseDown is declared at top of file to avoid temporal dead zone
    let lastMouseX = 0;
    let lastMouseY = 0;
    // Accumulate fractional pixel movements for whole-pixel snapping
    let accumulatedDeltaX = 0;
    let accumulatedDeltaY = 0;
    
    // Helper function to quantize viewOffset to ensure whole-pixel buffer alignment
    // Fragment shader: bufferX = (screenX - width/2) / viewScale - viewOffsetX * baseScale + width/2
    // For whole pixel alignment: viewOffsetX * baseScale must be integer
    // So: viewOffsetX must be quantized to 1/baseScale increments
    function quantizeViewOffset() {
        const baseScale = getViewBaseScale();
        if (baseScale > 0) {
            const quantizeStep = 1.0 / baseScale;
            state.viewOffsetX = Math.round(state.viewOffsetX / quantizeStep) * quantizeStep;
            state.viewOffsetY = Math.round(state.viewOffsetY / quantizeStep) * quantizeStep;
        }
    }
    
    // Adaptive rendering: track interaction and last render time
    // SCOPE: isInteracting, interactionTimeout, and markInteraction() declared at top of file to avoid temporal dead zone
    let lastRenderTime = performance.now(); // Initialize to current time so first render happens immediately
    const IDLE_RENDER_INTERVAL = 100; // 10 Hz when idle (100ms = 10/sec)
    
    // Left or right mouse button drag to pan
    // NOTE: Mouse interactions (drag/zoom) work during attract mode and do NOT stop it
    canvas.addEventListener('mousedown', (e) => {
        if (e.button === 0 || e.button === 2) { // Left or right mouse button
            // Prevent context menu on right-click drag
            if (e.button === 2) {
                e.preventDefault();
            }
            isMouseDown = true;
            markInteraction(); // User is interacting
            lastMouseX = e.clientX;
            lastMouseY = e.clientY;
            // Reset accumulated deltas when starting a new drag
            accumulatedDeltaX = 0;
            accumulatedDeltaY = 0;
            canvas.style.cursor = 'grabbing';
        }
    });
    
    canvas.addEventListener('mousemove', (e) => {
        if (isMouseDown) {
            markInteraction(); // User is dragging
            const deltaX = e.clientX - lastMouseX;
            const deltaY = e.clientY - lastMouseY;
            
            // FIX: Only allow whole pixel displacements to avoid needing rebase for smooth results
            // Accumulate fractional pixel movements and only apply when they reach a whole pixel
            // This ensures crisp rendering without fractional pixel sampling artifacts
            accumulatedDeltaX += deltaX;
            accumulatedDeltaY += deltaY;
            
            // Extract whole pixel components
            const wholePixelsX = Math.round(accumulatedDeltaX);
            const wholePixelsY = Math.round(accumulatedDeltaY);
            
            // Skip update if no whole pixels accumulated
            if (wholePixelsX === 0 && wholePixelsY === 0) {
                // Still update lastMouseX/Y to track actual mouse position
                lastMouseX = e.clientX;
                lastMouseY = e.clientY;
                return;
            }
            
                // Convert screen pixels to fractal space
                // Scale by current zoom level
                const pxRatio = getCanvasPixelRatio(canvas);
                const baseScale = getViewBaseScale();
                const scale = baseScale * state.viewScale;
                state.viewOffsetX += wholePixelsX * pxRatio.x / scale;
                state.viewOffsetY += wholePixelsY * pxRatio.y / scale;
            
            // FIX: Quantize viewOffset to ensure whole-pixel buffer alignment
            // This prevents antialiasing artifacts by ensuring buffer coordinates are always whole pixels
            quantizeViewOffset();
            
            // Remove applied whole pixels from accumulation, keep fractional parts
            accumulatedDeltaX -= wholePixelsX;
            accumulatedDeltaY -= wholePixelsY;
            
            // Update last mouse position to actual current position
            lastMouseX = e.clientX;
            lastMouseY = e.clientY;
            
            // In CPU mode, trigger a clear and update params so view updates are visible
            // In GPU mode, updateParams() updates shader uniforms
            if (useCpuRng) {
                needsClear = true; // Clear and re-render with new view
                updateParams(); // Update fractal bounds from view offset
            } else {
                updateParams(); // Update shader parameters
            }
        }
    });
    
    canvas.addEventListener('mouseup', (e) => {
        if (e.button === 0 || e.button === 2) { // Left or right mouse button
            isMouseDown = false;
            // Reset accumulated deltas when drag ends
            accumulatedDeltaX = 0;
            accumulatedDeltaY = 0;
            canvas.style.cursor = 'default';
            scheduleAutoRebase();
        }
    });
    
    canvas.addEventListener('mouseleave', () => {
        isMouseDown = false;
        canvas.style.cursor = 'default';
        scheduleAutoRebase();
    });
    
    // Prevent context menu on right-click (when dragging)
    canvas.addEventListener('contextmenu', (e) => {
        if (isMouseDown) {
            e.preventDefault(); // Prevent context menu during drag
        }
    });
    
    // Mouse wheel to zoom
    // NOTE: Mouse wheel works during attract mode and does NOT stop it
    canvas.addEventListener('wheel', (e) => {
        e.preventDefault();
        
        markInteraction(); // User is zooming
        
        // Zoom factor (subtle zoom)
        const zoomFactor = 1.0 + (e.deltaY > 0 ? -0.1 : 0.1);
        const newScale = Math.max(0.1, Math.min(10.0, state.viewScale * zoomFactor));
        
        // Zoom towards mouse position
            const rect = canvas.getBoundingClientRect();
            const mouseX = e.clientX - rect.left;
            const mouseY = e.clientY - rect.top;
            const pxRatio = getCanvasPixelRatio(canvas);
            const canvasMouseX = mouseX * pxRatio.x;
            const canvasMouseY = mouseY * pxRatio.y;
        
        // Calculate zoom center in fractal space
        const baseScale = getViewBaseScale();
        const oldScale = baseScale * state.viewScale;
        const fractalX = (canvasMouseX - canvasWidth * 0.5 - state.viewOffsetX * oldScale) / oldScale;
        const fractalY = (canvasMouseY - canvasHeight * 0.5 - state.viewOffsetY * oldScale) / oldScale;
        
        // Update scale
        state.viewScale = newScale;
        
        // Adjust offset to zoom towards mouse position
        const newScaleValue = baseScale * state.viewScale;
        state.viewOffsetX = (canvasMouseX - canvasWidth * 0.5) / newScaleValue - fractalX;
        state.viewOffsetY = (canvasMouseY - canvasHeight * 0.5) / newScaleValue - fractalY;
        
        // FIX: Quantize viewOffset after zoom to ensure whole-pixel buffer alignment
        quantizeViewOffset();

        // PRECISION FIX: Aggressive rebase when viewScale exceeds Sterbenz-safe range
        // When viewScale > 2 or < 0.5, subtraction precision degrades (outside 2× ratio).
        // Auto-rebase keeps coordinates fresh and maintains precision at deep zoom.
        if (state.viewScale > 2.0 || state.viewScale < 0.5) {
            performRebase(false);
            return; // performRebase already calls updateParams and clears
        }

        // In CPU mode, trigger a clear so view updates are visible
        if (useCpuRng) {
            needsClear = true; // Clear and re-render with new view
        }
        updateParams(); // Update shader parameters / fractal bounds
        scheduleAutoRebase();
    });

    updateParams();
    updateStatusDisplay(); // Initialize status display
    clearDensity();
    updateIterCount();
    
    // Initialize toggle button state
    updateToggleButton();
    
    // Auto-start the frame loop
    startFrame();

    // --- ADAPTIVE STATE ---
    // workgroupCount is declared above (before updateParams) so it's accessible everywhere
    const TARGET_MS = 16.0; // Target 60 FPS (16.6ms)
    let lastFrameTime = performance.now();
    const MAX_DISPATCH = 65535; // WebGPU Limit per single dispatch call
    let consecutiveSlowFrames = 0; // Track consecutive slow frames for backoff
    let backpressureScaleSmoothed = 1.0; // EMA for queue pressure scaling

    // Throttle updateIterCount() to reduce DOM thrash
    let lastIterCountUpdate = 0;
    const ITER_COUNT_THROTTLE_MS = 250; // Update 4× per second

    // --- Main Loop ---

    // CPU compute function (runs simulation in JavaScript using JS RNG)
    // Fast path: maintains persistent JS-side buffers and uploads via writeBuffer (no mapAsync stalls)
    async function runCpuCompute(iterationsPerFrame, clearFirst = false) {
        if (!webgpuAvailable || !device || !densityBuffer) {
            // Fallback to pure CPU mode
            runCpuComputePure(iterationsPerFrame);
            if (cpuCtx) renderCpuToCanvas();
            return;
        }
        
        const renderDims = getRenderDimensions();
        const w = renderDims.width;
        const h = renderDims.height;
        const size = w * h;
        ensureJsBuffers(size);
        
        if (clearFirst) {
            jsDensityBuffer.fill(0);
            jsColorBufferR.fill(0);
            jsColorBufferG.fill(0);
            jsColorBufferB.fill(0);
        }
        
        // Calculate fractal bounds
        const originalRangeX = state.originalFractalMaxX - state.originalFractalMinX;
        const originalRangeY = state.originalFractalMaxY - state.originalFractalMinY;
        const fractalRangeX = state.fractalMaxX - state.fractalMinX;
        const fractalRangeY = state.fractalMaxY - state.fractalMinY;
        const baseScale = getViewBaseScale();
        const baseRange = 4.0;
        const maxRange = Math.max(fractalRangeX, fractalRangeY);
        const scale = baseScale * (baseRange / maxRange) * 0.95;
        const fractalCenterX = (state.fractalMinX + state.fractalMaxX) * 0.5;
        const fractalCenterY = (state.fractalMinY + state.fractalMaxY) * 0.5;
        
        // Run simulation for each thread
        const numThreads = Math.min(Math.floor(iterationsPerFrame / 128), 5000);
        for (let threadIndex = 0; threadIndex < numThreads; threadIndex++) {
            const rnd = getRandom2D_JS(threadIndex + cumulativeThreadCount, state.seed, state.rngMode);
            let x = state.originalFractalMinX + rnd[0] * originalRangeX;
            let y = state.originalFractalMinY + rnd[1] * originalRangeY;
            
            // Warmup
            for (let i = 0; i < 12; i++) {
                const nx = Math.sin(state.a * y) - Math.cos(state.b * x);
                const ny = Math.sin(state.c * x) - Math.cos(state.d * y);
                x = nx;
                y = ny;
            }
            
            // Accumulation
            let prevX = x;
            let prevY = y;
            for (let i = 0; i < 128; i++) {
                const nx = Math.sin(state.a * y) - Math.cos(state.b * x);
                const ny = Math.sin(state.c * x) - Math.cos(state.d * y);
                
                // Fix numerical cancellation: subtract first, then scale
                // This avoids losing precision when scale is very large
                const dx = x - fractalCenterX;
                const dy = y - fractalCenterY;
                const sx = Math.floor(dx * scale + w * 0.5);
                const sy = Math.floor(dy * scale + h * 0.5);
                if (sx >= 0 && sx < w && sy >= 0 && sy < h) {
                    const pixelIndex = sy * w + sx;
                    const prevCount = jsDensityBuffer[pixelIndex];
                    const count = prevCount + 1;
                    jsDensityBuffer[pixelIndex] = count;
                    
                    if (state.colorMethod >= 1) {
                        const logCount = 31 - Math.clz32((prevCount | 1) >>> 0);
                        const tier = Math.max(0, logCount - 6);
                        let shouldUpdateColor = true;
                        if (tier > 0) {
                            const throttleBits = Math.min(tier + 3, 10);
                            const mask = (1 << throttleBits) - 1;
                            shouldUpdateColor = (prevCount & mask) === 0;
                        }

                        if (!shouldUpdateColor) {
                            prevX = x;
                            prevY = y;
                            x = nx;
                            y = ny;
                            continue;
                        }

                        const colorUpdates = colorUpdateCountJS(prevCount);

                        let r, g, b;
                        
                        if (state.colorMethod === 5) {
                            // Iteration Hues mode: approximate shader path
                            const MAX_ITER_LOG = 12.0; // log2(4096)
                            const countF = Math.max(prevCount, 1);
                            const logCount = Math.log2(countF);
                            const normalized = Math.max(0, Math.min(1, logCount / MAX_ITER_LOG));
                            const hueRGB = iterationHueToRGB(normalized);
                            r = hueRGB[0] * COLOR_RANGE;
                            g = hueRGB[1] * COLOR_RANGE;
                            b = hueRGB[2] * COLOR_RANGE;
                        } else {
                            const baseR = Math.abs(prevY - ny);
                            const baseG = Math.abs(prevX - ny);
                            const baseB = Math.abs(prevX - nx);
                            const scaledR = baseR * 2.5;
                            const scaledG = baseG * 2.5;
                            const scaledB = baseB * 2.5;
                            
                            r = scaledR; g = scaledG; b = scaledB;
                            switch (state.colorMethod) {
                                case 2: r = scaledG; g = scaledB; b = scaledR; break;
                                case 3: r = scaledB; g = scaledR; b = scaledG; break;
                                case 4: r = scaledB; g = scaledG; b = scaledR; break;
                            }
                        }

                        const rClamped = Math.min(Math.max(r, 0), COLOR_RANGE);
                        const gClamped = Math.min(Math.max(g, 0), COLOR_RANGE);
                        const bClamped = Math.min(Math.max(b, 0), COLOR_RANGE);
                        const alpha = 1.0 / colorUpdates;
                        jsColorBufferR[pixelIndex] += (rClamped - jsColorBufferR[pixelIndex]) * alpha;
                        jsColorBufferG[pixelIndex] += (gClamped - jsColorBufferG[pixelIndex]) * alpha;
                        jsColorBufferB[pixelIndex] += (bClamped - jsColorBufferB[pixelIndex]) * alpha;
                    }
                }
                
                prevX = x;
                prevY = y;
                x = nx;
                y = ny;
            }
        }
        
        // Upload results to GPU without mapping
        device.queue.writeBuffer(densityBuffer, 0, jsDensityBuffer);
        if (state.colorMethod >= 1) {
            device.queue.writeBuffer(colorBufferR, 0, jsColorBufferR);
            device.queue.writeBuffer(colorBufferG, 0, jsColorBufferG);
            device.queue.writeBuffer(colorBufferB, 0, jsColorBufferB);
        }
    }

    // Must match shader's colorUpdateCount() exactly.
    function colorUpdateCountJS(lastCount) {
        if (lastCount === 0) return 1;
        const logCount = 31 - Math.clz32(lastCount | 1);
        const tier = Math.max(0, logCount - 6);
        if (tier <= 0) {
            return lastCount + 1;
        }
        const throttleBits = Math.min(tier + 3, 10);
        const stride = 1 << throttleBits;
        const tierStart = 1 << (tier + 6);
        const priorTierUpdates = 8 * Math.max(0, tier - 1);
        const updatesInTier = Math.floor((lastCount - tierStart) / stride);
        return 128 + 1 + priorTierUpdates + updatesInTier;
    }

    async function frame() {
        if (!isRunning) return;
        
        // Pause when tab is hidden to avoid wasting resources
        if (document.hidden) {
            animationFrameId = requestAnimationFrame(frame);
            return;
        }

        // 0. SMOOTH PARAMETER INTERPOLATION (only when randomize is pressed)
        // -------------------------------------------------------------
        if (state.isLerping) {
            const lerp = (a, b, t) => a + (b - a) * t;
            // Ease-in-out function (smoothstep): slow-quick-slow curve
            // This creates a natural time-symmetric animation that starts slow, speeds up in the middle, then slows down
            const easeInOut = (t) => t * t * (3 - 2 * t);
            
            // Advance progress by paramLerpSpeed each frame (maintains same total duration)
            state.lerpProgress = Math.min(1, state.lerpProgress + state.paramLerpSpeed);
            
            // Apply ease-in-out curve to progress for slow-quick-slow effect
            const easedProgress = easeInOut(state.lerpProgress);
            
            // Interpolate from start to target using eased progress
            state.a = lerp(state.lerpStartA, state.targetA, easedProgress);
            state.b = lerp(state.lerpStartB, state.targetB, easedProgress);
            state.c = lerp(state.lerpStartC, state.targetC, easedProgress);
            state.d = lerp(state.lerpStartD, state.targetD, easedProgress);
            
            // Check if we've reached the end
            if (state.lerpProgress >= 1) {
                // Snap to exact target values and stop lerping
                state.a = state.targetA;
                state.b = state.targetB;
                state.c = state.targetC;
                state.d = state.targetD;
                state.isLerping = false;
            }
            
            // Update UI sliders to reflect interpolated values
            const pa = document.getElementById('param-a');
            const pb = document.getElementById('param-b');
            const pc = document.getElementById('param-c');
            const pd = document.getElementById('param-d');
            const va = document.getElementById('val-a');
            const vb = document.getElementById('val-b');
            const vc = document.getElementById('val-c');
            const vd = document.getElementById('val-d');
            if (pa && va) { pa.value = state.a; va.innerText = state.a.toFixed(3); }
            if (pb && vb) { pb.value = state.b; vb.innerText = state.b.toFixed(3); }
            if (pc && vc) { pc.value = state.c; vc.innerText = state.c.toFixed(3); }
            if (pd && vd) { pd.value = state.d; vd.innerText = state.d.toFixed(3); }
            
            // Update previous slider values
            if (typeof previousSliderValues !== 'undefined') {
                previousSliderValues.set('a', state.a);
                previousSliderValues.set('b', state.b);
                previousSliderValues.set('c', state.c);
                previousSliderValues.set('d', state.d);
            }
            
            // Clear each frame during lerp to avoid ghosting.
            needsClear = true;
            
            // Push interpolated params every frame for smooth morph.
            updateParams();
        }

        // 1. Measure + predict (PID-like scheduler).
        let dt = TARGET_FRAME_TIME_MS; // Default fallback
        const now = performance.now();
        const wallTime = now - lastFrameTime;
        lastFrameTime = now;

        // Attract mode: update params on the main loop so it respects GPU backpressure.
        if (attractModeActive) {
            updateAttractMode(now);
        }
        
            // Prefer GPU timing; fall back to submission time or wallTime.
            let gpuTimeMs = null;
            if (enableTimestamps && lastTotalTimeNs !== null) {
                dt = lastTotalTimeNs / 1e6; // Convert nanoseconds to milliseconds
                gpuTimeMs = dt;
            } else if (lastSubmittedGpuMs !== null) {
                dt = lastSubmittedGpuMs;
                gpuTimeMs = dt;
            } else {
                dt = wallTime;
            }

        // Calibration: record timing per frame.
        if (calibrationActive && calibrationFrameCount > 0) {
            calibrationTimings.push({
                frameTime: wallTime,
                gpuTime: gpuTimeMs,
                frameIndex: calibrationFrameCount
            });
        }
        if (calibrationActive) {
            calibrationFrameCount++;
        }

        // Update perf model using the previous frame's timing.
            const haveReliableTiming = (enableTimestamps && lastTotalTimeNs !== null) || (lastSubmittedGpuMs !== null);
        if (haveReliableTiming && dt > 0 && workgroupCount > 0 && !isNaN(dt) && isFinite(dt)) {
            const instantaneousRate = dt / workgroupCount;
            
            // EMA smooths jitter (90% history / 10% new).
            avgTimePerWorkgroup = (0.9 * avgTimePerWorkgroup) + (0.1 * instantaneousRate);
            
            // Fit workgroups into the target frame time.
            const safeGroupCount = Math.floor(TARGET_FRAME_TIME_MS / avgTimePerWorkgroup);
            
            // Clamp to sane bounds.
            const newWorkgroupCount = Math.max(100, Math.min(2000000, safeGroupCount));
            
            // Fast drop for heavy overruns; otherwise smooth with deadband.
            const prevCount = workgroupCount;
            if (dt > TARGET_FRAME_TIME_MS * 1.5) {
                workgroupCount = Math.max(100, Math.floor(prevCount * 0.5));
            } else {
                const error = newWorkgroupCount - prevCount;
                const errorRatio = Math.abs(error) / Math.max(prevCount, 1);
                if (errorRatio < WORKGROUP_DEADBAND_RATIO) {
                    workgroupCount = prevCount;
                } else {
                    const alpha = error > 0 ? WORKGROUP_UP_ALPHA : WORKGROUP_DOWN_ALPHA;
                    const blended = prevCount + (error * alpha);
                    const maxChange = prevCount * 0.2;
                    const delta = Math.max(-maxChange, Math.min(maxChange, blended - prevCount));
                    workgroupCount = prevCount + delta;
                }
            }
            
            // Integerize to avoid fractional jitter.
            workgroupCount = Math.max(100, (workgroupCount | 0));
        } else {
            // No reliable GPU timing yet; keep conservative baseline.
            workgroupCount = Math.min(workgroupCount, 40000);
        }
        
        // Cap workgroups during attract mode for consistent framerate.
        if (attractModeActive) {
            workgroupCount = Math.min(workgroupCount, 40000);
        }

        // CALIBRATION MODE: Override all adaptive scheduling for precise benchmark
        if (calibrationActive) {
            workgroupCount = CALIBRATION_WORKGROUPS;  // Fixed count for repeatability

            // Stop after a fixed amount of work so every device runs identical math.
            if (calibrationFrameCount >= CALIBRATION_FRAME_LIMIT) {
                finishCalibration();
                // Continue with normal frame after calibration ends
            }
        }

        // CRITICAL: Cap workgroups during interaction for responsive UI
        // Prevents GPU queue depth saturation that causes slider freeze
        // Use adaptive budget based on measured frame time for smoother UI response
        if (isInteracting && !calibrationActive) {
            // Target 8ms frame time during interaction for responsive UI
            const INTERACTION_TARGET_MS = 8.0;
            const interactionBudget = INTERACTION_TARGET_MS / avgTimePerWorkgroup;
            workgroupCount = Math.min(workgroupCount, Math.max(500, interactionBudget));
        }

        // CAP MAX WORKGROUPS WHEN IDLE to prevent GPU queue saturation and UI freeze.
        // Without this, the PID scheduler can ramp to millions of workgroups on fast GPUs,
        // starving the browser compositor and making input feel "stuck" until the backlog drains.
        if (!attractModeActive && !isInteracting && !isMouseDown && !calibrationActive) {
            workgroupCount = Math.min(workgroupCount, 150000);
        }

        // Smooth backpressure: scale workgroups based on queue depth instead of hard skipping.
        if (webgpuAvailable && device && !calibrationActive && inFlightFrames >= 2) {
            const queuePressure = inFlightFrames / MAX_IN_FLIGHT_FRAMES;
            const targetScale = Math.max(0.5, 1 / (1 + queuePressure));
            backpressureScaleSmoothed = (0.8 * backpressureScaleSmoothed) + (0.2 * targetScale);
            workgroupCount = Math.max(100, Math.floor(workgroupCount * backpressureScaleSmoothed));
        } else {
            backpressureScaleSmoothed = (0.8 * backpressureScaleSmoothed) + 0.2;
        }

        // Force integer to prevent fractional jitter (interactionBudget can be fractional)
        workgroupCount = Math.max(100, (workgroupCount | 0));

        const currentMaxRange = Math.max(
            state.fractalMaxX - state.fractalMinX,
            state.fractalMaxY - state.fractalMinY
        );
        const iterationCount = getAdaptiveIterationCount(currentMaxRange);

        // 2. BACKPRESSURE ALREADY APPLIED - continue with reduced workgroupCount if needed

        // 3. CLEAR + RESET (if needed) - do this FIRST before updating params
        let needsClearPass = false;
        if (needsClear) {
            needsClear = false;
            state.totalIterations = 0;
            frameCounter = 0; // Reset frame counter on clear
            frameOffsetCounter = 0; // Reset frame offset counter on clear
            frameId = 0; // Reset frame ID on clear
            cumulativeThreadCount = 0; // Reset cumulative thread count on clear
            updateIterCount(); // Always update immediately on clear
            lastIterCountUpdate = now; // Reset throttle timer
            needsClearPass = true;
            // Keep avgTimePerWorkgroup: GPU cost doesn't change just because buffers cleared.
        }

        if (needsTimingReset) {
            needsTimingReset = false;
            avgTimePerWorkgroup = 0.02; // Calibrated for 128 iterations per thread
        }

        // 3b. ADVANCE COUNTERS BEFORE updateParams() - CRITICAL FIX for density normalization
        // The fragment shader reads invTotal from params to normalize density.
        // If we increment AFTER updateParams(), the shader gets stale data and over-brightens.
        // Since we've passed the early-out check, we're committed to submitting this frame,
        // so it's safe to increment now. This ensures the shader sees the correct value.
        const iterationsPerFrame = workgroupCount * 64 * iterationCount;
        state.totalIterations += iterationsPerFrame;
        frameId++;
        cumulativeThreadCount += workgroupCount * 64;
        
        // Update iteration count display (throttled) - do this here so it's consistent
        if (now - lastIterCountUpdate > ITER_COUNT_THROTTLE_MS) {
            updateIterCount();
            lastIterCountUpdate = now;
        }

        // 4. UPLOAD PARAMS (after advancing counters, so shader sees correct values)
        // Write to inactive buffer - this queues the write asynchronously
        updateParams();
        
        // Increment frame offset counter - advance RNG sequence offset every FRAME_OFFSET_INTERVAL frames
        // For R2 Precision mode (mode 4), always advance to avoid repeated sequences
        if (state.rngMode === 4) {
            // R2 Precision: always advance frame offset for unique sequences
            frameCounter++;
        } else {
            // Other modes: advance every FRAME_OFFSET_INTERVAL frames
            frameOffsetCounter++;
            if (frameOffsetCounter >= FRAME_OFFSET_INTERVAL) {
                frameCounter++;
                frameOffsetCounter = 0;
            }
        }
        
        // 4b. SWAP PARAMS BUFFERS (double-buffering: swap before encoder creation)
        // Swap active/inactive buffers BEFORE creating encoder for maximum responsiveness
        // WebGPU's command queue ensures writeBuffer() completes before commands execute,
        // so swapping here is safe and gives us the best slider responsiveness
        const temp = paramsBufferActive;
        paramsBufferActive = paramsBufferInactive;
        paramsBufferInactive = temp;
        // Swap bind group references (zero allocation - no GPU resource creation)
        const tempComputeBG = computeBindGroupActive;
        computeBindGroupActive = computeBindGroupInactive;
        computeBindGroupInactive = tempComputeBG;
        const tempRenderBG = renderBindGroupActive;
        renderBindGroupActive = renderBindGroupInactive;
        renderBindGroupInactive = tempRenderBG;
        
        // If WebGPU not available, this function should not be called
        // CPU mode uses its own loop (startCpuRenderLoop) started by startFrame()
        if (!webgpuAvailable) {
            console.error('frame() called in CPU mode - this should not happen');
            return;
        }
        
        // CPU RNG path: run compute in JavaScript
        if (useCpuRng) {
            try {
                await runCpuCompute(iterationsPerFrame, needsClearPass);
                // Counters already advanced before updateParams() - display already updated
            } catch (e) {
                console.error('CPU compute error:', e);
            }
        }
        
        // Always run compute and render (throttling disabled)
        const computeEncoder = device.createCommandEncoder();

        // 4a. CLEAR (if needed) - use native clearBuffer (fast + avoids 65,535 workgroup limits)
        // FIX: Native buffer clearing replaces custom compute shader clearing pipeline.
        // Dispatching a clear shader requires calculating workgroup counts. On 4K/8K screens,
        // the buffer size requires more workgroups than the hardware limit (65,535), causing
        // the WebGPU device to crash/reset. Native clearBuffer is faster, simpler, and has no size limits.
        if (needsClearPass && webgpuAvailable && !useCpuRng) {
            if (densityBuffer) computeEncoder.clearBuffer(densityBuffer);
            if (colorBufferR) computeEncoder.clearBuffer(colorBufferR);
            if (colorBufferG) computeEncoder.clearBuffer(colorBufferG);
            if (colorBufferB) computeEncoder.clearBuffer(colorBufferB);
        } else if (needsClearPass && !webgpuAvailable) {
            // Clear CPU buffers
            if (cpuDensityBuffer) cpuDensityBuffer.fill(0);
            if (cpuColorBufferR) cpuColorBufferR.fill(0);
            if (cpuColorBufferG) cpuColorBufferG.fill(0);
            if (cpuColorBufferB) cpuColorBufferB.fill(0);
        }

        // 5. COMPUTE PASS - always runs
            // Timestamp queries: DO NOT use passEncoder.writeTimestamp() (not supported in Firefox and
            // gated behind experimental features in Chromium). Prefer commandEncoder.writeTimestamp().
            const canWriteTimestamps =
                !!(enableTimestamps && timestampQuerySet && timestampBuffer && timestampReadbackBuffer &&
                    computeEncoder && typeof computeEncoder.writeTimestamp === 'function');
        
        if (canWriteTimestamps) {
            computeEncoder.writeTimestamp(timestampQuerySet, 0); // Before compute pass
        }
        
        const passEncoder = computeEncoder.beginComputePass();
        
        // Skip GPU compute if using CPU RNG or WebGPU unavailable
        if (!useCpuRng && webgpuAvailable && computePipeline) {
            passEncoder.setPipeline(computePipeline);
            passEncoder.setBindGroup(0, computeBindGroupActive);

            // Use 2D dispatch for >65535 workgroups to ensure unique indices
            const dispatchDimX = Math.min(workgroupCount, 65535);
            const dispatchDimY = Math.ceil(workgroupCount / 65535);
            passEncoder.dispatchWorkgroups(dispatchDimX, dispatchDimY);
            // Note: Don't increment cumulativeThreadCount here - only after successful submission
        }
        
        passEncoder.end();
        
        if (canWriteTimestamps) {
            computeEncoder.writeTimestamp(timestampQuerySet, 1); // After compute pass
        }
        
            // Prepare compute command buffer (but don't submit yet - will combine with render)
        let computeCommandBuffer = null;
        if (webgpuAvailable && device) {
            computeCommandBuffer = computeEncoder.finish();
        }

        // 6. RENDER PASSES (always render - throttling disabled)
        // If WebGPU not available, skip GPU rendering (CPU mode handles its own rendering)
        if (!webgpuAvailable) {
            animationFrameId = requestAnimationFrame(frame);
            return;
        }
        
        // Validate context and resources before rendering
        if (!context) {
            console.warn('WebGPU context not available, skipping render');
            animationFrameId = requestAnimationFrame(frame);
            return;
        }
        
        // Validate all required resources before rendering
        // Note: renderBindGroupActive was just swapped, so it now references the buffer we wrote to
        if (!renderPipeline || !renderBindGroupActive) {
            console.warn('Render pipeline or bind group not available, skipping render');
            animationFrameId = requestAnimationFrame(frame);
            return;
        }
        
        // Use resilient renderer if available, otherwise fall back to try-catch
        if (resilientRenderer && webgpuAvailable) {
            await resilientRenderer.render(async () => {
                // Get current texture - must be done before creating encoder
                let currentTexture;
                try {
                    currentTexture = context.getCurrentTexture();
                } catch (e) {
                    console.warn('Failed to get current texture:', e);
                    animationFrameId = requestAnimationFrame(frame);
                    return;
                    }
                    
                    const renderEncoder = device.createCommandEncoder();
                    if (canWriteTimestamps) {
                        renderEncoder.writeTimestamp(timestampQuerySet, 2); // Before render pass(es)
                    }
                    
                    // Direct render pass
                    const renderPass = renderEncoder.beginRenderPass({
                        colorAttachments: [{
                            view: currentTexture.createView(),
                            loadOp: 'clear', clearValue: { r: 0, g: 0, b: 0, a: 1 }, storeOp: 'store',
                        }]
                    });
                    renderPass.setPipeline(renderPipeline);
                    renderPass.setBindGroup(0, renderBindGroupActive);
                    renderPass.draw(6);
                    renderPass.end();

                    // Resolve timestamps (compute + render) if supported.
                    if (canWriteTimestamps) {
                        renderEncoder.writeTimestamp(timestampQuerySet, 3); // After render pass(es)
                        renderEncoder.resolveQuerySet(timestampQuerySet, 0, 4, timestampBuffer, 0);
                        renderEncoder.copyBufferToBuffer(timestampBuffer, 0, timestampReadbackBuffer, 0, 4 * 8);
                    }

                    // Submit compute + render together with in-flight tracking
                    // Only advance counters when we actually submit (correct accumulation math)
                    const renderCommandBuffer = renderEncoder.finish();
                
                if (webgpuAvailable && device) {
                    // Combine compute and render into single submit
                    const commandBuffers = [];
                    if (computeCommandBuffer) {
                        commandBuffers.push(computeCommandBuffer);
                    }
                    commandBuffers.push(renderCommandBuffer);
                    
                    const submitT0 = performance.now();
                    device.queue.submit(commandBuffers);
                    inFlightFrames++;
                    
                    // Note: Counters were already advanced before updateParams() to ensure
                    // the fragment shader sees the correct invTotal for density normalization.
                    // This prevents over-brightening when frames are submitted.
                    // Display update already handled before updateParams()
                    
                    // Decrement when GPU finishes this work (async, may lag slightly)
                    device.queue.onSubmittedWorkDone().then(() => {
                        // Total queue completion time for this submission (includes any backlog)
                        lastSubmittedGpuMs = performance.now() - submitT0;
                        inFlightFrames = Math.max(0, inFlightFrames - 1);
                    }).catch(() => {
                        // Ignore errors, but still decrement to prevent stuck state
                        inFlightFrames = Math.max(0, inFlightFrames - 1);
                    });
                }
            });
        } else {
            // Fallback to original try-catch for CPU mode or when renderer not initialized
            try {
                // Get current texture - must be done before creating encoder
                let currentTexture;
                try {
                    currentTexture = context.getCurrentTexture();
                } catch (e) {
                    console.warn('Failed to get current texture:', e);
                    animationFrameId = requestAnimationFrame(frame);
                    return;
                    }
                    
                    const renderEncoder = device.createCommandEncoder();
                    if (canWriteTimestamps) {
                        renderEncoder.writeTimestamp(timestampQuerySet, 2); // Before render pass(es)
                    }
                    
                    // Direct render pass
                    const renderPass = renderEncoder.beginRenderPass({
                        colorAttachments: [{
                            view: currentTexture.createView(),
                            loadOp: 'clear', clearValue: { r: 0, g: 0, b: 0, a: 1 }, storeOp: 'store',
                        }]
                    });
                    renderPass.setPipeline(renderPipeline);
                    renderPass.setBindGroup(0, renderBindGroupActive);
                    renderPass.draw(6);
                    renderPass.end();

                    // Resolve timestamps (compute + render) if supported.
                    if (canWriteTimestamps) {
                        renderEncoder.writeTimestamp(timestampQuerySet, 3); // After render pass(es)
                        renderEncoder.resolveQuerySet(timestampQuerySet, 0, 4, timestampBuffer, 0);
                        renderEncoder.copyBufferToBuffer(timestampBuffer, 0, timestampReadbackBuffer, 0, 4 * 8);
                    }

                    // Submit compute + render together with in-flight tracking
                    // Only advance counters when we actually submit (correct accumulation math)
                    const renderCommandBuffer = renderEncoder.finish();
                
                if (webgpuAvailable && device) {
                    // Combine compute and render into single submit
                    const commandBuffers = [];
                    if (computeCommandBuffer) {
                        commandBuffers.push(computeCommandBuffer);
                    }
                    commandBuffers.push(renderCommandBuffer);
                    
                    const submitT0 = performance.now();
                    device.queue.submit(commandBuffers);
                    inFlightFrames++;
                    
                    // Note: Counters were already advanced before updateParams() to ensure
                    // the fragment shader sees the correct invTotal for density normalization.
                    // This prevents over-brightening when frames are submitted.
                    // Display update already handled before updateParams()
                    
                    // Decrement when GPU finishes this work (async, may lag slightly)
                    device.queue.onSubmittedWorkDone().then(() => {
                        // Total queue completion time for this submission (includes any backlog)
                        lastSubmittedGpuMs = performance.now() - submitT0;
                        inFlightFrames = Math.max(0, inFlightFrames - 1);
                    }).catch(() => {
                        // Ignore errors, but still decrement to prevent stuck state
                        inFlightFrames = Math.max(0, inFlightFrames - 1);
                    });
                }
            } catch (error) {
                console.error('Render error:', error);
                // Continue frame loop even if render fails
            }
        }
        
        // If frame time was very slow, yield to UI thread to prevent freezing
        if (dt > 100.0) {
            // Very slow frame (>100ms), yield control briefly
            setTimeout(() => {
                animationFrameId = requestAnimationFrame(frame);
            }, 5);
            return;
        }
        
            // Read back timestamps for adaptive scaling (use for next frame)
            // NOTE: We only attempt this if we actually wrote timestamps via commandEncoder.writeTimestamp.
            if (enableTimestamps && timestampQuerySet && timestampBuffer && timestampReadbackBuffer &&
                typeof device.createCommandEncoder().writeTimestamp === 'function') {
                timestampReadbackBuffer.mapAsync(GPUMapMode.READ).then(() => {
                    const times = new BigUint64Array(timestampReadbackBuffer.getMappedRange());
                    const computeTimeNs = Number(times[1] - times[0]);
                    const renderTimeNs = Number(times[3] - times[2]);
                    const totalTimeNs = Number(times[3] - times[0]);
                    // Store for use in next frame's adaptive logic
                    lastComputeTimeNs = computeTimeNs;
                    lastTotalTimeNs = totalTimeNs;
                    
                    // Optional: log performance info (debug only)
                    if (GPU_TIMING_DEBUG_LOGGING) {
                        console.log(`Performance: Compute=${(computeTimeNs/1e6).toFixed(2)}ms, Render=${(renderTimeNs/1e6).toFixed(2)}ms, Total=${(totalTimeNs/1e6).toFixed(2)}ms`);
                    }
                    timestampReadbackBuffer.unmap();
                }).catch(() => {
                    // Ignore errors (may happen if buffer is still in use)
                    // Fall back to wall time for this frame
                    lastComputeTimeNs = null;
                    lastTotalTimeNs = null;
                });
            }
        
        // During interaction, yield to allow input events to process
        // Use queueMicrotask for true yield without delay (setTimeout has ~4ms minimum)
        if (isInteracting) {
            await new Promise(resolve => queueMicrotask(resolve));
        }
        
        animationFrameId = requestAnimationFrame(frame);
    }

    function startFrame() {
        // Dispatch to appropriate render loop based on WebGPU availability
        if (!webgpuAvailable) {
            // CPU mode: use dedicated CPU loop (setTimeout-based, yields frequently)
            startCpuRenderLoop();
        } else {
            // GPU mode: use requestAnimationFrame loop
            animationFrameId = requestAnimationFrame(frame);
        }
    }

    // Start attract mode by default
    state.isLerping = false;
    state.lerpProgress = 1;
    startAttractMode();
}

// Initialize immediately (no external config override).
init();
