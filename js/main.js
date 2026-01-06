/**
 * @fileoverview Main application controller for RandomNumberLab
 * 
 * ARCHITECTURE ROLE: Application layer - orchestrates all modules
 * ┌─────────────────────────────────────────────────────────────────┐
 * │  main.js (this file)                                            │
 * │    ├─→ config.js: Constants, enums, utilities                   │
 * │    ├─→ state.js: Global application state                       │
 * │    ├─→ gpu.js: GPU buffer creation                              │
 * │    ├─→ cpu-generators.js: CPU fallback RNG implementations      │
 * │    ├─→ visualization.js: Canvas rendering and accumulation      │
 * │    └─→ statistics.js: Statistical analysis                      │
 * └─────────────────────────────────────────────────────────────────┘
 * 
 * RESPONSIBILITIES:
 * - WebGPU initialization and device management
 * - Sequence generation (GPU/CPU routing)
 * - UI event handling and state updates
 * - Continuous generation loop management
 * - Buffer pooling and resource cleanup
 * 
 * DATA FLOW:
 * User Input → state.sequenceType → generateSequence() → 
 *   [GPU/CPU path] → samples array → visualize() → canvas
 * 
 * Compatibility: Chrome 113+, Firefox 141+, Safari 18+, Edge 113+
 * Backend: Direct3D 12 (Windows), Metal (macOS/iOS), Vulkan (Linux/Android)
 */

import { CONFIG, SEQUENCE_TYPES, SEQUENCE_TO_SHADER_MODE, RNG_SHADER_MODE, RNG_EXPLANATIONS, getElementByIdStrict, R2_CONSTANTS, PARAMS_BUFFER_SIZE, PARAMS_LAYOUT, UINT32_MAX_PLUS_ONE } from './config.js';
import { state } from './state.js';
import { createDirectionsBuffer, createScrambleSeedsBuffer, createPixelPermutationBuffer } from './gpu.js';
import {
    generateSobolSamples_CPU,
    generateLCG_CPU,
    generateXorShift_CPU,
    generatePCG_CPU,
    generateR2_CPU,
    generateExactUniformSamples_CPU,
    generatePixelPermutation
} from './cpu-generators.js';
import {
    computeCorrelation,
    computeDiscrepancy,
    computeChiSquared,
    computeCoverage,
    computeEfficiency,
    formatCorrelation,
    formatDiscrepancy,
    formatPValue,
    resetStatisticsCache
} from './statistics.js';
import { visualize, clearCanvas, redrawAtZoom, extractViewport } from './visualization.js';

/**
 * Register a GPU resource for automatic cleanup during device loss
 * @param {GPUBuffer|GPUComputePipeline|GPUBindGroup|GPUShaderModule} resource - GPU resource to track
 * @returns {*} The resource (for chaining)
 */
function registerGPUResource(resource) {
    if (!resource || !state.gpuResources) return resource;
    state.gpuResources.add(resource);
    return resource;
}

/**
 * Clean up all registered GPU resources
 * Called during device loss events to prevent memory leaks
 */
function cleanupGPUResources() {
    if (!state.gpuResources || state.gpuResources.size === 0) return;
    
    console.log(`Cleaning up ${state.gpuResources.size} GPU resources`);
    state.gpuResources.forEach(resource => {
        try {
            if (resource && typeof resource.destroy === 'function') {
                resource.destroy();
            } else if (resource && typeof resource.release === 'function') {
                resource.release();
            }
        } catch (e) {
            console.warn('Resource cleanup error:', e);
        }
    });
    state.gpuResources.clear();
}

// Initialize
export async function init() {
    try {
        const statusEl = document.getElementById("browserStatus");
        statusEl.textContent = "Detecting WebGPU support...";

        if (!navigator.gpu) {
            showBrowserWarning(
                "WebGPU not supported. " +
                "Try Chrome 113+, Edge 113+, or Safari 17+. " +
                "On older browsers, CPU mode will be used automatically."
            );
            state.useCPU = true;
            state.webgpuAvailable = false;
            setupEventListeners();
            await generateSequence();
            return;
        }

        const adapter = await navigator.gpu.requestAdapter();
        if (!adapter) {
            showBrowserWarning(
                "No WebGPU adapter found. " +
                "Your GPU may not support WebGPU, or it may be disabled. " +
                "Check chrome://gpu for details. Using CPU fallback."
            );
            state.useCPU = true;
            state.webgpuAvailable = false;
            setupEventListeners();
            await generateSequence();
            return;
        }

        state.gpuDevice = await adapter.requestDevice();
        
        // Load shader from external file
        let shaderCode;
        try {
            const response = await fetch('shaders/sobol.wgsl');
            if (!response.ok) {
                throw new Error(`Failed to load shader: ${response.statusText}`);
            }
            shaderCode = await response.text();
        } catch (error) {
            console.warn(`Shader load failed: ${error.message}, CPU modes only`);
            state.webgpuAvailable = false;
            updateModeAvailability();
            // Don't return — proceed with CPU-only operation
            setupEventListeners();
            updateGenerateButtonState();
            await generateSequence();  // Generate with CPU fallback
            return;
        }
        
        // Create shader module
        const shaderModule = registerGPUResource(state.gpuDevice.createShaderModule({
            code: shaderCode
        }));
        
        // Validate shader compilation
        const compilationInfo = await shaderModule.getCompilationInfo();
        for (const message of compilationInfo.messages) {
            if (message.type === 'error') {
                console.error('Shader error:', message.message);
                showBrowserWarning('Shader compilation failed: ' + message.message);
                return;
            }
        }
        
        // Create compute pipeline
        state.computePipeline = registerGPUResource(state.gpuDevice.createComputePipeline({
            layout: 'auto',
            compute: {
                module: shaderModule,
                entryPoint: 'main'
            }
        }));
        
        // Create static buffers once (hoisted from generateSequence)
        state.directionsBuffer = registerGPUResource(createDirectionsBuffer());
        state.scrambleSeedsBuffer = registerGPUResource(createScrambleSeedsBuffer());
        
        // Create pixel permutation buffer for Exact Uniform mode
        // Use default seed (12345) for initial creation - will be randomized on clear
        const canvas = getElementByIdStrict('canvas', 'canvas');
        const { w: canvasWidth, h: canvasHeight } = ensureCanvasResolution(canvas, CONFIG.MIN_CANVAS_SIZE);
        state.pixelPermutationBuffer = registerGPUResource(createPixelPermutationBuffer(canvasWidth, canvasHeight));
        state.pixelPermutationDims = [canvasWidth, canvasHeight];
        
        // Setup device loss recovery
        state.gpuDevice.lost.then((info) => {
            console.error('WebGPU device lost:', info.message);
            cleanupGPUResources();
            state.gpuDevice = null;
            state.computePipeline = null;
            state.directionsBuffer = null;
            state.scrambleSeedsBuffer = null;
            state.pixelPermutationBuffer = null;
            state.pixelPermutationDims = null;
            state.gpuBufferPool = null; // Clear buffer pool on device loss
            
            // Attempt recovery
            showBrowserWarning('GPU device lost, attempting recovery...');
            setTimeout(init, 1000);
        });
        
        updateBrowserStatus("WebGPU supported");
        setupEventListeners();
        updateGenerateButtonState(); // Set initial button state
        updateGpuCpuDescription(); // Set initial description
        updateTechDetails(); // Set initial tech details
        
        // Setup periodic statistics cache reset
        if (!state.statisticsCacheInterval) {
            state.statisticsCacheInterval = setInterval(() => {
                resetStatisticsCache();
            }, 5000);
        }
        
        // Setup ResizeObserver for responsive canvas
        if (canvas && !state.resizeObserver) {
            state.resizeObserver = new ResizeObserver(entries => {
                for (const entry of entries) {
                    if (entry.target.id === 'canvas') {
                        const { width, height } = entry.contentRect;
                        if (width > 0 && height > 0) {
                            // Debounce resize handling
                            clearTimeout(state.resizeTimeout);
                            state.resizeTimeout = setTimeout(() => {
                                state.accumulationImageData = null;
                                state.baseAccumulationImageData = null;
                                if (state.data) {
                                    visualize(
                                        state.data.map(p => p[0] || 0),
                                        state.data.map(p => p[1] || 0),
                                        false
                                    );
                                }
                            }, 100);
                        }
                    }
                }
            });
            state.resizeObserver.observe(canvas);
        }
        
        await generateSequence();
    } catch (error) {
        console.error('WebGPU init error:', error);
        showBrowserWarning(`WebGPU initialization error: ${error.message}`);
    }
}

export function showBrowserWarning(message) {
    const status = document.getElementById('browserStatus');
    status.textContent = message + ' — All modes available via CPU';
    status.className = 'browser-info unsupported';
    state.webgpuAvailable = false;
    state.useCPU = true;  // Force CPU mode
    updateModeAvailability();
    
    // Still allow generation — CPU works for everything
    setupEventListeners();
    updateGenerateButtonState();
    // Use setTimeout to defer execution since this function isn't async
    setTimeout(() => generateSequence(), 0);
}

export function updateBrowserStatus(message) {
    const status = document.getElementById('browserStatus');
    status.textContent = message;
    status.className = 'browser-info supported';
    state.webgpuAvailable = true;
    updateModeAvailability();
}

export function updateGpuCpuDescription() {
    const desc = document.getElementById('gpuCpuDescription');
    if (!desc) return;
    
    const descriptions = {
        [SEQUENCE_TYPES.SOBOL]: 'GPU: parallel Gray code | CPU: sequential loop',
        [SEQUENCE_TYPES.SCRAMBLED]: 'GPU: parallel + hash scramble | CPU: sequential + LCG scramble',
        [SEQUENCE_TYPES.EXACT]: 'GPU: hash-based permutation | CPU: Fisher-Yates shuffle',
        [SEQUENCE_TYPES.LCG]: 'GPU: parallel LCG | CPU: sequential LCG',
        [SEQUENCE_TYPES.XORSHIFT]: 'GPU: parallel xorshift32 | CPU: sequential xorshift32',
        [SEQUENCE_TYPES.PCG]: 'GPU: parallel PCG-XSH-RS | CPU: sequential PCG-XSH-RS',
        [SEQUENCE_TYPES.R2]: 'GPU: parallel plastic constant | CPU: sequential plastic constant'
    };
    
    desc.textContent = descriptions[state.sequenceType] || 'Compare GPU vs CPU performance';
}

/**
 * Update the tech-details panel based on current RNG method
 */
export function updateTechDetails() {
    const info = RNG_EXPLANATIONS[state.sequenceType];
    if (!info) return;
    
    const titleEl = document.querySelector('.tech-details h2');
    if (titleEl) titleEl.textContent = info.title;
    
    const descEl = document.querySelector('.tech-details > p:first-of-type');
    if (descEl) descEl.innerHTML = info.description.replace(/\n\s+/g, ' ');
    
    const formulaEl = document.querySelector('.tech-details .formula');
    if (formulaEl) formulaEl.innerHTML = info.formula.split('\n').join('<br>');
    
    // Create or update pros/cons
    let prosConsEl = document.getElementById('prosCons');
    if (!prosConsEl) {
        const techDetails = document.querySelector('.tech-details');
        if (techDetails) {
            prosConsEl = document.createElement('div');
            prosConsEl.id = 'prosCons';
            prosConsEl.className = 'pros-cons';
            techDetails.appendChild(prosConsEl);
        }
    }
    if (prosConsEl) {
        prosConsEl.innerHTML = `
            <div class="pros"><strong>✓</strong> ${info.strengths.join(' · ')}</div>
            <div class="cons"><strong>✗</strong> ${info.weaknesses.join(' · ')}</div>
            <div class="use-case"><strong>Use:</strong> ${info.useCase}</div>
        `;
    }
}


export function updateModeAvailability() {
    const gpuCpuGroup = document.getElementById('gpuCpuToggleGroup');
    const useCPUCb = document.getElementById('useCPU');
    
    if (!state.webgpuAvailable) {
        // Force CPU mode when no WebGPU
        state.useCPU = true;
        if (useCPUCb) {
            useCPUCb.checked = true;
            useCPUCb.disabled = true;  // Can't uncheck — no GPU available
        }
        if (gpuCpuGroup) {
            gpuCpuGroup.classList.add('forced-cpu');
        }
        
        // Update description to indicate forced CPU
        const desc = document.getElementById('gpuCpuDescription');
        if (desc) {
            desc.textContent = 'WebGPU unavailable — using CPU implementation';
        }
    } else {
        // WebGPU available — user can choose
        if (useCPUCb) {
            useCPUCb.disabled = false;
        }
        if (gpuCpuGroup) {
            gpuCpuGroup.classList.remove('forced-cpu');
        }
        updateGpuCpuDescription();
    }
    
    // Never disable mode buttons — all have CPU support
    document.querySelectorAll('.toggle-btn').forEach(btn => {
        btn.disabled = false;
        btn.title = '';
        btn.style.opacity = '1';
    });
}

export function setupEventListeners() {
    // STRICT: Validate input range
    document.getElementById('samples').addEventListener('input', (e) => {
        const val = parseInt(e.target.value, 10);
        if (!isNaN(val) && val >= CONFIG.MIN_SAMPLES && val <= CONFIG.MAX_SAMPLES) {
            state.samples = val;
            document.getElementById('samplesValue').textContent = state.samples;
        }
    });


    document.querySelectorAll('.toggle-btn').forEach(btn => {
        btn.addEventListener('click', (e) => {
            document.querySelectorAll('.toggle-btn').forEach(b => b.classList.remove('active'));
            e.target.classList.add('active');
            
            // STRICT: Ensure dataset.type is valid
            const newType = e.target.dataset.type;
            if (Object.values(SEQUENCE_TYPES).includes(newType)) {
                state.sequenceType = newType;
            } else {
                console.error(`Invalid sequence type: ${newType}`);
                return;
            }
            
            // Clear stale data on mode switch
            state.data = null;
            
            // Update description text based on mode
            updateGpuCpuDescription();
            updateTechDetails();
            
            // Reset and regenerate when switching types
            if (!state.continuousMode) {
                state.offset = 0;
                state.frameCount = 0;
                state.accumulationImageData = null;
                state.baseAccumulationImageData = null;
                generateSequence();
            }
        });
    });
    
    // GPU/CPU toggle handler
    const useCPUCb = document.getElementById('useCPU');
    if (useCPUCb) {
        useCPUCb.addEventListener('change', (e) => {
            state.useCPU = e.target.checked;
            // Reset and regenerate when toggling
            if (!state.continuousMode) {
                state.offset = 0;
                state.frameCount = 0;
                state.accumulationImageData = null;
                state.baseAccumulationImageData = null;
                state.exactPermutation = null;
                generateSequence();
            } else {
                // If in continuous mode, just update button state to reflect current status
                // Don't restart generation - let it continue with new CPU/GPU setting
                updateGenerateButtonState();
            }
        });
    }

    document.getElementById('generateBtn').addEventListener('click', () => {
        if (state.continuousMode && state.continuousRunning) {
            // Stop continuous generation
            stopContinuousGeneration();
            state.continuousMode = false;
            updateGenerateButtonState();
        } else {
            // Start continuous generation
            state.continuousMode = true;
            // Reset accumulation when starting
            state.offset = 0;
            state.frameCount = 0;
            state.accumulationImageData = null;
            state.baseAccumulationImageData = null;
            startContinuousGeneration();
            updateGenerateButtonState();
        }
    });
    
    document.getElementById('clearBtn').addEventListener('click', () => {
        clearCanvas();
    });
    
    // Export PNG functionality
    document.getElementById('exportBtn').addEventListener('click', () => {
        const canvas = getElementByIdStrict('canvas', 'canvas');
        const link = document.createElement('a');
        link.download = `sobol-${state.sequenceType}-${Date.now()}.png`;
        link.href = canvas.toDataURL('image/png');
        link.click();
    });
    
    // Keyboard shortcuts
    document.addEventListener('keydown', (e) => {
        if (e.target.tagName === 'INPUT') return;  // Don't capture in inputs
        
        switch(e.key.toLowerCase()) {
            case ' ':  // Space: toggle generation
                e.preventDefault();
                document.getElementById('generateBtn').click();
                break;
            case 'c':  // C: clear
                clearCanvas();
                break;
            case '1': case '2': case '4':  // Zoom levels
                const zoomBtn = document.querySelector(`.zoom-btn[data-zoom="${e.key}"]`);
                if (zoomBtn) zoomBtn.click();
                break;
            case 'escape':
                stopContinuousGeneration();
                break;
        }
    });
    
    // Zoom button handlers
    document.querySelectorAll('.zoom-btn').forEach(btn => {
        btn.addEventListener('click', (e) => {
            const zoom = parseInt(e.target.dataset.zoom);
            state.zoom = zoom;
            
            // Update button active state
            document.querySelectorAll('.zoom-btn').forEach(b => {
                if (parseInt(b.dataset.zoom) === zoom) {
                    b.classList.add('active');
                } else {
                    b.classList.remove('active');
                }
            });
            
            // Redraw at new zoom level without resetting accumulation
            if (state.data) {
                // Always use dimension 1 for X and dimension 2 for Y
                visualize(state.data.map(p => p[0] || 0), 
                         state.data.map(p => p[1] || 0), 
                         state.continuousMode);
            } else if (state.baseAccumulationImageData) {
                // Just redraw the cached accumulation at new zoom
                redrawAtZoom();
            }
        });
    });
}

export function startContinuousGeneration() {
    state.offset = 0;
    state.frameCount = 0;
    state.accumulationImageData = null;
    state.continuousRunning = true;
    
    function frame() {
        if (!state.continuousRunning) return;
        generateSequence().then(() => {
            if (state.continuousRunning) {
                requestAnimationFrame(frame);
            }
        });
    }
    requestAnimationFrame(frame);
    updateGenerateButtonState();
}

export function stopContinuousGeneration() {
    state.continuousRunning = false;
    state.continuousMode = false;
    updateGenerateButtonState();
}

/**
 * Ensure canvas resolution matches display size or minimum size
 * @param {HTMLCanvasElement} canvas - Canvas element
 * @param {number} minSize - Minimum canvas size
 * @returns {{w: number, h: number}} Canvas width and height
 */
function ensureCanvasResolution(canvas, minSize) {
    const displayW = canvas.clientWidth || canvas.offsetWidth || 800;
    const displayH = canvas.clientHeight || canvas.offsetHeight || 400;
    
    // Clamp to both minSize and MAX_CANVAS_SIZE to prevent thrashing
    const w = Math.min(Math.max(displayW, minSize), CONFIG.MAX_CANVAS_SIZE);
    const h = Math.min(Math.max(displayH, minSize), CONFIG.MAX_CANVAS_SIZE);

    // Only set if different to avoid layout thrashing
    if (canvas.width !== w) canvas.width = w;
    if (canvas.height !== h) canvas.height = h;

    return { w, h };
}

export function updateGenerateButtonState() {
    const btn = document.getElementById('generateBtn');
    if (!btn) return;
    
    const isRunning = state.continuousMode && state.continuousRunning;
    
    // Remove previous state classes
    btn.classList.remove('running', 'stopped');
    
    if (isRunning) {
        btn.textContent = 'Stop';
        btn.classList.add('running');
    } else {
        btn.textContent = 'Generate Sequence';
        btn.classList.add('stopped');
    }
}

export function updateUI(xData, yData) {
    const canvas = getElementByIdStrict('canvas', 'canvas');
    const width = canvas.width || 800;
    const height = canvas.height || 400;
    const totalPoints = state.frameCount * state.samples;
    
    // Compute all statistics
    const coverage = computeCoverage(width, height);
    const efficiency = computeEfficiency(width, height, totalPoints);
    const correlation = computeCorrelation(xData, yData);
    
    // Only compute expensive stats periodically or on small datasets
    const computeExpensive = xData.length < 5000 || state.frameCount % 10 === 0;
    
    // Update displays
    document.getElementById('totalPoints').textContent = totalPoints.toLocaleString();
    document.getElementById('gpuTime').textContent = state.lastGenerationTime.toFixed(2);
    document.getElementById('dataSize').textContent = 
        ((totalPoints * 2 * 4) / 1024).toFixed(1) + ' KB';
    document.getElementById('memoryBuffer').textContent = 
        (state.samples * 2 * 4).toLocaleString() + ' B';
    
    // Update time label based on CPU/GPU mode
    const timeLabel = document.getElementById('timeLabel');
    if (timeLabel) {
        timeLabel.textContent = state.useCPU ? 'CPU Time (ms)' : 'GPU Time (ms)';
    }
    
    // New statistics
    document.getElementById('coverage').textContent = 
        (coverage * 100).toFixed(1) + '%';
    document.getElementById('efficiency').textContent = 
        efficiency > 0 ? efficiency.toFixed(3) : '—';
    document.getElementById('correlation').textContent = 
        formatCorrelation(correlation);
    
    if (computeExpensive) {
        const discrepancy = computeDiscrepancy(xData, yData);
        const chiSqP = computeChiSquared(xData, yData);
        
        document.getElementById('discrepancy').textContent = 
            formatDiscrepancy(discrepancy, xData.length);
        document.getElementById('chisquare').textContent = 
            formatPValue(chiSqP);
    }
    
    // Throughput
    if (state.lastGenerationTime > 0.001) {  // Guard against near-zero (< 1μs)
        const pps = Math.round(state.samples / (state.lastGenerationTime / 1000));
        document.getElementById('throughput').textContent = 
            pps > 1000000 ? (pps/1000000).toFixed(1) + 'M' :
            pps > 1000 ? (pps/1000).toFixed(0) + 'K' : 
            String(pps);
    } else {
        document.getElementById('throughput').textContent = '—';  // Instant generation
    }
}

/**
 * Generate Sobol low-discrepancy sequence points on GPU
 * @param {number} numSamples - Number of 2D points to generate
 * @param {number} offset - Starting index in sequence
 * @param {boolean} scrambled - Apply Owen scrambling
 * @returns {Promise<void>}
 */
export async function generateSequence() {
    const btn = document.getElementById('generateBtn');
    const wasDisabled = btn.disabled;
    btn.disabled = true;
    btn.textContent = 'Generating...';
    
    try {
        const startTime = performance.now();
        const numSamples = state.samples;
        const canvas = getElementByIdStrict('canvas', 'canvas');
        const { w: canvasWidth, h: canvasHeight } = ensureCanvasResolution(canvas, CONFIG.MIN_CANVAS_SIZE);
        
        let samples;
        
        // ========================================
        // CPU PATH — Either forced or user choice
        // ========================================
        if (state.useCPU || !state.webgpuAvailable) {
            
            // Switch uses constants now
            switch (state.sequenceType) {
                case SEQUENCE_TYPES.SOBOL:
                    samples = generateSobolSamples_CPU(numSamples, state.offset, false, canvasWidth, canvasHeight);
                    break;
                case SEQUENCE_TYPES.SCRAMBLED:
                    samples = generateSobolSamples_CPU(numSamples, state.offset, true, canvasWidth, canvasHeight);
                    break;
                case SEQUENCE_TYPES.EXACT:
                    samples = generateExactUniformSamples_CPU(numSamples, canvasWidth, canvasHeight, state.offset);
                    break;
                case SEQUENCE_TYPES.LCG:
                    samples = generateLCG_CPU(numSamples, state.offset, CONFIG.PRNG_SEEDS.lcg[0]);
                    break;
                case SEQUENCE_TYPES.XORSHIFT:
                    samples = generateXorShift_CPU(numSamples, state.offset, CONFIG.PRNG_SEEDS.lcg[0]);
                    break;
                case SEQUENCE_TYPES.PCG:
                    samples = generatePCG_CPU(numSamples, state.offset, CONFIG.PRNG_SEEDS.lcg[0]);
                    break;
                case SEQUENCE_TYPES.R2:
                    // CPU uses f64, so it can handle large offsets without precision loss
                    // No wrapping needed - pass offset directly
                    samples = generateR2_CPU(numSamples, state.offset);
                    break;
                default:
                    // Exhaustive check: all SEQUENCE_TYPES values should be handled above
                    // If this fires, a new sequence type was added but not implemented
                    console.error(`Unhandled sequence type: ${state.sequenceType}`);
                    if (state.verbose) {
                        throw new Error(`Unhandled sequence type: ${state.sequenceType}`);
                    }
                    // Production fallback to prevent crash
                    samples = generateSobolSamples_CPU(numSamples, state.offset, false, canvasWidth, canvasHeight);
            }
            
            // Reuse pre-allocated buffers to reduce GC pressure
            if (!state.xDataBuffer || state.xDataBuffer.length !== numSamples) {
                state.xDataBuffer = new Float32Array(numSamples);
                state.yDataBuffer = new Float32Array(numSamples);
            }
            for (let i = 0; i < numSamples; i++) {
                state.xDataBuffer[i] = samples[i][0];
                state.yDataBuffer[i] = samples[i][1];
            }
            const xData = state.xDataBuffer;
            const yData = state.yDataBuffer;
            
            state.lastGenerationTime = performance.now() - startTime;
            state.data = samples;
            
            updateUI(xData, yData);
            visualize(xData, yData, state.continuousMode);
            
            if (state.continuousMode) {
                // Guard: If generation took too long (>200ms), yield to browser before next frame
                // This keeps the UI responsive without modifying user's sample count
                if (state.lastGenerationTime > 200) {
                    await new Promise(resolve => setTimeout(resolve, 0));
                }
                if (state.sequenceType === SEQUENCE_TYPES.EXACT) {
                    const totalPixels = canvasWidth * canvasHeight;
                    state.offset = (state.offset + numSamples) % totalPixels;
                } else if (state.sequenceType === SEQUENCE_TYPES.R2) {
                    // R2: Both CPU (f64) and GPU (CPU-precomputed start) handle large offsets
                    state.offset += numSamples;
                } else {
                    state.offset += numSamples;
                }
            } else {
                state.offset = 0;
                state.frameCount = 0;
            }
            return;
        }
        
        // ========================================
        // GPU PATH — WebGPU available and enabled
        // ========================================
        if (!state.gpuDevice || !state.computePipeline) {
            // Should not reach here if webgpuAvailable is false, but handle gracefully
            state.useCPU = true;
            return generateSequence();
        }
        
        // GPU timing starts here
        const gpuStartTime = performance.now();
        const timings = state.performanceTimings;
        if (timings) timings.bufferSetupStart = performance.now();
        
        // Persistent GPU buffer pooling: reuse buffers when dimensions match
        const outputBufferSize = numSamples * 8; // vec2<f32> = 8 bytes per sample
        const paramsBufferSize = PARAMS_BUFFER_SIZE;
        
        // Check if existing cached buffers match current needs
        if (!state.gpuBufferPool || 
            state.gpuBufferPool.samples !== numSamples || 
            state.gpuBufferPool.width !== canvasWidth ||
            state.gpuBufferPool.height !== canvasHeight) {
            
            // Cleanup old buffers if they exist
            if (state.gpuBufferPool) {
                if (state.gpuBufferPool.output) {
                    state.gpuResources.delete(state.gpuBufferPool.output);
                    state.gpuBufferPool.output.destroy();
                }
                if (state.gpuBufferPool.params) {
                    state.gpuResources.delete(state.gpuBufferPool.params);
                    state.gpuBufferPool.params.destroy();
                }
                if (state.gpuBufferPool.readback) {
                    state.gpuResources.delete(state.gpuBufferPool.readback);
                    state.gpuBufferPool.readback.destroy();
                }
            }
            
            // Allocate new persistent buffers
            state.gpuBufferPool = {
                samples: numSamples,
                width: canvasWidth,
                height: canvasHeight,
                output: registerGPUResource(state.gpuDevice.createBuffer({
                    size: outputBufferSize,
                    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
                })),
                params: registerGPUResource(state.gpuDevice.createBuffer({
                    size: paramsBufferSize,
                    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
                })),
                readback: registerGPUResource(state.gpuDevice.createBuffer({
                    size: outputBufferSize,
                    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
                }))
            };
        }
        
        if (timings) {
            timings.bufferSetupEnd = performance.now();
            timings.dispatchStart = performance.now();
        }
        
        const outputBuffer = state.gpuBufferPool.output;
        
        // Use static buffers (created once in init)
        if (!state.directionsBuffer || !state.scrambleSeedsBuffer) {
            throw new Error('Static buffers not initialized');
        }
        const directionsBuffer = state.directionsBuffer;
        
        // STRICT: Update scramble seeds 
        // Use constants for comparison
        const useScrambling = state.sequenceType === SEQUENCE_TYPES.SCRAMBLED ? 1 : 0;
        const scrambleSeedsArray = new Uint32Array(2);
        
        // Use constants
        if ([SEQUENCE_TYPES.LCG, SEQUENCE_TYPES.XORSHIFT, SEQUENCE_TYPES.PCG].includes(state.sequenceType)) {
            scrambleSeedsArray[0] = CONFIG.PRNG_SEEDS.lcg[0];
            scrambleSeedsArray[1] = CONFIG.PRNG_SEEDS.lcg[1];
        } else {
            scrambleSeedsArray[0] = CONFIG.PRNG_SEEDS.sobol[0];
            scrambleSeedsArray[1] = CONFIG.PRNG_SEEDS.sobol[1];
        }
        state.gpuDevice.queue.writeBuffer(state.scrambleSeedsBuffer, 0, scrambleSeedsArray);
        const scrambleSeedsBuffer = state.scrambleSeedsBuffer;
        
        // OPTIMIZATION: Use pre-calculated map instead of creating object per frame
        const mode = SEQUENCE_TO_SHADER_MODE[state.sequenceType] ?? RNG_SHADER_MODE.SOBOL;
        
        // For exact uniform GPU mode, use pre-created permutation buffer
        let pixelPermutationBuffer = null;
        if (mode === RNG_SHADER_MODE.EXACT) {
            const totalPixels = canvasWidth * canvasHeight;
            
            // Guard: Prevent u32 overflow (max safe: 2^32 - 1 = 4,294,967,295)
            if (totalPixels > 0xFFFFFFFF) {
                console.error(`Canvas too large for Exact Uniform mode: ${canvasWidth}×${canvasHeight}`);
                showBrowserWarning('Canvas dimensions exceed Exact Uniform limits');
                return;
            }
            
            // CHECK for reset signal OR dimension mismatch
            if (state.forcePermutationReset || 
                !state.pixelPermutationBuffer || 
                state.pixelPermutationBuffer.size !== totalPixels * 4 ||
                !state.pixelPermutationDims ||
                state.pixelPermutationDims[0] !== canvasWidth ||
                state.pixelPermutationDims[1] !== canvasHeight) {
                
                // Destroy old buffer if it exists
                if (state.pixelPermutationBuffer) {
                    state.gpuResources.delete(state.pixelPermutationBuffer);
                    state.pixelPermutationBuffer.destroy();
                    state.pixelPermutationBuffer = null;
                }
                
                // GENERATE RANDOM SEED for GPU to ensure unique pattern
                const seed = Math.floor(Math.random() * UINT32_MAX_PLUS_ONE);
                
                // Create new buffer using the existing helper
                // This generates a FRESH permutation, ensuring a new "layer" sequence
                state.pixelPermutationBuffer = registerGPUResource(createPixelPermutationBuffer(canvasWidth, canvasHeight, seed));
                
                // Store dimensions for future checks
                state.pixelPermutationDims = [canvasWidth, canvasHeight];
                
                // Increment version to force BindGroup recreation
                state.permutationVersion = (state.permutationVersion || 0) + 1;
                
                // Reset flags
                state.forcePermutationReset = false;
                state.offset = 0;
            }
            
            pixelPermutationBuffer = state.pixelPermutationBuffer;
        }
        
        // Update params buffer
        // FIX: For R2 sequence, use frameCount-based offset to ensure each frame generates different samples
        // This prevents the same pattern from being drawn repeatedly
        // NOTE: frameCount is incremented in visualize() AFTER generation, so we use frameCount + 1
        // to get the correct offset for the current frame
        let effectiveOffset = state.offset;
        
        // Compute R2 starting point on CPU with f64 precision (if R2 mode)
        let r2StartX = 0.0;
        let r2StartY = 0.0;
        if (mode === RNG_SHADER_MODE.R2) {
            // Compute starting point using f64 precision (JavaScript default)
            // This avoids f32 precision loss from large offset * alpha
            const { ALPHA1: alpha1, ALPHA2: alpha2 } = R2_CONSTANTS;
            
            // Use full f64 precision - no wrapping needed
            let x = (effectiveOffset * alpha1) % 1;
            let y = (effectiveOffset * alpha2) % 1;
            // Ensure positive modulo result
            if (x < 0) x += 1;
            if (y < 0) y += 1;
            r2StartX = x;
            r2StartY = y;
        }
        
        // Write params buffer using PARAMS_LAYOUT offsets
        const paramsData = new ArrayBuffer(PARAMS_BUFFER_SIZE);
        const paramsU32View = new Uint32Array(paramsData, 0, 8);
        const paramsF32View = new Float32Array(paramsData, PARAMS_LAYOUT.r2StartX, 2);
        
        paramsU32View[PARAMS_LAYOUT.numSamples / 4] = numSamples;
        paramsU32View[PARAMS_LAYOUT.numDimensions / 4] = 2; 
        paramsU32View[PARAMS_LAYOUT.offset / 4] = effectiveOffset;
        paramsU32View[PARAMS_LAYOUT.useScrambling / 4] = useScrambling;
        paramsU32View[PARAMS_LAYOUT.mode / 4] = mode;
        paramsU32View[PARAMS_LAYOUT.canvasWidth / 4] = canvasWidth;
        paramsU32View[PARAMS_LAYOUT.canvasHeight / 4] = canvasHeight;
        paramsU32View[PARAMS_LAYOUT._pad / 4] = 0; // Pad
        
        paramsF32View[0] = r2StartX;
        paramsF32View[1] = r2StartY;
        
        state.gpuDevice.queue.writeBuffer(state.gpuBufferPool.params, 0, paramsData);
        const paramsBuffer = state.gpuBufferPool.params;
        const readbackBuffer = state.gpuBufferPool.readback;
        
        // ─────────────────────────────────────────────────────────────────────────────
        // BIND GROUP CACHING STRATEGY
        // ─────────────────────────────────────────────────────────────────────────────
        // Updated to include permutationVersion in the cache key
        // This ensures we create a new bind group when the permutation buffer changes
        // ─────────────────────────────────────────────────────────────────────────────
        const bindGroupVersion = `${numSamples}-${canvasWidth}-${canvasHeight}-${mode}-${pixelPermutationBuffer ? 'exact' : 'normal'}-${state.permutationVersion || 0}`;
        
        // For binding 3: Use pixelPermutationBuffer if available (EXACT mode), otherwise use state.pixelPermutationBuffer
        // which is always initialized in init(). The shader only reads from it in EXACT mode anyway.
        const binding3Buffer = pixelPermutationBuffer || state.pixelPermutationBuffer;
        
        if (!binding3Buffer) {
            console.error('pixelPermutationBuffer not initialized! This should not happen.');
            return; // Safety check
        }
        
        let bindGroup = state.cachedBindGroup;
        if (!bindGroup || state.cachedBindGroupVersion !== bindGroupVersion) {
            // Create new bind group when buffers change
            bindGroup = registerGPUResource(state.gpuDevice.createBindGroup({
                layout: state.computePipeline.getBindGroupLayout(0),
                entries: [
                    { binding: 0, resource: { buffer: outputBuffer } },
                    { binding: 1, resource: { buffer: directionsBuffer } },
                    { binding: 2, resource: { buffer: scrambleSeedsBuffer } },
                    { binding: 3, resource: { buffer: binding3Buffer } }, // Always valid: pixelPermutationBuffer (initialized in init)
                    { binding: 4, resource: { buffer: paramsBuffer } }
                ]
            }));
            state.cachedBindGroup = bindGroup;
            state.cachedBindGroupVersion = bindGroupVersion;
        }
        
        // Dispatch compute shader
        const encoder = state.gpuDevice.createCommandEncoder();
        const pass = encoder.beginComputePass();
        pass.setPipeline(state.computePipeline);
        pass.setBindGroup(0, bindGroup);
        const workgroupCount = Math.ceil(numSamples / CONFIG.WORKGROUP_SIZE);
        pass.dispatchWorkgroups(workgroupCount);
        pass.end();
        
        // Copy output to readback buffer
        encoder.copyBufferToBuffer(outputBuffer, 0, readbackBuffer, 0, outputBuffer.size);
        
        // Submit
        state.gpuDevice.queue.submit([encoder.finish()]);
        
        if (timings) timings.dispatchEnd = performance.now();
        
        // Read back results
        await readbackBuffer.mapAsync(GPUMapMode.READ);
        
        if (timings) {
            timings.readbackEnd = performance.now();
            console.debug('GPU Timings:', {
                bufferSetup: (timings.bufferSetupEnd - timings.bufferSetupStart).toFixed(2) + 'ms',
                dispatch: (timings.dispatchEnd - timings.dispatchStart).toFixed(2) + 'ms',
                readback: (timings.readbackEnd - timings.dispatchEnd).toFixed(2) + 'ms'
            });
        }
        const resultArray = new Float32Array(readbackBuffer.getMappedRange());
        
        // Convert to sequence format
        // IMPORTANT: The shader ALWAYS generates 2D Sobol points (X and Y coordinates)
        // regardless of the state.dimensions setting. For 2D visualization, we always
        // use the X and Y coordinates from the shader output.
        // Reuse pre-allocated buffers to reduce GC pressure
        if (!state.xDataBuffer || state.xDataBuffer.length !== numSamples) {
            state.xDataBuffer = new Float32Array(numSamples);
            state.yDataBuffer = new Float32Array(numSamples);
        }
        
        const sequence = [];
        for (let i = 0; i < numSamples; i++) {
            // Shader always outputs 2D (vec2<f32> = 2 floats per sample)
            const x = resultArray[i * 2];      // Dimension 1 (X coordinate)
            const y = resultArray[i * 2 + 1];  // Dimension 2 (Y coordinate)
            
            // Always use dimension 1 (X) and dimension 2 (Y) from shader
            const displayX = x;  // Dimension 1
            const displayY = y;  // Dimension 2
            
            state.xDataBuffer[i] = displayX;
            state.yDataBuffer[i] = displayY;
            
            // Build sequence array: always 2D (X and Y)
            sequence.push([x, y]);
        }
        
        readbackBuffer.unmap();
        
        // Use pre-allocated buffers
        const xData = state.xDataBuffer;
        const yData = state.yDataBuffer;
        
        // Note: Buffers are now pooled and reused, so no cleanup needed here
        // Only static buffers (directionsBuffer, scrambleSeedsBuffer, pixelPermutationBuffer)
        // and pooled buffers (gpuBufferPool) are cached and reused
        
        state.lastGenerationTime = performance.now() - gpuStartTime;
        state.data = sequence;

        updateUI(xData, yData);
        visualize(xData, yData, state.continuousMode);
        
        // Update offset for next continuous generation
        if (state.continuousMode) {
            // Guard: If generation took too long (>200ms), yield to browser before next frame
            // This keeps the UI responsive without modifying user's sample count
            if (state.lastGenerationTime > 200) {
                await new Promise(resolve => setTimeout(resolve, 0));
            }
            if (state.sequenceType === SEQUENCE_TYPES.EXACT && !state.useCPU) {
                // For exact uniform mode, wrap offset to cycle through permutation
                const totalPixels = canvasWidth * canvasHeight;
                state.offset = (state.offset + numSamples) % totalPixels;
            } else if (state.sequenceType === SEQUENCE_TYPES.R2) {
                // R2: Both CPU (f64) and GPU (CPU-precomputed start) handle large offsets
                state.offset += numSamples;
            } else {
                // For all other methods, increment offset (Sobol and PRNGs can handle large indices)
                state.offset += numSamples;
            }
        } else {
            state.offset = 0;
            state.frameCount = 0;
        }
    } catch (error) {
        console.error('Generation error:', error);
        showBrowserWarning(`Generation failed: ${error.message}`);
    } finally {
        // Always re-enable button and update state based on continuous mode
        // Don't restore wasDisabled - button should be enabled unless there's a specific reason
        btn.disabled = false;
        updateGenerateButtonState();
    }
}
