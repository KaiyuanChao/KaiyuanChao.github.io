/**
 * @fileoverview Canvas rendering and accumulation for RandomNumberLab
 * 
 * ARCHITECTURE ROLE: Presentation layer - visual output and accumulation
 * ┌─────────────────────────────────────────────────────────────────┐
 * │  visualization.js                                               │
 * │    ← config.js: Constants (CONFIG, SEQUENCE_TYPES)             │
 * │    ← state.js: Global state (accumulation buffers, zoom)     │
 * │    ← main.js: Called after sequence generation                 │
 * └─────────────────────────────────────────────────────────────────┘
 * 
 * EXPORTS:
 * - visualize: Main rendering function (additive accumulation)
 * - clearCanvas: Reset accumulation and clear display
 * - redrawAtZoom: Redraw at current zoom level
 * - extractViewport: Extract viewport from full-resolution buffer
 * 
 * FEATURES:
 * - Progressive accumulation with configurable intensity
 * - Zoom levels (1x, 2x, 4x) with viewport extraction
 * - Full-resolution accumulation buffer (always at canvas size)
 * - Automatic canvas resizing with buffer management
 */

import { CONFIG, SEQUENCE_TYPES, clamp01, getElementByIdStrict } from './config.js';
import { state } from './state.js';

export function extractViewport(fullWidth, fullHeight, displayWidth, displayHeight, zoom) {
    const fullData = state.baseAccumulationImageData.data;
    const displayData = state.accumulationImageData.data;
    
    // Calculate viewport size in full resolution coordinates
    // 1x: shows full bitmap (fullWidth x fullHeight)
    // 2x: shows top-left quadrant (fullWidth/2 x fullHeight/2)
    // 4x: shows top-left 1/16th (fullWidth/4 x fullHeight/4)
    const viewportWidth = Math.floor(fullWidth / zoom);
    const viewportHeight = Math.floor(fullHeight / zoom);
    
    // Always show top-left viewport (starting at 0,0 in full bitmap)
    // Scale up the viewport pixels to fill display (nearest-neighbor upscaling)
    for (let y = 0; y < displayHeight; y++) {
        const viewportY = Math.floor(y / zoom);
        for (let x = 0; x < displayWidth; x++) {
            const viewportX = Math.floor(x / zoom);
            
            if (viewportX < viewportWidth && viewportY < viewportHeight) {
                // Source pixel in full bitmap (top-left viewport)
                const fullIdx = (viewportY * fullWidth + viewportX) * 4;
                // Destination pixel in display (scaled up)
                const displayIdx = (y * displayWidth + x) * 4;
                
                displayData[displayIdx] = fullData[fullIdx];
                displayData[displayIdx + 1] = fullData[fullIdx + 1];
                displayData[displayIdx + 2] = fullData[fullIdx + 2];
                displayData[displayIdx + 3] = fullData[fullIdx + 3];
            }
        }
    }
}

export function redrawAtZoom() {
    const canvas = getElementByIdStrict('canvas', 'canvas');
    const width = canvas.width || canvas.clientWidth || 800;
    const height = canvas.height || canvas.clientHeight || 400;
    const zoom = state.zoom || 1;
    const fullWidth = width;
    const fullHeight = height;
    
    if (state.baseAccumulationImageData && state.accumulationImageData) {
        extractViewport(fullWidth, fullHeight, width, height, zoom);
        const ctx = canvas.getContext('2d');
        ctx.putImageData(state.accumulationImageData, 0, 0);
        
        // Border
        ctx.strokeStyle = '#2d3e52';
        ctx.lineWidth = 1;
        ctx.strokeRect(0, 0, width, height);
    }
}

export function clearCanvas() {
    // Clear the canvas and reset accumulation, but DON'T stop continuous generation
    // ImageData buffers are automatically garbage collected when references are cleared
    
    // Reset accumulation state
    state.frameCount = 0;
    state.accumulationImageData = null;
    state.baseAccumulationImageData = null;
    
    // Reset exact uniform permutation
    if (state.sequenceType === SEQUENCE_TYPES.EXACT) {
        if (state.useCPU) {
            state.exactPermutation = null; // CPU: Destroy array to force fresh shuffle
        } else {
            // GPU: Signal main.js to destroy and recreate the permutation buffer
            // This ensures we get a FRESH random pattern, matching CPU behavior
            state.forcePermutationReset = true;
        }
        // Reset offset for both CPU and GPU to ensure equivalence
        state.offset = 0;
    } else if (state.sequenceType === SEQUENCE_TYPES.R2) {
        // R2 sequence is optimal when starting from index 0
        // Large offsets cause sparse rectangular grid patterns
        // Reset offset to maintain optimal low-discrepancy properties
        state.offset = 0;
    }
    
    // Clear the canvas
    const canvas = getElementByIdStrict('canvas', 'canvas');
    const ctx = canvas.getContext('2d');
    const width = canvas.width || canvas.clientWidth || 800;
    const height = canvas.height || canvas.clientHeight || 400;
    
    ctx.fillStyle = 'rgba(15, 20, 25, 1.0)';
    ctx.fillRect(0, 0, width, height);
    
    // Border
    ctx.strokeStyle = '#2d3e52';
    ctx.lineWidth = 1;
    ctx.strokeRect(0, 0, width, height);
    
    // Note: We don't stop continuous generation or reset offset
    // The generation will continue and start accumulating from the current offset
}

export function visualize(xData, yData, accumulate = false) {
    if (!xData || xData.length === 0) {
        return;
    }
    
    // Data guard: validate data before rendering to prevent canvas blanking on invalid data
    if (!xData.every(v => isFinite(v) && !isNaN(v))) {
        // Invalid data detected - skip rendering
        const canvas = getElementByIdStrict('canvas', 'canvas');
        const ctx = canvas.getContext('2d');
        ctx.fillStyle = '#ff0000';
        ctx.font = '16px monospace';
        ctx.fillText('Invalid data detected (NaN/Infinity)', 10, 30);
        return;
    }
    
    // STRICT: Validate array lengths match to prevent silent index overrun
    if (!yData || yData.length !== xData.length) {
        console.error(`visualize: xData.length=${xData.length} !== yData.length=${yData?.length}`);
        return;
    }
    
    const canvas = getElementByIdStrict('canvas', 'canvas');
    // Set canvas dimensions based on display size
    const displayWidth = canvas.clientWidth || 800;
    const displayHeight = canvas.clientHeight || 400;
    
    // Use centralized config instead of local constant
    const cappedWidth = Math.min(Math.max(displayWidth, CONFIG.MIN_CANVAS_SIZE), CONFIG.MAX_CANVAS_SIZE);
    const cappedHeight = Math.min(Math.max(displayHeight, CONFIG.MIN_CANVAS_SIZE), CONFIG.MAX_CANVAS_SIZE);
    
    // Ensure minimum dimensions
    const width = cappedWidth;
    const height = cappedHeight;
    
    if (canvas.width !== width || canvas.height !== height) {
        // ImageData buffers are automatically garbage collected when references are cleared
        canvas.width = width;
        canvas.height = height;
        // Reset accumulation when canvas resizes
        state.accumulationImageData = null;
        state.baseAccumulationImageData = null;
        state.frameCount = 0;
    }
    
    const ctx = canvas.getContext('2d');
    const zoom = state.zoom || 1;
    
    // Always accumulate at full resolution (1x equivalent = canvas size)
    // Zoom levels just show different viewports into this full bitmap
    const fullWidth = width;
    const fullHeight = height;

    // Initialize or get full resolution accumulation buffer
    if (!state.baseAccumulationImageData || !accumulate) {
        // Initialize full resolution accumulation buffer with dark background
        state.baseAccumulationImageData = ctx.createImageData(fullWidth, fullHeight);
        const initData = state.baseAccumulationImageData.data;
        // Initialize with dark background color
        for (let i = 0; i < initData.length; i += 4) {
            initData[i] = CONFIG.COLORS.background[0];     // R
            initData[i + 1] = CONFIG.COLORS.background[1]; // G
            initData[i + 2] = CONFIG.COLORS.background[2]; // B
            initData[i + 3] = 255; // A (fully opaque)
        }
        state.frameCount = 0;
    }
    
    // Create display buffer at current zoom level (for viewport extraction)
    if (!state.accumulationImageData || state.accumulationImageData.width !== width || state.accumulationImageData.height !== height) {
        state.accumulationImageData = ctx.createImageData(width, height);
    }
    
    // Work with full resolution accumulation buffer (always at 1x = canvas size)
    const fullData = state.baseAccumulationImageData.data;
    state.frameCount++;
    
    // Simple additive accumulation with very small increments to prevent saturation
    // Use a small constant that allows for long accumulation
    const intensityPerHit = CONFIG.INTENSITY_PER_HIT;
    
    // Accumulate points in full resolution buffer (always at canvas size)
    // Optimized hot loop: use for-loop with inline clamp to avoid function call overhead
    const len = xData.length;
    const pointR = CONFIG.COLORS.point[0];
    const pointG = CONFIG.COLORS.point[1];
    const pointB = CONFIG.COLORS.point[2];
    for (let i = 0; i < len; i++) {
        // Inline clamp: (x < 0 ? 0 : x > 1 ? 1 : x) is faster than function call
        const x = xData[i];
        const y = yData[i];
        const clampedX = x < 0 ? 0 : x > 1 ? 1 : x;
        const clampedY = y < 0 ? 0 : y > 1 ? 1 : y;
        
        // Map to full resolution pixel coordinates (1x = full canvas)
        const px = Math.floor(clampedX * fullWidth) | 0;  // |0 is faster than Math.floor for positive numbers
        const py = Math.floor(clampedY * fullHeight) | 0;
        
        if (px >= 0 && px < fullWidth && py >= 0 && py < fullHeight) {
            const idx = (py * fullWidth + px) * 4;
            
            // Additive color accumulation (cyan/teal) with small constant intensity
            fullData[idx] = Math.min(255, fullData[idx] + intensityPerHit * pointR);     // R: cyan component
            fullData[idx + 1] = Math.min(255, fullData[idx + 1] + intensityPerHit * pointG); // G: cyan component
            fullData[idx + 2] = Math.min(255, fullData[idx + 2] + intensityPerHit * pointB); // B: cyan component
            fullData[idx + 3] = 255; // Alpha (fully opaque)
        }
    }
    
    // Extract viewport from full buffer based on zoom level
    extractViewport(fullWidth, fullHeight, width, height, zoom);
    
    // Draw viewport buffer
    ctx.putImageData(state.accumulationImageData, 0, 0);
    
    // Debug: draw a test point if no valid data
    // Check for valid data: non-empty, no NaN/Infinity, and at least one non-zero value
    const hasValidData = xData.length > 0 && 
                         xData.every(v => isFinite(v) && !isNaN(v)) &&
                         xData.some(v => v !== 0);
    if (!hasValidData) {
        ctx.fillStyle = '#ff0000';
        ctx.font = '16px monospace';
        const hasNaN = xData.some(v => isNaN(v));
        const hasInf = xData.some(v => !isFinite(v));
        const allZero = xData.length > 0 && xData.every(v => v === 0);
        let msg = 'No valid data - check console';
        if (hasNaN) msg += ' (NaN detected)';
        else if (hasInf) msg += ' (Infinity detected)';
        else if (allZero) msg += ' (all zeros)';
        else if (xData.length === 0) msg += ' (empty array)';
        ctx.fillText(msg, 10, 30);
    }

    // Border
    ctx.strokeStyle = '#2d3e52';
    ctx.lineWidth = 1;
    ctx.strokeRect(0, 0, width, height);
}
