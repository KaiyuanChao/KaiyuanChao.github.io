/**
 * @fileoverview Statistical analysis functions for RandomNumberLab
 * 
 * ARCHITECTURE ROLE: Analysis layer - statistical computations
 * ┌─────────────────────────────────────────────────────────────────┐
 * │  statistics.js                                                  │
 * │    ← config.js: Constants (CONFIG)                              │
 * │    ← state.js: Global state (frameCount, samples)                │
 * │    → main.js: Called during UI updates                          │
 * └─────────────────────────────────────────────────────────────────┘
 * 
 * EXPORTS:
 * - computeCorrelation: Pearson correlation coefficient
 * - computeDiscrepancy: Star discrepancy (Monte Carlo approximation)
 * - computeChiSquared: Chi-squared uniformity test
 * - computeCoverage: Percentage of unique pixels hit
 * - computeEfficiency: Unique pixels / total points
 * - Formatting functions: formatCorrelation, formatDiscrepancy, formatPValue
 * 
 * OPTIMIZATION:
 * - Caching system prevents redundant calculations
 * - Cache automatically resets every 5 seconds or after 100 hits
 */

import { CONFIG } from './config.js';
import { state } from './state.js';

/**
 * Statistics cache to avoid redundant calculations
 * Tracks data fingerprints and caches expensive computations
 */
const statisticsCache = {
    dataFingerprint: null,
    lastComputed: {},
    cacheHits: 0,
    cacheMisses: 0,
    lastResetTime: Date.now()
};

/**
 * Create lightweight fingerprint based on data characteristics
 * @param {Array<number>} xData - X coordinates
 * @param {Array<number>} yData - Y coordinates
 * @param {number} totalPoints - Total number of points
 * @returns {string} Fingerprint string
 */
function getDataFingerprint(xData, yData, totalPoints) {
    if (!xData || xData.length === 0) return 'empty';
    // Create lightweight fingerprint based on data characteristics
    // Use first point, length, and total points to detect changes
    return `${xData.length}:${totalPoints}:${Math.round(xData[0]*1000)}:${Math.round(yData[0]*1000)}:${Math.round(xData[xData.length-1]*1000)}:${Math.round(yData[yData.length-1]*1000)}`;
}

/**
 * Pearson correlation coefficient between X and Y
 * @param {Array<number>} xData - X coordinates
 * @param {Array<number>} yData - Y coordinates
 * @returns {number} r in [-1, 1], should be ~0 for good RNG
 */
export function computeCorrelation(xData, yData) {
    const n = xData.length;
    if (n < 2) return 0;
    
    let sumX = 0, sumY = 0, sumXY = 0, sumX2 = 0, sumY2 = 0;
    for (let i = 0; i < n; i++) {
        sumX += xData[i];
        sumY += yData[i];
        sumXY += xData[i] * yData[i];
        sumX2 += xData[i] * xData[i];
        sumY2 += yData[i] * yData[i];
    }
    
    const num = n * sumXY - sumX * sumY;
    const den = Math.sqrt((n * sumX2 - sumX * sumX) * (n * sumY2 - sumY * sumY));
    return den === 0 ? 0 : num / den;
}

/**
 * Star discrepancy (Monte Carlo approximation)
 * @param {Array<number>} xData - X coordinates
 * @param {Array<number>} yData - Y coordinates
 * @param {number} numTests - Number of test points
 * @returns {number} D* in [0, 1], lower = better
 */
function computeDiscrepancyImpl(xData, yData, numTests = 200) {
    const n = xData.length;
    if (n < 1) return 1;
    
    let maxDisc = 0;
    
    // Test at random points + grid points
    for (let t = 0; t < numTests; t++) {
        // Mix of random and systematic test points
        const tx = t < numTests/2 ? Math.random() : (t - numTests/2) / (numTests/2);
        const ty = t < numTests/2 ? Math.random() : ((t * 7) % (numTests/2)) / (numTests/2);
        
        let count = 0;
        for (let i = 0; i < n; i++) {
            if (xData[i] < tx && yData[i] < ty) count++;
        }
        
        const empirical = count / n;
        const expected = tx * ty;
        maxDisc = Math.max(maxDisc, Math.abs(empirical - expected));
    }
    
    return maxDisc;
}

/**
 * Cached version of computeDiscrepancy
 * Only recomputes when data actually changes
 * @param {Array<number>} xData - X coordinates
 * @param {Array<number>} yData - Y coordinates
 * @param {number} totalPoints - Total number of points
 * @param {number} numTests - Number of test points
 * @returns {number} D* in [0, 1], lower = better
 */
export function computeDiscrepancy(xData, yData, numTests = 200) {
    const fingerprint = getDataFingerprint(xData, yData, state.frameCount * state.samples);
    if (statisticsCache.dataFingerprint === fingerprint && 
        statisticsCache.lastComputed.discrepancy !== undefined) {
        statisticsCache.cacheHits++;
        return statisticsCache.lastComputed.discrepancy;
    }
    
    statisticsCache.cacheMisses++;
    const result = computeDiscrepancyImpl(xData, yData, numTests);
    statisticsCache.dataFingerprint = fingerprint;
    statisticsCache.lastComputed.discrepancy = result;
    return result;
}

/**
 * Chi-squared uniformity test on a grid (implementation)
 * @param {Array<number>} xData - X coordinates
 * @param {Array<number>} yData - Y coordinates
 * @param {number} gridSize - Grid size (default 10x10)
 * @returns {number} p-value, >0.05 = likely uniform
 */
function computeChiSquaredImpl(xData, yData, gridSize = 10) {
    const n = xData.length;
    if (n < gridSize * gridSize) return 1; // Not enough data
    
    // Count points in each cell
    const cells = new Uint32Array(gridSize * gridSize);
    for (let i = 0; i < n; i++) {
        const cx = Math.min(gridSize - 1, Math.floor(xData[i] * gridSize));
        const cy = Math.min(gridSize - 1, Math.floor(yData[i] * gridSize));
        cells[cy * gridSize + cx]++;
    }
    
    // Expected count per cell
    const expected = n / (gridSize * gridSize);
    
    // Chi-squared statistic
    let chiSq = 0;
    for (let i = 0; i < cells.length; i++) {
        const diff = cells[i] - expected;
        chiSq += (diff * diff) / expected;
    }
    
    // Approximate p-value using chi-squared CDF (df = gridSize²-1)
    // Using Wilson-Hilferty approximation
    const df = gridSize * gridSize - 1;
    const z = Math.pow(chiSq / df, 1/3) - (1 - 2/(9*df));
    const stdZ = Math.sqrt(2/(9*df));
    const pValue = 1 - normalCDF(z / stdZ);
    
    return Math.max(0, Math.min(1, pValue));
}

/**
 * Cached version of computeChiSquared
 * Only recomputes when data actually changes
 * @param {Array<number>} xData - X coordinates
 * @param {Array<number>} yData - Y coordinates
 * @param {number} gridSize - Grid size (default 10x10)
 * @returns {number} p-value, >0.05 = likely uniform
 */
export function computeChiSquared(xData, yData, gridSize = 10) {
    const fingerprint = getDataFingerprint(xData, yData, state.frameCount * state.samples);
    if (statisticsCache.dataFingerprint === fingerprint && 
        statisticsCache.lastComputed.chiSquared !== undefined) {
        statisticsCache.cacheHits++;
        return statisticsCache.lastComputed.chiSquared;
    }
    
    statisticsCache.cacheMisses++;
    const result = computeChiSquaredImpl(xData, yData, gridSize);
    statisticsCache.dataFingerprint = fingerprint;
    statisticsCache.lastComputed.chiSquared = result;
    return result;
}

/**
 * Standard normal CDF approximation
 * @param {number} x - Input value
 * @returns {number} CDF value
 */
function normalCDF(x) {
    const a1 =  0.254829592, a2 = -0.284496736, a3 = 1.421413741;
    const a4 = -1.453152027, a5 = 1.061405429, p = 0.3275911;
    const sign = x < 0 ? -1 : 1;
    x = Math.abs(x) / Math.sqrt(2);
    const t = 1 / (1 + p * x);
    const y = 1 - (((((a5*t + a4)*t) + a3)*t + a2)*t + a1) * t * Math.exp(-x*x);
    return 0.5 * (1 + sign * y);
}

/**
 * Coverage: percentage of unique pixels hit
 * Uses the baseAccumulationImageData to count lit pixels
 * @param {number} canvasWidth - Canvas width
 * @param {number} canvasHeight - Canvas height
 * @returns {number} Coverage percentage (0-1)
 */
export function computeCoverage(canvasWidth, canvasHeight) {
    if (!state.baseAccumulationImageData) return 0;
    
    const data = state.baseAccumulationImageData.data;
    const bgR = CONFIG.COLORS.background[0];
    const bgG = CONFIG.COLORS.background[1];
    const bgB = CONFIG.COLORS.background[2];
    let hitPixels = 0;
    const totalPixels = canvasWidth * canvasHeight;
    
    for (let i = 0; i < data.length; i += 4) {
        // Check if pixel differs from background
        if (data[i] !== bgR || data[i+1] !== bgG || data[i+2] !== bgB) {
            hitPixels++;
        }
    }
    
    return hitPixels / totalPixels;
}

/**
 * Efficiency: unique pixels / total points generated
 * @param {number} canvasWidth - Canvas width
 * @param {number} canvasHeight - Canvas height
 * @param {number} totalPoints - Total points generated
 * @returns {number} Efficiency ratio
 */
export function computeEfficiency(canvasWidth, canvasHeight, totalPoints) {
    if (totalPoints === 0) return 0;
    const coverage = computeCoverage(canvasWidth, canvasHeight);
    const uniquePixels = coverage * canvasWidth * canvasHeight;
    return uniquePixels / totalPoints;
}

/**
 * Format correlation value with indicator
 * @param {number} r - Correlation coefficient
 * @returns {string} Formatted string
 */
export function formatCorrelation(r) {
    const absR = Math.abs(r);
    const indicator = absR < 0.02 ? '✓' : absR < 0.1 ? '~' : '⚠';
    return `${r.toFixed(3)} ${indicator}`;
}

/**
 * Format discrepancy value with rating
 * @param {number} d - Discrepancy value
 * @param {number} n - Number of samples
 * @returns {string} Formatted string
 */
export function formatDiscrepancy(d, n) {
    // Compare to theoretical bounds
    const sobolBound = Math.pow(Math.log2(n + 1), 2) / n;  // O(log²N/N)
    const prngBound = 1 / Math.sqrt(n);                    // O(1/√N)
    
    let rating;
    if (d < sobolBound * 2) rating = '★★★';
    else if (d < prngBound) rating = '★★';
    else if (d < prngBound * 2) rating = '★';
    else rating = '—';
    
    return `${d.toFixed(4)} ${rating}`;
}

/**
 * Format p-value with indicator
 * @param {number} p - P-value
 * @returns {string} Formatted string
 */
export function formatPValue(p) {
    if (p > 0.1) return `${p.toFixed(2)} ✓`;
    if (p > 0.05) return `${p.toFixed(2)} ~`;
    if (p > 0.01) return `${p.toFixed(2)} ⚠`;
    return `${p.toFixed(3)} ✗`;
}

/**
 * Reset statistics cache periodically to prevent stale data
 * Should be called periodically (e.g., every 5 seconds)
 */
export function resetStatisticsCache() {
    const now = Date.now();
    // Reset cache every 5 seconds or if cache hits exceed threshold
    if (now - statisticsCache.lastResetTime > 5000 || statisticsCache.cacheHits > 100) {
        if (statisticsCache.cacheHits > 100 && state.verbose) {
            const total = statisticsCache.cacheHits + statisticsCache.cacheMisses;
            const efficiency = total > 0 ? Math.round(statisticsCache.cacheHits / total * 100) : 0;
            console.log(`Statistics cache efficiency: ${efficiency}% (${statisticsCache.cacheHits} hits, ${statisticsCache.cacheMisses} misses)`);
        }
        statisticsCache.dataFingerprint = null;
        statisticsCache.lastComputed = {};
        statisticsCache.cacheHits = 0;
        statisticsCache.cacheMisses = 0;
        statisticsCache.lastResetTime = now;
    }
}
