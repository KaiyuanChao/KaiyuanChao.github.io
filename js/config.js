/**
 * @fileoverview Central configuration and type definitions for RandomNumberLab
 * 
 * ARCHITECTURE ROLE: Foundation layer - imported by ALL other modules
 * ┌─────────────────────────────────────────────────────────────────┐
 * │  config.js ← state.js ← cpu-generators.js ← gpu.js ← main.js   │
 * │      ↑           ↑              ↑             ↑         ↑      │
 * │      └───────────┴──────────────┴─────────────┴─────────┘      │
 * │                    visualization.js, statistics.js             │
 * └─────────────────────────────────────────────────────────────────┘
 * 
 * EXPORTS:
 * - CONFIG: Runtime constants (samples, canvas, colors)
 * - SEQUENCE_TYPES: String enum for UI/logic ("sobol", "lcg", etc.)
 * - RNG_SHADER_MODE: Integer enum matching shader's mode parameter
 * - SEQUENCE_TO_SHADER_MODE: Maps SEQUENCE_TYPES → RNG_SHADER_MODE
 * - SOBOL_DIRECTIONS: Direction vectors for Sobol sequence generation
 * - Utility functions: clamp, clamp01, getElementByIdStrict
 * 
 * INVARIANTS:
 * - All numeric constants are frozen (Object.freeze)
 * - SEQUENCE_TYPES values must match HTML select option values
 * - RNG_SHADER_MODE values must match shader's mode checks (0-6)
 */

// Configuration constants
export const CONFIG = Object.freeze({
    MIN_SAMPLES: 8,
    MAX_SAMPLES: 8192,
    DEFAULT_SAMPLES: 1024,
    MIN_CANVAS_SIZE: 400,
    MAX_CANVAS_SIZE: 2048, // Prevent memory exhaustion on high-DPI displays
    INTENSITY_PER_HIT: 0.3,
    JITTER_AMOUNT: 0.3,
    WORKGROUP_SIZE: 64,
    PRNG_SEEDS: {
        lcg: [0xDEADBEEF, 0xCAFEBABE],
        sobol: [0x12345678, 0x9ABCDEF0]
    },
    COLORS: {
        background: [15, 20, 25],
        point: [33, 128, 160]  // RGB multipliers
    }
});

// Numeric constants for RNG normalization and hashing
export const UINT32_MAX_PLUS_ONE = 4294967296;  // 2^32, used for u32→[0,1) normalization
export const GOLDEN_RATIO_HASH = 0x9E3779B9;   // Knuth multiplicative hash constant

/**
 * GPU Params buffer layout - MUST match shaders/sobol.wgsl Params struct
 * Total size: 40 bytes (10 × 4-byte fields)
 * 
 * Offset  Type   Name           Description
 * ──────────────────────────────────────────────────────────
 * 0       u32    numSamples     Number of points to generate
 * 4       u32    numDimensions  Always 2 for this application
 * 8       u32    offset         Starting index in sequence
 * 12      u32    useScrambling  1 = apply Cranley-Patterson rotation
 * 16      u32    mode           RNG_SHADER_MODE enum value (0-6)
 * 20      u32    canvasWidth    Output canvas width in pixels
 * 24      u32    canvasHeight   Output canvas height in pixels
 * 28      u32    _pad           Padding for 16-byte alignment
 * 32      f32    r2StartX       R2 sequence starting X (CPU-computed)
 * 36      f32    r2StartY       R2 sequence starting Y (CPU-computed)
 * ──────────────────────────────────────────────────────────
 */
export const PARAMS_BUFFER_SIZE = 40;
export const PARAMS_LAYOUT = Object.freeze({
    numSamples: 0,
    numDimensions: 4,
    offset: 8,
    useScrambling: 12,
    mode: 16,
    canvasWidth: 20,
    canvasHeight: 24,
    _pad: 28,
    r2StartX: 32,
    r2StartY: 36
});

// R2 sequence constants derived from the plastic constant φ₂
// φ₂ is the unique real root of x³ = x + 1, approximately 1.32471795724474602596
// 
// These values MUST match shaders/sobol.wgsl alpha1/alpha2 constants.
// The shader uses f32, so precision is limited to ~7 decimal digits there.
export const R2_CONSTANTS = Object.freeze({
    PHI2: 1.32471795724474602596,        // Plastic constant
    ALPHA1: 0.7548776662466927,           // 1/φ₂ (f64 precision)
    ALPHA2: 0.5698402909980532            // 1/φ₂² (f64 precision)
});

// DEPRECATED: MAX_R2_OFFSET is no longer needed with CPU-precomputed starting point
// The current R2 implementation passes fract(offset * alpha) computed with f64 to the GPU.
// The offset can grow indefinitely since f64 handles it. The wrapping logic is now dead code.
// export const MAX_R2_OFFSET = 8388608;  // 2^23 (8M - matches shader wrap)

/**
 * @typedef {Object} AppConfig
 * @property {number} MIN_SAMPLES
 * @property {number} MAX_SAMPLES
 * @property {number} DEFAULT_SAMPLES
 * @property {number} MIN_CANVAS_SIZE
 * @property {number} MAX_CANVAS_SIZE
 * @property {number} INTENSITY_PER_HIT
 * @property {number} JITTER_AMOUNT
 * @property {number} WORKGROUP_SIZE
 * @property {{lcg: number[], sobol: number[]}} PRNG_SEEDS
 * @property {{background: number[], point: number[]}} COLORS
 */

// STRICT TYPING: Sequence identifiers (string enum for UI/logic)
/** @enum {string} */
export const SEQUENCE_TYPES = Object.freeze({
    SOBOL: 'sobol',
    SCRAMBLED: 'scrambled',
    EXACT: 'exact',
    LCG: 'lcg',
    XORSHIFT: 'xorshift',
    PCG: 'pcg',
    R2: 'r2'
});

// STRICT TYPING: Shader Integer Modes (for GPU compute shader)
/** @enum {number} */
export const RNG_SHADER_MODE = Object.freeze({
    SOBOL: 0,
    SCRAMBLED: 1, // Logic handled via useScrambling flag, but enum reserved
    EXACT: 2,
    LCG: 3,
    XORSHIFT: 4,
    PCG: 5,
    R2: 6
});

// MAPPING: Connects string types to shader modes (Moved from main.js to prevent allocation loop)
export const SEQUENCE_TO_SHADER_MODE = Object.freeze({
    [SEQUENCE_TYPES.SOBOL]: RNG_SHADER_MODE.SOBOL,
    [SEQUENCE_TYPES.SCRAMBLED]: RNG_SHADER_MODE.SOBOL, // Scrambled uses Sobol mode + flag
    [SEQUENCE_TYPES.EXACT]: RNG_SHADER_MODE.EXACT,
    [SEQUENCE_TYPES.LCG]: RNG_SHADER_MODE.LCG,
    [SEQUENCE_TYPES.XORSHIFT]: RNG_SHADER_MODE.XORSHIFT,
    [SEQUENCE_TYPES.PCG]: RNG_SHADER_MODE.PCG,
    [SEQUENCE_TYPES.R2]: RNG_SHADER_MODE.R2
});

// Sobol direction vectors - 32-bit values for 32 bits per dimension
// Standard Sobol: v_i = m_i * 2^(32-i) where bit i (0=LSB, 31=MSB) uses v_i
// Dimension 0: m_i = 1 for all i → v_i = 2^(32-i)  
// Dimension 1: standard Sobol m_i = [1,1,3,1,5,3,3,9,1,3,5,11,9,11,13,15,1,3,9,7,13,11,7,15,5,7,11,3,13,31,5,17]
// Note: Array index [bit] where bit=0 is LSB, so we reverse the typical order
// Sobol direction vectors - Standard Sobol direction numbers
// Array indexed by bit position: bit 0 = LSB, bit 31 = MSB
// Note: In Sobol, direction[0] (LSB) uses the largest value (0x80000000)
export const SOBOL_DIRECTIONS = [
    // Dimension 0 (X): powers of 2, bit 0 (LSB) = 0x80000000, bit 31 (MSB) = 0x00000001
    [0x80000000, 0x40000000, 0x20000000, 0x10000000, 0x08000000, 0x04000000, 0x02000000, 0x01000000,
     0x00800000, 0x00400000, 0x00200000, 0x00100000, 0x00080000, 0x00040000, 0x00020000, 0x00010000,
     0x00008000, 0x00004000, 0x00002000, 0x00001000, 0x00000800, 0x00000400, 0x00000200, 0x00000100,
     0x00000080, 0x00000040, 0x00000020, 0x00000010, 0x00000008, 0x00000004, 0x00000002, 0x00000001],
    // Dimension 1 (Y): standard Sobol direction numbers (from SOBOL_OPTIMIZED_IMPLEMENTATION.wgsl)
    [0x80000000, 0xC0000000, 0xA0000000, 0xF0000000, 0x88000000, 0xCC000000, 0xAA000000, 0xFF000000,
     0x80800000, 0xC0C00000, 0xA0A00000, 0xF0F00000, 0x88880000, 0xCCCC0000, 0xAAAA0000, 0xFFFF0000,
     0x80808000, 0xC0C0C000, 0xA0A0A000, 0xF0F0F000, 0x88888800, 0xCCCCCC00, 0xAAAAAA00, 0xFFFFFF00,
     0x80808080, 0xC0C0C0C0, 0xA0A0A0A0, 0xF0F0F0F0, 0x88888888, 0xCCCCCCCC, 0xAAAAAAAA, 0xFFFFFFFF]
];

// RNG Method Explanations for the tech-details panel
export const RNG_EXPLANATIONS = Object.freeze({
    sobol: {
        title: 'Sobol Sequence',
        category: 'Quasi-Random (Low-Discrepancy)',
        description: `A deterministic low-discrepancy sequence that fills space more uniformly than 
            pseudo-random numbers. Uses Gray code indexing with direction vectors for O(1) generation 
            per sample. Ideal for Monte Carlo integration where it achieves O(log²N/N) convergence 
            vs O(1/√N) for PRNGs.`,
        formula: `gray(n) = n ⊕ (n >> 1)
x_n = ⊕{direction[i] : bit i set in gray(n)}
Complexity: O(1) per sample, fully parallelizable`,
        strengths: ['Excellent space-filling', 'Deterministic', 'Fast convergence for integration'],
        weaknesses: ['Visible lattice structure', 'Not cryptographically secure'],
        useCase: 'Monte Carlo integration, numerical analysis, quasi-random sampling'
    },
    
    scrambled: {
        title: 'Owen-Scrambled Sobol',
        category: 'Randomized Quasi-Random',
        description: `Sobol sequence with Cranley-Patterson rotation applied. Adds random offset 
            (mod 1) to break the visible lattice structure while preserving low-discrepancy 
            properties. Combines benefits of quasi-random uniformity with pseudo-random appearance.`,
        formula: `x_scrambled = fract(x_sobol + hash(seed, index))
Preserves: Low discrepancy O(log²N/N)
Removes: Visible lattice artifacts`,
        strengths: ['Low discrepancy + random appearance', 'Unbiased estimator', 'No visible patterns'],
        weaknesses: ['Slightly slower than pure Sobol', 'Seed-dependent results'],
        useCase: 'Rendering (path tracing), when visual randomness matters'
    },
    
    exact: {
        title: 'Exact Uniform (Permutation)',
        category: 'Perfect Coverage',
        description: `Guarantees every pixel is hit exactly once before any repetition. Uses 
            Fisher-Yates shuffle (CPU) or hash-based permutation (GPU) to create a random 
            ordering of all pixel coordinates. Perfect for coverage-critical applications.`,
        formula: `permutation = Fisher-Yates(0..N-1)
pixel[i] = permutation[i mod N]
Coverage: 100% after N samples (N = width × height)`,
        strengths: ['Perfect coverage guarantee', '100% efficiency until full', 'No clustering or gaps'],
        weaknesses: ['Memory: O(N) for permutation', 'Canvas-size dependent', 'Not useful for integration'],
        useCase: 'Dithering, progressive rendering, coverage testing'
    },
    
    lcg: {
        title: 'Linear Congruential Generator',
        category: 'Classic PRNG',
        description: `The simplest and fastest PRNG: state = (a × state + c) mod m. Uses 
            Numerical Recipes constants (a=1664525, c=1013904223, m=2³²). Each index generates 
            independent X,Y by advancing state twice. Period is 2³² with full-period constants.`,
        formula: `state = 1664525 × state + 1013904223 (mod 2³²)
x = state₁ / 2³²
y = state₂ / 2³²
Period: 2³² (4.29 billion states)`,
        strengths: ['Extremely fast', 'Tiny state (4 bytes)', 'Well-understood mathematically'],
        weaknesses: ['Fails spectral tests', 'Sequential correlation', 'Low bits less random'],
        useCase: 'Quick prototyping, non-critical randomness, educational purposes'
    },
    
    xorshift: {
        title: 'XorShift32 (Marsaglia)',
        category: 'Fast PRNG',
        description: `Marsaglia's xorshift uses only XOR and bit shifts — no multiplication needed. 
            The shift constants (13, 17, 5) were carefully chosen to maximize period and pass basic 
            statistical tests. Often faster than LCG on modern hardware due to no multiply.`,
        formula: `state ^= state << 13
state ^= state >> 17  
state ^= state << 5
Period: 2³² - 1 (zero state forbidden)`,
        strengths: ['Very fast (no multiply)', 'Better than LCG statistically', 'Simple to implement'],
        weaknesses: ['Fails some TestU01 tests', 'Zero state is absorbing', 'Linear in GF(2)'],
        useCase: 'Games, procedural generation, when speed > quality'
    },
    
    pcg: {
        title: 'PCG (Permuted Congruential Generator)',
        category: 'Modern High-Quality PRNG',
        description: `O'Neill's PCG (2014) combines an LCG core with a permutation output function. 
            The XSH-RS variant applies: xorshift-high, then random-rotate. Passes all standard 
            statistical tests (TestU01, PractRand) while remaining competitive in speed.`,
        formula: `state = state × 747796405 + 2891336453
output = permute(old_state)
permute = XSH-RS (xorshift-high, random-shift)
Period: 2³² with excellent statistical quality`,
        strengths: ['Passes TestU01/PractRand', 'Fast generation', 'Excellent quality', 'Small state'],
        weaknesses: ['More complex than XorShift', 'Not cryptographically secure'],
        useCase: 'Simulations, games, any application needing quality randomness'
    },
    
    r2: {
        title: 'R₂ Sequence (Plastic Constant)',
        category: 'Quasi-Random (Additive Recurrence)',
        description: `Uses the plastic constant φ₂ ≈ 1.3247 (the unique real root of x³ = x + 1). 
            The R₂ sequence is the optimal 2D low-discrepancy sequence using additive recurrence — 
            provably better than Halton for 2D, competitive with Sobol, with no lattice artifacts.`,
        formula: `φ₂ = 1.32471795724474602596... (plastic constant)
α₁ = 1/φ₂ ≈ 0.7549
α₂ = 1/φ₂² ≈ 0.5699
x_n = fract(0.5 + n × α₁)
y_n = fract(0.5 + n × α₂)`,
        strengths: ['No lattice artifacts', 'Optimal 2D additive recurrence', 'O(1) trivial computation', 'Infinite sequence'],
        weaknesses: ['Less known than Sobol', 'Optimal specifically for 2D', 'Fixed sequence (no scrambling)'],
        useCase: 'Blue noise sampling, stratification, 2D quasi-Monte Carlo'
    }
});

/**
 * Clamp value to [min, max] range
 * @param {number} value
 * @param {number} min
 * @param {number} max
 * @returns {number}
 */
export const clamp = (value, min, max) => Math.max(min, Math.min(max, value));

/** Clamp to unit interval [0, 1] */
export const clamp01 = (value) => clamp(value, 0, 1);

/**
 * Coerce a JavaScript number to unsigned 32-bit integer.
 * JS bitwise operators produce signed i32; this converts to u32 range [0, 2³²).
 * Equivalent to: value >>> 0
 * 
 * @param {number} value - Value to coerce (typically result of Math.imul or bitwise ops)
 * @returns {number} Unsigned 32-bit integer in range [0, 4294967295]
 */
export const toU32 = (value) => value >>> 0;

/**
 * Get element by ID with strict type checking and null safety
 * @template {keyof HTMLElementTagNameMap} T
 * @param {string} id - DOM element ID
 * @param {T} tagName - Expected HTML tag name
 * @returns {HTMLElementTagNameMap[T]} The element, typed correctly
 * @throws {Error} If element not found
 * @throws {Error} If element tag doesn't match expected tagName
 */
export function getElementByIdStrict(id, tagName) {
    const el = document.getElementById(id);
    if (!el) {
        throw new Error(`Element #${id} not found`);
    }
    if (el.tagName.toLowerCase() !== tagName.toLowerCase()) {
        throw new Error(`#${id} is ${el.tagName}, expected ${tagName}`);
    }
    return /** @type {HTMLElementTagNameMap[T]} */ (el);
}