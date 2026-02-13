# ExoticFractalPlotter

Interactive Peter de Jong attractor renderer using WebGPU compute + fragment shaders, with automatic CPU fallback.

## Requirements

- Node.js 18+ (for `npm` scripts)
- A modern browser with WebGPU support for GPU mode

## Quick Start

This link is up temporarily for one-click testing: https://kaiyuanchao.github.io/ExoticFractalPlotter.html

To run locally:

1. Install dependencies:
   
   ```bash
   npm install
   ```

2. Start local server:
   
   ```bash
   npm start
   ```

3. Open:
   `http://localhost:3000/ExoticFractalPlotter.html`

Notes:

- Serve over `http://` or `https://` (not `file://`), because the app loads module JS and WGSL at runtime.
- `npm start` opens the browser. `npm run serve` starts the same server without auto-open.

## Required Runtime Files

```text
ExoticFractalPlotter.html
js/exotic-fractalplotter.js
shaders/fractal-accumulation.wgsl
```

Optional:

- `favicon.svg`

## How To Use

UI controls:

- `Variable A/B/C/D`: attractor parameters
- `Exposure`, `Gamma`, `Contrast`: tone controls
- `Color Mode`: density or trajectory color methods
- `Rebase`: make current view the new full bounds
- `Reset`: reset view/bounds
- `Randomize`: random parameter exploration
- `Animate`: continuous parameter motion
- `Clear`: clear accumulation
- `Save Screen`: export PNG with embedded state metadata
- `Load Fractal`: load state from a PNG saved by this app

Mouse:

- Drag: pan
- Wheel: zoom toward cursor

Button modifiers:

- `Shift + Rebase`: reset view transform to 1:1 (preserve pixels)
- `Ctrl/Cmd + Rebase`: toggle auto-rebase
- `Shift + Reset`: also reset exposure/gamma/contrast to defaults
- `Ctrl/Cmd + Randomize`: history back
- `Shift + Randomize`: subtle cycle
- `Alt + Randomize`: micro cycle
- `Shift + Animate`: slow animate
- `Ctrl/Cmd + Animate`: ultra-slow animate

## Keyboard Shortcuts

- `Space` or `A`: toggle animate
- `R`: reverse animate direction (when animate is active)
- `G`: cycle RNG mode
- `B`: rebase
- `Shift+B` or `Shift+R`: reset view transform to 1:1 (preserve pixels)
- `Ctrl+S` / `Cmd+S`: new RNG seed (keeps accumulation)
- `F1`: toggle help overlay
- `Esc`: close help overlay
- `Shift+C`: run calibration benchmark (base reproducible seed)
- `Alt+C`: run calibration benchmark (alternate reproducible seed)

## CPU Fallback

- If WebGPU is unavailable, the app automatically runs in CPU mode.
- Force CPU mode manually with:
  `http://localhost:3000/ExoticFractalPlotter.html?forceCPU=true`

## Development

No build step is required.

Useful scripts:

- `npm run lint`
- `npm run lint:fix`
- `npm run test:all`
- `npm run test:all:300k`
- `npm run test:r2`
- `npm run test:sobol`

## Architecture (Short)

- `js/exotic-fractalplotter.js` orchestrates UI, state, WebGPU setup, and CPU fallback.
- `shaders/fractal-accumulation.wgsl` contains:
  - compute entry points for accumulation
  - vertex + fragment entry points for rendering

Core map:

- `x' = sin(a*y) - cos(b*x)`
- `y' = sin(c*x) - cos(d*y)`

## License

MIT (`LICENSE`).
