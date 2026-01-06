# ExoticFractalPlotter

A high-performance WebGPU-based strange attractor fractal renderer featuring real-time progressive accumulation, interactive pan/zoom controls, and multiple coloring modes. Renders the Peter de Jong attractor with millions of iterations per frame using GPU compute shaders.

## Features

- **Real-time Progressive Rendering**: Accumulates fractal points over time with stable averaging
- **WebGPU Acceleration**: Leverages GPU compute shaders for parallel point generation
- **Interactive Controls**: Pan, zoom, and adjust rendering parameters in real-time
- **Multiple Coloring Modes**: Density-based, iteration-based, and custom color palettes
- **Progressive Refinement**: Stable averages with bounded precision, no quantization artifacts
- **View Transform System**: Pan and zoom without re-accumulation
- **Rebase System**: Bake view transforms into fractal parameters for optimal performance
- **Adaptive Performance**: Automatically scales workgroup count based on GPU capabilities
- **Export Functionality**: Save high-resolution renders with embedded metadata

## Requirements

- Modern browser with WebGPU support:
  - Chrome/Edge 113+ (Windows/Linux/Mac)
  - Firefox 110+ (Windows/Linux/Mac)
  - Safari 18+ (Mac/iOS)
- WebGPU-compatible GPU (discrete or integrated)

## Installation

1. Clone or download this repository
2. Install dependencies:
   ```bash
   npm install
   ```
3. Start the development server:
   ```bash
   npm start
   ```
   This will open the application in your default browser at `http://localhost:3000`

Alternatively, use any static file server to serve the `ExoticFractalPlotter.html` file.

## Usage

### Basic Controls

- **Variable A, B, C, D**: Adjust the attractor parameters to explore different fractal patterns
- **Exposure**: Control overall brightness
- **Gamma**: Adjust midtone contrast
- **Contrast**: Fine-tune image contrast
- **Color Mode**: Switch between different coloring methods:
  - Default: Density-based heat map
  - Classic: Velocity-based coloring
  - Exotic/Vibrant/Surprise: RGB channel permutations
  - Iteration Hues: Logarithmic iteration count to color mapping

### Interactive Features

- **Pan**: Click and drag to pan the view
- **Zoom**: Use mouse wheel to zoom in/out
- **Rebase**: Click "Rebase" to bake the current view transform into fractal parameters (improves performance)
- **Reset**: Reset view to original bounds
- **Randomize**: Generate random parameter values
- **Attract Mode**: Automatically cycle through parameter variations
- **Clear**: Reset accumulation and start fresh

### Presets

Select from predefined parameter sets:
- Classic
- Swirl
- Dream
- Chaos
- Spiral
- Vortex

### Export

- **Save Screen**: Save the current viewport as PNG
- **Load Fractal**: Load fractal parameters from PNG metadata

## Architecture

### Compute Shader Pipeline

1. **Point Generation**: Each GPU thread generates 128 iterations of the attractor equation
2. **Atomic Accumulation**: Points are accumulated into density and color buffers using atomic operations
3. **Running Average**: Color values use running average accumulation for stable, bounded precision
4. **Progressive Refinement**: Accumulation continues over time, refining the image quality

### Rendering Pipeline

1. **Fragment Shader**: Reads accumulated density and color buffers
2. **Tone Mapping**: Applies Reinhard tone mapping for HDR compression
3. **Post-processing**: Gamma correction and contrast adjustment
4. **Display**: Renders to screen with view transform applied

### Performance Optimization

- **Adaptive Workgroup Scaling**: Automatically adjusts workgroup count based on GPU performance
- **Stochastic Color Updates**: Reduces atomic contention on hot pixels
- **View Transform Caching**: Pan/zoom operations don't require re-accumulation
- **Frame Budget Management**: Maintains target framerate by adjusting iteration count

## Technical Details

### Shader Files

- `shaders/fractal-accumulation.wgsl`: Main compute and render shaders
- `shaders/downsample.wgsl`: Supersampling downsampling shader

### Key Algorithms

- **Peter de Jong Attractor**: `x' = sin(a·y) - cos(b·x)`, `y' = sin(c·x) - cos(d·y)`
- **Quasi-Random Sampling**: Supports Sobol sequences and R2 sequences for uniform coverage
- **Running Average**: `newAvg = oldAvg + (sample - oldAvg) / count`
- **Reinhard Tone Mapping**: `L_out = L_in / (1 + L_in)` for stable HDR compression

### Browser Compatibility

The renderer uses WebGPU features that require strict WGSL compliance. The shader code is compatible with:
- Chrome/Dawn (lenient validation)
- Firefox/Naga (strict validation)

All atomic operations are inlined to comply with WGSL specification requirements.

## Development

### Project Structure

```
ExoticFractalPlotter.html    # Main application
shaders/
  ├── fractal-accumulation.wgsl  # Compute and render shaders
  └── downsample.wgsl             # Supersampling shader
package.json                      # Dependencies and scripts
```

### Building

No build step required. The application runs directly in the browser.

### Testing

Run test scripts for RNG generators:
```bash
npm run test:all
```

## License

MIT License

Copyright (c) 2025

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Acknowledgments

- WebGPU API specification and browser implementations
- Peter de Jong for the attractor equations
- Sobol sequence research and implementations
- R2 sequence (Martin Roberts) for optimal 2D low-discrepancy sequences
