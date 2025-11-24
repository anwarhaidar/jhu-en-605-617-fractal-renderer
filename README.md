# CUDA Fractal Renderer

High-performance fractal rendering using CUDA with an interactive OpenGL viewer.

**Author:** Anwar Sleiman Haidar  
**Institution:** Johns Hopkins University  
**Program:** MS Artificial Intelligence  
**Course:** EN.605.617 - Introduction to GPU Programming  
**Date:** November 2025  

## Demo

[![Watch the Demo](https://img.shields.io/badge/YouTube-Demo-red?logo=youtube)](https://youtu.be/nSteTBq1gIo)

## Features

- Multiple fractal types: Mandelbrot, Julia, Burning Ship, Tricorn
- Real-time interactive exploration with mouse/keyboard controls
- 6 color schemes with animation support
- Performance benchmarking mode to measure true GPU computational speed
- Both single and double precision support
- Display vs. Compute FPS measurement modes
- Organized output management with automatic directory creation

## Requirements

- NVIDIA GPU with CUDA Compute Capability 5.0+
- CUDA Toolkit 10.0+
- OpenGL 3.3+
- GLFW3 and GLEW libraries

## Installation

### Ubuntu/Debian (Desktop/Server)
```bash
sudo apt-get install nvidia-cuda-toolkit libglew-dev libglfw3-dev
make check-deps  # verify installation
make all
```

### NVIDIA Jetson (Orin, Xavier, etc.)
**Do NOT install nvidia-cuda-toolkit** - CUDA is pre-installed via JetPack.
```bash
sudo apt-get install libglew-dev libglfw3-dev freeglut3-dev
make check-deps  # verify installation
make all
```

## Usage

### Interactive Viewer (Recommended)
```bash
make run-viewer
```

**Controls:**
- **Mouse drag:** Pan view
- **Mouse wheel:** Zoom in/out
- **1-4:** Switch fractal types (Mandelbrot, Julia, Burning Ship, Tricorn)
- **Q/W/E/R/T/Y:** Change color schemes (Grayscale, Rainbow, Fire, Ocean, Psychedelic, Electric)
- **+/-:** Adjust iteration count
- **0:** Set iterations to 64 (minimum/fast)
- **9:** Set iterations to 2048 (maximum/detailed)
- **B:** Toggle benchmark mode (Display FPS vs. Compute FPS)
- **Space:** Toggle animation
- **Arrow keys:** Pan view
- **C:** Center view
- **V:** Reset zoom to 1.0
- **X:** Reset view, colors, and animation (keeps fractal type and iterations)
- **M:** Redisplay menu/controls
- **J/K/I/L:** Adjust Julia parameters (Julia mode only)
- **ESC:** Exit

### Console Renderer
```bash
make run              # Generate sample images to output/ directory
make run-benchmark    # Run performance tests (saves 24 PPM files to output/)
```

**File Organization:**
- All generated PPM files are automatically saved to `output/` directory
- Directory is created automatically on first run
- Use `make show-output` to list generated files
- Use `make clean-output` to remove only generated images

## Performance Notes

### RTX Pro 6000 (Blackwell) - Measured Performance:

**Computational Performance (Benchmark Mode ON):**
- **64 iterations:** 900+ FPS (1.8 billion iterations/second)
- **256 iterations:** 700+ FPS (3.6 billion iterations/second)
- **2048 iterations:** 240+ FPS initial, 60 FPS at deep zoom (9.9 billion iterations/second peak)
- Performance scales with fractal complexity (lower FPS at boundaries)
- Maximum useful zoom: 10^14 (100 trillion times magnification) before precision limits

**Display Performance (Benchmark Mode OFF):**
- **10,000+ FPS** - Pure OpenGL texture display throughput
- Shows GPU memory bandwidth is not the bottleneck
- Drops to ~100 FPS during animation (due to continuous recomputation)
- Monitor limitation: 60Hz refresh rate only displays 60 FPS regardless of compute speed

**Key Performance Insights:**
- 32x more iterations (64→2048) results in only 4x slowdown at initial view (efficient early escape)
- Double precision maintains accuracy to 10^14 zoom level
- Float precision speedup over double on Blackwell architecture (block size dependent):
  - 16x16 blocks: 32.47x speedup (optimal - 112,110 Mpixels/s)
  - 32x32 blocks: 26.70x speedup (97,297 Mpixels/s)
  - 8x8 blocks: 128.77x speedup (88,767 Mpixels/s, but lower throughput)
- Optimal block size: 16x16 achieves highest throughput for modern GPUs
- Excess computational FPS ensures zero lag and perfect responsiveness

**Note:** Performance will vary significantly on other GPUs. This implementation is tested and optimized for Blackwell architecture (sm_100).

### Jetson Orin Performance:
The Jetson Orin (sm_87) with 2048 CUDA cores and unified memory architecture provides excellent performance for embedded GPU computing:
- Lower absolute FPS compared to desktop GPUs, but excellent for edge computing
- Unified memory eliminates CPU↔GPU transfer overhead
- Power-efficient: ~60W total system power (vs 300W+ for desktop GPUs)
- Performance varies by Orin model (AGX Orin 32GB/64GB, Orin NX, Orin Nano)

## Understanding FPS Modes

**Display FPS (Default):**
- Measures how fast the GPU can display an already-computed fractal
- Always very high (limited by memory bandwidth)
- Not affected by iteration count
- Shows texture blitting performance

**Compute FPS (Press 'B'):**
- Measures actual fractal computation speed
- Varies significantly with iteration count and zoom level
- Shows true GPU parallel computing performance
- Use this for benchmarking and performance comparisons

**Why High Compute FPS Matters:**
Even though monitors typically refresh at 60Hz, high computational FPS provides:
- Lower input latency
- Smoother interaction during panning/zooming
- Headroom for additional computational tasks
- Better performance in complex regions

## Building for Your GPU

Check your GPU architecture:
```bash
nvidia-smi --query-gpu=compute_cap --format=csv
```

**Note:** Blackwell GPUs (sm_100) require CUDA Toolkit 12.6 or later.

Build with specific architecture:
```bash
make CUDA_ARCH=sm_75 all   # For RTX 2080 (Turing)
make CUDA_ARCH=sm_86 all   # For RTX 3080/3090 (Ampere)
make CUDA_ARCH=sm_87 all   # For Jetson Orin (Ampere)
make CUDA_ARCH=sm_89 all   # For RTX 4090 (Ada Lovelace)
make CUDA_ARCH=sm_100 all  # For RTX Pro 6000 (Blackwell fallback)
make CUDA_ARCH=sm_120 all  # For RTX Pro 6000 (Blackwell CC 12.0)
```

## Output Management

### Generated Files
All benchmark and animation files are organized in the `output/` directory:
```bash
make run-benchmark       # Generates 24 fractal images in output/
make show-output        # List generated files
```

### Cleaning Options
```bash
make clean              # Remove everything (builds + output)
make clean-build        # Keep generated images, remove executables  
make clean-output       # Keep executables, remove generated images
```

### File Conversion
Convert large PPM files to smaller formats:
```bash
# Convert all PPM files to PNG (requires ImageMagick)
for f in output/*.ppm; do convert "$f" "${f%.ppm}.png"; done

# Remove original PPM files after conversion
make clean-output
```

## Known Issues

- Interactive viewer may have rendering lag on integrated GPUs
- High iteration counts (>1024) can cause timeouts on some systems
- PPM files are large and saved to output/ directory; consider converting to PNG with ImageMagick
- Display FPS can be misleading - use Benchmark Mode (B) for true performance metrics
- Numerical precision limits become visible beyond 10^14 zoom (pixelation)

## Project Structure

```
fractal_engine.cu        - Core CUDA kernels and console app
interactive_viewer.cpp   - OpenGL interactive interface with benchmark mode
Makefile                 - Build system with output management
README.md                - Project documentation
output/                  - Generated PPM files and animations (auto-created)
```

## Advanced Usage

### Animation Generation
Enable zoom animation generation during benchmark:
```bash
./fractal_console --animate  # Generates zoom sequence frames
```

### Custom Resolutions
Run benchmarks at different resolutions:
```bash
./fractal_console --resolution 3840 2160  # 4K resolution
./fractal_console --resolution 7680 4320  # 8K resolution
```

### Performance Analysis
Get detailed performance metrics:
```bash
make run-benchmark       # Shows block size optimization results
make check-deps          # Verify GPU detection and libraries
```

## Contributing

This project demonstrates advanced CUDA programming techniques including:
- Memory coalescing optimization
- Thread block configuration tuning  
- Early escape optimization algorithms
- CUDA-OpenGL interoperability
- Cross-platform GPU architecture detection
- Performance measurement and analysis

## License

Academic project for educational purposes at Johns Hopkins University.

## Acknowledgments

- Johns Hopkins University Whiting School of Engineering
- EN.605.617 Introduction to GPU Programming
- NVIDIA CUDA development community