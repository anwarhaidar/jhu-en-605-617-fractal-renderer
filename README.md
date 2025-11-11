# CUDA Fractal Renderer

High-performance fractal rendering using CUDA with an interactive OpenGL viewer.

**Author:** Anwar Sleiman Haidar  
**Institution:** Johns Hopkins University  
**Program:** MS Artifical Intelligence  
**Course:** EN.605.617 - GPU Programming  
**Date:** November 2025  

## Features

- Multiple fractal types: Mandelbrot, Julia, Burning Ship, Tricorn
- Real-time interactive exploration with mouse/keyboard controls
- 6 color schemes with animation support
- Performance benchmarking mode
- Both single and double precision support

## Requirements

- NVIDIA GPU with CUDA Compute Capability 5.0+
- CUDA Toolkit 10.0+
- OpenGL 3.3+
- GLFW3 and GLEW libraries

## Installation

### Ubuntu/Debian
```bash
sudo apt-get install nvidia-cuda-toolkit libglew-dev libglfw3-dev
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
- **Space:** Toggle animation
- **Arrow keys:** Pan view
- **C:** Center view
- **V:** Reset zoom to 1.0
- **F1:** Reset view, colors, and animation (keeps fractal type and iterations)
- **M:** Redisplay menu/controls
- **J/K/I/L:** Adjust Julia parameters (Julia mode only)
- **ESC:** Exit

### Console Renderer
```bash
make run              # Generate sample images
make run-benchmark    # Run performance tests
```

## Known Issues

- Interactive viewer may have rendering lag on integrated GPUs
- High iteration counts (>1024) can cause timeouts on some systems
- PPM files are large; consider converting to PNG with ImageMagick

## Performance Notes

Typical performance by GPU:
- **RTX Pro 6000 (Blackwell):** 1920x1080 @ 256 iterations: ~600-700 FPS
- **GTX 1080:** 1920x1080 @ 256 iterations: ~60 FPS
- Double precision is 2-3x slower than float (up to 32x slower on some architectures)
- Optimal block size: 16x16 for most modern GPUs
- Float precision provides 26x speedup over double on Blackwell architecture

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
make CUDA_ARCH=sm_100 all  # For RTX Pro 6000 (Blackwell)
```

## Project Structure

```
fractal_engine.cu        - Core CUDA kernels and console app
interactive_viewer.cpp   - OpenGL interactive interface
Makefile                 - Build system
README.md                - Project documentation
```

