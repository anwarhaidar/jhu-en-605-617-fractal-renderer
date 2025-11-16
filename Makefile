# CUDA Fractal Renderer - Cross-Platform (Desktop & Jetson)
NVCC = nvcc

# Detect if running on Jetson
IS_JETSON := $(shell if [ -f /etc/nv_tegra_release ] || [ -f /sys/module/tegra_fuse/parameters/tegra_chip_id ]; then echo "yes"; else echo "no"; fi)

# Try to auto-detect GPU architecture with fallback
ifeq ($(IS_JETSON),yes)
    # Try to detect Jetson model
    JETSON_MODEL := $(shell if [ -f /sys/module/tegra_fuse/parameters/tegra_chip_id ]; then \
        chip_id=$$(cat /sys/module/tegra_fuse/parameters/tegra_chip_id); \
        if [ "$$chip_id" = "234" ] || [ "$$chip_id" = "235" ]; then echo "orin"; \
        elif [ "$$chip_id" = "233" ]; then echo "xavier"; \
        elif [ "$$chip_id" = "210" ]; then echo "tx2"; \
        else echo "unknown"; fi; \
        else echo "unknown"; fi)
    
    # Set architecture based on Jetson model
    ifeq ($(JETSON_MODEL),orin)
        DETECTED_CC := 8.7
        DEFAULT_ARCH := sm_87
    else ifeq ($(JETSON_MODEL),xavier)
        DETECTED_CC := 7.2
        DEFAULT_ARCH := sm_72
    else ifeq ($(JETSON_MODEL),tx2)
        DETECTED_CC := 6.2
        DEFAULT_ARCH := sm_62
    else
        # Try deviceQuery fallback for Jetson
        DETECTED_CC := $(shell if [ -x /usr/local/cuda/samples/bin/aarch64/linux/release/deviceQuery ]; then \
            /usr/local/cuda/samples/bin/aarch64/linux/release/deviceQuery 2>/dev/null | grep "CUDA Capability" | head -1 | sed 's/.*Major\/Minor version number:\s*\([0-9]*\)\.\([0-9]*\).*/\1.\2/'; \
            else echo ""; fi)
        DEFAULT_ARCH := sm_87
    endif
    $(info Detected Jetson platform: $(JETSON_MODEL))
else
    # Desktop/Server GPU detection using nvidia-smi
    DETECTED_CC := $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '[:space:]')
    DEFAULT_ARCH := sm_86
endif

# Debug output (uncomment to see detection)
# $(info Detected Compute Capability: $(DETECTED_CC))

CUDA_ARCH ?= $(shell \
	if [ "$(DETECTED_CC)" = "12.1" ]; then echo "sm_120"; \
	elif [ "$(DETECTED_CC)" = "12.0" ] || [ "$(DETECTED_CC)" = "12" ]; then echo "sm_120"; \
	elif [ "$(DETECTED_CC)" = "10.0" ] || [ "$(DETECTED_CC)" = "10" ]; then echo "sm_100"; \
	elif [ "$(DETECTED_CC)" = "9.0" ] || [ "$(DETECTED_CC)" = "9" ]; then echo "sm_90"; \
	elif [ "$(DETECTED_CC)" = "8.9" ]; then echo "sm_89"; \
	elif [ "$(DETECTED_CC)" = "8.7" ]; then echo "sm_87"; \
	elif [ "$(DETECTED_CC)" = "8.6" ]; then echo "sm_86"; \
	elif [ "$(DETECTED_CC)" = "8.0" ] || [ "$(DETECTED_CC)" = "8" ]; then echo "sm_80"; \
	elif [ "$(DETECTED_CC)" = "7.5" ]; then echo "sm_75"; \
	elif [ "$(DETECTED_CC)" = "7.2" ]; then echo "sm_72"; \
	elif [ "$(DETECTED_CC)" = "7.0" ]; then echo "sm_70"; \
	elif [ "$(DETECTED_CC)" = "6.2" ]; then echo "sm_62"; \
	elif [ "$(DETECTED_CC)" = "6.1" ]; then echo "sm_61"; \
	else echo "$(DEFAULT_ARCH)"; fi)

# Platform-specific library paths
ifeq ($(IS_JETSON),yes)
    # Jetson uses different library paths
    CUDA_LIBS = -L/usr/local/cuda/lib64 -L/usr/local/cuda/targets/aarch64-linux/lib -lcudart
    # Jetson might have different OpenGL paths
    OPENGL_LIBS = -L/usr/lib/aarch64-linux-gnu -lGL -lGLEW -lglfw
else
    # Desktop/Server paths
    CUDA_LIBS = -L/usr/local/cuda/lib64 -lcudart
    OPENGL_LIBS = -lGL -lGLEW -lglfw
endif

# Flags
NVCC_FLAGS = -O3 -arch=$(CUDA_ARCH) -std=c++11
MATH_LIBS = -lm

# Executables
CONSOLE_APP = fractal_console
INTERACTIVE_APP = fractal_viewer

# Default target
all: $(CONSOLE_APP) $(INTERACTIVE_APP)
	@echo "Build complete using architecture: $(CUDA_ARCH)"
	@echo "Detected Compute Capability: $(DETECTED_CC)"
ifeq ($(IS_JETSON),yes)
	@echo "Platform: Jetson ($(JETSON_MODEL))"
endif

# Console application
$(CONSOLE_APP): fractal_engine.o
	$(NVCC) $(NVCC_FLAGS) -o $@ $^ $(CUDA_LIBS) $(MATH_LIBS)

# Interactive viewer
$(INTERACTIVE_APP): fractal_engine_lib.o interactive_viewer.o
	$(NVCC) $(NVCC_FLAGS) -o $@ $^ $(CUDA_LIBS) $(OPENGL_LIBS) $(MATH_LIBS)

# Regular build 
fractal_engine.o: fractal_engine.cu
	$(NVCC) $(NVCC_FLAGS) -c -o $@ $<

# Library version (no main) 
fractal_engine_lib.o: fractal_engine.cu
	$(NVCC) $(NVCC_FLAGS) -DLIB_MODE -c -o $@ $<

# C++ files
interactive_viewer.o: interactive_viewer.cpp
	$(NVCC) $(NVCC_FLAGS) -x c++ -c -o $@ $<

clean:
	rm -f *.o $(CONSOLE_APP) $(INTERACTIVE_APP)
	rm -f *.ppm

# Quick run targets
run: $(CONSOLE_APP)
	./$(CONSOLE_APP)

run-viewer: $(INTERACTIVE_APP)
	./$(INTERACTIVE_APP)

run-benchmark: $(CONSOLE_APP)
	./$(CONSOLE_APP) --benchmark

# Check dependencies and GPU detection
check-deps:
	@echo ""
	@echo "=== System Information ==="
ifeq ($(IS_JETSON),yes)
	@echo "Platform: NVIDIA Jetson ($(JETSON_MODEL))"
	@echo "Jetson Detection: /etc/nv_tegra_release found"
else
	@echo "Platform: Desktop/Server GPU"
endif
	@echo "GPU Compute Capability: $(DETECTED_CC)"
	@echo "Selected Architecture: $(CUDA_ARCH)"
	@echo ""
	@echo "CUDA Version:"
	@nvcc --version || echo "CUDA not found"
	@echo ""
ifeq ($(IS_JETSON),no)
	@echo "GPU Information:"
	@nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv || echo "nvidia-smi failed"
else
	@echo "Jetson Information:"
	@cat /etc/nv_tegra_release 2>/dev/null || echo "Tegra release info not found"
	@echo ""
	@echo "Memory Info:"
	@free -h | head -2
endif
	@echo ""
	@echo "Libraries:"
	@pkg-config --exists glfw3 && echo "GLFW: OK" || echo "GLFW: missing"
	@pkg-config --exists glew && echo "GLEW: OK" || echo "GLEW: missing"
	@echo ""
	@echo "Architecture Mapping:"
	@echo "  12.0, 12.1 → sm_120 (Blackwell)"
	@echo "  10.0 → sm_100 (Hopper)"
	@echo "  9.0 → sm_90 (Hopper)"
	@echo "  8.9 → sm_89 (Ada Lovelace)"
	@echo "  8.7 → sm_87 (Jetson AGX Orin)"
	@echo "  8.6 → sm_86 (Ampere)"
	@echo "  7.2 → sm_72 (Jetson AGX Xavier)"
	@echo "  6.2 → sm_62 (Jetson TX2)"
	@echo ""

help:
	@echo ""
	@echo "CUDA Fractal Renderer - Cross-Platform"
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@echo "  all           - Build console and viewer"
	@echo "  clean         - Remove build files"
	@echo "  run           - Run console renderer"
	@echo "  run-viewer    - Run interactive viewer"
	@echo "  run-benchmark - Run performance tests"
	@echo "  check-deps    - Verify dependencies and GPU detection"
	@echo "  help          - Show this help"
	@echo ""
	@echo "Override GPU architecture if needed:"
	@echo "  make CUDA_ARCH=sm_120 all  # Blackwell (12.0)"
	@echo "  make CUDA_ARCH=sm_100 all  # Fallback for Blackwell"
	@echo "  make CUDA_ARCH=sm_89 all   # RTX 4090"
	@echo "  make CUDA_ARCH=sm_87 all   # Jetson AGX Orin"
	@echo "  make CUDA_ARCH=sm_86 all   # RTX 3080/3090"
	@echo "  make CUDA_ARCH=sm_72 all   # Jetson AGX Xavier"
	@echo ""
	@echo "Current detection: CC $(DETECTED_CC) → $(CUDA_ARCH)"
ifeq ($(IS_JETSON),yes)
	@echo "Platform: Jetson ($(JETSON_MODEL))"
endif
	@echo ""
	

.PHONY: all clean run run-viewer run-benchmark check-deps help
