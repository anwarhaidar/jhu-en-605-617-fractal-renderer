# CUDA Fractal Renderer
NVCC = nvcc

# Try to auto-detect GPU architecture with fallback
DETECTED_CC := $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '[:space:]')
CUDA_ARCH ?= $(shell \
	if [ "$(DETECTED_CC)" = "12.0" ]; then echo "sm_100"; \
	elif [ "$(DETECTED_CC)" = "10.0" ]; then echo "sm_100"; \
	elif [ "$(DETECTED_CC)" = "9.0" ]; then echo "sm_90"; \
	elif [ "$(DETECTED_CC)" = "8.9" ]; then echo "sm_89"; \
	elif [ "$(DETECTED_CC)" = "8.6" ]; then echo "sm_86"; \
	elif [ "$(DETECTED_CC)" = "8.0" ]; then echo "sm_80"; \
	elif [ "$(DETECTED_CC)" = "7.5" ]; then echo "sm_75"; \
	elif [ "$(DETECTED_CC)" = "7.0" ]; then echo "sm_70"; \
	elif [ "$(DETECTED_CC)" = "6.1" ]; then echo "sm_61"; \
	else echo "sm_70"; fi)

# Flags
NVCC_FLAGS = -O3 -arch=$(CUDA_ARCH) -std=c++11
CUDA_LIBS = -L/usr/local/cuda/lib64 -lcudart
OPENGL_LIBS = -lGL -lGLEW -lglfw
MATH_LIBS = -lm

# Executables
CONSOLE_APP = fractal_console
INTERACTIVE_APP = fractal_viewer

# Default target
all: $(CONSOLE_APP) $(INTERACTIVE_APP)

# Console application
$(CONSOLE_APP): fractal_engine.o
	$(NVCC) $(NVCC_FLAGS) -o $@ $^ $(CUDA_LIBS) $(MATH_LIBS)

# Interactive viewer
$(INTERACTIVE_APP): fractal_engine_lib.o interactive_viewer.o
	$(NVCC) $(NVCC_FLAGS) -o $@ $^ $(CUDA_LIBS) $(OPENGL_LIBS) $(MATH_LIBS)

# Regular build
fractal_engine.o: fractal_engine.cu
	$(NVCC) $(NVCC_FLAGS) -dc -o $@ $<

# Library version (no main)
fractal_engine_lib.o: fractal_engine.cu
	$(NVCC) $(NVCC_FLAGS) -DLIB_MODE -dc -o $@ $<

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

check-deps:
	@echo ""
	@echo "Checking dependencies..."
	@nvcc --version || echo "CUDA not found"
	@pkg-config --exists glfw3 && echo "GLFW: OK" || echo "GLFW: missing"
	@pkg-config --exists glew && echo "GLEW: OK" || echo "GLEW: missing"
	@echo ""

help:
	@echo ""
	@echo "CUDA Fractal Renderer"
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@echo "  all           - Build console and viewer"
	@echo "  clean         - Remove build files"
	@echo "  run           - Run console renderer"
	@echo "  run-viewer    - Run interactive viewer"
	@echo "  run-benchmark - Run performance tests"
	@echo "  check-deps    - Verify dependencies"
	@echo "  help          - Show this help"
	@echo ""
	

.PHONY: all clean run run-viewer run-benchmark check-deps help