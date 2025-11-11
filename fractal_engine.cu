// fractal_engine.cu
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>

// Fractal types
typedef enum {
    MANDELBROT,
    JULIA,
    BURNING_SHIP,
    TRICORN
} FractalType;

// Color schemes
typedef enum {
    GRAYSCALE,
    HSV_RAINBOW,
    FIRE,
    OCEAN,
    PSYCHEDELIC,
    ELECTRIC
} ColorScheme;

// Fractal parameters structure
struct FractalParams {
    double x_center, y_center;
    double zoom;
    double julia_cx, julia_cy;
    int max_iterations;
    FractalType type;
    ColorScheme color_scheme;
    double animation_time;
};

// Color conversion utilities
__device__ float3 hsv_to_rgb(float h, float s, float v) {
    float c = v * s;
    float x = c * (1.0f - fabsf(fmodf(h / 60.0f, 2.0f) - 1.0f));
    float m = v - c;
    
    float3 rgb;
    if (h < 60) {
        rgb.x = c; rgb.y = x; rgb.z = 0;
    } else if (h < 120) {
        rgb.x = x; rgb.y = c; rgb.z = 0;
    } else if (h < 180) {
        rgb.x = 0; rgb.y = c; rgb.z = x;
    } else if (h < 240) {
        rgb.x = 0; rgb.y = x; rgb.z = c;
    } else if (h < 300) {
        rgb.x = x; rgb.y = 0; rgb.z = c;
    } else {
        rgb.x = c; rgb.y = 0; rgb.z = x;
    }
    
    rgb.x += m; rgb.y += m; rgb.z += m;
    return rgb;
}

// Advanced color mapping
__device__ uchar3 apply_color_scheme(int iteration, int max_iter, ColorScheme scheme, double time) {
    if (iteration == max_iter) {
        return make_uchar3(0, 0, 0); // Black for points in set
    }
    
    float t = (float)iteration / max_iter;
    float3 color;
    
    switch (scheme) {
        case GRAYSCALE:
            {
                unsigned char gray = (unsigned char)(255 * t);
                return make_uchar3(gray, gray, gray);
            }
        
        case HSV_RAINBOW:
            {
                float hue = fmodf(360.0f * t + time * 50.0f, 360.0f);
                color = hsv_to_rgb(hue, 0.8f, 1.0f);
                break;
            }
        
        case FIRE:
            {
                color.x = fminf(1.0f, t * 2.0f);
                color.y = fmaxf(0.0f, t * 2.0f - 1.0f);
                color.z = fmaxf(0.0f, t * 4.0f - 3.0f);
                break;
            }
        
        case OCEAN:
            {
                color.x = t * 0.3f;
                color.y = t * 0.6f;
                color.z = fminf(1.0f, t * 1.5f);
                break;
            }
        
        case PSYCHEDELIC:
            {
                float phase = time * 2.0f;
                color.x = 0.5f + 0.5f * sinf(t * 10.0f + phase);
                color.y = 0.5f + 0.5f * sinf(t * 15.0f + phase + 2.0f);
                color.z = 0.5f + 0.5f * sinf(t * 20.0f + phase + 4.0f);
                break;
            }
        
        case ELECTRIC:
            {
                float intensity = powf(t, 0.3f);
                color.x = intensity;
                color.y = intensity * 0.8f;
                color.z = fminf(1.0f, intensity * 2.0f);
                break;
            }
    }
    
    return make_uchar3(
        (unsigned char)(255 * fminf(1.0f, fmaxf(0.0f, color.x))),
        (unsigned char)(255 * fminf(1.0f, fmaxf(0.0f, color.y))),
        (unsigned char)(255 * fminf(1.0f, fmaxf(0.0f, color.z)))
    );
}

// Main Mandelbrot kernel
__global__ void mandelbrot_kernel_optimized(uchar3* image, int width, int height,
                                           FractalParams params) {
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (px >= width || py >= height) return;
    
    // Calculate complex plane coordinates with zoom
    double range = 4.0 / params.zoom;
    double x_min = params.x_center - range;
    double y_min = params.y_center - range * height / width;
    
    double x0 = x_min + (range * 2.0 * px) / (width - 1);
    double y0 = y_min + (range * 2.0 * height / width * py) / (height - 1);
    
    int iteration = 0;
    double x, y;
    
    // Different fractal computations
    switch (params.type) {
        case MANDELBROT:
            {
                x = 0.0; y = 0.0;
                while (x*x + y*y <= 4.0 && iteration < params.max_iterations) {
                    double temp = x*x - y*y + x0;
                    y = 2.0*x*y + y0;
                    x = temp;
                    iteration++;
                }
                break;
            }
        
        case JULIA:
            {
                x = x0; y = y0;
                double cx = params.julia_cx;
                double cy = params.julia_cy;
                
                // Animate Julia parameters
                cx += 0.1 * sin(params.animation_time * 0.5);
                cy += 0.1 * cos(params.animation_time * 0.3);
                
                while (x*x + y*y <= 4.0 && iteration < params.max_iterations) {
                    double temp = x*x - y*y + cx;
                    y = 2.0*x*y + cy;
                    x = temp;
                    iteration++;
                }
                break;
            }
        
        case BURNING_SHIP:
            {
                x = 0.0; y = 0.0;
                while (x*x + y*y <= 4.0 && iteration < params.max_iterations) {
                    double temp = x*x - y*y + x0;
                    y = 2.0*fabs(x)*fabs(y) + y0;
                    x = temp;
                    iteration++;
                }
                break;
            }
        
        case TRICORN:
            {
                x = 0.0; y = 0.0;
                while (x*x + y*y <= 4.0 && iteration < params.max_iterations) {
                    double temp = x*x - y*y + x0;
                    y = -2.0*x*y + y0;
                    x = temp;
                    iteration++;
                }
                break;
            }
    }
    
    // Apply color mapping
    uchar3 color = apply_color_scheme(iteration, params.max_iterations, 
                                      params.color_scheme, params.animation_time);
    
    // Store result
    int index = py * width + px;
    image[index] = color;
}

// Single precision kernel for performance comparison
__global__ void mandelbrot_kernel_float(uchar3* image, int width, int height,
                                        float x_center, float y_center, float zoom,
                                        int max_iterations, ColorScheme color_scheme) {
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (px >= width || py >= height) return;
    
    float range = 4.0f / zoom;
    float x_min = x_center - range;
    float y_min = y_center - range * height / width;
    
    float x0 = x_min + (range * 2.0f * px) / (width - 1);
    float y0 = y_min + (range * 2.0f * height / width * py) / (height - 1);
    
    float x = 0.0f, y = 0.0f;
    int iteration = 0;
    
    while (x*x + y*y <= 4.0f && iteration < max_iterations) {
        float temp = x*x - y*y + x0;
        y = 2.0f*x*y + y0;
        x = temp;
        iteration++;
    }
    
    uchar3 color = apply_color_scheme(iteration, max_iterations, color_scheme, 0.0);
    int index = py * width + px;
    image[index] = color;
}

// Save image as PPM format
void save_ppm(const char* filename, uchar3* image, int width, int height) {
    FILE* file = fopen(filename, "wb");
    if (!file) {
        printf("Error: Could not open file %s\n", filename);
        return;
    }
    
    fprintf(file, "P6\n%d %d\n255\n", width, height);
    fwrite(image, sizeof(uchar3), width * height, file);
    fclose(file);
    
    printf("Image saved: %s (%dx%d)\n", filename, width, height);
}

// Error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            printf("CUDA error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(error)); \
            exit(1); \
        } \
    } while(0)

// Animation frame generation
void generate_zoom_animation(const char* base_filename, int width, int height,
                           double center_x, double center_y, int num_frames) {
    size_t image_size = width * height * sizeof(uchar3);
    uchar3* h_image = (uchar3*)malloc(image_size);
    uchar3* d_image;
    CUDA_CHECK(cudaMalloc(&d_image, image_size));
    
    dim3 block_size(16, 16);
    dim3 grid_size((width + block_size.x - 1) / block_size.x,
                   (height + block_size.y - 1) / block_size.y);
    
    for (int frame = 0; frame < num_frames; frame++) {
        FractalParams params;
        params.x_center = center_x;
        params.y_center = center_y;
        params.zoom = pow(1.1, frame);  // Exponential zoom
        params.julia_cx = -0.7;
        params.julia_cy = 0.27015;
        params.max_iterations = 256;
        params.type = MANDELBROT;
        params.color_scheme = HSV_RAINBOW;
        params.animation_time = frame * 0.1;
        
        mandelbrot_kernel_optimized<<<grid_size, block_size>>>(
            d_image, width, height, params);
        CUDA_CHECK(cudaDeviceSynchronize());
        
        CUDA_CHECK(cudaMemcpy(h_image, d_image, image_size, cudaMemcpyDeviceToHost));
        
        char filename[256];
        sprintf(filename, "%s_frame_%04d.ppm", base_filename, frame);
        save_ppm(filename, h_image, width, height);
        
        printf("Generated frame %d/%d\r", frame + 1, num_frames);
        fflush(stdout);
    }
    printf("\nAnimation frames generated!\n");
    
    free(h_image);
    CUDA_CHECK(cudaFree(d_image));
}


// CPU version for comparison
void mandelbrot_cpu(uchar3* image, int width, int height, FractalParams params) {
    for (int py = 0; py < height; py++) {
        for (int px = 0; px < width; px++) {
            double range = 4.0 / params.zoom;
            double x_min = params.x_center - range;
            double y_min = params.y_center - range * height / width;
            
            double x0 = x_min + (range * 2.0 * px) / (width - 1);
            double y0 = y_min + (range * 2.0 * height / width * py) / (height - 1);
            
            int iteration = 0;
            double x = 0.0, y = 0.0;
            
            while (x*x + y*y <= 4.0 && iteration < params.max_iterations) {
                double temp = x*x - y*y + x0;
                y = 2.0*x*y + y0;
                x = temp;
                iteration++;
            }
            
            unsigned char gray = (iteration == params.max_iterations) ? 0 : 
                               (unsigned char)(255 * iteration / params.max_iterations);
            image[py * width + px] = make_uchar3(gray, gray, gray);
        }
    }
}

#ifndef LIB_MODE

// Main function
int main(int argc, char** argv) {
    printf("=== CUDA Fractal Renderer ===\n\n");
    
    // Default parameters
    int width = 1920, height = 1080;
    bool run_benchmarks = false;
    bool generate_animation = false;
    bool run_cpu_gpu_comparison = false;  
    // Parse command line arguments
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--benchmark") == 0) run_benchmarks = true;
        else if (strcmp(argv[i], "--animate") == 0) generate_animation = true;
        else if (strcmp(argv[i], "--cpu-vs-gpu") == 0) run_cpu_gpu_comparison = true;
        else if (strcmp(argv[i], "--resolution") == 0 && i + 2 < argc) {
            width = atoi(argv[++i]);
            height = atoi(argv[++i]);
        }
    }
    
    printf("Resolution: %dx%d\n", width, height);
    
    // Allocate memory
    size_t image_size = width * height * sizeof(uchar3);
    uchar3* h_image = (uchar3*)malloc(image_size);
    uchar3* d_image;
    CUDA_CHECK(cudaMalloc(&d_image, image_size));
    
    // Configure CUDA execution parameters
    dim3 block_size(16, 16);
    dim3 grid_size((width + block_size.x - 1) / block_size.x,
                   (height + block_size.y - 1) / block_size.y);
    // CPU vs GPU comparison - place AFTER memory allocation
    if (run_cpu_gpu_comparison) {
        printf("\n=== CPU vs GPU Performance Comparison ===\n");
        
        FractalParams params = {-0.5, 0.0, 1.0, -0.7, 0.27015, 256, MANDELBROT, GRAYSCALE, 0.0};
        
        // CPU timing using simple clock
        clock_t start_cpu = clock();
        mandelbrot_cpu(h_image, width, height, params);
        clock_t end_cpu = clock();
        float cpu_time = ((float)(end_cpu - start_cpu) / CLOCKS_PER_SEC) * 1000;
        
        // GPU timing  
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);
        mandelbrot_kernel_optimized<<<grid_size, block_size>>>(d_image, width, height, params);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float gpu_time;
        cudaEventElapsedTime(&gpu_time, start, stop);
        
        printf("Resolution: %dx%d\n", width, height);
        printf("CPU Time: %.1f ms\n", cpu_time);
        printf("GPU Time: %.1f ms\n", gpu_time);
        printf("Speedup: %.1fx faster on GPU\n", cpu_time / gpu_time);
        
        // Cleanup and exit
        free(h_image);
        CUDA_CHECK(cudaFree(d_image));
        printf("\nComparison complete!\n");
        return 0;
    }
    // Performance benchmarking
    if (run_benchmarks) {
        printf("\n=== Performance Benchmarks ===\n");
        
        dim3 block_sizes[] = {{8,8}, {16,16}, {32,32}};
        const char* block_names[] = {"8x8", "16x16", "32x32"};
        
        for (int i = 0; i < 3; i++) {
            dim3 test_block_size = block_sizes[i];
            dim3 test_grid_size((width + test_block_size.x - 1) / test_block_size.x,
                               (height + test_block_size.y - 1) / test_block_size.y);
            
            printf("\nBlock size %s:\n", block_names[i]);
            
            // Create timing events
            cudaEvent_t start, stop;
            CUDA_CHECK(cudaEventCreate(&start));
            CUDA_CHECK(cudaEventCreate(&stop));
            
            // Test double precision
            FractalParams params_double = {-0.5, 0.0, 1.0, -0.7, 0.27015, 256, MANDELBROT, HSV_RAINBOW, 0.0};
            
            CUDA_CHECK(cudaEventRecord(start));
            mandelbrot_kernel_optimized<<<test_grid_size, test_block_size>>>(d_image, width, height, params_double);
            CUDA_CHECK(cudaEventRecord(stop));
            CUDA_CHECK(cudaEventSynchronize(stop));
            
            float double_time;
            CUDA_CHECK(cudaEventElapsedTime(&double_time, start, stop));
            
            // Test single precision
            CUDA_CHECK(cudaEventRecord(start));
            mandelbrot_kernel_float<<<test_grid_size, test_block_size>>>(d_image, width, height, -0.5f, 0.0f, 1.0f, 256, HSV_RAINBOW);
            CUDA_CHECK(cudaEventRecord(stop));
            CUDA_CHECK(cudaEventSynchronize(stop));
            
            float float_time;
            CUDA_CHECK(cudaEventElapsedTime(&float_time, start, stop));
            
            int double_mpixels = (int)((width * height) / (double_time / 1000.0f) / 1000000);
            int float_mpixels = (int)((width * height) / (float_time / 1000.0f) / 1000000);
            
            printf("  Double precision: %.2f ms, %d Mpixels/s\n", double_time, double_mpixels);
            printf("  Single precision: %.2f ms, %d Mpixels/s\n", float_time, float_mpixels);
            printf("  Speedup (float): %.2fx\n", double_time / float_time);
            
            CUDA_CHECK(cudaEventDestroy(start));
            CUDA_CHECK(cudaEventDestroy(stop));
        }
    }
    
    // Generate sample images for all fractal types and color schemes
    printf("\n=== Generating Sample Images ===\n");
    
    FractalType fractals[] = {MANDELBROT, JULIA, BURNING_SHIP, TRICORN};
    const char* fractal_names[] = {"mandelbrot", "julia", "burning_ship", "tricorn"};
    
    ColorScheme colors[] = {GRAYSCALE, HSV_RAINBOW, FIRE, OCEAN, PSYCHEDELIC, ELECTRIC};
    const char* color_names[] = {"grayscale", "rainbow", "fire", "ocean", "psychedelic", "electric"};
    
    for (int f = 0; f < 4; f++) {
        for (int c = 0; c < 6; c++) {
            FractalParams params;
            params.x_center = (fractals[f] == MANDELBROT || fractals[f] == TRICORN) ? -0.5 : 0.0;
            params.y_center = 0.0;
            params.zoom = 1.0;
            params.julia_cx = -0.7;
            params.julia_cy = 0.27015;
            params.max_iterations = 256;
            params.type = fractals[f];
            params.color_scheme = colors[c];
            params.animation_time = 0.0;
            
            mandelbrot_kernel_optimized<<<grid_size, block_size>>>(
                d_image, width, height, params);
            CUDA_CHECK(cudaDeviceSynchronize());
            
            CUDA_CHECK(cudaMemcpy(h_image, d_image, image_size, cudaMemcpyDeviceToHost));
            
            char filename[256];
            sprintf(filename, "%s_%s.ppm", fractal_names[f], color_names[c]);
            save_ppm(filename, h_image, width, height);
            
            printf("Generated: %s\n", filename);
        }
    }
    
    // Generate zoom animation
    if (generate_animation) {
        printf("\n=== Generating Zoom Animation ===\n");
        generate_zoom_animation("mandelbrot_zoom", width/2, height/2, 
                               -0.7453, 0.11307, 50);
    }
    
    // Cleanup
    free(h_image);
    CUDA_CHECK(cudaFree(d_image));
    
    printf("\n=== Rendering Complete! ===\n");
    printf("Usage: %s [--benchmark] [--animate] [--resolution W H]\n", argv[0]);
    
    return 0;
}

#endif // LIB_MODE

// External interface for interactive viewer
extern "C" void launch_fractal_kernel(unsigned char* d_image, int width, int height, FractalParams params) {
    // Convert to uchar3 format for existing kernel
    uchar3* d_image_uchar3 = reinterpret_cast<uchar3*>(d_image);
    
    dim3 block_size(16, 16);
    dim3 grid_size((width + block_size.x - 1) / block_size.x,
                   (height + block_size.y - 1) / block_size.y);
    
    
    // Call existing kernel
    mandelbrot_kernel_optimized<<<grid_size, block_size>>>(d_image_uchar3, width, height, params);
    cudaDeviceSynchronize();
}
