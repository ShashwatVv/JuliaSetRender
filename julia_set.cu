/**
 * @author: Shashwat Vaibhav
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define WIDTH 1024
#define HEIGHT 1024
#define BLOCK_SIZE 16

#define REAL_MIN -1.5f
#define REAL_MAX 1.5f
#define IMAG_MIN -1.5f
#define IMAG_MAX 1.5f

/* Iteration could be varied, 300 served the purpose */
#define MAX_ITER 300

/**  
* @brief: This is a basic function computing the num of iterations till the values explode
* @details: This function returns the i-th value while iterating as per the equation
*           Z_n+1 = Z_n^2 + C, where
*           Z = x + iy , C = cx + icy
*           The math will unfold as, during the iteration:
*           x_new => x_old^2 - y_old^2 + cx
*           y_new => 2 * x_old * y_old + cy
* @where: This would run on GPU (or device)
*/
__device__ int julia(float x, float y, float cx, float cy) 
{
    int i = 0;
    while (i < MAX_ITER && x*x + y*y < 4.0f) {
        float xtemp = x*x - y*y + cx;
        y = 2.0f * x * y + cy;
        x = xtemp;
        ++i;
    }
    return i;
}

/**
 * @brief: A basic color mapping based on iterations.
 * 
 * @details: This function will assign RGB mapping, based upon the count of 
 *           iterations retuned by the call to julia(). This will assign a the values
 *           based upon this non-linear equation:
 *           r(t) = k1 * (1-t) * t^3 * 255
 *           g(t) = k2 * (1-t)^2 * t^2 * 255
 *           b(t) = k3 * (1-t)^3 * t * 255 
 *           --> k1, k2, k3 => any real constants  
 * 
 *           Note: set r, g and b = k * t * 255 for grayscale. 
 * 
 * @where: This would run on GPU (or device)
 */

__device__ void iterationToRGB(int iter, unsigned char* r, unsigned char* g, unsigned char* b) 
{
    /* Just normalizing the iter value/count returned */
    float t = (float)iter / MAX_ITER;

    *r = (unsigned char)(9 * (1 - t) * t * t * t * 255);
    *g = (unsigned char)(15 * (1 - t) * (1 - t) * t * t * 255);
    *b = (unsigned char)(8.5 * (1 - t) * (1 - t) * (1 - t) * t * 255);
}

/**
 * @brief: This is the kernel function to be called from main
 * 
 * @details: This function computes color of each pixel based on the equation
 *           desribing the julia set. It stores the result in a buffer. Each cuda
 *           thread does compute one pixel.
 * 
 * @where: This would run on GPU (device)
 */
__global__ void juliaKernelRGB(unsigned char* output, float cx, float cy) 
{
    int x, y, iter, idx;
    float real, imag;
    unsigned char r, g, b;

    /* compute x and y co-ordinates of the pixel */
    x = blockIdx.x * blockDim.x + threadIdx.x;
    y = blockIdx.y * blockDim.y + threadIdx.y;
    
    /* check the bounds for pixel. Should be within the image */
    if (x >= WIDTH || y >= HEIGHT) return;
    
    /* fit the pixels mapping between these ranges of real and imaginary numbers */
    real = REAL_MIN + (REAL_MAX - REAL_MIN) * x / WIDTH;
    imag = IMAG_MIN + (IMAG_MAX - IMAG_MIN) * y / HEIGHT;
    
    /* get iter count from julia () */
    iter = julia(real, imag, cx, cy);
    
    /* map iteration to pixels */
    iterationToRGB(iter, &r, &g, &b);
    
    /* store the RGB values to output buffer 
       These r, g, b values for each pixels are stored sequentially.
    */
    idx = (y * WIDTH + x) * 3;
    output[idx] = r;
    output[idx + 1] = g;
    output[idx + 2] = b;
}

/**
 * @brief: Saves the data to the ppm file
 * 
 * @details: This function will store the raw rgb data in a ppm file 
 * 
 * @where: This would run on the host (CPU)
 */
void save_ppm_color(const char* filename, unsigned char* data) 
{
    FILE* fp = fopen(filename, "wb");
    if (!fp) {
        perror("fopen");
        return;
    }

    fprintf(fp, "P6\n%d %d\n255\n", WIDTH, HEIGHT);
    fwrite(data, sizeof(unsigned char), WIDTH * HEIGHT * 3, fp);
    fclose(fp);
}

/**
 * @brief: expected main :D
 * 
 * @details: The cuda runtime will be called executing the kernel on different threads
 *           finally the data will be saved into a ppm file.
 * 
 * @where: This would run on host and will synchronize after kernel call to the device 
 */
int main() 
{
    /* ptr host_output to hold image data on CPU 
       ptr device_output will hold image data on GPU
    */
    unsigned char* host_output;
    unsigned char* device_output;
    
    float cx, cy;

    size_t size = WIDTH * HEIGHT * 3 * sizeof(unsigned char);
    host_output = (unsigned char*)malloc(size);
    
    /* I don't want to use the CudaError status returned, I will void it */
    (void)cudaMalloc(&device_output, size);

    /* Each block would have 16*16 = 256 threads */
    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);

    dim3 numBlocks((WIDTH + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (HEIGHT + threadsPerBlock.y - 1) / threadsPerBlock.y);
    
    /* Random value initialized for C: change it for different patterns */               
    cx = -0.7f;
    cy = 0.27015f;
    
    /* Call the julia kernel */
    juliaKernelRGB<<<numBlocks, threadsPerBlock>>>(device_output, cx, cy);
    
    /* if you are using Linux, need to call this and wait for threads to finish */
    /* On windows, I didn't face the issue */
    cudaDeviceSynchronize();

    (void)cudaMemcpy(host_output, device_output, size, cudaMemcpyDeviceToHost);

    save_ppm_color("julia_color.ppm", host_output);
    
    printf("Colored Julia set saved as julia_color.ppm\n");

    cudaFree(device_output);
    free(host_output);

    return 0;
}
