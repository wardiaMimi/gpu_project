#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#include "../includes/stb_image.h"
#include "../includes/stb_image_write.h"
#include <stdio.h>
#include <stdlib.h>

__constant__ int laplacianKernel[3][3] = {
    {0, 1, 0},
    {1, -4, 1},
    {0, 1, 0}
};

// Kernel to convert the input image to grayscale
__global__ void convertToGrayscaleKernel(unsigned char* inputImage, unsigned char* grayscaleImage, int width, int height, int channels) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < width * height) {
        int sum = 0;
        for (int c = 0; c < channels; ++c) {
            sum += inputImage[index * channels + c];
        }
        grayscaleImage[index] = (unsigned char)(sum / channels);
    }
}

// Kernel to apply Laplace filter to the grayscale image
__global__ void laplacianFilter(const unsigned char* inputImage, unsigned char* outputImage, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int result = 0;

        // Apply Laplacian operator using the kernel
        for (int i = -1; i <= 1; ++i) {
            for (int j = -1; j <= 1; ++j) {
                int imgX = x + j;
                int imgY = y + i;

                // Check boundaries
                if (imgX >= 0 && imgX < width && imgY >= 0 && imgY < height) {
                    result += inputImage[imgY * width + imgX] * laplacianKernel[i + 1][j + 1];
                }
            }
        }

        // Clamp the result to the valid range [0, 255]
        result = result < 0 ? 0 : (result > 255 ? 255 : result);

        // Store the result in the output image
        outputImage[y * width + x] = (unsigned char)result;
    }
}



// Wrapper function for launching both kernels with timing
void GrayscaleAndLaplacianFilterCuda(unsigned char* inputImage, unsigned char* grayscaleImage, unsigned char* outputImage, int width, int height, int channels) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
   
    
    dim3 dimBlock(256);
    // grayscale  kernel processes each pixel in the input image independently, without considering neighboring pixels. 
    
    dim3 dimGrid((width * height + dimBlock.x - 1) / dimBlock.x);

    // laplacian  kernel processes on each pixel in the image by considering its neighboring pixels. 
    // these neighboring pixels are required to calculate the Laplacian value for a given pixel.
    // so the grid size is adjusted to ensure that neighboring pixels are available for computation within each block.

    dim3 dimGridFilter((width + dimBlock.x - 1) / dimBlock.x, (height + dimBlock.y - 1) / dimBlock.y);


    // Record start time
    cudaEventRecord(start);

    // Launch the grayscale kernel
    convertToGrayscaleKernel<<<dimGrid, dimBlock>>>(inputImage, grayscaleImage, width, height, channels);

    // Launch the Laplacian filter kernel
    laplacianFilter<<<dimGridFilter, dimBlock>>>(grayscaleImage, outputImage, width, height);

    // Record stop time
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Grayscale and Laplacian Filter Execution Time using GPU: %f ms\n", milliseconds);
}

int main() {
    int width, height, channels;
    unsigned char* inputImage = stbi_load("./images/memorial.jpg", &width, &height, &channels, 3);    unsigned char* grayscaleImage = (unsigned char*)malloc(width * height);
     if (inputImage == NULL) {
    fprintf(stderr, "Error loading image.\n");
    return 1;
    }

    unsigned char* outputImage = (unsigned char*)malloc(width * height);

    // Device memory
    unsigned char* d_inputImage;
    unsigned char* d_grayscaleImage;
    unsigned char* d_outputImage;
    cudaMalloc((void**)&d_inputImage, width * height * channels);
    cudaMalloc((void**)&d_grayscaleImage, width * height);
    cudaMalloc((void**)&d_outputImage, width * height);

    // Copy input image from host to device
    cudaMemcpy(d_inputImage, inputImage, width * height * channels, cudaMemcpyHostToDevice);

    // Call the CUDA function to perform grayscale conversion and Laplacian filter with timing
    GrayscaleAndLaplacianFilterCuda(d_inputImage, d_grayscaleImage, d_outputImage, width, height, channels);

    // Copy result from device to host
    cudaMemcpy(outputImage, d_outputImage, width * height, cudaMemcpyDeviceToHost);

    // Save the result
    stbi_write_png("output_laplacian_gpu.png", width, height, 1, outputImage, width);

    // Free memory
    stbi_image_free(inputImage);
    free(grayscaleImage);
    free(outputImage);
    cudaFree(d_inputImage);
    cudaFree(d_grayscaleImage);
    cudaFree(d_outputImage);

    return 0;
}
