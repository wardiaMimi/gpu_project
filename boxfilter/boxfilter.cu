#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#include "../includes/stb_image.h"
#include "../includes/stb_image_write.h"

#include <stdio.h>
#include <stdlib.h>

// Kernel definition
__global__ void boxFilterKernel(unsigned char *image, int width, int height, int channels, int ksize, unsigned char *result) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height) {
    // Perform filtering calculation for each channel
    for (int c = 0; c < channels; ++c) {
      int sum = 0;
      int count = 0;

      // Unroll the filtering loop for performance
      for (int m = -ksize / 2; m <= ksize / 2; m += 2) {
        for (int n = -ksize / 2; n <= ksize / 2; n += 2) {
          int xx = x + n;
          int yy = y + m;

          if (xx >= 0 && xx < width && yy >= 0 && yy < height) {
            sum += image[(yy * width + xx) * channels + c];
            ++count;
          }
        }
      }

      result[(y * width + x) * channels + c] = sum / count;
    }
  }
}

int main() {
  // Load image 
  int width, height, channels;
  unsigned char *image = stbi_load("./images/flower.jpg", &width, &height, &channels, 0);

  if (image == NULL) {
    fprintf(stderr, "Error loading image.\n");
    return 1;
  }

  // Determine kernel size
  int ksize = 10;

  // Allocate device memory
  unsigned char *d_image, *d_result;
  cudaMalloc(&d_image, width * height * channels);
  cudaMalloc(&d_result, width * height * channels);

  // Copy image data to device
  cudaMemcpy(d_image, image, width * height * channels, cudaMemcpyHostToDevice);

  // Determine grid and block dimensions
  dim3 blocks((width + 16 - 1) / 16, (height + 16 - 1) / 16);
  dim3 threads(16, 16);

  // Create CUDA events for timing
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // Record start time
  cudaEventRecord(start);

  // Launch kernel
  boxFilterKernel<<<blocks, threads>>>(d_image, width, height, channels, ksize, d_result);

  // Record stop time
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  // Calculate elapsed time
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);

  printf("Time taken to apply the Box filter using GPU: %.2f milliseconds\n", milliseconds);

  // Copy result  device to host
  cudaMemcpy(image, d_result, width * height * channels, cudaMemcpyDeviceToHost);

  // Save the output image using stb_image_write
  if (stbi_write_png("output_boxfilter_gpu.png", width, height, channels, image, width * channels) == 0) {
    fprintf(stderr, "Error writing image.\n");
    return 1;
  }

  // Free device memory
  cudaFree(d_image);
  cudaFree(d_result);

  // Free image data
  stbi_image_free(image);

  return 0;
}
