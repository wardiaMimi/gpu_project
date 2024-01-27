#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../includes/stb_image.h"
#include "../includes/stb_image_write.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h> 

// Function to convert a color image to grayscale
void convertToGrayscale(unsigned char *inputImage, unsigned char *grayscaleImage, int width, int height, int channels)
{
    for (int i = 0; i < width * height; ++i)
    {
        int sum = 0;
        for (int c = 0; c < channels; ++c)
        {
            sum += inputImage[i * channels + c];
        }
        grayscaleImage[i] = (unsigned char)(sum / channels);
    }
}

// Function to apply Laplacian filter to a grayscale image
void laplacianFilter(unsigned char *inputImage, unsigned char *outputImage, int width, int height)
{
    // Laplacian kernel
    int laplacianKernel[3][3] = {
        {0, 1, 0},
        {1, -4, 1},
        {0, 1, 0}};
    // Iterate over each pixel in the image (excluding borders)
    for (int y = 1; y < height - 1; ++y)
    {
        for (int x = 1; x < width - 1; ++x)
        {
            int result = 0;
            // Apply Laplacian operator using the kernel
            for (int i = -1; i <= 1; ++i)
            {
                for (int j = -1; j <= 1; ++j)
                {
                    result += inputImage[(y + i) * width + (x + j)] * laplacianKernel[i + 1][j + 1];
                }
            }

            //  the result must be in the range [0, 255]
            result = result < 0 ? 0 : (result > 255 ? 255 : result);

            // Store the result in the output image
            outputImage[y * width + x] = (unsigned char)result;
        }
    }
}

int main()
{
    // Load input image
    int width, height, channels;
    unsigned char *inputImage = stbi_load("./images/memorial.jpg", &width, &height, &channels, 3);

    if (!inputImage)
    {
        fprintf(stderr, "Error loading image\n");
        return 1;
    }

    // Allocate memory for the grayscale image
    unsigned char *grayscaleImage = (unsigned char *)malloc(width * height);

    //  time before grayscale conversion and Laplacian 
    clock_t start_time_total = clock();

    // convert the input image to grayscale
    convertToGrayscale(inputImage, grayscaleImage, width, height, channels);

    // allocate memory for the output image
    unsigned char *outputImage = (unsigned char *)malloc(width * height);

    // apply Laplacian filter to the grayscale image
    laplacianFilter(grayscaleImage, outputImage, width, height);

    //  time after grayscale conversion and Laplacian 
    clock_t end_time_total = clock();

    // calculate elapsed time for total processing
    double elapsed_time_total = ((double)(end_time_total - start_time_total)) * 1000 / CLOCKS_PER_SEC;

    printf("Total time taken for grayscale conversion and Laplacian filter using CPU: %.2f milliseconds\n", elapsed_time_total);

    // save the result
    stbi_write_png("output_laplacian_cpu.png", width, height, 1, outputImage, width);

    // free memry
    stbi_image_free(inputImage);
    free(grayscaleImage);
    free(outputImage);

    return 0;
}
