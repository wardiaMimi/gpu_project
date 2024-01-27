#define STB_IMAGE_IMPLEMENTATION
#include "../includes/stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../includes/stb_image_write.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h> 

// Function to apply a 2D box filter to an image
void boxFilter(unsigned char *image, int width, int height, int channels, int ksize)
{
    unsigned char *result = malloc(width * height * channels);

    for (int i = 0; i < height; ++i)
    {
        for (int j = 0; j < width; ++j)
        {
            for (int c = 0; c < channels; ++c)
            {
                int sum = 0;
                int count = 0;

                for (int m = -ksize / 2; m <= ksize / 2; ++m)
                {
                    for (int n = -ksize / 2; n <= ksize / 2; ++n)
                    {
                        int x = j + n;
                        int y = i + m;

                        if (x >= 0 && x < width && y >= 0 && y < height)
                        {
                            sum += image[(y * width + x) * channels + c];
                            ++count;
                        }
                    }
                }

                result[(i * width + j) * channels + c] = sum / count;
            }
        }
    }

    // Copy the result back to the original image
    for (int i = 0; i < width * height * channels; ++i)
    {
        image[i] = result[i];
    }

    free(result);
}

int main()
{
    // Load image 
    int width, height, channels;
    unsigned char *image = stbi_load("./images/flower.jpg", &width, &height, &channels, 0);

    if (image == NULL)
    {
        fprintf(stderr, "Error loading image.\n");
        return 1;
    }

    //  time before applying the filter
    clock_t start_time = clock();

    // Apply box filter
    int ksize = 10;
    boxFilter(image, width, height, channels, ksize);

    //  time after applying the filter
    clock_t end_time = clock();

    // Calculate the elapsed time in milliseconds
    double elapsed_time = ((double)(end_time - start_time)) * 1000 / CLOCKS_PER_SEC;

    printf("Time taken to apply the filter using CPU: %.2f milliseconds\n", elapsed_time);

    // Save the filtered image 
    if (stbi_write_png("output_boxfilter_cpu.png", width, height, channels, image, width * channels) == 0)
    {
        fprintf(stderr, "Error writing image.\n");
        return 1;
    }

    // Free the image data
    stbi_image_free(image);

    return 0;
}
