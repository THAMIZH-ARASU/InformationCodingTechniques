import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Define the quantization table (8x8)
quantization_table = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99]
])

# Define the inverse quantization table (8x8)
inverse_quantization_table = np.array([
    [1/16, 1/11, 1/10, 1/16, 1/24, 1/40, 1/51, 1/61],
    [1/12, 1/12, 1/14, 1/19, 1/26, 1/58, 1/60, 1/55],
    [1/14, 1/13, 1/16, 1/24, 1/40, 1/57, 1/69, 1/56],
    [1/14, 1/17, 1/22, 1/29, 1/51, 1/87, 1/80, 1/62],
    [1/18, 1/22, 1/37, 1/56, 1/68, 1/109, 1/103, 1/77],
    [1/24, 1/35, 1/55, 1/64, 1/81, 1/104, 1/113, 1/92],
    [1/49, 1/64, 1/78, 1/87, 1/103, 1/121, 1/120, 1/101],
    [1/72, 1/92, 1/95, 1/98, 1/112, 1/100, 1/103, 1/99]
])

def block_quantization(block):
    return np.round(block / quantization_table)

def block_inverse_quantization(block):
    return np.round(block * inverse_quantization_table)

def main():
    # Open the image
    img = Image.open('input.bmp')
    img = img.convert('L')  # Convert to grayscale

    # Get the image dimensions
    width, height = img.size

    # Create a new image for the quantized and inverse quantized images
    quantized_img = np.zeros((height, width), dtype=np.uint8)
    inverse_quantized_img = np.zeros((height, width), dtype=np.uint8)

    # Apply quantization and inverse quantization block by block
    for i in range(0, height, 8):
        for j in range(0, width, 8):
            block = np.array(img.crop((j, i, j+8, i+8)))
            quantized_block = block_quantization(block)
            inverse_quantized_block = block_inverse_quantization(quantized_block)

            # Assign the quantized and inverse quantized blocks to the new images
            quantized_img[i:i+8, j:j+8] = quantized_block
            inverse_quantized_img[i:i+8, j:j+8] = inverse_quantized_block

    # Display the original, quantized, and inverse quantized images
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(img, cmap='gray')
    axs[0].set_title('Original Image')
    axs[1].imshow(quantized_img, cmap='gray')
    axs[1].set_title('Quantized Image')
    axs[2].imshow(inverse_quantized_img, cmap='gray')
    axs[2].set_title('Inverse Quantized Image')
    plt.show()

if __name__ == "__main__":
    main()