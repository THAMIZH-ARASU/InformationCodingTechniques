import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import math

def compress_color_image(image: np.ndarray, block_size: int, compression_factor: float):
    # Split the image into RGB channels
    channels = cv.split(image)

    dct_channels = []
    quantized_channels = []
    dequantized_channels = []
    idct_channels = []
    reconstructed_channels = []

    # Create a basic quantization matrix (8x8 example, you can customize this)
    quantization_matrix = np.array([
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99]
    ])

    # Scale the quantization matrix by the compression factor
    quantization_matrix = np.round(quantization_matrix * compression_factor)

    # Loop over each channel (R, G, B)
    for ch in channels:
        h, w = ch.shape

        dct_block = np.zeros_like(ch, dtype=np.float32)
        quantized_block = np.zeros_like(ch, dtype=np.float32)
        dequantized_block = np.zeros_like(ch, dtype=np.float32)
        idct_block = np.zeros_like(ch, dtype=np.float32)

        for i in range(0, h, block_size):
            for j in range(0, w, block_size):
                block = ch[i:i+block_size, j:j+block_size]
                
                # Perform DCT on each block
                dct_sub_block = cv.dct(np.float32(block))
                dct_block[i:i+block_size, j:j+block_size] = dct_sub_block

                # Quantize the DCT coefficients
                quantized_sub_block = np.round(dct_sub_block / quantization_matrix)
                quantized_block[i:i+block_size, j:j+block_size] = quantized_sub_block

                # Dequantize the coefficients
                dequantized_sub_block = quantized_sub_block * quantization_matrix
                dequantized_block[i:i+block_size, j:j+block_size] = dequantized_sub_block

                # Perform IDCT on dequantized block
                idct_sub_block = cv.idct(dequantized_sub_block)
                idct_block[i:i+block_size, j:j+block_size] = idct_sub_block

        dct_channels.append(dct_block)
        quantized_channels.append(quantized_block)
        dequantized_channels.append(dequantized_block)
        idct_channels.append(idct_block)

    # Merge channels back together
    dct_image = cv.merge(dct_channels)
    quantized_image = cv.merge(quantized_channels)
    dequantized_image = cv.merge(dequantized_channels)
    idct_image = cv.merge(idct_channels)

    # Clip pixel values to valid range [0, 255] for the final image
    reconstructed_image = np.clip(idct_image, 0, 255).astype(np.uint8)

    return dct_image, quantized_image, dequantized_image, idct_image, reconstructed_image

def calculate_psnr(original: np.ndarray, reconstructed: np.ndarray) -> float:
    mse = np.mean((original.astype(np.float32) - reconstructed.astype(np.float32)) ** 2)
    if mse == 0:
        return 100  # Perfect match
    return 10 * math.log10(255 ** 2 / mse)

def plot_images(original: np.ndarray, dct: np.ndarray, quantized: np.ndarray, dequantized: np.ndarray, idct: np.ndarray, reconstructed: np.ndarray):
    fig, axes = plt.subplots(2, 3, figsize=(12, 12))

    # Plot original image
    axes[0, 0].imshow(cv.cvtColor(original, cv.COLOR_BGR2RGB))
    axes[0, 0].set_title("Original Image")
    axes[0, 0].axis('off')

    # Plot DCT image
    axes[0, 1].imshow(np.log1p(np.abs(dct)), cmap='gray')  # Log scale for better visualization
    axes[0, 1].set_title("DCT Image")
    axes[0, 1].axis('off')

    # Plot Quantized image
    axes[0, 2].imshow(np.log1p(np.abs(quantized)), cmap='gray')  # Log scale for better visualization
    axes[0, 2].set_title("Quantized Image")
    axes[0, 2].axis('off')

    # Plot Dequantized image
    axes[1, 0].imshow(np.log1p(np.abs(dequantized)), cmap='gray')
    axes[1, 0].set_title("Dequantized Image")
    axes[1, 0].axis('off')

    # Plot IDCT image
    axes[1, 1].imshow(cv.cvtColor(idct.astype(np.uint8), cv.COLOR_BGR2RGB))
    axes[1, 1].set_title("IDCT Image")
    axes[1, 1].axis('off')

    # Plot Reconstructed image
    axes[1, 2].imshow(cv.cvtColor(reconstructed, cv.COLOR_BGR2RGB))
    axes[1, 2].set_title("Reconstructed Image")
    axes[1, 2].axis('off')

    plt.tight_layout()
    plt.show()

# User input for image path and compression factor
image_path = input("Enter the image path: ")
compression_factor = float(input("Enter the compression factor (e.g., 1 for no compression, >1 for stronger compression): "))

# Load the color image
original_image = cv.imread(image_path)

# Ensure the image is loaded properly
if original_image is None:
    print("Error loading image!")
else:
    block_size = 8
    # Perform compression
    dct_image, quantized_image, dequantized_image, idct_image, reconstructed_image = compress_color_image(original_image, block_size, compression_factor)

    # Calculate PSNR
    psnr_value = calculate_psnr(original_image, reconstructed_image)
    print(f"PSNR: {psnr_value:.2f} dB")

    # Plot all stages
    plot_images(original_image, dct_image, quantized_image, dequantized_image, idct_image, reconstructed_image)