import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import os
import math

def compress_color_image(image: np.ndarray, block_size: int, compression_factor: float):
    # Split the image into RGB channels
    channels = cv.split(image)
    
    dct_channels = []
    quantized_channels = []
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
        # Split the image into 8x8 blocks
        h, w = ch.shape
        dct_block = np.zeros_like(ch, dtype=np.float32)
        
        for i in range(0, h, block_size):
            for j in range(0, w, block_size):
                block = ch[i:i+block_size, j:j+block_size]
                
                # Perform DCT on each block
                dct_block[i:i+block_size, j:j+block_size] = cv.dct(np.float32(block))
                
        dct_channels.append(dct_block)
        
        # Quantization step
        quantized_block = np.zeros_like(dct_block, dtype=np.float32)
        for i in range(0, h, block_size):
            for j in range(0, w, block_size):
                block = dct_block[i:i+block_size, j:j+block_size]
                
                # Quantize each 8x8 block
                quantized_block[i:i+block_size, j:j+block_size] = np.round(block / quantization_matrix)
        
        quantized_channels.append(quantized_block)
        
        # Inverse DCT to reconstruct the image
        reconstructed_block = np.zeros_like(quantized_block, dtype=np.float32)
        for i in range(0, h, block_size):
            for j in range(0, w, block_size):
                block = quantized_block[i:i+block_size, j:j+block_size]
                
                # Apply inverse DCT
                reconstructed_block[i:i+block_size, j:j+block_size] = cv.idct(block * quantization_matrix)
        
        reconstructed_channels.append(reconstructed_block)
        
    # Merge channels back together
    dct_image = cv.merge(dct_channels)
    quantized_image = cv.merge(quantized_channels)
    reconstructed_image = cv.merge(reconstructed_channels)
    
    # Clip pixel values to valid range [0, 255] after processing
    reconstructed_image = np.clip(reconstructed_image, 0, 255).astype(np.uint8)
    
    return dct_image, quantized_image, reconstructed_image

def calculate_psnr(original: np.ndarray, reconstructed: np.ndarray) -> float:
    mse = np.mean((original.astype(np.float32) - reconstructed.astype(np.float32)) ** 2)
    if mse == 0:
        return 100  # Perfect match
    return 10 * math.log10(255 ** 2 / mse)

def calculate_compression_ratio(original: np.ndarray, quantized: np.ndarray) -> float:
    original_size = original.size * original.itemsize  # Total number of elements * size of each element (bytes)
    quantized_size = quantized.size * quantized.itemsize
    return original_size / quantized_size

def plot_images(original: np.ndarray, dct: np.ndarray, quantized: np.ndarray, reconstructed: np.ndarray, psnr_value: float, compression_ratio: float):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Plot original image
    axes[0, 0].imshow(cv.cvtColor(original, cv.COLOR_BGR2RGB))  # Convert from BGR to RGB
    axes[0, 0].set_title("Original Image")
    axes[0, 0].axis('off')
    
    # Plot DCT image
    axes[0, 1].imshow(cv.cvtColor(dct, cv.COLOR_BGR2RGB))  # Convert from BGR to RGB
    axes[0, 1].set_title("DCT Image")
    axes[0, 1].axis('off')
    
    # Plot Quantized image
    axes[1, 0].imshow(cv.cvtColor(quantized, cv.COLOR_BGR2RGB))  # Convert from BGR to RGB
    axes[1, 0].set_title("Quantized Image")
    axes[1, 0].axis('off')
    
    # Plot Reconstructed image
    axes[1, 1].imshow(cv.cvtColor(reconstructed, cv.COLOR_BGR2RGB))  # Convert from BGR to RGB
    axes[1, 1].set_title("Reconstructed Image")
    axes[1, 1].axis('off')
    
    # Align PSNR and Compression Ratio text
    text_psnr = f'PSNR: {psnr_value:.2f} dB'
    text_cr = f'Compression Ratio: {compression_ratio:.2f}'
    
    # Add text with appropriate alignment
    plt.figtext(0.5, 0.96, text_psnr, ha='center', fontsize=14, color='green', va='bottom')
    plt.figtext(0.5, 0.93, text_cr, ha='center', fontsize=14, color='blue', va='bottom')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.88)  # Adjust space for the text area
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
    # Compression and reconstruction
    block_size = 8
    dct_image, quantized_image, reconstructed_image = compress_color_image(original_image, block_size, compression_factor)

    # Calculate compression ratio and PSNR
    compression_ratio = calculate_compression_ratio(original_image, quantized_image)
    psnr_value = calculate_psnr(original_image, reconstructed_image)

    # Display compression ratio and PSNR
    print(f"Compression Ratio: {compression_ratio:.2f}")
    print(f"PSNR: {psnr_value:.2f} dB")

    # Plot images with PSNR and compression ratio values
    plot_images(original_image, dct_image, quantized_image, reconstructed_image, psnr_value, compression_ratio)
    
    # Save images (optional)
    cv.imwrite('original_image.jpg', original_image)
    cv.imwrite('dct_image.jpg', dct_image)
    cv.imwrite('quantized_image.jpg', quantized_image)
    cv.imwrite('reconstructed_image.jpg', reconstructed_image)
