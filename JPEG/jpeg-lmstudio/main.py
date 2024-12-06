import numpy as np
import cv2

def rgb_to_ycbcr(image):
    R = image[:, :, 0]
    G = image[:, :, 1]
    B = image[:, :, 2]

    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Cb = -0.168736 * R - 0.331264 * G + 0.5 * B + 128
    Cr = 0.5 * R - 0.418688 * G - 0.081312 * B + 128

    return np.dstack((Y, Cb, Cr)).astype(np.uint8)

def ycbcr_to_rgb(ycbcr_image):
    Y = ycbcr_image[:, :, 0]
    Cb = ycbcr_image[:, :, 1] - 128
    Cr = ycbcr_image[:, :, 2] - 128

    R = Y + 1.402 * Cr
    G = Y - 0.34414 * Cb - 0.71414 * Cr
    B = Y + 1.772 * Cb

    return np.dstack((R, G, B)).clip(0, 255).astype(np.uint8)

def dct_2d(block):
    return cv2.dct(block.astype(np.float32))

def idct_2d(block):
    return cv2.idct(block.astype(np.float32))

def quantize(block, quality_factor):
    if quality_factor < 50:
        scale = 5000 // quality_factor
    else:
        scale = 200 - 2 * quality_factor

    q_matrix = np.array([
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 61, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99]
    ])
    q_matrix = (q_matrix * scale + 50) // 100
    return np.round(block / q_matrix)

def dequantize(block, quality_factor):
    if quality_factor < 50:
        scale = 5000 // quality_factor
    else:
        scale = 200 - 2 * quality_factor

    q_matrix = np.array([
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 61, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99]
    ])
    q_matrix = (q_matrix * scale + 50) // 100
    return block * q_matrix

def jpeg_compress(image, quality_factor=50):
    ycbcr_image = rgb_to_ycbcr(image)
    compressed_image = np.zeros_like(ycbcr_image, dtype=np.float32)

    for c in range(3):
        channel = ycbcr_image[:, :, c].astype(np.float32)
        height, width = channel.shape

        for i in range(0, height, 8):
            for j in range(0, width, 8):
                block = channel[i:i + 8, j:j + 8]
                if block.shape == (8, 8):  # Ensure full blocks
                    dct_block = dct_2d(block)
                    quantized_block = quantize(dct_block, quality_factor)
                    compressed_image[i:i + 8, j:j + 8, c] = quantized_block

    return compressed_image.astype(np.int32), ycbcr_image

def jpeg_decompress(compressed_image, original_image, quality_factor=50):
    height, width, _ = original_image.shape
    decompressed_image = np.zeros_like(original_image, dtype=np.float32)

    for c in range(3):
        channel = compressed_image[:, :, c].astype(np.float32)

        for i in range(0, height, 8):
            for j in range(0, width, 8):
                block = channel[i:i + 8, j:j + 8]
                if block.shape == (8, 8):  # Ensure full blocks
                    dequantized_block = dequantize(block, quality_factor)
                    idct_block = idct_2d(dequantized_block)
                    decompressed_image[i:i + 8, j:j + 8, c] = idct_block

    return ycbcr_to_rgb(decompressed_image).astype(np.uint8)

# Example usage
if __name__ == "__main__":
    # Load an image
    image = cv2.imread('Lena256B.bmp')

    # Compress the image
    compressed_image, ycbcr_image = jpeg_compress(image, quality_factor=50)

    # Decompress the image
    decompressed_image = jpeg_decompress(compressed_image, ycbcr_image, quality_factor=50)

    # Save the decompressed image
    cv2.imwrite('output.jpg', decompressed_image)
