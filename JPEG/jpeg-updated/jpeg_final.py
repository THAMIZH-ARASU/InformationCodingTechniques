import numpy as np
from PIL import Image
from scipy.fftpack import dct, idct
import heapq
from collections import defaultdict
import os
import json


# Zigzag pattern mapping for an 8x8 block
zigzag_indices = [
    (0, 0), (0, 1), (1, 0), (2, 0), (1, 1), (0, 2), (0, 3), (1, 2),
    (2, 1), (3, 0), (4, 0), (3, 1), (2, 2), (1, 3), (0, 4), (0, 5),
    (1, 4), (2, 3), (3, 2), (4, 1), (5, 0), (6, 0), (5, 1), (4, 2),
    (3, 3), (2, 4), (1, 5), (0, 6), (0, 7), (1, 6), (2, 5), (3, 4),
    (4, 3), (5, 2), (6, 1), (7, 0), (7, 1), (6, 2), (5, 3), (4, 4),
    (3, 5), (2, 6), (1, 7), (2, 7), (3, 6), (4, 5), (5, 4), (6, 3),
    (7, 2), (7, 3), (6, 4), (5, 5), (4, 6), (3, 7), (4, 7), (5, 6),
    (6, 5), (7, 4), (7, 5), (6, 6), (5, 7), (6, 7), (7, 6), (7, 7)
]

Q_MATRIX = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99]
])

def zigzag_order(block):
    """Apply zigzag ordering to an 8x8 block."""
    return [block[i][j] for i, j in zigzag_indices]

def inverse_zigzag_order(data, size=8):
    """Reconstruct an 8x8 block from zigzag ordering."""
    block = np.zeros((size, size))
    for idx, (i, j) in enumerate(zigzag_indices):
        block[i][j] = data[idx]
    return block

def huffman_encode(data):
    """Perform Huffman encoding."""
    if not data:  # Handle empty data gracefully
        return "", {}

    # Frequency dictionary
    freq = defaultdict(int)
    for value in data:
        freq[value] += 1

    # Priority queue (heap)
    heap = [[weight, [symbol, ""]] for symbol, weight in freq.items()]
    heapq.heapify(heap)

    while len(heap) > 1:
        lo = heapq.heappop(heap)
        hi = heapq.heappop(heap)
        for pair in lo[1:]:
            pair[1] = "0" + pair[1]
        for pair in hi[1:]:
            pair[1] = "1" + pair[1]
        heapq.heappush(heap, [lo[0] + hi[0]] + lo[1:] + hi[1:])

    # Validate heap structure
    if len(heap) != 1 or not heap[0][1:]:
        raise ValueError("Invalid heap structure during Huffman encoding")

    # Extract symbol-code mapping
    huffman_dict = {}
    for entry in heap[0][1:]:
        if len(entry) != 2:
            raise ValueError(f"Unexpected entry in heap: {entry}")
        symbol, code = entry
        huffman_dict[symbol] = code

    # Encode the data
    encoded_data = "".join(huffman_dict[value] for value in data)
  
    return encoded_data, huffman_dict


def huffman_decode(encoded_data, huffman_dict):
    """Decode Huffman encoded data."""
    reverse_dict = {code: symbol for symbol, code in huffman_dict.items()}
    decoded_data = []
    buffer = ""
    for bit in encoded_data:
        buffer += bit
        if buffer in reverse_dict:
            decoded_data.append(reverse_dict[buffer])
            buffer = ""
    return decoded_data

def rgb_to_ycbcr(rgb):
    rgb = rgb.astype(np.float32)
    y = 0.299 * rgb[:,:,0] + 0.587 * rgb[:,:,1] + 0.114 * rgb[:,:,2]
    cb = -0.1687 * rgb[:,:,0] - 0.3313 * rgb[:,:,1] + 0.5 * rgb[:,:,2] + 128
    cr = 0.5 * rgb[:,:,0] - 0.4187 * rgb[:,:,1] - 0.0813 * rgb[:,:,2] + 128
    return np.stack([y, cb, cr], axis=-1)

def ycbcr_to_rgb(ycbcr):
    y = ycbcr[:,:,0]
    cb = ycbcr[:,:,1] - 128
    cr = ycbcr[:,:,2] - 128
    r = y + 1.402 * cr
    g = y - 0.34414 * cb - 0.71414 * cr
    b = y + 1.772 * cb
    return np.clip(np.stack([r, g, b], axis=-1), 0, 255).astype(np.uint8)

def pad_image(image, block_size=8):
    h, w = image.shape[:2]
    new_h = int(np.ceil(h / block_size) * block_size)
    new_w = int(np.ceil(w / block_size) * block_size)
    padded = np.zeros((new_h, new_w, image.shape[2]), dtype=image.dtype)
    padded[:h, :w, :] = image
    return padded

def process_channel(channel, quality=50):
    """Compress and decompress a single channel."""
    block_size = 8
    h, w = channel.shape
    compressed_blocks = []
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            block = channel[i:i+block_size, j:j+block_size]
            dct_block = dct(dct(block.T, norm='ortho').T, norm='ortho')
            q_factor = 50 / quality
            quantized = np.round(dct_block / (Q_MATRIX * q_factor))
            zigzagged = zigzag_order(quantized)
            compressed_blocks.extend(zigzagged)
    return compressed_blocks


def decompress_channel(compressed_data, huffman_dict, original_shape, quality=50):
    """Decompress a single channel from compressed data."""
    block_size = 8
    h, w = original_shape

    # Decode Huffman
    decoded_data = huffman_decode(compressed_data, huffman_dict)

    # Reconstruct blocks
    decompressed = np.zeros((h, w), dtype=np.float32)
    block_index = 0
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            if block_index < len(decoded_data) // (block_size * block_size):
                # Zigzag to block
                zigzagged_block = decoded_data[block_index * (block_size * block_size):(block_index + 1) * (block_size * block_size)]
                quantized_block = inverse_zigzag_order(zigzagged_block)
                
                # Dequantize
                q_factor = 50 / quality
                dequantized_block = quantized_block * (Q_MATRIX * q_factor)
                
                # IDCT
                idct_block = idct(idct(dequantized_block.T, norm='ortho').T, norm='ortho')
                
                # Place block in image
                decompressed[i:i+block_size, j:j+block_size] = idct_block
                block_index += 1
    return decompressed

def compress_image(input_path, quality):
    """Compress an image and return compressed data."""
    img = Image.open(input_path).convert("RGB")
    img_array = np.array(img)

    # Convert to YCbCr
    ycbcr = rgb_to_ycbcr(img_array)

    # Process channels
    y_blocks = process_channel(ycbcr[:,:,0], quality)
    cb_blocks = process_channel(ycbcr[:,:,1], quality)
    cr_blocks = process_channel(ycbcr[:,:,2], quality)

    # Huffman encode
    y_encoded, y_huffman_dict = huffman_encode(y_blocks)
    cb_encoded, cb_huffman_dict = huffman_encode(cb_blocks)
    cr_encoded, cr_huffman_dict = huffman_encode(cr_blocks)

    return {
        "encoded_data": {
            "y": y_encoded, "cb": cb_encoded, "cr": cr_encoded
        },
        "huffman_dicts": {
            "y": y_huffman_dict, "cb": cb_huffman_dict, "cr": cr_huffman_dict
        },
        "original_shape": ycbcr.shape
    }

def decompress_image(compressed_data, output_path, quality):
    """Decompress an image from compressed data."""
    encoded_data = compressed_data["encoded_data"]
    huffman_dicts = compressed_data["huffman_dicts"]
    original_shape = compressed_data["original_shape"]

    # Decompress channels
    y = decompress_channel(encoded_data["y"], huffman_dicts["y"], original_shape[:2], quality)
    cb = decompress_channel(encoded_data["cb"], huffman_dicts["cb"], original_shape[:2], quality)
    cr = decompress_channel(encoded_data["cr"], huffman_dicts["cr"], original_shape[:2], quality)

    # Reconstruct YCbCr
    ycbcr = np.stack([y, cb, cr], axis=-1)

    # Convert back to RGB
    reconstructed_rgb = ycbcr_to_rgb(ycbcr)

    # Save reconstructed image
    Image.fromarray(reconstructed_rgb.astype(np.uint8)).save(output_path)
    print(f"Decompressed image saved to {output_path}")


if __name__ == "__main__":
    # Define paths and quality
    input_path = "Birds.bmp"  # Input BMP image
    compressed_data_path = "compressed_data.json"  # Path to save compressed data
    decompressed_output_path = "reconstructed_image.bmp"  # Path to save decompressed BMP
    compressed_jpg_output_path = "compressed_image.jpg"  # Path to save JPEG
    quality = 50  # Compression quality

    try:
        # Step 1: Load the input image
        img = Image.open(input_path).convert("RGB")
        img_array = np.array(img)

        # Step 2: Convert image to YCbCr color space
        ycbcr = rgb_to_ycbcr(img_array)

        # Step 3: Compress each channel (Y, Cb, Cr)
        y_blocks = process_channel(ycbcr[:, :, 0], quality)
        cb_blocks = process_channel(ycbcr[:, :, 1], quality)
        cr_blocks = process_channel(ycbcr[:, :, 2], quality)

        # Step 4: Huffman encode each channel
        print(f"Processing Y channel blocks for Huffman encoding...")
        y_encoded, y_huffman_dict = huffman_encode(y_blocks)
       

        print(f"Processing Cb channel blocks for Huffman encoding...")
        cb_encoded, cb_huffman_dict = huffman_encode(cb_blocks)
        

        print(f"Processing Cr channel blocks for Huffman encoding...")
        cr_encoded, cr_huffman_dict = huffman_encode(cr_blocks)
       
        # Step 5: Save the compressed data to a file
        compressed_data = {
            "encoded_data": {
                "y": y_encoded,
                "cb": cb_encoded,
                "cr": cr_encoded
            },
            "huffman_dicts": {
                "y": y_huffman_dict,
                "cb": cb_huffman_dict,
                "cr": cr_huffman_dict
            },
            "original_shape": ycbcr.shape
        }
        """
        with open(compressed_data_path, "w") as f:
            json.dump(compressed_data, f)
        print(f"Compressed data saved to {compressed_data_path}")

        # Step 6: Load compressed data for decompression
        with open(compressed_data_path, "r") as f:
            compressed_data = json.load(f)
        """
        

        # Decode and reconstruct each channel
        y = decompress_channel(compressed_data["encoded_data"]["y"], compressed_data["huffman_dicts"]["y"],
                               tuple(compressed_data["original_shape"][:2]), quality)
        cb = decompress_channel(compressed_data["encoded_data"]["cb"], compressed_data["huffman_dicts"]["cb"],
                                tuple(compressed_data["original_shape"][:2]), quality)
        cr = decompress_channel(compressed_data["encoded_data"]["cr"], compressed_data["huffman_dicts"]["cr"],
                                tuple(compressed_data["original_shape"][:2]), quality)

        # Combine channels into a YCbCr image
        ycbcr_reconstructed = np.stack([y, cb, cr], axis=-1)

        # Convert back to RGB
        rgb_reconstructed = ycbcr_to_rgb(ycbcr_reconstructed)

        # Step 7: Save the reconstructed image as BMP
        Image.fromarray(rgb_reconstructed.astype(np.uint8)).save(decompressed_output_path)
        print(f"Decompressed image saved to {decompressed_output_path}")

        # Step 8: Save the compressed version as JPEG
        compressed_jpg = Image.fromarray(rgb_reconstructed.astype(np.uint8))
        compressed_jpg.save(compressed_jpg_output_path, format="JPEG", quality=quality)
        print(f"Compressed JPEG image saved to {compressed_jpg_output_path}")

    except Exception as e:
        print(f"Error: {str(e)}")