import numpy as np
from PIL import Image
from scipy.fftpack import dct, idct
import heapq

# Huffman encoding utilities
class HuffmanNode:
    def __init__(self, symbol, freq):
        self.symbol = symbol
        self.freq = freq
        self.left = None
        self.right = None

    def __lt__(self, other):
        return self.freq < other.freq

def build_huffman_tree(frequency):
    heap = [HuffmanNode(symbol, freq) for symbol, freq in frequency.items()]
    heapq.heapify(heap)

    while len(heap) > 1:
        left = heapq.heappop(heap)
        right = heapq.heappop(heap)
        merged = HuffmanNode(None, left.freq + right.freq)
        merged.left = left
        merged.right = right
        heapq.heappush(heap, merged)

    return heap[0]

def build_huffman_codes(node, prefix="", codes={}):
    if node is None:
        return
    if node.symbol is not None:
        codes[node.symbol] = prefix
    build_huffman_codes(node.left, prefix + "0", codes)
    build_huffman_codes(node.right, prefix + "1", codes)
    return codes

def huffman_encode(data):
    data = [str(item) for item in data]
    frequency = {symbol: data.count(symbol) for symbol in set(data)}
    tree = build_huffman_tree(frequency)
    codes = build_huffman_codes(tree)
    encoded_data = "".join([codes[symbol] for symbol in data])
    return encoded_data, tree

def huffman_decode(encoded_data, tree):
    decoded_data = []
    node = tree
    for bit in encoded_data:
        node = node.left if bit == "0" else node.right
        if node.symbol is not None:
            decoded_data.append(eval(node.symbol))  # Convert back to tuple or original type
            node = tree
    return decoded_data

# Quantization matrix
QUANTIZATION_MATRIX = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99]
])

# Image processing functions
def rgb_to_ycbcr(img):
    return img.convert("YCbCr")

def ycbcr_to_rgb(img):
    return img.convert("RGB")

def split_into_blocks(img_array, block_size=8):
    h, w = img_array.shape[:2]
    blocks = []
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            blocks.append(img_array[i:i+block_size, j:j+block_size])
    return blocks

def merge_blocks(blocks, img_shape, block_size=8):
    h, w = img_shape[:2]
    img_array = np.zeros((h, w, img_shape[2]), dtype=np.float32)
    index = 0
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            img_array[i:i+block_size, j:j+block_size] = blocks[index]
            index += 1
    return img_array

def apply_dct(block):
    if block.ndim == 3:
        return np.stack([dct(dct(block[..., c].T, norm='ortho').T, norm='ortho') for c in range(block.shape[-1])], axis=-1)
    else:
        return dct(dct(block.T, norm='ortho').T, norm='ortho')

def apply_idct(block):
    if block.ndim == 3:
        return np.stack([idct(idct(block[..., c].T, norm='ortho').T, norm='ortho') for c in range(block.shape[-1])], axis=-1)
    else:
        return idct(idct(block.T, norm='ortho').T, norm='ortho')

def quantize(block):
    return np.round(block / QUANTIZATION_MATRIX).astype(int)

def dequantize(block):
    return (block * QUANTIZATION_MATRIX).astype(float)

def run_length_encode(block):
    flattened = block.flatten()
    rle = []
    count = 1
    for i in range(1, len(flattened)):
        if flattened[i] == flattened[i - 1]:
            count += 1
        else:
            rle.append((flattened[i - 1], count))
            count = 1
    rle.append((flattened[-1], count))
    return rle

def run_length_decode(rle, shape):
    decoded = []
    for value, count in rle:
        decoded.extend([value] * count)
    return np.array(decoded).reshape(shape)

def process_image(input_file, output_file):
    # Load the BMP image
    img = Image.open(input_file)
    ycbcr_img = rgb_to_ycbcr(img)
    img_array = np.array(ycbcr_img, dtype=np.float32)

    # Split into 8x8 blocks
    blocks = split_into_blocks(img_array)

    # Process each block
    processed_blocks = []
    rle_encoded_blocks = []
    huffman_trees = []
    for block in blocks:
        block_dct = apply_dct(block)
        block_quantized = quantize(block_dct)
        rle = run_length_encode(block_quantized)
        huffman_encoded, tree = huffman_encode(rle)
        processed_blocks.append(block_quantized)
        rle_encoded_blocks.append(huffman_encoded)
        huffman_trees.append(tree)

    # Decode and reconstruct the image
    decoded_blocks = []
    for encoded_rle, tree, quantized_block in zip(rle_encoded_blocks, huffman_trees, processed_blocks):
        rle_decoded = huffman_decode(encoded_rle, tree)
        block_dequantized = dequantize(run_length_decode(rle_decoded, quantized_block.shape))
        block_idct = apply_idct(block_dequantized)
        decoded_blocks.append(block_idct)

    # Merge blocks and save the decoded image
    decoded_img_array = merge_blocks(decoded_blocks, img_array.shape)
    decoded_img = Image.fromarray(np.clip(decoded_img_array, 0, 255).astype(np.uint8), "YCbCr")
    decoded_img = ycbcr_to_rgb(decoded_img)
    decoded_img.save(output_file)

# Example usage
input_bmp = "input.bmp"
output_bmp = "output.bmp"
process_image(input_bmp, output_bmp)
