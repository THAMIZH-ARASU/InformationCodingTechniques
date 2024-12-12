import numpy as np
import cv2

# Function to apply DCT in 8x8 blocks
def apply_dct(image):
    # Convert the image to float32 (required for DCT)
    image_float = np.float32(image)
    
    # Get the dimensions of the image
    height, width = image.shape

    # Create an empty array to store DCT coefficients
    dct_coefficients = np.zeros_like(image_float)

    # Apply DCT in 8x8 blocks
    for i in range(0, height, 8):
        for j in range(0, width, 8):
            block = image_float[i:i+8, j:j+8]
            # Apply DCT to the block and store the result
            dct_block = cv2.dct(block)
            dct_coefficients[i:i+8, j:j+8] = dct_block

    return dct_coefficients

# Read the input image
image = cv2.imread('Input.bmp', cv2.IMREAD_GRAYSCALE)

# Apply DCT to the image
dct_coefficients = apply_dct(image)

# Write the DCT coefficients to a file
with open('dct.txt', 'w') as f:
    # Write the DCT coefficients in a readable format
    for row in dct_coefficients:
        f.write(' '.join([f'{x:0.2f}' for x in row]) + '\n')

print("DCT coefficients have been written to dct.txt.")
