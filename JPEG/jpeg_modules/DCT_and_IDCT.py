import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt

# Function to apply DCT to an image
def apply_dct(image):
    # Convert the image to YCrCb color space
    image_ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    
    # Split the image into Y, Cr, and Cb channels
    y, cr, cb = cv2.split(image_ycrcb)
    
    # Convert the channels to floating-point format
    y = np.float32(y)
    cr = np.float32(cr)
    cb = np.float32(cb)
    
    # Apply DCT to each channel
    dct_y = cv2.dct(y)
    dct_cr = cv2.dct(cr)
    dct_cb = cv2.dct(cb)
    
    # Combine the DCT channels
    dct_image = cv2.merge([dct_y, dct_cr, dct_cb])
    
    return dct_image

# Function to apply inverse DCT to an image
def apply_idct(image):
    # Split the image into Y, Cr, and Cb channels
    y, cr, cb = cv2.split(image)
    
    # Apply inverse DCT to each channel
    idct_y = cv2.idct(y)
    idct_cr = cv2.idct(cr)
    idct_cb = cv2.idct(cb)
    
    # Combine the IDCT channels
    idct_image = cv2.merge([idct_y, idct_cr, idct_cb])
    
    # Convert the image back to BGR color space
    idct_image = cv2.cvtColor(idct_image, cv2.COLOR_YCrCb2BGR)
    
    return idct_image

# Read the BMP file
image = cv2.imread('input.bmp')

# Apply DCT to the image
dct_image = apply_dct(image)

# Apply inverse DCT to the original image
idct_original = apply_idct(apply_dct(image))

# Apply inverse DCT to the DCT image
idct_dct = apply_idct(dct_image)

# Save the result as a JPEG file
cv2.imwrite('output.jpg', idct_dct)

# Display the original, DCT on original, IDCT on original, and IDCT on DCT images
plt.figure(figsize=(15, 10))
plt.subplot(2, 2, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Original Image')
plt.subplot(2, 2, 2)
plt.imshow(cv2.cvtColor(dct_image, cv2.COLOR_YCrCb2RGB))
plt.title('DCT on Original')
plt.subplot(2, 2, 3)
plt.imshow(cv2.cvtColor(idct_original, cv2.COLOR_BGR2RGB))
plt.title('IDCT on Original')
plt.subplot(2, 2, 4)
plt.imshow(cv2.cvtColor(idct_dct, cv2.COLOR_BGR2RGB))
plt.title('IDCT on DCT')
plt.show()