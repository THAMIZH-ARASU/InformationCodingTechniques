import cv2
import numpy as np
import matplotlib.pyplot as plt

def rgb_to_ycbcr(image):
    """Converts an RGB image to YCbCr color space."""
    return cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)

def ycbcr_to_rgb(image):
    """Converts a YCbCr image to RGB color space."""
    return cv2.cvtColor(image, cv2.COLOR_YCrCb2RGB)

def chroma_subsampling_400(ycbcr_image):
    """Applies 4:0:0 chroma subsampling on the YCbCr image."""
    y, cb, cr = cv2.split(ycbcr_image)
    
    # Set the chroma channels (Cb and Cr) to zero
    cb_zeros = np.zeros_like(cb)
    cr_zeros = np.zeros_like(cr)
    
    # Merge back the channels with zeroed chroma
    subsampled_image = cv2.merge((y, cb_zeros, cr_zeros))
    return subsampled_image


def chroma_subsampling_420(ycbcr_image):
    """Applies 4:2:0 chroma subsampling on the YCbCr image."""
    y, cb, cr = cv2.split(ycbcr_image)
    
    # Downsample the Cb and Cr channels
    cb_downsampled = cv2.resize(cb, (cb.shape[1] // 4, cb.shape[0] // 4), interpolation=cv2.INTER_AREA)
    cr_downsampled = cv2.resize(cr, (cr.shape[1] // 4, cr.shape[0] // 4), interpolation=cv2.INTER_AREA)

    # Upsample back to original size
    cb_upsampled = cv2.resize(cb_downsampled, (cb.shape[1], cb.shape[0]), interpolation=cv2.INTER_LINEAR)
    cr_upsampled = cv2.resize(cr_downsampled, (cr.shape[1], cr.shape[0]), interpolation=cv2.INTER_LINEAR)

    # Merge the channels back together
    subsampled_image = cv2.merge((y, cb_upsampled, cr_upsampled))
    return subsampled_image

def chroma_subsampling_411(ycbcr_image):
    """Applies 4:1:1 chroma subsampling on the YCbCr image."""
    y, cb, cr = cv2.split(ycbcr_image)
    
    # Downsample the Cb and Cr channels horizontally by a factor of 4
    cb_downsampled = cv2.resize(cb, (cb.shape[1] // 4, cb.shape[0]), interpolation=cv2.INTER_AREA)
    cr_downsampled = cv2.resize(cr, (cr.shape[1] // 4, cr.shape[0]), interpolation=cv2.INTER_AREA)
    
    # Upsample back to the original size
    cb_upsampled = cv2.resize(cb_downsampled, (cb.shape[1], cb.shape[0]), interpolation=cv2.INTER_LINEAR)
    cr_upsampled = cv2.resize(cr_downsampled, (cr.shape[1], cr.shape[0]), interpolation=cv2.INTER_LINEAR)
    
    # Merge the channels back together
    subsampled_image = cv2.merge((y, cb_upsampled, cr_upsampled))
    return subsampled_image

def save_image(filename, image):
    """Saves an image to a file."""
    cv2.imwrite(filename, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

def main():
    # Step 1: Read the BMP image
    input_filename = "input.bmp"
    original_image = cv2.imread(input_filename)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

    # Step 2: Convert RGB to YCbCr
    ycbcr_image = rgb_to_ycbcr(original_image)
    save_image("ycbcr_image.bmp", ycbcr_image)

    # Step 3: Apply chroma subsampling (4:2:0)
    #subsampled_image = chroma_subsampling_420(ycbcr_image)
    #save_image("subsampled_image.bmp", subsampled_image)

    # Step 3: Apply chroma subsampling (4:0:0)
    #subsampled_image = chroma_subsampling_400(ycbcr_image)
    #save_image("subsampled_image.bmp", subsampled_image)

    # Step 3: Apply chroma subsampling (4:1:1)
    subsampled_image = chroma_subsampling_411(ycbcr_image)
    save_image("subsampled_image.bmp", subsampled_image)

    # Step 4: Convert back to RGB
    reconstructed_rgb_image = ycbcr_to_rgb(subsampled_image)
    save_image("reconstructed_rgb_image.bmp", reconstructed_rgb_image)

    # Plot all images
    fig, axes = plt.subplots(1, 4, figsize=(16, 8))

    axes[0].imshow(original_image)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    axes[1].imshow(ycbcr_image)
    axes[1].set_title("YCbCr Image")
    axes[1].axis("off")

    axes[2].imshow(subsampled_image)
    axes[2].set_title("Subsampled YCbCr Image")
    axes[2].axis("off")

    axes[3].imshow(reconstructed_rgb_image)
    axes[3].set_title("Reconstructed RGB Image")
    axes[3].axis("off")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
