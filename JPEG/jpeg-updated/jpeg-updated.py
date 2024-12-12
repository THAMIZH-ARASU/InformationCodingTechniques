import numpy as np
from PIL import Image
from scipy.fftpack import dct, idct
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import os

# Standard JPEG luminance quantization matrix
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

def rgb_to_ycbcr(rgb):
    """Convert RGB to YCbCr color space."""
    rgb = rgb.astype(np.float32)
    y = 0.299 * rgb[:,:,0] + 0.587 * rgb[:,:,1] + 0.114 * rgb[:,:,2]
    cb = -0.1687 * rgb[:,:,0] - 0.3313 * rgb[:,:,1] + 0.5 * rgb[:,:,2] + 128
    cr = 0.5 * rgb[:,:,0] - 0.4187 * rgb[:,:,1] - 0.0813 * rgb[:,:,2] + 128
    return np.stack([y, cb, cr], axis=-1)

def ycbcr_to_rgb(ycbcr):
    """Convert YCbCr to RGB color space."""
    y = ycbcr[:,:,0]
    cb = ycbcr[:,:,1] - 128
    cr = ycbcr[:,:,2] - 128
    
    r = y + 1.402 * cr
    g = y - 0.34414 * cb - 0.71414 * cr
    b = y + 1.772 * cb
    
    rgb = np.stack([r, g, b], axis=-1)
    return np.clip(rgb, 0, 255).astype(np.uint8)

def pad_image(image, block_size=8):
    """Pad image to be divisible by block size."""
    h, w = image.shape[:2]
    new_h = int(np.ceil(h / block_size) * block_size)
    new_w = int(np.ceil(w / block_size) * block_size)
    
    padded = np.zeros((new_h, new_w, image.shape[2]), dtype=image.dtype)
    padded[:h, :w, :] = image
    return padded

def compress_channel(channel, quality=50):
    """Compress a single color channel."""
    block_size = 8
    compressed_blocks = []
    h, w = channel.shape
    
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            block = channel[i:i+block_size, j:j+block_size]
            if block.shape[0] == block_size and block.shape[1] == block_size:
                # Apply DCT
                dct_block = dct(dct(block.T, norm='ortho').T, norm='ortho')
                
                # Quantize
                q_factor = 50 if quality == 50 else (1 if quality > 50 else quality / 50)
                quantized_block = np.round(dct_block / (Q_MATRIX * q_factor))
                
                # Compress
                compressed_blocks.append(quantized_block)
    print(compressed_blocks)
    
    return compressed_blocks

def decompress_channel(compressed_blocks, original_shape, quality=50):
    """Decompress a single color channel."""
    block_size = 8
    h, w = original_shape
    decompressed = np.zeros((h, w), dtype=np.float32)
    
    block_index = 0
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            if block_index < len(compressed_blocks):
                # Dequantize
                q_factor = 50 if quality == 50 else (1 if quality > 50 else quality / 50)
                dequantized_block = compressed_blocks[block_index] * (Q_MATRIX * q_factor)
                
                # Apply IDCT
                idct_block = idct(idct(dequantized_block.T, norm='ortho').T, norm='ortho')
                
                # Place block in image
                decompressed[i:i+block_size, j:j+block_size] = idct_block
                block_index += 1
    
    return decompressed

class JPEGCompressionApp:
    def __init__(self, master):
        self.master = master
        master.title("JPEG Compression Tool")
        master.geometry("500x450")

        # Input Image Path
        self.input_path_var = tk.StringVar()
        self.input_path_label = tk.Label(master, textvariable=self.input_path_var, wraplength=400)
        self.input_path_label.pack(pady=10)

        # Compression Quality
        quality_frame = tk.Frame(master)
        quality_frame.pack(pady=5)
        
        tk.Label(quality_frame, text="Compression Quality:").pack(side=tk.LEFT)
        self.quality_var = tk.IntVar(value=50)
        self.quality_scale = ttk.Scale(
            quality_frame, 
            from_=1, 
            to=100, 
            variable=self.quality_var, 
            orient=tk.HORIZONTAL, 
            length=200
        )
        self.quality_scale.pack(side=tk.LEFT, padx=10)
        
        self.quality_label = tk.Label(quality_frame, textvariable=self.quality_var)
        self.quality_label.pack(side=tk.LEFT)

        # Buttons
        button_frame = tk.Frame(master)
        button_frame.pack(pady=10)

        tk.Button(button_frame, text="Select Image", command=self.select_image).pack(side=tk.LEFT, padx=5)
        tk.Button(button_frame, text="Compress", command=self.compress_image).pack(side=tk.LEFT, padx=5)

        # Log Area
        self.log_text = tk.Text(master, height=10, width=60)
        self.log_text.pack(pady=10)

    def log(self, message):
        """Log messages to the text widget."""
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)

    def select_image(self):
        """Open file dialog to select an image."""
        filepath = filedialog.askopenfilename(
            filetypes=[
                ("Image files", "*.bmp *.png *.jpg *.jpeg"),
                ("All files", "*.*")
            ]
        )
        if filepath:
            self.input_path_var.set(filepath)
            self.log(f"Selected image: {filepath}")

    def compress_image(self):
        """Perform JPEG-like compression."""
        # Validate input
        filepath = self.input_path_var.get()
        if not filepath:
            messagebox.showerror("Error", "Please select an image first")
            return

        try:
            # Read image
            original_image = Image.open(filepath).convert("RGB")
            image_array = np.array(original_image)
            
            # Get quality
            quality = self.quality_var.get()
            self.log(f"Compression quality: {quality}")

            # Convert to YCbCr
            ycbcr_image = rgb_to_ycbcr(image_array)
            
            # Pad image to be divisible by 8
            padded_ycbcr = pad_image(ycbcr_image)
            
            # Compress each channel
            compressed_y = compress_channel(padded_ycbcr[:,:,0], quality)
            compressed_cb = compress_channel(padded_ycbcr[:,:,1], quality)
            compressed_cr = compress_channel(padded_ycbcr[:,:,2], quality)
            
            # Decompress channels
            decompressed_y = decompress_channel(compressed_y, padded_ycbcr[:,:,0].shape, quality)
            decompressed_cb = decompress_channel(compressed_cb, padded_ycbcr[:,:,1].shape, quality)
            decompressed_cr = decompress_channel(compressed_cr, padded_ycbcr[:,:,2].shape, quality)
            
            # Reconstruct YCbCr image
            reconstructed_ycbcr = np.stack([
                decompressed_y, 
                decompressed_cb, 
                decompressed_cr
            ], axis=-1)
            
            # Convert back to RGB
            reconstructed_rgb = ycbcr_to_rgb(reconstructed_ycbcr)
            
            # Crop back to original size
            h, w = image_array.shape[:2]
            reconstructed_rgb = reconstructed_rgb[:h, :w, :]

            # Prepare save paths
            base_path = os.path.splitext(filepath)[0]
            compressed_path = f"{base_path}_compressed.jpg"
            reconstructed_path = f"{base_path}_reconstructed.bmp"

            # Save compressed (reconstructed) image
            Image.fromarray(reconstructed_rgb).save(compressed_path)
            self.log(f"Compressed image saved to: {compressed_path}")

            # Save original for comparison
            Image.fromarray(image_array).save(reconstructed_path)
            self.log(f"Original image saved to: {reconstructed_path}")

            messagebox.showinfo("Success", "Compression completed successfully!")

        except Exception as e:
            messagebox.showerror("Error", str(e))
            self.log(f"Error: {str(e)}")

def main():
    root = tk.Tk()
    app = JPEGCompressionApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()