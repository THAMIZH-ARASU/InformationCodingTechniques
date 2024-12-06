from PIL import Image

def compress_bmp_to_jpeg(input_bmp_path, output_jpeg_path):
    # Open the BMP image
    with Image.open(input_bmp_path) as img:
        # Save the image in JPEG format
        img.save(output_jpeg_path, "JPEG")
    print(f"Image compressed and saved to {output_jpeg_path}")

def decompress_jpeg_to_bmp(input_jpeg_path, output_bmp_path):
    # Open the JPEG image
    with Image.open(input_jpeg_path) as img:
        # Save the image in BMP format
        img.save(output_bmp_path, "BMP")
    print(f"Image decompressed and saved to {output_bmp_path}")

# Example usage
input_bmp_path = 'Birds.bmp'
output_jpeg_path = 'Birds.jpeg'
output_bmp_path = 'reconstructed.bmp'

compress_bmp_to_jpeg(input_bmp_path, output_jpeg_path)
decompress_jpeg_to_bmp(output_jpeg_path, output_bmp_path)
