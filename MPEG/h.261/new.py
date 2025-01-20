import os
import numpy as np
import cv2

# Parameters
BLOCK_SIZE = 16  # Size of macroblocks used in motion estimation
SEARCH_RANGE = 8  # Range for block matching in motion estimation

# Helper function for block matching
def block_matching(ref_frame, target_block, x, y, search_range):
    # Ensure target_block is complete
    if target_block.shape != (BLOCK_SIZE, BLOCK_SIZE):
        return (0, 0)  # Return zero motion vector for incomplete blocks
        
    min_mse = float('inf')
    best_match = (0, 0)

    for dx in range(-search_range, search_range + 1):
        for dy in range(-search_range, search_range + 1):
            x_offset = x + dx
            y_offset = y + dy

            if (
                0 <= x_offset <= ref_frame.shape[1] - BLOCK_SIZE
                and 0 <= y_offset <= ref_frame.shape[0] - BLOCK_SIZE
            ):
                ref_block = ref_frame[y_offset:y_offset + BLOCK_SIZE, x_offset:x_offset + BLOCK_SIZE]
                
                # Verify shapes match before calculating MSE
                if ref_block.shape == target_block.shape:
                    mse = np.mean((target_block.astype(float) - ref_block.astype(float)) ** 2)

                    if mse < min_mse:
                        min_mse = mse
                        best_match = (dx, dy)

    return best_match

# Motion estimation
def motion_estimation(ref_frame, target_frame):
    height, width = target_frame.shape
    motion_vectors = []

    # Adjust the range to ensure we only process complete blocks
    height_range = height - (height % BLOCK_SIZE)
    width_range = width - (width % BLOCK_SIZE)

    for y in range(0, height_range, BLOCK_SIZE):
        for x in range(0, width_range, BLOCK_SIZE):
            target_block = target_frame[y:y + BLOCK_SIZE, x:x + BLOCK_SIZE]
            
            # Only process if we have a complete block
            if target_block.shape == (BLOCK_SIZE, BLOCK_SIZE):
                dx, dy = block_matching(ref_frame, target_block, x, y, SEARCH_RANGE)
                motion_vectors.append(((x, y), (dx, dy)))

    return motion_vectors

# Motion compensation
def motion_compensation(ref_frame, motion_vectors):
    height, width = ref_frame.shape
    compensated_frame = np.zeros((height, width), dtype=np.uint8)

    for (x, y), (dx, dy) in motion_vectors:
        # Check boundaries before copying blocks
        if (y + dy + BLOCK_SIZE <= ref_frame.shape[0] and 
            x + dx + BLOCK_SIZE <= ref_frame.shape[1] and 
            y + BLOCK_SIZE <= compensated_frame.shape[0] and 
            x + BLOCK_SIZE <= compensated_frame.shape[1]):
            
            ref_block = ref_frame[y + dy:y + dy + BLOCK_SIZE, x + dx:x + dx + BLOCK_SIZE]
            compensated_frame[y:y + BLOCK_SIZE, x:x + BLOCK_SIZE] = ref_block

    return compensated_frame

# Intra-frame coding (using DCT)
def intra_frame_coding(frame):
    # Convert to float32 for DCT
    float_frame = np.float32(frame)
    # Process the frame in 8x8 blocks to match typical DCT processing
    height, width = float_frame.shape
    dct_frame = np.zeros_like(float_frame)
    
    for y in range(0, height - 7, 8):
        for x in range(0, width - 7, 8):
            block = float_frame[y:y + 8, x:x + 8]
            dct_frame[y:y + 8, x:x + 8] = cv2.dct(block)
    
    return dct_frame

# Main process
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file: {video_path}")
        return

    # Create output directory for frames
    output_dir = os.path.splitext(video_path)[0] + "_frames"
    os.makedirs(output_dir, exist_ok=True)

    ret, prev_frame = cap.read()
    if not ret:
        print("Failed to read the first frame.")
        cap.release()
        return

    prev_frame_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    frame_index = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Save original frame as image
        cv2.imwrite(os.path.join(output_dir, f"original_frame_{frame_index:04d}.png"), frame_gray)

        # Intra-frame coding
        intra_coded_frame = intra_frame_coding(frame_gray)

        # Motion estimation
        motion_vectors = motion_estimation(prev_frame_gray, frame_gray)

        # Motion compensation
        compensated_frame = motion_compensation(prev_frame_gray, motion_vectors)

        # Save motion compensated frame as image
        cv2.imwrite(os.path.join(output_dir, f"compensated_frame_{frame_index:04d}.png"), compensated_frame)

        frame_index += 1
        prev_frame_gray = frame_gray

    cap.release()
    print(f"Processed {frame_index} frames")
    print(f"Frames saved to directory: {output_dir}")

if __name__ == "__main__":
    video_path = "example.mp4"
    process_video(video_path)