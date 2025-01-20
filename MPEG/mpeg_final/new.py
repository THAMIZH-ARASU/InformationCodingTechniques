import cv2
import numpy as np
import os

class H261VideoEncoder:
    def __init__(self, block_size=16, search_range=8):
        self.block_size = block_size
        self.search_range = search_range

    def motion_estimation(self, current_frame, reference_frame):
        height, width = current_frame.shape
        # Calculate number of blocks, rounding down
        num_blocks_y = height // self.block_size
        num_blocks_x = width // self.block_size
        
        # Initialize motion vectors array with correct dimensions
        motion_vectors = np.zeros((num_blocks_y, num_blocks_x, 2), dtype=int)
        
        # Only process complete blocks
        for y in range(0, num_blocks_y * self.block_size, self.block_size):
            for x in range(0, num_blocks_x * self.block_size, self.block_size):
                best_match = (0, 0)
                min_error = float('inf')
                current_block = current_frame[y:y + self.block_size, x:x + self.block_size]
                
                for dy in range(-self.search_range, self.search_range + 1):
                    for dx in range(-self.search_range, self.search_range + 1):
                        ref_x = x + dx
                        ref_y = y + dy
                        if (0 <= ref_x < width - self.block_size + 1 and 
                            0 <= ref_y < height - self.block_size + 1):
                            ref_block = reference_frame[ref_y:ref_y + self.block_size, 
                                                      ref_x:ref_x + self.block_size]
                            error = np.sum((current_block - ref_block) ** 2)
                            if error < min_error:
                                min_error = error
                                best_match = (dy, dx)
                
                motion_vectors[y // self.block_size, x // self.block_size] = best_match
        
        return motion_vectors

    def motion_compensation(self, reference_frame, motion_vectors):
        height, width = reference_frame.shape
        num_blocks_y = height // self.block_size
        num_blocks_x = width // self.block_size
        
        # Initialize compensated frame with zeros
        compensated_frame = np.zeros_like(reference_frame)
        
        # Only process complete blocks
        for y in range(0, num_blocks_y * self.block_size, self.block_size):
            for x in range(0, num_blocks_x * self.block_size, self.block_size):
                dy, dx = motion_vectors[y // self.block_size, x // self.block_size]
                ref_x = x + dx
                ref_y = y + dy
                
                if (0 <= ref_x < width - self.block_size + 1 and 
                    0 <= ref_y < height - self.block_size + 1):
                    compensated_frame[y:y + self.block_size, x:x + self.block_size] = \
                        reference_frame[ref_y:ref_y + self.block_size, ref_x:ref_x + self.block_size]
        
        return compensated_frame

    def draw_motion_vectors(self, frame, motion_vectors):
        frame_with_vectors = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        
        for y in range(motion_vectors.shape[0]):
            for x in range(motion_vectors.shape[1]):
                dy, dx = motion_vectors[y, x]
                start_point = (x * self.block_size + self.block_size // 2, 
                             y * self.block_size + self.block_size // 2)
                end_point = (start_point[0] + dx, start_point[1] + dy)
                cv2.arrowedLine(frame_with_vectors, start_point, end_point, 
                              (0, 0, 255), 1, tipLength=0.4)
        
        return frame_with_vectors

def process_video(input_path, output_motion_estimation, output_motion_compensation, 
                 block_size=16, search_range=8):
    encoder = H261VideoEncoder(block_size=block_size, search_range=search_range)
    cap = cv2.VideoCapture(input_path)
    
    if not cap.isOpened():
        print("Error: Cannot open video file.")
        return
    
    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Adjust dimensions to be multiples of block_size
    frame_width_adjusted = (frame_width // block_size) * block_size
    frame_height_adjusted = (frame_height // block_size) * block_size
    
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out_motion_estimation = cv2.VideoWriter(output_motion_estimation, fourcc, fps, 
                                          (frame_width, frame_height))
    out_motion_compensation = cv2.VideoWriter(output_motion_compensation, fourcc, fps, 
                                            (frame_width, frame_height), isColor=False)
    
    prev_frame = None
    print(f"Processing {frame_count} frames...")
    
    for i in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break
            
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Crop frame to adjusted dimensions
        gray_frame = gray_frame[:frame_height_adjusted, :frame_width_adjusted]
        
        if prev_frame is None:
            print("Processing Intra-frame (first frame)...")
            out_motion_estimation.write(cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2BGR))
            out_motion_compensation.write(gray_frame)
        else:
            print(f"Processing P-frame ({i+1}/{frame_count})...")
            motion_vectors = encoder.motion_estimation(gray_frame, prev_frame)
            motion_estimation_frame = encoder.draw_motion_vectors(prev_frame, motion_vectors)
            compensated_frame = encoder.motion_compensation(prev_frame, motion_vectors)
            
            # Pad back to original dimensions if needed
            if frame_height_adjusted < frame_height or frame_width_adjusted < frame_width:
                motion_estimation_frame = cv2.copyMakeBorder(
                    motion_estimation_frame, 0, frame_height - frame_height_adjusted,
                    0, frame_width - frame_width_adjusted, cv2.BORDER_CONSTANT)
                compensated_frame = cv2.copyMakeBorder(
                    compensated_frame, 0, frame_height - frame_height_adjusted,
                    0, frame_width - frame_width_adjusted, cv2.BORDER_CONSTANT)
            
            out_motion_estimation.write(motion_estimation_frame)
            out_motion_compensation.write(compensated_frame)
        
        prev_frame = gray_frame
        if i % 10 == 0:
            print(f"Processed {i}/{frame_count} frames...")
    
    cap.release()
    out_motion_estimation.release()
    out_motion_compensation.release()
    print(f"Processing complete! Output files:")
    print(f" - Motion Estimation video: {output_motion_estimation}")
    print(f" - Motion Compensation video: {output_motion_compensation}")

def convert_mp4_to_avi(input_mp4, output_avi):
    cap = cv2.VideoCapture(input_mp4)
    if not cap.isOpened():
        print("Error: Cannot open MP4 video file.")
        return
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Use XVID codec for AVI format
    out = cv2.VideoWriter(output_avi, fourcc, fps, (frame_width, frame_height))

    print(f"Converting {input_mp4} to {output_avi}...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)  # Write the current frame to the AVI file
    cap.release()
    out.release()
    print(f"Conversion complete! Output file: {output_avi}")
def process_decoding(input_motion_estimation, input_motion_compensation, output_avi):
    cap_estimation = cv2.VideoCapture(input_motion_estimation)
    cap_compensation = cv2.VideoCapture(input_motion_compensation)    
    if not cap_estimation.isOpened() or not cap_compensation.isOpened():
        print("Error: Cannot open one or both MP4 video files.")
        return
    frame_width = int(cap_estimation.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap_estimation.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap_estimation.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Use XVID codec for AVI format
    out = cv2.VideoWriter(output_avi, fourcc, fps, (frame_width, frame_height))
    print(f"Decoding {input_motion_estimation} and {input_motion_compensation} into {output_avi}...")
    while True:
        ret_estimation, frame_estimation = cap_estimation.read()
        ret_compensation, frame_compensation = cap_compensation.read()
        if not ret_estimation or not ret_compensation:
            break
        original_frame = frame_compensation  # or any other frame reconstruction logic
        out.write(original_frame)  # Write the reconstructed frame to the AVI file
    cap_estimation.release()
    cap_compensation.release()
    out.release()
    print(f"Decoding complete! Output file: {output_avi}")

if __name__ == "__main__":
    input_path = input("Enter the path to the input AVI video file for encoding: ").strip()
    output_motion_estimation = "motion_estimation.mp4"
    output_motion_compensation = "motion_compensation.mp4"
    if not os.path.exists(input_path):
        print("Error: Input file does not exist.")
        exit(1)
    process_video(input_path, output_motion_estimation, output_motion_compensation, block_size=16, search_range=8)
    output_avi = input("Enter the output AVI file name: ").strip()
    process_decoding(output_motion_estimation, output_motion_compensation, output_avi)