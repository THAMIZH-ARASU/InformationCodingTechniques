import numpy as np
import cv2
from abc import ABC, abstractmethod
from typing import Tuple, List, Optional
import os

# Abstract base class for compression strategy
class CompressionStrategy(ABC):
    @abstractmethod
    def compress(self, frame: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def decompress(self, frame: np.ndarray) -> np.ndarray:
        pass

# Strategy for Intra-frame compression
class IntraFrameStrategy(CompressionStrategy):
    def __init__(self, quality_factor: int = 10):
        self.quality_factor = quality_factor

    def compress(self, frame: np.ndarray) -> np.ndarray:
        # Convert to YCbCr color space
        if len(frame.shape) == 3:
            frame_ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
            y, cr, cb = cv2.split(frame_ycrcb)
        else:
            y = frame
            cr = cb = None

        # Apply DCT transformation
        dct_y = cv2.dct(np.float32(y))
        
        # Quantization
        quantized_y = np.round(dct_y / self.quality_factor)
        
        if cr is not None and cb is not None:
            dct_cr = cv2.dct(np.float32(cr))
            dct_cb = cv2.dct(np.float32(cb))
            quantized_cr = np.round(dct_cr / self.quality_factor)
            quantized_cb = np.round(dct_cb / self.quality_factor)
            return np.array([quantized_y, quantized_cr, quantized_cb])
        
        return quantized_y

    def decompress(self, compressed_frame: np.ndarray) -> np.ndarray:
        if isinstance(compressed_frame, list) or (isinstance(compressed_frame, np.ndarray) and len(compressed_frame) == 3):
            quantized_y, quantized_cr, quantized_cb = compressed_frame
            
            # Inverse quantization and DCT
            y = cv2.idct(quantized_y * self.quality_factor)
            cr = cv2.idct(quantized_cr * self.quality_factor)
            cb = cv2.idct(quantized_cb * self.quality_factor)
            
            # Merge channels and convert back to BGR
            frame_ycrcb = cv2.merge([y, cr, cb])
            return cv2.cvtColor(frame_ycrcb, cv2.COLOR_YCrCb2BGR)
        else:
            return cv2.idct(compressed_frame * self.quality_factor)

# Strategy for P-frame compression with motion estimation
class PFrameStrategy(CompressionStrategy):
    def __init__(self, reference_frame: np.ndarray, block_size: int = 16, search_range: int = 16):
        self.reference_frame = reference_frame
        self.block_size = block_size
        self.search_range = search_range

    def motion_estimation(self, current_block: np.ndarray, search_area: np.ndarray) -> Tuple[Tuple[int, int], np.ndarray]:
        min_mad = float('inf')
        motion_vector = (0, 0)
        
        h, w = search_area.shape[:2]
        block_h, block_w = current_block.shape[:2]
        
        for i in range(h - block_h + 1):
            for j in range(w - block_w + 1):
                candidate_block = search_area[i:i+block_h, j:j+block_w]
                mad = np.mean(np.abs(current_block - candidate_block))
                
                if mad < min_mad:
                    min_mad = mad
                    motion_vector = (i - self.search_range, j - self.search_range)
        
        return motion_vector, min_mad

    def compress(self, frame: np.ndarray) -> Tuple[List[Tuple[int, int]], np.ndarray]:
        h, w = frame.shape[:2]
        motion_vectors = []
        residual = np.zeros_like(frame)
        
        for i in range(0, h, self.block_size):
            for j in range(0, w, self.block_size):
                current_block = frame[i:i+self.block_size, j:j+self.block_size]
                
                # Define search area in reference frame
                search_start_y = max(0, i - self.search_range)
                search_end_y = min(h, i + self.block_size + self.search_range)
                search_start_x = max(0, j - self.search_range)
                search_end_x = min(w, j + self.block_size + self.search_range)
                
                search_area = self.reference_frame[search_start_y:search_end_y, 
                                                 search_start_x:search_end_x]
                
                motion_vector, _ = self.motion_estimation(current_block, search_area)
                motion_vectors.append((motion_vector, (i, j)))
                
                # Calculate residual
                ref_block = self.reference_frame[i+motion_vector[0]:i+motion_vector[0]+self.block_size,
                                               j+motion_vector[1]:j+motion_vector[1]+self.block_size]
                residual[i:i+self.block_size, j:j+self.block_size] = current_block - ref_block
        
        return motion_vectors, residual

    def decompress(self, compressed_data: Tuple[List[Tuple[int, int]], np.ndarray]) -> np.ndarray:
        motion_vectors, residual = compressed_data
        reconstructed_frame = np.zeros_like(self.reference_frame)
        
        for (motion_vector, block_pos) in motion_vectors:
            i, j = block_pos
            ref_block = self.reference_frame[i+motion_vector[0]:i+motion_vector[0]+self.block_size,
                                           j+motion_vector[1]:j+motion_vector[1]+self.block_size]
            reconstructed_frame[i:i+self.block_size, j:j+self.block_size] = \
                ref_block + residual[i:i+self.block_size, j:j+self.block_size]
        
        return reconstructed_frame

class H261Codec:
    def __init__(self, gop_size: int = 12):
        self.gop_size = gop_size
        
    def compress_video(self, input_path: str, output_path: str) -> None:
        cap = cv2.VideoCapture(input_path)
        
        # Get video properties
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Create output file
        compressed_frames = []
        frame_types = []  # 'I' for intra-frame, 'P' for p-frame
        
        frame_num = 0
        reference_frame = None
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_num % self.gop_size == 0:  # I-frame
                strategy = IntraFrameStrategy()
                compressed_frame = strategy.compress(frame)
                frame_types.append('I')
                reference_frame = frame
            else:  # P-frame
                if reference_frame is None:
                    raise ValueError("No reference frame available for P-frame compression")
                strategy = PFrameStrategy(reference_frame)
                compressed_frame = strategy.compress(frame)
                frame_types.append('P')
                reference_frame = strategy.decompress(compressed_frame)
            
            compressed_frames.append(compressed_frame)
            frame_num += 1
            
        cap.release()
        
        # Save compressed data
        np.savez_compressed(
            output_path,
            frames=compressed_frames,
            frame_types=frame_types,
            fps=fps,
            width=width,
            height=height,
            gop_size=self.gop_size
        )
        
    def decompress_video(self, input_path: str, output_path: str) -> None:
        # Load compressed data
        data = np.load(input_path)
        compressed_frames = data['frames']
        frame_types = data['frame_types']
        fps = data['fps']
        width = data['width']
        height = data['height']
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        reference_frame = None
        
        for i, (compressed_frame, frame_type) in enumerate(zip(compressed_frames, frame_types)):
            if frame_type == 'I':
                strategy = IntraFrameStrategy()
                decompressed_frame = strategy.decompress(compressed_frame)
                reference_frame = decompressed_frame
            else:  # P-frame
                strategy = PFrameStrategy(reference_frame)
                decompressed_frame = strategy.decompress(compressed_frame)
                reference_frame = decompressed_frame
            
            # Ensure frame is in uint8 format
            decompressed_frame = np.clip(decompressed_frame, 0, 255).astype(np.uint8)
            out.write(decompressed_frame)
            
        out.release()

def main():
    # Get input from user
    input_path = input("Enter the path to the input video file: ")
    compressed_output = input("Enter the path for compressed output file (e.g., compressed.npz): ")
    decompressed_output = input("Enter the path for decompressed output video (e.g., decompressed.mp4): ")
    
    # Create codec instance
    codec = H261Codec()
    
    # Compress video
    print("Compressing video...")
    codec.compress_video(input_path, compressed_output)
    print(f"Video compressed and saved to {compressed_output}")
    
    # Decompress video
    print("Decompressing video...")
    codec.decompress_video(compressed_output, decompressed_output)
    print(f"Video decompressed and saved to {decompressed_output}")

if __name__ == "__main__":
    main()