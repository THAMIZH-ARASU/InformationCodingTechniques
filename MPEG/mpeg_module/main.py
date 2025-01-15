import numpy as np
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional
import cv2

# Abstract base classes for Strategy Pattern
class CompressionStrategy(ABC):
    @abstractmethod
    def compress(self, frame: np.ndarray) -> dict:
        pass

    @abstractmethod
    def decompress(self, data: dict) -> np.ndarray:
        pass

# Concrete strategies for different frame types
class IntraFrameStrategy(CompressionStrategy):
    def __init__(self, block_size: int = 16, quantization_parameter: int = 8):
        self.block_size = block_size
        self.qp = quantization_parameter

    def compress(self, frame: np.ndarray) -> dict:
        height, width = frame.shape[:2]
        compressed_blocks = []
        
        # Process frame in macroblocks
        for y in range(0, height, self.block_size):
            for x in range(0, width, self.block_size):
                block = frame[y:y+self.block_size, x:x+self.block_size]
                if block.size == self.block_size * self.block_size:
                    # Apply DCT
                    dct_block = cv2.dct(block.astype(np.float32))
                    # Quantization
                    quantized = np.round(dct_block / self.qp)
                    compressed_blocks.append({
                        'position': (x, y),
                        'data': quantized
                    })
        
        return {
            'type': 'I-frame',
            'blocks': compressed_blocks,
            'shape': frame.shape,
            'qp': self.qp
        }

    def decompress(self, data: dict) -> np.ndarray:
        height, width = data['shape'][:2]
        frame = np.zeros((height, width), dtype=np.uint8)
        
        for block in data['blocks']:
            x, y = block['position']
            # Dequantization
            dequantized = block['data'] * data['qp']
            # Inverse DCT
            reconstructed = cv2.idct(dequantized)
            frame[y:y+self.block_size, x:x+self.block_size] = reconstructed
        
        return frame

class PFrameStrategy(CompressionStrategy):
    def __init__(self, block_size: int = 16, search_window: int = 16):
        self.block_size = block_size
        self.search_window = search_window

    def motion_estimation(self, current_block: np.ndarray, reference_frame: np.ndarray, 
                        block_x: int, block_y: int) -> Tuple[Tuple[int, int], float]:
        min_error = float('inf')
        best_motion_vector = (0, 0)
        
        # Search window boundaries
        start_x = max(0, block_x - self.search_window)
        end_x = min(reference_frame.shape[1] - self.block_size, block_x + self.search_window)
        start_y = max(0, block_y - self.search_window)
        end_y = min(reference_frame.shape[0] - self.block_size, block_y + self.search_window)
        
        # Three-step search algorithm
        step_size = self.search_window // 2
        center_x, center_y = block_x, block_y
        
        while step_size >= 1:
            for dy in [-step_size, 0, step_size]:
                for dx in [-step_size, 0, step_size]:
                    y = center_y + dy
                    x = center_x + dx
                    
                    if start_x <= x <= end_x and start_y <= y <= end_y:
                        reference_block = reference_frame[y:y+self.block_size, x:x+self.block_size]
                        if reference_block.shape == current_block.shape:
                            error = np.mean((current_block - reference_block) ** 2)
                            
                            if error < min_error:
                                min_error = error
                                best_motion_vector = (x - block_x, y - block_y)
            
            center_x += best_motion_vector[0]
            center_y += best_motion_vector[1]
            step_size //= 2
            
        return best_motion_vector, min_error

    def compress(self, frame: np.ndarray, reference_frame: np.ndarray) -> dict:
        height, width = frame.shape[:2]
        motion_vectors = []
        residuals = []
        
        for y in range(0, height, self.block_size):
            for x in range(0, width, self.block_size):
                current_block = frame[y:y+self.block_size, x:x+self.block_size]
                
                if current_block.size == self.block_size * self.block_size:
                    # Motion estimation
                    motion_vector, error = self.motion_estimation(
                        current_block, reference_frame, x, y)
                    
                    # Get reference block using motion vector
                    ref_x = x + motion_vector[0]
                    ref_y = y + motion_vector[1]
                    reference_block = reference_frame[
                        ref_y:ref_y+self.block_size, 
                        ref_x:ref_x+self.block_size
                    ]
                    
                    # Calculate residual
                    residual = current_block - reference_block
                    
                    motion_vectors.append({
                        'position': (x, y),
                        'vector': motion_vector
                    })
                    residuals.append({
                        'position': (x, y),
                        'data': residual
                    })
        
        return {
            'type': 'P-frame',
            'motion_vectors': motion_vectors,
            'residuals': residuals,
            'shape': frame.shape
        }

    def decompress(self, data: dict, reference_frame: np.ndarray) -> np.ndarray:
        height, width = data['shape'][:2]
        frame = np.zeros((height, width), dtype=np.uint8)
        
        for mv, residual in zip(data['motion_vectors'], data['residuals']):
            x, y = mv['position']
            dx, dy = mv['vector']
            
            # Get reference block
            ref_x = x + dx
            ref_y = y + dy
            reference_block = reference_frame[
                ref_y:ref_y+self.block_size, 
                ref_x:ref_x+self.block_size
            ]
            
            # Add residual to reference block
            reconstructed = reference_block + residual['data']
            frame[y:y+self.block_size, x:x+self.block_size] = reconstructed
        
        return frame

# Factory Pattern for frame compression
class FrameCompressorFactory:
    @staticmethod
    def create_compressor(frame_type: str) -> CompressionStrategy:
        if frame_type.upper() == 'I':
            return IntraFrameStrategy()
        elif frame_type.upper() == 'P':
            return PFrameStrategy()
        else:
            raise ValueError(f"Unsupported frame type: {frame_type}")

# Main H261 Codec class
class H261Codec:
    def __init__(self, gop_size: int = 12):
        self.gop_size = gop_size
        self.factory = FrameCompressorFactory()
    
    def compress_sequence(self, frames: List[np.ndarray]) -> List[dict]:
        compressed_sequence = []
        reference_frame = None
        
        for i, frame in enumerate(frames):
            # Determine frame type
            if i % self.gop_size == 0:
                frame_type = 'I'
            else:
                frame_type = 'P'
            
            # Get appropriate compressor
            compressor = self.factory.create_compressor(frame_type)
            
            # Compress frame
            if frame_type == 'I':
                compressed = compressor.compress(frame)
                reference_frame = frame
            else:
                compressed = compressor.compress(frame, reference_frame)
                # Update reference frame
                reference_frame = self.decompress_frame(compressed, reference_frame)
            
            compressed_sequence.append(compressed)
        
        return compressed_sequence
    
    def decompress_frame(self, compressed_frame: dict, 
                        reference_frame: Optional[np.ndarray] = None) -> np.ndarray:
        frame_type = compressed_frame['type'].split('-')[0]
        decompressor = self.factory.create_compressor(frame_type)
        
        if frame_type == 'I':
            return decompressor.decompress(compressed_frame)
        else:
            return decompressor.decompress(compressed_frame, reference_frame)
    
    def decompress_sequence(self, compressed_sequence: List[dict]) -> List[np.ndarray]:
        decompressed_frames = []
        reference_frame = None
        
        for compressed_frame in compressed_sequence:
            decompressed = self.decompress_frame(compressed_frame, reference_frame)
            decompressed_frames.append(decompressed)
            
            if compressed_frame['type'] == 'I-frame':
                reference_frame = decompressed
            else:
                reference_frame = decompressed
        
        return decompressed_frames
    


# Example usage
import numpy as np

# Create some sample frames (grayscale)
frames = [(np.random.rand(480, 640) * 255).astype(np.uint8) for _ in range(24)]

# Initialize codec
codec = H261Codec(gop_size=12)

# Compress sequence
compressed_sequence = codec.compress_sequence(frames)

# Decompress sequence
reconstructed_frames = codec.decompress_sequence(compressed_sequence)