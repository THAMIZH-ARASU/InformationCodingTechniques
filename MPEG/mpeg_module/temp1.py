import cv2
import numpy as np

class VideoCompressor:
    def __init__(self, input_file):
        self.input_file = input_file
        self.cap = cv2.VideoCapture(input_file)
        self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.output_file = 'compressed.mp4'
        self.motion_vectors_file = 'motion_estimation.mp4'
        self.decompressed_file = 'decompressed.avi'

    def compress(self):
        # Initialize video writer for compressed video
        out = cv2.VideoWriter(self.output_file, self.fourcc, 30.0, (int(self.cap.get(3)), int(self.cap.get(4))))
        motion_vectors_out = cv2.VideoWriter(self.motion_vectors_file, self.fourcc, 30.0, (int(self.cap.get(3)), int(self.cap.get(4))))

        ret, prev_frame = self.cap.read()
        if not ret:
            print("Failed to read video")
            return

        prev_frame = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

        while True:
            ret, curr_frame = self.cap.read()
            if not ret:
                break

            curr_frame_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
            motion_vector = self.motion_estimation(prev_frame, curr_frame_gray)

            # Draw motion vectors on the current frame
            motion_vectors_out.write(self.draw_motion_vectors(curr_frame, motion_vector))

            # Write the current frame to the output video
            out.write(curr_frame)

            prev_frame = curr_frame_gray

        self.cap.release()
        out.release()
        motion_vectors_out.release()

    def motion_estimation(self, prev_frame, curr_frame):
        # Simple block matching for motion estimation
        h, w = prev_frame.shape
        block_size = 16
        motion_vector = np.zeros((h // block_size, w // block_size, 2), dtype=np.int)

        for i in range(0, h - block_size, block_size):
            for j in range(0, w - block_size, block_size):
                block = prev_frame[i:i + block_size, j:j + block_size]
                best_match = (0, 0)
                min_error = float('inf')

                for x in range(-block_size, block_size + 1):
                    for y in range(-block_size, block_size + 1):
                        if 0 <= i + x < h - block_size and 0 <= j + y < w - block_size:
                            candidate_block = curr_frame[i + x:i + x + block_size, j + y:j + y + block_size]
                            error = np.sum((block - candidate_block) ** 2)
                            if error < min_error:
                                min_error = error
                                best_match = (x, y)

                motion_vector[i // block_size, j // block_size] = best_match

        return motion_vector

    def draw_motion_vectors(self, frame, motion_vector):
        h, w, _ = frame.shape
        for i in range(motion_vector.shape[0]):
            for j in range(motion_vector.shape[1]):
                x = j * 16 + 8
                y = i * 16 + 8
                dx, dy = motion_vector[i, j]
                cv2.arrowedLine(frame, (x, y), (x + dx, y + dy), (0, 255, 0), 1, tipLength=0.2)
        return frame

class VideoDecompressor:
    def __init__(self, input_file):
        self.input_file = input_file
        self.cap = cv2.VideoCapture(input_file)
        self.output_file = 'decompressed.avi'
        self.fourcc = cv2.VideoWriter_fourcc(*'XVID')

    def decompress(self):
        out = cv2.VideoWriter(self.output_file, self.fourcc, 30.0, (int(self.cap.get(3)), int(self.cap.get(4))))

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            # Here you would typically apply the inverse of the compression algorithm
            # For simplicity, we will just write the frame directly
            out.write(frame)

        self.cap.release()
        out.release()

def main():
    input_file = input("Enter the path of the input AVI file: ")
    compressor = VideoCompressor(input_file)
    compressor.compress()
    
    decompressor = VideoDecompressor(compressor.output_file)
    decompressor.decompress()

if __name__ == "__main__":
    main()
