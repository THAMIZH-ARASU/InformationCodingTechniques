import cv2
import numpy as np
import os

class VideoCompression:
    def __init__(self, input_file, output_file):
        self.input_file = input_file
        self.output_file = output_file
        self.motion_vectors_file = "motion_estimation.mp4"

    def compress(self):
        cap = cv2.VideoCapture(self.input_file)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(self.output_file, fourcc, cap.get(cv2.CAP_PROP_FPS), 
                              (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

        motion_vector_writer = cv2.VideoWriter(self.motion_vectors_file, fourcc, cap.get(cv2.CAP_PROP_FPS), 
                                                (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
        
        ret, prev_frame = cap.read()
        if not ret:
            print("Failed to read the video.")
            return
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            motion_vectors = self._motion_estimation(prev_gray, gray)

            motion_frame = self._visualize_motion_vectors(prev_gray, motion_vectors)
            motion_vector_writer.write(motion_frame)

            predicted_frame = self._motion_compensated_prediction(prev_frame, motion_vectors)
            compressed_frame = cv2.addWeighted(frame, 0.7, predicted_frame, 0.3, 0)

            out.write(compressed_frame)
            prev_gray = gray
            prev_frame = frame

        cap.release()
        out.release()
        motion_vector_writer.release()
        print(f"Compression completed. Saved as {self.output_file} and motion vectors saved as {self.motion_vectors_file}")

    def decompress(self, compressed_file, decompressed_file):
        cap = cv2.VideoCapture(compressed_file)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(decompressed_file, fourcc, cap.get(cv2.CAP_PROP_FPS), 
                              (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)
        
        cap.release()
        out.release()
        print(f"Decompression completed. Saved as {decompressed_file}")

    def _motion_estimation(self, prev_gray, gray):
        # Optical flow for motion estimation
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        return flow

    def _visualize_motion_vectors(self, frame, motion_vectors):
        hsv = np.zeros_like(frame)
        hsv[..., 1] = 255
        mag, ang = cv2.cartToPolar(motion_vectors[..., 0], motion_vectors[..., 1])
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        return bgr

    def _motion_compensated_prediction(self, prev_frame, motion_vectors):
        height, width = prev_frame.shape[:2]
        grid_y, grid_x = np.mgrid[0:height, 0:width].astype(np.float32)
        remap_x = grid_x + motion_vectors[..., 0]
        remap_y = grid_y + motion_vectors[..., 1]
        predicted_frame = cv2.remap(prev_frame, remap_x, remap_y, interpolation=cv2.INTER_LINEAR)
        return predicted_frame

def main():
    input_file = input("Enter the path of the input AVI file: ")
    compressed_file = "compressed.mp4"
    decompressed_file = "decompressed.avi"

    if not os.path.exists(input_file):
        print("Input file not found.")
        return

    compressor = VideoCompression(input_file, compressed_file)
    compressor.compress()
    compressor.decompress(compressed_file, decompressed_file)

if __name__ == "__main__":
    main()
