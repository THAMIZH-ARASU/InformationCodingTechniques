import matplotlib.pyplot as plt
import librosa
import numpy as np

def plot_signals_comparison(input_file, output_file):
    """
    Plot the input signal, output signal, and their difference.
    
    Parameters:
        input_file (str): Path to the input WAV file
        output_file (str): Path to the output WAV file
    """
    # Load both signals
    input_signal, sr_in = librosa.load(input_file, sr=None)
    output_signal, sr_out = librosa.load(output_file, sr=None)
    
    # Calculate the error/difference
    error = input_signal - output_signal
    
    # Create time arrays for x-axis
    time = np.arange(len(input_signal)) / sr_in
    
    # Create the plot
    plt.figure(figsize=(15, 10))
    
    # Plot input signal
    plt.subplot(3, 1, 1)
    plt.plot(time, input_signal, 'b-', label='Input Signal')
    plt.title('Input Signal')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.legend()
    
    # Plot output signal
    plt.subplot(3, 1, 2)
    plt.plot(time, output_signal, 'g-', label='Output Signal')
    plt.title('Reconstructed Output Signal')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.legend()
    
    # Plot error
    plt.subplot(3, 1, 3)
    plt.plot(time, error, 'r-', label='Error')
    plt.title('Error (Input - Output)')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.legend()
    
    # Adjust layout and display
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    plot_signals_comparison('input.wav', 'input_audio.wav')