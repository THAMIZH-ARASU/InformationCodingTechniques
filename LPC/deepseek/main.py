import matplotlib.pyplot as plt
import numpy as np
import librosa
import soundfile as sf
import pickle

def lpc_encode(signal, order):
    """
    Perform Linear Predictive Coding (LPC) on the input signal.
    
    Parameters:
        signal (numpy array): The input audio signal.
        order (int): The order of the LPC analysis.
        
    Returns:
        numpy array: The LPC coefficients.
        numpy array: The error signal (residual).
    """
    # Compute LPC coefficients
    a = librosa.lpc(signal, order=order)
    
    # Compute the residual (error) signal
    residual = signal - np.convolve(signal, a, mode='same')
    
    return a, residual

def lpc_decode(a, residual):
    """
    Reconstruct the signal from LPC coefficients and residual.
    
    Parameters:
        a (numpy array): The LPC coefficients.
        residual (numpy array): The error signal (residual).
        
    Returns:
        numpy array: The reconstructed signal.
    """
    # Reconstruct the signal using the LPC coefficients and residual
    reconstructed_signal = np.convolve(residual, np.hstack((1, -a)), mode='same')
    
    return reconstructed_signal

def save_lpc_data(a, residual, file_path):
    """
    Save LPC coefficients and residual signal to a file.
    
    Parameters:
        a (numpy array): The LPC coefficients.
        residual (numpy array): The error signal (residual).
        file_path (str): Path to save the file.
    """
    with open(file_path, 'wb') as f:
        pickle.dump({'a': a, 'residual': residual}, f)

def load_lpc_data(file_path):
    """
    Load LPC coefficients and residual signal from a file.
    
    Parameters:
        file_path (str): Path to the file containing LPC data.
        
    Returns:
        numpy array: The LPC coefficients.
        numpy array: The error signal (residual).
    """
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data['a'], data['residual']

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

# Modified main function to include plotting
def main():
    # Load a WAV file
    input_file = 'input.wav'
    lpc_data_file = 'lpc_data.pkl'
    output_file = 'output.wav'
    
    # Load the audio file
    signal, sr = librosa.load(input_file, sr=None)
    
    # Set the LPC order (typically 8-16 for speech signals)
    order = 12
    
    # Encode the signal using LPC
    a, residual = lpc_encode(signal, order)
    
    # Save the LPC coefficients and residual to a file
    save_lpc_data(a, residual, lpc_data_file)
    print(f"LPC data saved to {lpc_data_file}")
    
    # Load the LPC coefficients and residual from the file
    a_loaded, residual_loaded = load_lpc_data(lpc_data_file)
    
    # Decode the signal using the loaded LPC data
    reconstructed_signal = lpc_decode(a_loaded, residual_loaded)
    
    # Save the reconstructed signal to a WAV file
    sf.write(output_file, reconstructed_signal, sr)
    
    print(f"LPC decoding completed. Output saved to {output_file}")
    
    # Plot the signals comparison
    plot_signals_comparison(input_file, output_file)

if __name__ == "__main__":
    main()