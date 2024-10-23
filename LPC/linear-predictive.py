import wave
import struct

def write_wave_file(filename, sample_rate, signal):
    """Write a WAV file."""
    with wave.open(filename, 'wb') as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 16 bits
        wav_file.setframerate(sample_rate)
        # Clamp signal values to the range of 16-bit PCM
        clamped_signal = [max(-32768, min(32767, int(sample))) for sample in signal]
        wav_file.writeframes(struct.pack(f'{len(clamped_signal)}h', *clamped_signal))

def linear_predictive_encode(signal, order=2):
    """Perform Linear Predictive Encoding."""
    N = len(signal)
    r = [0] * (order + 1)

    # Calculate the autocorrelation coefficients
    for i in range(order + 1):
        if i == 0:
            r[i] = sum(x * x for x in signal)
        else:
            r[i] = sum(signal[j] * signal[j - i] for j in range(i, N))

    # Create the autocorrelation matrix
    A = [[0] * (order + 1) for _ in range(order + 1)]
    print("Initializing Auto-correlation matrix")
    print(A)

    for i in range(order + 1):
        for j in range(order + 1):
            A[i][j] = r[abs(i - j)]
    print("\n\nCreating the Auto-correlation matrix")
    print(A)
    
    # Solve for LPC coefficients using Gaussian elimination
    lpc_coeffs = [0] * (order + 1)
    for i in range(order + 1):
        if A[i][i] == 0:
            print("Zero encountered in diagonal, unable to proceed.")
            return [1.0] + [0.0] * order  # Fallback to trivial coefficients

        # Normalize the row
        for j in range(i + 1, order + 1):
            ratio = A[j][i] / A[i][i]
            for k in range(order + 1):
                A[j][k] -= ratio * A[i][k]
            r[j] -= ratio * r[i]

    # Back substitution to find the coefficients
    for i in range(order, -1, -1):
        lpc_coeffs[i] = r[i] / A[i][i]

        for j in range(i + 1, order + 1):
            r[i] -= A[i][j] * lpc_coeffs[j]

    return lpc_coeffs

def linear_predictive_decode(lpc_coeffs, input_signal):
    """Decode the signal using LPC coefficients."""
    output_signal = [0] * len(input_signal)

    for n in range(len(input_signal)):
        output_signal[n] = input_signal[n]
        for k in range(1, len(lpc_coeffs)):
            if n - k >= 0:
                output_signal[n] -= lpc_coeffs[k] * output_signal[n - k]

    return output_signal

def main():
    input_filename = 'input_audio.wav'  # Replace with your WAV file name
    output_filename = 'decoded_audio.wav'
    encoded_file = 'encoded_signal.lpc'
    
    # Read the input audio file
    with wave.open(input_filename, 'rb') as wav_file:
        sample_rate = wav_file.getframerate()
        num_samples = wav_file.getnframes()
        signal = wav_file.readframes(num_samples)
        signal = [int.from_bytes(signal[i:i + 2], 'little', signed=True) for i in range(0, len(signal), 2)]

    print(f"Original signal (first 10 samples): {signal[:10]}")

    if len(signal) < 3:
        print("Input audio signal is too short for LPC analysis.")
        return

    # Perform Linear Predictive Encoding
    lpc_coeffs = linear_predictive_encode(signal)
    print(f"LPC Coefficients: {lpc_coeffs}")

    # Save LPC coefficients and original signal to file
    with open(encoded_file, 'w') as f:
        f.write('LPC Coefficients: ' + ', '.join(map(str, lpc_coeffs)) + '\n')
        f.write('Original Signal: ' + ', '.join(map(str, signal)) + '\n')

    # Perform Linear Predictive Decoding
    decoded_signal = linear_predictive_decode(lpc_coeffs, signal)
    write_wave_file(output_filename, sample_rate, decoded_signal)

    print(f"Decoded signal saved to {output_filename}.")

    # Check if the original signal and decoded signal are similar
    if all(abs(signal[i] - decoded_signal[i]) < 1 for i in range(len(signal))):
        print("Encoding and decoding are correct!")
    else:
        print("There was an error in encoding/decoding.")

if __name__ == "__main__":
    main()
