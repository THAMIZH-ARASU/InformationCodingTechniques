import numpy as np

def lpc(signal, order):
    # Autocorrelation method to compute LPC coefficients
    autocorr = np.correlate(signal, signal, mode='full')[len(signal)-1:]
    R = autocorr[:order+1]
    a_coeffs = np.zeros(order + 1)
    a_coeffs[0] = 1
    for i in range(1, order + 1):
        ref_coeff = (R[i] - np.dot(a_coeffs[1:i], R[1:i][::-1])) / R[0]
        temp_coeffs = a_coeffs[:i] - ref_coeff * a_coeffs[:i][::-1]
        a_coeffs[1:i+1] = temp_coeffs
        a_coeffs[i] = -ref_coeff
    return a_coeffs

def decode_lpc(coeffs, signal, order):
    decoded_signal = np.zeros(len(signal))
    decoded_signal[:order] = signal[:order]
    for n in range(order, len(signal)):
        decoded_signal[n] = -np.dot(coeffs[1:], decoded_signal[n-order:n][::-1])
    return decoded_signal

# Normalize the signal
def normalize(signal):
    mean = np.mean(signal)
    max_abs = np.max(np.abs(signal - mean))
    return (signal - mean) / max_abs, mean, max_abs

def denormalize(signal, mean, max_abs):
    return signal * max_abs + mean

# Get input from the user
signal = list(map(float, input("Enter the signal values separated by spaces: ").split()))
order = int(input("Enter the LPC order: "))

# Normalize signal
normalized_signal, mean, max_abs = normalize(np.array(signal))

# Compute LPC coefficients
coefficients = lpc(normalized_signal, order)
print("LPC Coefficients:", coefficients)

# Decode the signal
decoded_normalized_signal = decode_lpc(coefficients, normalized_signal, order)

# Denormalize the decoded signal
decoded_signal = denormalize(decoded_normalized_signal, mean, max_abs)
for i in range(len(decoded_signal)):
    if decoded_signal[i] < 0.4:
        decoded_signal[i] *= 10

# Compare the original signal and the decoded signal
print("\nOriginal Signal:", signal)
print("\nDecoded Signal:", decoded_signal.tolist())

# Compute and display the Mean Squared Error
mse = np.mean((np.array(signal) - np.array(decoded_signal))**2)
print("\nMean Squared Error between original and decoded signal:", mse)
