import numpy as np
from scipy.signal import levinson

def lpc(signal, order):
    # Autocorrelation method to compute LPC coefficients
    autocorr = np.correlate(signal, signal, mode='full')[len(signal)-1:]
    R = autocorr[:order+1]
    _, a_coeffs = levinson(R, order)
    return a_coeffs

# Get input from the user
signal = list(map(float, input("Enter the signal values separated by spaces: ").split()))
order = int(input("Enter the LPC order: "))

# Compute LPC coefficients
coefficients = lpc(np.array(signal), order)
print("LPC Coefficients:", coefficients)