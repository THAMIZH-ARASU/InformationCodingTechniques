import numpy as np

def levinson_durbin(r, order):
    """Levinson-Durbin recursion for solving Toeplitz systems."""
    a = np.zeros(order + 1)
    e = r[0]  # Initial error (residual energy)
    a[0] = 1  # First coefficient is always 1 for LPC
    
    for i in range(1, order + 1):
        # Compute reflection coefficient
        acc = 0
        for j in range(1, i):
            acc += a[j] * r[i - j]
        k = (r[i] - acc) / e
        
        # Update LPC coefficients
        new_a = a.copy()
        for j in range(1, i):
            new_a[j] = a[j] - k * a[i - j]
        new_a[i] = k
        
        a = new_a
        e *= (1 - k ** 2)  # Update error
    
    return a

def lpc(signal, order):
    # Autocorrelation method to compute LPC coefficients
    autocorr = np.correlate(signal, signal, mode='full')[len(signal)-1:]
    R = autocorr[:order+1]
    a_coeffs = levinson_durbin(R, order)
    return a_coeffs


# Read the encoded message back from the file for decoding
input_file = 'input_file.txt'
with open(input_file, 'r') as file:
    signal = list(map(float, file.read().split()))
# Get input from the user
#signal = list(map(float, input("Enter the signal values separated by spaces: ").split()))
#order = int(input("Enter the LPC order: "))
order = 2 # 2 is set as default
print("Signal in file: ", signal)
# Compute LPC coefficients
coefficients = lpc(np.array(signal), order)
print("LPC Coefficients:", coefficients)