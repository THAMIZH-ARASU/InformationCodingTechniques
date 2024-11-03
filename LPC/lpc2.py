import numpy as np
from scipy.io import wavfile

def levinson_durbin(r, order, reg=1e-6):
    """Levinson-Durbin recursion for solving Toeplitz systems with regularization."""
    a = np.zeros(order + 1)
    e = r[0] + reg
    a[0] = 1
    for i in range(1, order + 1):
        acc = sum(a[j] * r[i - j] for j in range(1, i))
        k = (r[i] - acc) / e
        new_a = a.copy()
        for j in range(1, i):
            new_a[j] = a[j] - k * a[i - j]
        new_a[i] = k
        a = new_a
        e *= (1 - k ** 2)
    return a

def lpc(signal, order):
    autocorr = np.correlate(signal, signal, mode='full')[len(signal)-1:]
    R = autocorr[:order+1]
    return levinson_durbin(R, order)

def encode_lpc(signal, order, filename):
    coefficients = lpc(signal, order)
    with open(filename, 'w') as f:
        f.write(' '.join(map(str, coefficients)))
    return coefficients

def decode_lpc(coefficients, length, noise_scale=0.01):
    signal = np.random.normal(0, noise_scale, length)
    for n in range(len(coefficients), length):
        acc = 0
        for i in range(1, len(coefficients)):
            if n - i >= 0:
                acc -= coefficients[i] * signal[n - i]
        acc = np.clip(acc, -1e4, 1e4) 
        signal[n] += acc
    return signal

input_wav = input("Enter a wav audio file for input: ")
sample_rate, signal = wavfile.read(input_wav)
if signal.ndim > 1:
    signal = signal[:, 0]
order = 4 
encoded_file = 'encoded.lpc'
coefficients = encode_lpc(signal, order, encoded_file)
decoded_signal = decode_lpc(coefficients, len(signal))
decoded_wav = 'decoded.wav'
decoded_signal = np.nan_to_num(decoded_signal / (np.max(np.abs(decoded_signal)) + 1e-6) * 32767)
decoded_signal = np.int16(decoded_signal)
wavfile.write(decoded_wav, sample_rate, decoded_signal)
print("LPC Coefficients:", coefficients)
print("Decoded signal saved to", decoded_wav)
signal = signal / (np.max(np.abs(signal)) + 1e-6)
decoded_signal = decoded_signal / (np.max(np.abs(decoded_signal)) + 1e-6)
correlation = np.corrcoef(signal[:len(decoded_signal)], decoded_signal)[0, 1]
#correlation = 1
if correlation > 0.9:
    print("Encoding and decoding are close!")
else:
    print("There may be a difference between original and decoded signals.")
