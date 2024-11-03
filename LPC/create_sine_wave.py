import wave
import numpy as np

def create_sine_wave(filename, duration=0.5, frequency=440, sample_rate=44100):
    """Create a sine wave audio file."""
    # Generate the time values
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    # Generate the sine wave
    sine_wave = 0.5 * np.sin(2 * np.pi * frequency * t)  # Amplitude is 0.5
    # Convert to 16-bit PCM format
    sine_wave = np.int16(sine_wave * 32767)  # Scale to int16 range

    # Write to a WAV file
    with wave.open(filename, 'wb') as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 16 bits
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(sine_wave.tobytes())

# Create the audio file
create_sine_wave('input_audio.wav')
print("input_audio.wav created.")
