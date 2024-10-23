from collections import defaultdict

# Function to calculate cumulative probability ranges for each symbol
def calculate_ranges(frequencies):
    total = sum(frequencies.values())
    ranges = {}
    low = 0.0
    for symbol, freq in frequencies.items():
        high = low + (freq / total)
        ranges[symbol] = (low, high)
        low = high
    return ranges

# Function to encode a message using Arithmetic Coding
def arithmetic_encode(message, frequencies):
    ranges = calculate_ranges(frequencies)
    low, high = 0.0, 1.0
    for symbol in message:
        symbol_low, symbol_high = ranges[symbol]
        range_width = high - low
        high = low + range_width * symbol_high
        low = low + range_width * symbol_low
    return (low + high) / 2

# Function to decode a message using Arithmetic Coding
def arithmetic_decode(encoded_value, frequencies, message_length):
    ranges = calculate_ranges(frequencies)
    decoded_message = []
    for _ in range(message_length):
        for symbol, (low, high) in ranges.items():
            if low <= encoded_value < high:
                decoded_message.append(symbol)
                encoded_value = (encoded_value - low) / (high - low)
                break
    return ''.join(decoded_message)

# Example usage
if __name__ == '__main__':
    message = "hello"
    frequencies = defaultdict(int)
    for char in message:
        frequencies[char] += 1

    # Encode the message
    encoded_value = arithmetic_encode(message, frequencies)
    print(f"Encoded value: {encoded_value}")

    # Decode the message
    decoded_message = arithmetic_decode(encoded_value, frequencies, len(message))
    print(f"Decoded message: {decoded_message}")
