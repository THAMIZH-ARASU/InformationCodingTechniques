from collections import defaultdict

# Function to sort symbols by frequency
def sort_by_frequency(frequencies):
    return sorted(frequencies.items(), key=lambda item: item[1], reverse=True)

# Recursive function to assign Shannon-Fano codes
def shannon_fano_code(symbols, code_dict, prefix=""):
    if len(symbols) == 1:
        symbol, _ = symbols[0]
        code_dict[symbol] = prefix
        return

    # Find the split point
    total = sum(freq for _, freq in symbols)
    cumulative = 0
    split_index = 0
    for i, (_, freq) in enumerate(symbols):
        cumulative += freq
        if cumulative >= total / 2:
            split_index = i + 1
            break

    # Assign codes recursively
    shannon_fano_code(symbols[:split_index], code_dict, prefix + "0")
    shannon_fano_code(symbols[split_index:], code_dict, prefix + "1")

# Function to encode a message using Shannon-Fano Coding
def shannon_fano_encode(message):
    # Count frequencies of each symbol
    frequencies = defaultdict(int)
    for char in message:
        frequencies[char] += 1

    # Sort symbols by frequency
    sorted_symbols = sort_by_frequency(frequencies)

    # Generate Shannon-Fano codes
    code_dict = {}
    shannon_fano_code(sorted_symbols, code_dict)

    # Encode the message
    encoded_message = ''.join(code_dict[char] for char in message)
    return encoded_message, code_dict

# Function to decode a message using Shannon-Fano Coding
def shannon_fano_decode(encoded_message, code_dict):
    # Reverse the code dictionary for decoding
    reverse_code_dict = {v: k for k, v in code_dict.items()}
    decoded_message = ''
    current_code = ''
    
    for bit in encoded_message:
        current_code += bit
        if current_code in reverse_code_dict:
            decoded_message += reverse_code_dict[current_code]
            current_code = ''
    
    return decoded_message

# Example usage
if __name__ == '__main__':
    message = "aaryan"
    
    # Encode the message
    encoded_message, code_dict = shannon_fano_encode(message)
    print(f"Encoded message: {encoded_message}")
    print(f"Code dictionary: {code_dict}")

    # Decode the message
    decoded_message = shannon_fano_decode(encoded_message, code_dict)
    print(f"Decoded message: {decoded_message}")
