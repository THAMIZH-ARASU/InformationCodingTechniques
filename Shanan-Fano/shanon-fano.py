def shannon_fano_recursive(symbols_freq, code=''):
    if len(symbols_freq) == 1:
        return {symbols_freq[0][0]: code}

    total_freq = sum([freq for symbol, freq in symbols_freq])
    cumulative_freq = 0
    split_point = 0

    for i in range(len(symbols_freq)):
        cumulative_freq += symbols_freq[i][1]
        if cumulative_freq >= total_freq / 2:
            split_point = i + 1
            break

    left_part = shannon_fano_recursive(symbols_freq[:split_point], code + '0')
    right_part = shannon_fano_recursive(symbols_freq[split_point:], code + '1')

    left_part.update(right_part)
    return left_part

def encode_message(message, codes):
    return ''.join(codes[char] for char in message)

def decode_message(encoded_message, codes):
    reverse_codes = {v: k for k, v in codes.items()}
    current_code = ''
    decoded_message = ''

    for bit in encoded_message:
        current_code += bit
        if current_code in reverse_codes:
            decoded_message += reverse_codes[current_code]
            current_code = ''

    return decoded_message

def main():
   
    input_file = input("Enter the text file name: ")
    with open(input_file, 'r') as file:
        message = file.read().strip() 
    print("Original message: ", message)
    # Calculate frequency of each symbol
    symbols_freq = {char: message.count(char) for char in set(message)}
    symbols_freq = sorted(symbols_freq.items(), key=lambda x: x[1], reverse=True)

    codes = shannon_fano_recursive(symbols_freq)

    encoded_message = encode_message(message, codes)
    print("Encoded Message: ", encoded_message)
    encoded_file = 'encoded_message.fano'
    with open(encoded_file, 'w') as file:
        file.write(encoded_message)

    print(f"Encoded message saved to {encoded_file}")

    decoded_message = decode_message(encoded_message, codes)
    print("Decoded Message: ", decoded_message)
    decoded_file = 'decoded_message.txt'
    with open(decoded_file, 'w') as file:
        file.write(decoded_message)

    print(f"Decoded message saved to {decoded_file}")

    if message == decoded_message:
        print("Success! The decoded message matches the original.")
    else:
        print("Error! The decoded message does not match the original.")

# Run the main function
if __name__ == "__main__":
    main()
