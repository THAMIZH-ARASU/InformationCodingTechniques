# Function to perform Run-Length Encoding
def run_length_encode(message):
    encoded = []
    count = 1

    for i in range(1, len(message)):
        if message[i] == message[i - 1]:
            count += 1
        else:
            encoded.append(f"{count}{message[i - 1]}")
            count = 1

    # Add the last group
    encoded.append(f"{count}{message[-1]}")
    return ''.join(encoded)

# Function to decode the encoded message
def run_length_decode(encoded_message):
    decoded = []
    count = ''

    for char in encoded_message:
        if char.isdigit():
            count += char
        else:
            decoded.append(char * int(count))
            count = ''

    return ''.join(decoded)

# Main code to handle file I/O
def main():
    # Reading the input message from a file
    input_file = 'input_message.txt'
    with open(input_file, 'r') as file:
        message = file.read().strip()

    # Encode the message
    encoded_message = run_length_encode(message)
    print(f"Encoded: {encoded_message}")
    
    # Save the encoded message to a file
    encoded_file = 'encoded_message.rle'
    with open(encoded_file, 'w') as file:
        file.write(encoded_message)

    print(f"Encoded message saved to {encoded_file}")

    # Read the encoded message back from the file for decoding
    with open(encoded_file, 'r') as file:
        encoded_message = file.read().strip()

    # Decode the message
    decoded_message = run_length_decode(encoded_message)
    print(f"Decoded: {decoded_message}")

    # Save the decoded message to a file
    decoded_file = 'decoded_message.txt'
    with open(decoded_file, 'w') as file:
        file.write(decoded_message)

    print(f"Decoded message saved to {decoded_file}")

    # Check if encoding-decoding is correct
    if message == decoded_message:
        print("Success! The decoded message matches the original.")
    else:
        print("Error! The decoded message does not match the original.")

# Run the main function
if __name__ == "__main__":
    main()
