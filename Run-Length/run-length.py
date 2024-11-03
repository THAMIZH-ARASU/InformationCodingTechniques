def run_length_encode(message):
    encoded = []
    count = 1
    for i in range(1, len(message)):
        if message[i] == message[i - 1]:
            count += 1
        else:
            encoded.append(f"{count}{message[i - 1]}")
            count = 1
    encoded.append(f"{count}{message[-1]}")
    return ''.join(encoded)

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

def main():
    input_file = input("Enter the text file name: ")
    with open(input_file, 'r') as file:
        message = file.read().strip()
    print("Original message :", message)
    encoded_message = run_length_encode(message)
    print(f"Encoded: {encoded_message}")
    encoded_file = 'encoded_message.rle'
    with open(encoded_file, 'w') as file:
        file.write(encoded_message)
    print(f"Encoded message saved to {encoded_file}")
    with open(encoded_file, 'r') as file:
        encoded_message = file.read().strip()
    decoded_message = run_length_decode(encoded_message)
    print(f"Decoded: {decoded_message}")
    decoded_file = 'decoded_message.txt'
    with open(decoded_file, 'w') as file:
        file.write(decoded_message)
    print(f"Decoded message saved to {decoded_file}")
    if message == decoded_message:
        print("Success! The decoded message matches the original.")
    else:
        print("Error! The decoded message does not match the original.")

if __name__ == "__main__":
    main()
