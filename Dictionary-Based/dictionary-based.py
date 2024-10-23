# Function to perform Dictionary-based Encoding (LZW)
def lzw_encode(message):
    # Initialize the dictionary with single character entries
    dictionary = {chr(i): i for i in range(256)}
    current_string = ""
    encoded_output = []

    for char in message:
        combined_string = current_string + char
        if combined_string in dictionary:
            current_string = combined_string
        else:
            # Output the code for the current string
            encoded_output.append(dictionary[current_string])
            # Add the new string to the dictionary
            dictionary[combined_string] = len(dictionary)
            current_string = char  # Start a new current string

    if current_string:
        encoded_output.append(dictionary[current_string])  # Output the last string

    return encoded_output

# Function to decode the encoded message
def lzw_decode(encoded_output):
    # Initialize the dictionary with single character entries
    dictionary = {i: chr(i) for i in range(256)}
    current_code = encoded_output[0]
    current_string = dictionary[current_code]
    decoded_output = [current_string]

    for code in encoded_output[1:]:
        if code in dictionary:
            entry = dictionary[code]
        else:
            entry = current_string + current_string[0]  # Handle special case
        decoded_output.append(entry)

        # Add new entry to the dictionary
        dictionary[len(dictionary)] = current_string + entry[0]
        current_string = entry

    return ''.join(decoded_output)

# Main code to handle file I/O
def main():
    # Reading the input message from a file
    input_file = 'input_message.txt'
    with open(input_file, 'r') as file:
        message = file.read().strip()

    # Encode the message
    encoded_output = lzw_encode(message)
    print(f"Encoded: {encoded_output}")
    
    # Save the encoded message to a file
    encoded_file = 'encoded_message.lzw'
    with open(encoded_file, 'w') as file:
        file.write(','.join(map(str, encoded_output)))

    print(f"Encoded message saved to {encoded_file}")

    # Read the encoded message back from the file for decoding
    with open(encoded_file, 'r') as file:
        encoded_output = list(map(int, file.read().split(',')))

    # Decode the message
    decoded_message = lzw_decode(encoded_output)
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
