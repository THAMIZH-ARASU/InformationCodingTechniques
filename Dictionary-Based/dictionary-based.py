# Function to perform Dictionary-based Encoding (LZW)
def lzw_encode(message):
    dictionary = {chr(i): i for i in range(256)}
    current_string = ""
    encoded_output = []
    for char in message:
        combined_string = current_string + char
        if combined_string in dictionary:
            current_string = combined_string
        else:
            encoded_output.append(dictionary[current_string])
            dictionary[combined_string] = len(dictionary)
            current_string = char  
    if current_string:
        encoded_output.append(dictionary[current_string])  
    return encoded_output

def lzw_decode(encoded_output):
    dictionary = {i: chr(i) for i in range(256)}
    current_code = encoded_output[0]
    current_string = dictionary[current_code]
    decoded_output = [current_string]
    for code in encoded_output[1:]:
        if code in dictionary:
            entry = dictionary[code]
        else:
            entry = current_string + current_string[0] 
        decoded_output.append(entry)
        dictionary[len(dictionary)] = current_string + entry[0]
        current_string = entry
    return ''.join(decoded_output)

def main():
    input_file = input("Enter the text file name: ")
    with open(input_file, 'r') as file:
        message = file.read().strip()
    print("Original message: ", message)
    encoded_output = lzw_encode(message)
    print(f"Encoded: {encoded_output}")
    encoded_file = 'encoded_message.lzw'
    with open(encoded_file, 'w') as file:
        file.write(','.join(map(str, encoded_output)))
    print(f"Encoded message saved to {encoded_file}")
    with open(encoded_file, 'r') as file:
        encoded_output = list(map(int, file.read().split(',')))
    decoded_message = lzw_decode(encoded_output)
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
