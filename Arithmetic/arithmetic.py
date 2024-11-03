from collections import defaultdict
from decimal import Decimal, getcontext

getcontext().prec = 500

def calculate_ranges(message):
    frequency = defaultdict(int)
    for char in message:
        frequency[char] += 1
    total_chars = len(message)
    ranges = {}
    lower_bound = Decimal(0)
    for char, count in frequency.items():
        ranges[char] = (lower_bound / total_chars, (lower_bound + count) / total_chars)
        lower_bound += count
    return ranges

# Function to perform Arithmetic Encoding
def arithmetic_encode(message):
    ranges = calculate_ranges(message)
    low = Decimal(0.0)
    high = Decimal(1.0)
    for char in message:
        range_width = high - low
        high = low + range_width * Decimal(ranges[char][1])
        low = low + range_width * Decimal(ranges[char][0])
    return (low + high) / 2  

def arithmetic_decode(encoded_value, message, ranges):
    low = Decimal(0.0)
    high = Decimal(1.0)
    decoded_message = ""

    for _ in range(len(message)):
        range_width = high - low
        value = (encoded_value - low) / range_width

        for char, (low_range, high_range) in ranges.items():
            if Decimal(low_range) <= value < Decimal(high_range):
                decoded_message += char
                high = low + range_width * Decimal(high_range)
                low = low + range_width * Decimal(low_range)
                break

    return decoded_message

def main():
    input_file = input("Enter the text file name: ")
    with open(input_file, 'r') as file:
        message = file.read().strip()
    
    print("Original message: ", message)
    encoded_value = arithmetic_encode(message)
    print(f"Encoded value: {encoded_value}")

    encoded_file = 'encoded_value.arith'
    with open(encoded_file, 'w') as file:
        file.write(str(encoded_value))

    print(f"Encoded value saved to {encoded_file}")

    ranges = calculate_ranges(message)

    decoded_message = arithmetic_decode(Decimal(encoded_value), message, ranges)
    print(f"Decoded Message: {decoded_message}")

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
