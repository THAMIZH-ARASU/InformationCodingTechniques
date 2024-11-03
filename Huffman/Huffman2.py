import heapq

class Node:
    def __init__(self, char, freq):
        self.char = char
        self.freq = freq
        self.left = None
        self.right = None

    def __lt__(self, other):
        return self.freq < other.freq

def build_huffman_tree(symbols_freq):
    heap = [Node(char, freq) for char, freq in symbols_freq]
    heapq.heapify(heap)
    while len(heap) > 1:
        left = heapq.heappop(heap)
        right = heapq.heappop(heap)
        merged = Node(None, left.freq + right.freq)
        merged.left = left
        merged.right = right
        heapq.heappush(heap, merged)

    return heap[0]

def generate_huffman_codes(node, code='', codes={}):
    if node is not None:
        if node.char is not None:
            codes[node.char] = code
        generate_huffman_codes(node.left, code + '0', codes)
        generate_huffman_codes(node.right, code + '1', codes)
    return codes

def encode_message(message, codes):
    return ''.join(codes[char] for char in message)

def decode_message(encoded_message, root):
    decoded_message = ''
    current_node = root
    for bit in encoded_message:
        current_node = current_node.left if bit == '0' else current_node.right
        if current_node.char is not None:
            decoded_message += current_node.char
            current_node = root
    return decoded_message

def main():
    input_file = input("Enter the text file name: ")
    with open(input_file, 'r') as file:
        message = file.read().strip()
    print("Original message:", message)

    symbols_freq = {char: message.count(char) for char in set(message)}
    symbols_freq = sorted(symbols_freq.items(), key=lambda x: x[1])

    huffman_tree_root = build_huffman_tree(symbols_freq)
    codes = generate_huffman_codes(huffman_tree_root)

    encoded_message = encode_message(message, codes)
    print("Encoded Message:", encoded_message)

    encoded_file = 'encoded_message.huff'
    with open(encoded_file, 'w') as file:
        file.write(encoded_message)

    print(f"Encoded message saved to {encoded_file}")

    decoded_message = decode_message(encoded_message, huffman_tree_root)
    print("Decoded Message:", decoded_message)

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
