from collections import Counter

# Create Huffman Tree nodes
class Node:
    def __init__(self, char, freq):
        self.char = char
        self.freq = freq
        self.left = None
        self.right = None

# Build the Huffman Tree
def build_tree(frequencies):
    nodes = [Node(char, freq) for char, freq in frequencies.items()]
    
    while len(nodes) > 1:
        nodes = sorted(nodes, key=lambda x: x.freq)
        left = nodes.pop(0)  # Smallest frequency node
        right = nodes.pop(0)  # Second smallest
        merged = Node(None, left.freq + right.freq)
        merged.left = left
        merged.right = right
        nodes.append(merged)  # Add merged node back to the list
    
    return nodes[0]

# Generate Huffman Codes
def generate_codes(node, code='', codes={}):
    if node:
        if node.char:  # Leaf node
            codes[node.char] = code
        generate_codes(node.left, code + '0', codes)
        generate_codes(node.right, code + '1', codes)
    return codes

# Encode the message
def encode(message, codes):
    return ''.join(codes[char] for char in message)

# Decode the encoded message
def decode(encoded, root):
    decoded = []
    node = root
    for bit in encoded:
        node = node.left if bit == '0' else node.right
        if node.char:  # Leaf node
            decoded.append(node.char)
            node = root  # Go back to root for next character
    return ''.join(decoded)

# Main program
message = input("Enter the message: ")
frequencies = Counter(message)

# Build the tree and get codes
root = build_tree(frequencies)
codes = generate_codes(root)

# Encode and decode
encoded_message = encode(message, codes)
print(f"Encoded: {encoded_message}")

decoded_message = decode(encoded_message, root)
print(f"Decoded: {decoded_message}")

# Check if encoding-decoding is correct
if message == decoded_message:
    print("Encoding and decoding are correct!")
