import tiktoken

enc = tiktoken.get_encoding('gpt2')
print("Number of takens in gpt2 vocab: " + str(enc.n_vocab))

encoded_msg = enc.encode("hii there")
decoded_msg = enc.decode(encoded_msg)

print("Message: hii there")
print("Encoded message: " + str(encoded_msg))
print("Decoded message: " + decoded_msg)