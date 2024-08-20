import torch

data1 = torch.tensor([1, 2, 3], dtype=torch.float)
data2 = torch.tensor([3, 5, 7, 9, 11], dtype=torch.float)
datalist = [data1, data2]

padded = torch.nn.utils.rnn.pad_sequence(datalist, batch_first=True)
print("padded---------------------------Padded Tensor Start------------------------")
print(padded)
print("padded---------------------------Padded Tensor End------------------------")

lengths = torch.tensor([len(x) for x in datalist])
print("lengths---------------------------Original Tensor Lengths Start------------------------")
print(lengths)
print("lengths---------------------------Original Lengths End------------------------")

rnn_model = torch.nn.RNN(input_size=1, hidden_size=8, batch_first=True)

linear_model = torch.nn.Linear(8, 1)
print("---------------------------padded.shape Start------------------------")
print(padded.shape)
print("---------------------------padded.shape End------------------------")
x = padded.reshape(padded.shape[0], padded.shape[1], 1)
print(x)

packed = torch.nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
print(packed)

output, hidden = rnn_model(packed)

print("output:", output)
print("hidden:", hidden)

unpacked, unpacked_len = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first=True)

print("unpacked:", unpacked)
last_hidden_indices = (lengths - 1).reshape(-1, 1, 1).repeat(1, 1, unpacked.shape[2])
print("last_hidden_indices:", last_hidden_indices)
last_hidden = unpacked.gather(1, last_hidden_indices)

print("last_hidden:", last_hidden)

predicted = linear_model(last_hidden.squeeze(1))
print("predicted:", predicted)
