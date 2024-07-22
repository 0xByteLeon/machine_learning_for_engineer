import torch

print(torch.tensor(1))

print(torch.tensor(1.0).item())

print(torch.tensor([1.0, 2.0, 3.0]))

print([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

print(torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]).dtype)
print(torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]).shape)

print(torch.tensor([[1, 2, 3], [-1, -2, -3]]).stride())

a = torch.tensor([[1, 2, 3], [4, 5, 6]])
b = torch.randint(1, 10, (3, 4))
print(a)
print(b)
print(a.mm(b))
print(a @ b)
print(a * b)
