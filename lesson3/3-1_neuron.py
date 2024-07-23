import torch

model = torch.nn.Linear(3, 1)

print(list(model.parameters()))

x = torch.tensor([1, 2, 3], dtype=torch.float)
y = torch.tensor([6], dtype=torch.float)

p = model(x)
print(p)

l = (p - y).abs()
print(f'loss: {l}')
l.backward()
print(list(model.parameters()))
for param in list(model.parameters()):
    print(param.grad)

# 1.1.1
model2 = torch.nn.Linear(3, 2)

print(list(model2.parameters()))

x2 = torch.tensor([1, 2, 3], dtype=torch.float)
y2 = torch.tensor([6, 7], dtype=torch.float)

p2 = model2(x2)

l2 = (p2 - y2).abs().mean()
print(f'loss: {l2}')

l2.backward()

for param in list(model2.parameters()):
    print(f'model2 grad: {param.grad}')
