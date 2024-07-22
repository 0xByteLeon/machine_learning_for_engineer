# lesson2.1: autograd
import torch
import matplotlib.pyplot as plt

weight = torch.tensor(1.0, requires_grad=True, dtype=torch.float64)
bias = torch.tensor(0.0, requires_grad=True, dtype=torch.float64)
lr = 0.01
# 准备训练集，验证集和测试集
training_set = [
    (torch.tensor(2.0), torch.tensor(5.0)),
    (torch.tensor(5.0), torch.tensor(11.0)),
    (torch.tensor(6.0), torch.tensor(13.0)),
    (torch.tensor(7.0), torch.tensor(15.0)),
    (torch.tensor(8.0), torch.tensor(17.0))
]

validating_set = [
    (torch.tensor(12.0), torch.tensor(25.0)),
    (torch.tensor(1.0), torch.tensor(3.0))
]

testing_set = [
    (torch.tensor(9.0), torch.tensor(19.0)),
    (torch.tensor(13.0), torch.tensor(27.0))
]

# 记录 weight 与 bias 的历史值
weight_history = [weight.item()]
bias_history = [bias.item()]

optimizer = torch.optim.SGD([weight, bias], lr=lr)
loss_fucntion = torch.nn.MSELoss()

for epoch in range(1, 10000):
    for x, y in training_set:
        p = weight * x + bias

        loss = loss_fucntion(p, y.double())

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        print(
            f'training epoch: {epoch}, x: {x}, y: {y}, predicted: {p}, loss: {loss.item()}, weight: {weight.item()}, bias: {bias.item()}')
        bias_history.append(bias.item())
        weight_history.append(weight.item())

    validating_accuracy = 0
    for x, y in validating_set:
        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)
        p = weight * x + bias
        validating_accuracy += 1 - abs(p - y) / y
        print(f'validating , x: {x}, y: {y}, predicted: {p}')
    validating_accuracy /= len(validating_set)

    if validating_accuracy > 0.99:
        break

testing_accuracy = 0

for x, y in testing_set:
    p = weight * x + bias
    print(f'validating , x: {x}, y: {y}, predicted: {p}')
    testing_accuracy += 1 - abs(p - y) / y
testing_accuracy /= len(validating_set)
print(f'testing accuracy: {testing_accuracy}')

plt.plot(weight_history, label='weight')
plt.plot(bias_history, label='bias')
plt.legend()
plt.show()
