# lesson2.2
import torch
import matplotlib.pyplot as plt

# 定义参数
weight = torch.tensor([[1.0]], requires_grad=True)  # 1 行 1 列
bias = torch.tensor(0.0, requires_grad=True)

# 创建损失计算器
loss_function = torch.nn.MSELoss()

# 创建参数调整器
optimizer = torch.optim.SGD([weight, bias], lr=0.01)

# training set, validating set, testing set
training_set_x = torch.tensor([[2.0], [5.0], [6.0], [7.0], [8.0]])  # 5 行 1 列，代表有 5 组，每组有 1 个输入
training_set_y = torch.tensor([[5.0], [11.0], [13.0], [15.0], [17.0]])  # 5 行 1 列，代表有 5 组，每组有 1 个输出
validating_set_x = torch.tensor([[12.0], [1.0]])  # 2 行 1 列，代表有 2 组，每组有 1 个输入
validating_set_y = torch.tensor([[25.0], [3.0]])  # 2 行 1 列，代表有 2 组，每组有 1 个输出
testing_set_x = torch.tensor([[9.0], [13.0]])  # 2 行 1 列，代表有 2 组，每组有 1 个输入
testing_set_y = torch.tensor([[19.0], [27.0]])  # 2 行 1 列，代表有 2 组，每组有 1 个输出

weight_histroy = [weight.item()]
bias_history = [bias.item()]

for epoch in range(1, 10000):
    print(f'epoch: {epoch}')
    predicted = training_set_x.mm(weight) + bias
    loss = loss_function(predicted, training_set_y)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    weight_histroy.append(weight[0][0].item())
    bias_history.append(bias.item())

    print(
        f'training epoch: {epoch}, x: {training_set_x}, y: {training_set_y}, predicted: {predicted}, loss: {loss.item()}, weight: {weight.item()}, bias: {bias.item()}')

    with torch.no_grad():
        vp = validating_set_x.mm(weight) + bias
        validating_accuracy = 1 - ((validating_set_y - vp).abs() / validating_set_y).mean()

        if validating_accuracy > 0.99:
            break

testing_predicted = testing_set_x.mm(weight) + bias
testing_accuracy = 1 - ((testing_set_y - testing_predicted).abs() / testing_set_y).mean()
print(f'testing accuracy: {testing_accuracy}')

plt.plot(weight_histroy, label='weight')
plt.plot(bias_history, label='bias')
plt.legend()
plt.show()
