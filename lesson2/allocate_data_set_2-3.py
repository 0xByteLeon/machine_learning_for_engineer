# lesson2.3
import torch
import matplotlib.pyplot as plt


class LinearRegressionModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.tensor([[1.0]])
        )  # torch.nn.Parameter function auto enable requires_grad=True
        self.bias = torch.nn.Parameter(
            torch.tensor(0.0)
        )
        self.weight_history = [self.weight[0][0].item()]
        self.bias_history = [self.bias.item()]

    def forward(self, x):
        return x.mm(self.weight) + self.bias

    def add_history(self):
        self.weight_history.append(self.weight[0][0].item())
        self.bias_history.append(self.bias.item())

    def show_history(self):
        plt.plot(self.weight_history, label='weight')
        plt.plot(self.bias_history, label='bias')
        plt.legend()
        plt.show()


model = LinearRegressionModel()

dataset = [(1, 3), (2, 5), (5, 11), (6, 13), (7, 15), (8, 17), (9, 19), (12, 25), (13, 27)]

dataset_tensor = torch.tensor(dataset, dtype=torch.float32)

# To assign an initial value to the random number generator so that the same random numbers are generated each time
# it runs, ensuring the reproducibility of the training process.
torch.random.manual_seed(0)
random_indices = torch.randperm(dataset_tensor.shape[0])
print(random_indices)

training_indices = random_indices[:int(dataset_tensor.shape[0] * 0.6)]
print(training_indices)

validating_indices = random_indices[int(dataset_tensor.shape[0] * 0.6):int(dataset_tensor.shape[0] * 0.8)]
print(validating_indices)

testing_set = random_indices[int(dataset_tensor.shape[0] * 0.8):]
print(testing_set)

training_set_x = dataset_tensor[training_indices][:, :1]
training_set_y = dataset_tensor[training_indices][:, 1:]
validating_set_x = dataset_tensor[validating_indices][:, :1]
validating_set_y = dataset_tensor[validating_indices][:, 1:]
testing_set_x = dataset_tensor[testing_set][:, :1]
testing_set_y = dataset_tensor[testing_set][:, 1:]

optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
loss_function = torch.nn.MSELoss()

for epoch in range(1, 10000):
    print(f'epoch: {epoch}')

    model.train()

    predicted = model(training_set_x)
    loss = loss_function(predicted, training_set_y)
    print(
        f'training epoch: {epoch}, x: {training_set_x}, y: {training_set_y}, predicted: {predicted}, loss: {loss.item()}, weight: {model.weight.item()}, bias: {model.bias.item()}')

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    model.add_history()

    model.eval()

    validating_predicted = model(validating_set_x)

    validating_accuracy = 1 - ((validating_set_y - validating_predicted).abs() / validating_set_y).mean()

    if validating_accuracy > 0.99:
        break
predicted = model(testing_set_x)
testing_accuracy = 1 - ((testing_set_y - predicted).abs() / testing_set_y).mean()

model.show_history()
