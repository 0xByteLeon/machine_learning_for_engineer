import torch

torch.random.manual_seed(0)

model = torch.nn.Linear(3, 1)

loss_function = torch.nn.MSELoss()

optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

dataset_x = torch.randn((20, 3))
dataset_y = dataset_x.mm(torch.tensor([[1.0], [2.0], [3.0]])) + 8
print(f'dataset_x: {dataset_x}')
print(f'dataset_y: {dataset_y}')

random_indices = torch.randperm(dataset_x.shape[0])

training_indices = random_indices[:int(dataset_x.shape[0] * 0.6)]
validating_indices = random_indices[int(dataset_x.shape[0] * 0.6):int(dataset_x.shape[0] * 0.8)]
testing_indices = random_indices[int(dataset_x.shape[0] * 0.8):]

training_set_x = dataset_x[training_indices]
training_set_y = dataset_y[training_indices]
validating_set_x = dataset_x[validating_indices]
validating_set_y = dataset_y[validating_indices]
testing_set_x = dataset_x[testing_indices]
testing_set_y = dataset_y[testing_indices]

for epoch in range(1, 10000):
    model.train()
    predicted = model(training_set_x)
    loss = loss_function(predicted, training_set_y)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    model.eval()
    validating_predicted = model(validating_set_x)
    validating_accuracy = 1 - ((validating_set_y - validating_predicted).abs() / validating_set_y).mean()
    if validating_accuracy > 0.99:
        break

testing_predicted = model(testing_set_x)
testing_accuracy = 1 - ((testing_set_y - testing_predicted).abs() / testing_set_y).mean()
print(f'testing accuracy: {testing_accuracy}')

print(f'weight: {model.weight}')
print(f'bias: {model.bias}')
