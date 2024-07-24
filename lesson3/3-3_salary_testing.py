import pandas
import torch
import gzip
from torch import nn

torch.set_default_device("mps")
salary_data_url = 'https://github.com/303248153/303248153.github.io/blob/master/ml-03/salary.csv'


class SalaryModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(8, 100)
        self.layer2 = nn.Linear(100, 50)
        self.layer3 = nn.Linear(50, 1)

    def forward(self, x):
        hidden1 = nn.functional.relu(self.layer1(x))
        hidden2 = nn.functional.relu(self.layer2(hidden1))
        y = self.layer3(hidden2)
        return y


torch.random.manual_seed(0)

df = pandas.read_csv('salary.csv')

dataset_tensor = torch.tensor(df.values, dtype=torch.float32)
# dataset_tensor = dataset_tensor[0:100]
random_indices = torch.randperm(dataset_tensor.shape[0])
training_indices = random_indices[:int(dataset_tensor.shape[0] * 0.6)]
validating_indices = random_indices[int(dataset_tensor.shape[0] * 0.6):int(dataset_tensor.shape[0] * 0.8)]
testing_indices = random_indices[int(dataset_tensor.shape[0] * 0.8):]

training_set_x = dataset_tensor[training_indices][:, :-1]
training_set_y = dataset_tensor[training_indices][:, -1:]
validating_set_x = dataset_tensor[validating_indices][:, :-1]
validating_set_y = dataset_tensor[validating_indices][:, -1:]
testing_set_x = dataset_tensor[testing_indices][:, :-1]
testing_set_y = dataset_tensor[testing_indices][:, -1:]

model = SalaryModel()
loss_function = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.0000001)
batch_capacity = 100

print(
    f'training set x length: {len(training_set_x)}, training set y length: {len(training_set_y)}, validating set x length: {len(validating_set_x)}, validating set y length: {len(validating_set_y)}, testing set x length: {len(testing_set_x)}, testing set y length: {len(testing_set_y)}')

for epoch in range(1, 100):
    print(f'epoch: {epoch}')

    model.train()

    for batch in range(0, training_set_x.shape[0], batch_capacity):
        predicted = model(training_set_x[batch:batch + batch_capacity])
        loss = loss_function(predicted, training_set_y[batch:batch + batch_capacity])
        # print(
        #     f'training epoch: {epoch}, x: {training_set_x[batch:batch + batch_capacity]}, y: {training_set_y[batch:batch + batch_capacity]}, predicted: {predicted}, loss: {loss.item()}')

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    model.eval()
    validating_accuracy = 0
    validating_num_batches = 0

    for validating_batch in range(0, validating_set_x.shape[0], batch_capacity):
        validating_batch_x = validating_set_x[validating_batch:validating_batch + batch_capacity]
        validating_batch_y = validating_set_y[validating_batch:validating_batch + batch_capacity]
        vp = model(validating_batch_x)
        batch_accuracy = 1 - ((validating_batch_y - vp).abs() / validating_batch_y).mean()
        validating_accuracy += batch_accuracy
        validating_num_batches += 1
    validating_accuracy /= validating_num_batches
    print(f'validating accuracy: {validating_accuracy}')
    if validating_accuracy > 0.99:
        break

testing_accuracy = 0
testing_num_batches = 0

for testing_batch in range(0, testing_set_x.shape[0], batch_capacity):
    testing_batch_x = testing_set_x[testing_batch:testing_batch + batch_capacity]
    testing_batch_y = testing_set_y[testing_batch:testing_batch + batch_capacity]
    predicted = model(testing_batch_x)
    batch_accuracy = 1 - ((testing_batch_y - predicted).abs() / testing_batch_y).mean()
    testing_num_batches += 1
testing_accuracy /= testing_num_batches
print(f'testing accuracy: {testing_accuracy}')

torch.save(model.state_dict(), gzip.GzipFile("salary_model.pt.gz", "wb"))

# torch.save(model.state_dict(), 'salary_model.pth')

# while True:
#     try:
#         print("enter input:")
#         r = list(map(float, input().split(",")))
#         x = torch.tensor(r).view(1, len(r))
#         print(model(x)[0, 0].item())
#     except Exception as e:
#         print("error:", e)
