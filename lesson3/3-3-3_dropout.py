import os
import itertools
import sys

import pandas
import torch
import gzip
from torch import nn
import matplotlib.pyplot as plt

torch.set_default_device("mps")
salary_data_url = 'https://github.com/303248153/303248153.github.io/blob/master/ml-03/salary.csv'

torch.random.manual_seed(0)


# Bad！！！！！！！ Don't use dropout
class SalaryModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(8, 100)
        self.layer2 = nn.Linear(100, 50)
        self.layer3 = nn.Linear(50, 1)
        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.2)

    def forward(self, x):
        hidden1 = self.dropout1(nn.functional.relu(self.layer1(x)))
        hidden2 = self.dropout2(nn.functional.relu(self.layer2(hidden1)))
        y = self.layer3(hidden2)
        return y


def save_tensor(tensor, filename):
    with gzip.open(filename, 'wb') as f:
        torch.save(tensor, f)


def load_tensor(filename):
    with gzip.open(filename, 'rb') as f:
        return torch.load(f)


def prepare(batch_size=2000, salary_path="data/salary.csv"):
    if not os.path.isdir("data"):
        os.makedirs("data")

    for batch, df in enumerate(pandas.read_csv(salary_path, chunksize=2000)):
        dataset_tensor = torch.tensor(df.values, dtype=torch.float32)
        # Normalize the data
        dataset_tensor *= torch.tensor([0.01, 1, 0.01, 0.2, 0.2, 0.2, 0.2, 0.2, 0.0001])

        random_indices = torch.randperm(dataset_tensor.shape[0])
        training_indices = random_indices[:int(dataset_tensor.shape[0] * 0.6)]
        validating_indices = random_indices[int(dataset_tensor.shape[0] * 0.6):int(dataset_tensor.shape[0] * 0.8)]
        testing_indices = random_indices[int(dataset_tensor.shape[0] * 0.8):]

        training_set = dataset_tensor[training_indices]
        validating_set = dataset_tensor[validating_indices]
        testing_set = dataset_tensor[testing_indices]

        save_tensor(training_set, f"data/training_set_{batch}.pt.gz")
        save_tensor(validating_set, f"data/validating_set_{batch}.pt.gz")
        save_tensor(testing_set, f"data/testing_set_{batch}.pt.gz")

        print(f"batch {batch} saved")


def train():
    model = SalaryModel()
    loss_function = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    training_accuracy_history = []
    validating_accuracy_history = []

    validating_accuracy_highest = 0
    validating_accuracy_highest_epoch = 0

    def read_batcher(path):
        for batch in itertools.count():
            filepath = f"{path}_{batch}.pt.gz"
            if not os.path.exists(filepath):
                break
            yield load_tensor(filepath)

    def calc_accuracy(predicted, y):
        return 1 - ((y - predicted).abs() / y.abs()).mean().item()

    for epoch in range(1, 1000):
        print(f'epoch: {epoch}')

        model.train()
        training_accuracy_list = []
        for training_set in read_batcher('data/training_set'):
            # Split the training set into batches of 100, Better for model generalization
            batch_step = 100
            for index in range(0, training_set.shape[0], batch_step):
                training_batch_x = training_set[index:index + batch_step, :-1]
                training_batch_y = training_set[index:index + batch_step, -1:]
                predicted = model(training_batch_x)
                loss = loss_function(predicted, training_batch_y)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                with torch.no_grad():
                    training_accuracy_list.append(calc_accuracy(predicted, training_batch_y))
                break
        training_accuracy = sum(training_accuracy_list) / len(training_accuracy_list)
        training_accuracy_history.append(training_accuracy)
        print(f'training accuracy: {training_accuracy}')

        model.eval()
        validating_accuracy_list = []
        for validating_set in read_batcher('data/validating_set'):
            for validating_batch in range(0, validating_set.shape[0], 100):
                validating_batch_x = validating_set[validating_batch:validating_batch + 100, :-1]
                validating_batch_y = validating_set[validating_batch:validating_batch + 100, -1:]
                predicted = model(validating_batch_x)
                validating_accuracy_list.append(calc_accuracy(predicted, validating_batch_y))
        validating_accuracy = sum(validating_accuracy_list) / len(validating_accuracy_list)
        validating_accuracy_history.append(validating_accuracy)
        print(f'validating accuracy: {validating_accuracy}')

        if validating_accuracy > validating_accuracy_highest:
            validating_accuracy_highest = validating_accuracy
            validating_accuracy_highest_epoch = epoch
            save_tensor(model.state_dict(), 'data/salary_model.pth.gz')
            print(f'saved model at epoch {epoch}')
        elif epoch - validating_accuracy_highest_epoch > 100:
            print(f'early stop at epoch {epoch}')
            break

    print(f'validating accuracy highest: {validating_accuracy_highest} at epoch {validating_accuracy_highest_epoch}')

    test_model = SalaryModel()
    test_model.load_state_dict(load_tensor('data/salary_model.pth.gz'))

    testing_accuracy_list = []

    for testing_batches in read_batcher('data/testing_set'):
        predicted = test_model(torch.tensor(testing_batches[:, :-1]))
        testing_accuracy_list.append(calc_accuracy(predicted, testing_batches[:, -1:]))
    testing_accuracy = sum(testing_accuracy_list) / len(testing_accuracy_list)
    print(f'testing accuracy: {testing_accuracy}')
    plt.plot(training_accuracy_history, label='training accuracy')
    plt.plot(validating_accuracy_history, label='validating accuracy')
    plt.legend()
    plt.show()


def eval_model():
    parameters = [
        "Age",
        "Gender (0: Male, 1: Female)",
        "Years of work experience",
        "Java Skill (0 ~ 5)",
        "NET Skill (0 ~ 5)",
        "JS Skill (0 ~ 5)",
        "CSS Skill (0 ~ 5)",
        "HTML Skill (0 ~ 5)"
    ]

    model = SalaryModel()
    model.load_state_dict(load_tensor('data/salary_model.pth.gz'))
    model.eval()
    while True:
        try:
            print("enter input:")
            x = torch.tensor([int(input(f'Your {parameter}: ')) for parameter in parameters],
                             dtype=torch.float32)
            x *= torch.tensor([0.01, 1, 0.01, 0.2, 0.2, 0.2, 0.2, 0.2])
            x = x.view(1, len(parameters))
            y = model(x)
            print(y[0, 0].item())
        except Exception as e:
            print("error:", e)


def main():
    if len(sys.argv) < 2:
        print("Usage: python 3-3_salary_testing.py prepare|train|eval")
        exit(1)

    torch.set_default_device("mps")

    for operation in sys.argv[1:]:
        if operation == 'prepare':
            prepare()
        elif operation == 'train':
            train()
        elif operation == 'eval':
            eval_model()
        else:
            print(f"Unknown operation: {operation}")
            exit(1)


if __name__ == '__main__':
    main()
