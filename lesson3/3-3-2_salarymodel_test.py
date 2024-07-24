import torch
import torch.nn as nn
import gzip


# keep the model definition same as the training script
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


model = SalaryModel()

with gzip.open('salary_model.pt.gz', 'rb') as f:
    model.load_state_dict(torch.load(f))

while True:
    try:
        print("enter input:")
        r = list(map(float, input().split(",")))
        x = torch.tensor(r).view(1, len(r))
        print(model(x)[0, 0].item())
    except Exception as e:
        print("error:", e)
