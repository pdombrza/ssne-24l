import torch
import torch.utils
import torch.nn as nn
import torch.optim as optim


class Evaluator(nn.Module):
    def __init__(self, input_dim, hidden_dim, class_amount):
        super().__init__()

        self.fc_1 = nn.Linear(input_dim, hidden_dim)
        self.fc_2 = nn.Linear(hidden_dim, 50)
        self.fc_out  = nn.Linear(50, class_amount)

        self.LeakyReLU = nn.LeakyReLU(0.2)

    def get_features(self, x):
        x = torch.flatten(x, 1)
        x = self.LeakyReLU(self.fc_1(x))
        x = self.LeakyReLU(self.fc_2(x))
        return x


    def forward(self, x):
        x = self.get_features(x)
        x = self.fc_out(x)
        return x


def train_evaluator(dataloader, num_epochs, class_amount, device, eval_input_dim, eval_hidden_dim):
    evaluator = Evaluator(eval_input_dim, eval_hidden_dim, class_amount).to(device)

    optimizer = optim.Adam(evaluator.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        total_loss = 0
        for data, targets in iter(dataloader):
            data = data.to(device)
            targets = targets.to(device)

            results = evaluator(data)
            loss = criterion(results, targets)
            total_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Evaluator epoch {epoch+1}, loss: {total_loss}")
    return evaluator