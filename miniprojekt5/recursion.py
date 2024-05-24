import torch
import torch.utils
from torch.utils.data import DataLoader, Dataset, random_split
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
import torch.optim as optim
import numpy as np

from functools import partial
import pickle


class SequenceDataset(Dataset):
    def __init__(self, data):
        self.data = [(torch.from_numpy(seq).float(), torch.tensor(label).long()) for seq, label in data]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        seq, label = self.data[index]
        return seq.unsqueeze(-1), label


class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def init_hidden(self, batch_size):
        hidden = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        state = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        return hidden, state

    def forward(self, x, len_x, hidden):
        all_outputs, hidden = self.lstm(x, hidden)
        out = all_outputs[torch.arange(all_outputs.size(0)), len_x]
        x = self.fc(out)
        return x, hidden


def load_data(path):
    with open(path, 'rb') as fh:
        data = pickle.load(fh)
    return data

def prepare_cuda():
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)

    torch.backends.cudnn.determinstic = True
    torch.backends.cudnn.benchmark = False


def pad_collate(batch, pad_value):
    xx, yy = zip(*batch)
    x_lens = [len(x) - 1 for x in xx]

    xx_pad = pad_sequence(xx, batch_first=True, padding_value=pad_value)
    yy = torch.stack(yy)

    return xx_pad, yy, x_lens


def train_lstm(lstm, optimizer, scheduler, loss_fun, train_loader, valid_loader, num_epochs, device):
    for epoch in range(num_epochs):
        losses_epoch = []
        for x, targets, x_len in train_loader:
            x, targets, x_len = x.to(device), targets.to(device), torch.tensor(x_len).to(device)
            hidden, state = lstm.init_hidden(x.size(0))
            hidden, state = hidden.to(device), state.to(device)
            preds, _ = lstm(x, x_len, (hidden, state))
            optimizer.zero_grad()
            loss = loss_fun(preds, targets)
            losses_epoch.append(loss.item())
            loss.backward()
            optimizer.step()
        if epoch % 5 == 0:
            valid_acc = 0
            batch_count = 0
            with torch.no_grad():
                for valid_x, valid_targets, valid_len_x in valid_loader:
                    valid_x, valid_targets, valid_len_x = valid_x.to(device), valid_targets.to(device), torch.tensor(valid_len_x).to(device)
                    valid_hidden, valid_state = lstm.init_hidden(valid_x.size(0))
                    valid_hidden, valid_state = valid_hidden.to(device), valid_state.to(device)
                    preds, _ = lstm(valid_x, valid_len_x, (valid_hidden, valid_state))
                    valid_acc += (torch.argmax(preds, dim=1) == targets).sum().item() / len(valid_targets)
                    batch_count += 1
            print(f"Epoch: {epoch}, loss: {np.mean(np.array(losses_epoch)):.4}, valid acc: {valid_acc/batch_count:.4}")
        scheduler.step()
    return lstm


def main():
    device = torch.device("cuda")
    train = load_data("train.pkl")
    prepare_cuda()

    train_dataset = SequenceDataset(train)
    train_size = int(0.8 * len(train_dataset))
    valid_size = len(train_dataset) - train_size
    train_dataset, valid_dataset = random_split(train_dataset, [train_size, valid_size])

    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=partial(pad_collate, pad_value=0), drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, collate_fn=partial(pad_collate, pad_value=0), drop_last=True)

    num_classes = 5
    lstm = LSTMClassifier(1, 25, 1, num_classes).to(device)
    optimizer = optim.Adam(lstm.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.99)
    loss_fun = nn.CrossEntropyLoss()
    num_epochs = 30

    lstm = train_lstm(lstm, optimizer, scheduler, loss_fun, train_loader, valid_loader, num_epochs, device)







if __name__ == "__main__":
    main()