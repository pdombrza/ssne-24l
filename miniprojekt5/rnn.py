import torch
import torch.utils
from torch.utils.data import DataLoader, Dataset, random_split
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd

from functools import partial
from collections import Counter
import pickle


class SequenceDataset(Dataset):
    def __init__(self, data):
        self.data = [(torch.from_numpy(seq).float(), torch.tensor(label).long()) for seq, label in data]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        seq, label = self.data[index]
        return seq.unsqueeze(-1), label


class RNNClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def init_hidden(self, batch_size):
        hidden = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        return hidden

    def forward(self, x, len_x, hidden):
        all_outputs, hidden = self.rnn(x, hidden)
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



def calculate_class_weights(class_counts):
    class_weights = torch.tensor([1 / class_count for class_count in class_counts.values()])
    return class_weights


def pad_collate(batch, pad_value):
    xx, yy = zip(*batch)
    x_lens = [len(x) - 1 for x in xx]

    xx_pad = pad_sequence(xx, batch_first=True, padding_value=pad_value)
    yy = torch.stack(yy)

    return xx_pad, yy, x_lens


def train_rnn(model, optimizer, scheduler, loss_fun, train_loader, valid_loader, num_epochs, device):
    for epoch in range(num_epochs):
        losses_epoch = []
        for x, targets, x_len in train_loader:
            x, targets, x_len = x.to(device), targets.to(device), torch.tensor(x_len).to(device)
            hidden = model.init_hidden(x.size(0)).to(device)
            preds, _ = model(x, x_len, hidden)
            optimizer.zero_grad()
            loss = loss_fun(preds, targets)
            losses_epoch.append(loss.item())
            loss.backward()
            optimizer.step()
        if epoch % 10 == 0:
            valid_acc = 0
            batch_count = 0
            with torch.no_grad():
                for valid_x, valid_targets, valid_len_x in valid_loader:
                    valid_x, valid_targets, valid_len_x = valid_x.to(device), valid_targets.to(device), torch.tensor(valid_len_x).to(device)
                    valid_hidden = model.init_hidden(valid_x.size(0)).to(device)
                    preds, _ = model(valid_x, valid_len_x, valid_hidden)
                    valid_acc += (torch.argmax(preds, dim=1) == valid_targets).sum().item() / len(valid_targets)
                    batch_count += 1
            print(f"Epoch: {epoch}, loss: {np.mean(np.array(losses_epoch)):.4}, valid acc: {valid_acc/batch_count:.4}")
        scheduler.step()
    return model


def main():
    device = torch.device("cuda")
    train = load_data("train.pkl")
    test = load_data("test_no_target.pkl")
    max_length = max(len(array) for array in test)
    test = [np.pad(array, (0, max_length - len(array)), 'constant', constant_values=0) for array in test]
    # test = torch.tensor(test)
    # test = test.type(torch.float32)
    # test = test.unsqueeze(2)
    # size = test.size()
    prepare_cuda()

    classes = [label for _, label in train]
    class_counts = dict(Counter(classes))
    class_weights = calculate_class_weights(class_counts).to(device)

    train_dataset = SequenceDataset(train)
    train_size = int(0.8 * len(train_dataset))
    valid_size = len(train_dataset) - train_size
    train_dataset, valid_dataset = random_split(train_dataset, [train_size, valid_size])


    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=partial(pad_collate, pad_value=0), drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, collate_fn=partial(pad_collate, pad_value=0), drop_last=True)
    # test_loader = DataLoader(test, batch_size=batch_size, shuffle=False, drop_last=False)

    num_classes = 5
    rnn = RNNClassifier(1, 25, 1, num_classes).to(device)
    optimizer = optim.Adam(rnn.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.99)
    loss_fun = nn.CrossEntropyLoss(weight=class_weights)
    num_epochs = 500

    rnn = train_rnn(rnn, optimizer, scheduler, loss_fun, train_loader, valid_loader, num_epochs, device)

    preds = torch.tensor([], dtype=torch.float32)
    with torch.no_grad():
        # test_data = test
        # test_data_len = torch.tensor(len(test_data))
        for iterator in range(1, len(test) + 1, 1):
            test_data, test_data_len = torch.tensor(test[iterator - 1:iterator], dtype=torch.float32, device=device, pin_memory=True).unsqueeze(2), torch.tensor(len(test[0]) - 1).to(device)
            test_hidden = rnn.init_hidden(1).to(device)
            # test_hidden = test_hidden.squeeze(dim = 0)
            # test_state = test_state.squeeze(dim = 0)
            # test_data_size, test_hidden_size, test_state_size = test_data.size(), test_hidden.size(), test_state.size()
            preds_tmp, _ = rnn(test_data, test_data_len, test_hidden)
            preds_tmp = torch.argmax(preds_tmp, dim=1)
            preds = torch.cat((preds, preds_tmp.to("cpu")), dim = 0)
    preds_array = preds.numpy()
    dataframe = pd.DataFrame(preds_array)
    # dataframe.to_csv("piatek_Dombrzalski_Kiełbus.csv", header=False, index=False)








if __name__ == "__main__":
    main()