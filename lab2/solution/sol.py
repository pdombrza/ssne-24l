import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
from sklearn.model_selection import train_test_split
from copy import deepcopy


class SimpleClassifier(nn.Module):

    def __init__(self, num_inputs, num_hidden, num_outputs):
        super().__init__()
        self.linear1 = nn.Linear(num_inputs, num_hidden, dtype=torch.float64)
        self.act_fn1 = nn.LeakyReLU()
        self.linear2 = nn.Linear(num_hidden, num_hidden, dtype=torch.float64)
        self.act_fn2 = nn.LeakyReLU()
        self.linear3 = nn.Linear(num_hidden, num_outputs, dtype=torch.float64)

    def forward(self, x):
        x = self.linear1(x)
        x = self.act_fn1(x)
        x = self.linear2(x)
        x = self.act_fn2(x)
        x = self.linear3(x)
        return x


def main():
    # Prepare cuda
    device = torch.device("cuda")
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.determinstic = True
    torch.backends.cudnn.benchmark = False

    # Prepare data
    data_train = pd.read_csv("df_train.csv")
    data_train_copy = deepcopy(data_train)
    data_train_attrs = data_train_copy.drop(["MedHouseVal", "Unnamed: 0"], axis=1)
    X_evaluate = pd.read_csv("X_test.csv")
    dataEvaluate = deepcopy(X_evaluate)
    dataEvaluate = dataEvaluate.drop("Unnamed: 0", axis=1)
    X_train, X_validate, y_train, y_validate = train_test_split(data_train_attrs, data_train["MedHouseVal"], test_size=0.3, random_state=42)
    X_validate, X_test, y_validate, y_test = train_test_split(X_validate, y_validate, test_size=0.33, random_state=42)

    # Normalize data
    scaler = StandardScaler()
    X_train_norm = scaler.fit_transform(X_train.values)
    X_test_norm = scaler.transform(X_test.values)
    X_evaluate = scaler.transform(dataEvaluate.values)
    trainDataset = data.TensorDataset(torch.from_numpy(X_train_norm),torch.from_numpy(y_train.values))
    trainDataLoader = data.DataLoader(trainDataset, batch_size=32, shuffle=True)
    testDataset = data.TensorDataset(torch.from_numpy(X_test_norm),torch.from_numpy(y_test.values))
    testDataLoader = data.DataLoader(testDataset, batch_size=32, shuffle=True)

    # Train model
    model = SimpleClassifier(num_inputs=8, num_hidden=30, num_outputs=1)
    optimizer = torch.optim.SGD(model.parameters(), lr = 0.01)
    model.to(device)
    lossCalc = nn.L1Loss()
    model.train()
    for epoch in range(100):
        for data_inputs, data_labels in trainDataLoader:
            optimizer.zero_grad()
            data_inputs = data_inputs.to(device)
            data_labels = data_labels.to(device)
            preds = model(data_inputs)
            preds = preds.squeeze(dim=1)
            loss = lossCalc(preds, data_labels)
            loss.backward()
            optimizer.step()
        print(f"Epoch: {epoch}, loss: {loss.item():.3}")

    # Test model
    model.eval()
    with torch.no_grad():
        preds = model(torch.from_numpy(X_test_norm).to(device))
        meanError = mae(torch.from_numpy(y_test.values).to("cpu"), preds.to("cpu"))
    print(f"Error of the model: {meanError}")

    # Calculate predictions on test data
    with torch.no_grad():
        preds = model(torch.from_numpy(X_evaluate).to(device))
    predsArray = preds.to("cpu").numpy()

    # Save predictions
    np.savetxt('predictions.csv', predsArray, delimiter=',', fmt="%f", header='', comments='')


def mae(y_true,y_pred):
    return np.absolute(np.subtract(y_true, y_pred)).mean()


if __name__ == "__main__":
    main()