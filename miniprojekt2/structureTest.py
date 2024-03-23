import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
from sklearn.model_selection import train_test_split
from copy import deepcopy


class SimpleClassifier(nn.Module):

    def __init__(self, inputsList: list[int], neuronsList: list[int], outputs: int, activationFunc: nn.Module = nn.ReLU()):
        super().__init__()
        self._layers = nn.Sequential()
        for inputs, neurons in zip(inputsList, neuronsList):
            self._layers.append(nn.Linear(inputs, neurons, dtype=torch.float64))
            self._layers.append(activationFunc)
        self._layers.append(nn.Linear(neuronsList[-1], outputs, dtype=torch.float64))

    def forward(self, x):
        return self._layers(x)


def prepare_cuda() -> None:
    # GPU operations have a separate seed we also want to set
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)

    # Additionally, some operations on a GPU are implemented stochastic for efficiency
    # We want to ensure that all operations are deterministic on GPU (if used) for reproducibility
    torch.backends.cudnn.determinstic = True
    torch.backends.cudnn.benchmark = False


def encode_categorical(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    dummies = pd.get_dummies(data=df, columns=columns, drop_first=True, dtype=float)
    return dummies



def split_data(df: pd.DataFrame, y_col: str) -> tuple:
    data_train_attrs = df.drop(columns=[y_col], axis=1)
    X_train, X_validate, y_train, y_validate = train_test_split(data_train_attrs, df[y_col], test_size=0.3, random_state=42)
    X_validate, X_test, y_validate, y_test = train_test_split(X_validate, y_validate, test_size=0.33, random_state=42)
    return X_train, y_train, X_validate, y_validate, X_test, y_test


def load_data(X: np.ndarray, y: np.ndarray) -> tuple:
    Y = torch.from_numpy(y.values)
    Y = Y.double()
    trainDataset = data.TensorDataset(torch.from_numpy(X),Y)
    dataloader = data.DataLoader(trainDataset, batch_size=32, shuffle=True)
    return dataloader


def transform_func(trueLabels: np.ndarray) -> int:
    for iterator, value in enumerate(trueLabels):
        if value < 100000:
            trueLabels[iterator] = 0
        elif value < 350000:
            trueLabels[iterator] = 1
        else:
            trueLabels[iterator] = 2

def main():
    # prepare cuda
    device = torch.device("cuda")
    prepare_cuda()

    df_train = pd.read_csv("pakiet/train_data.csv")
    df_test = pd.read_csv("pakiet/test_data.csv")


    categorical = ["HallwayType", "HeatingType", "AptManageType", "TimeToBusStop", "TimeToSubway", "SubwayStation"]
    df_train = encode_categorical(df_train, categorical)
    df_test = encode_categorical(df_test, categorical)



    X_train, y_train, X_validate, y_validate, X_test, y_test = split_data(df_train, y_col="SalePrice")


    scaler = StandardScaler()
    X_train_norm = scaler.fit_transform(X_train.values)
    X_validate_norm = scaler.transform(X_validate.values)
    X_evaluate = scaler.transform(df_test.values)

    trainDataLoader = load_data(X_train_norm, y_train)
    validateDataLoader = load_data(X_validate_norm, y_validate)

    # print(next(iter(trainDataLoader)))

    # Train model
    model = SimpleClassifier([27, 50], [50, 50], 1)
    optimizer = torch.optim.SGD(model.parameters(), lr = 0.001)
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
        preds = model(torch.from_numpy(X_validate_norm).to(device))
        meanError = mae(torch.from_numpy(y_validate.values).to("cpu"), preds.to("cpu"))
    print(f"Error of the model: {meanError}")
    predsArray = preds.to("cpu").numpy()
    predsArray = predsArray.reshape((predsArray.shape[0], ))
    transform_func(predsArray)
    predsList = list(predsArray)
    trueLabels = y_validate.values
    transform_func(trueLabels)
    trueList = list(trueLabels)
    true_preds = (predsArray == trueLabels).sum()
    num_preds = len(trueLabels)
    acc = true_preds / num_preds
    print(f"Accuracy of the model: {100.0*acc:4.2f}%")
    # # Calculate predictions on test data
    # with torch.no_grad():
    #     preds = model(torch.from_numpy(X_evaluate).to(device))
    # predsArray = preds.to("cpu").numpy()

    # # Save predictions
    # np.savetxt('predictions.csv', predsArray, delimiter=',', fmt="%f", header='', comments='')


def mae(y_true,y_pred):
    return np.absolute(np.subtract(y_true, y_pred)).mean()

def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


if __name__ == "__main__":
    main()