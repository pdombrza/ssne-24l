import time
import math
from copy import deepcopy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

TRAIN_PATH = "pakiet/train_data.csv"
TEST_PATH = "pakiet/test_data.csv"


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


def transform_func(x: float) -> int:
    if x < 100000:
        return 0
    if x < 350000:
        return 1
    return 2


def split_data(df: pd.DataFrame, y_col: str) -> tuple:
    data_train_attrs = df.drop(columns=[y_col], axis=1)
    X_train, X_validate, y_train, y_validate = train_test_split(data_train_attrs, df[y_col], test_size=0.3, random_state=42)
    X_validate, X_test, y_validate, y_test = train_test_split(X_validate, y_validate, test_size=0.33, random_state=42)
    return X_train, y_train, X_validate, y_validate, X_test, y_test


def load_data(X: np.ndarray, y: np.ndarray) -> tuple:
    trainDataset = data.TensorDataset(torch.from_numpy(X),torch.from_numpy(y.values))
    dataloader = data.DataLoader(trainDataset, batch_size=32, shuffle=True)
    return dataloader


def main():
    # prepare cuda
    device = torch.device("cuda")
    prepare_cuda()

    df_train = pd.read_csv(TRAIN_PATH)
    df_test = pd.read_csv(TEST_PATH)

    df_train["SalePrice"] = df_train["SalePrice"].map(transform_func) # This might be optional - second idea is to use regression and map answers to available classes
    categorical = ["HallwayType", "HeatingType", "AptManageType", "TimeToBusStop", "TimeToSubway", "SubwayStation"]
    df_train = encode_categorical(df_train, categorical)
    df_test = encode_categorical(df_test, categorical)

    X_train, y_train, X_validate, y_validate, X_test, y_test = split_data(df_train, y_col="SalePrice")


    scaler = StandardScaler()
    X_train_norm = scaler.fit_transform(X_train.values)
    X_test_norm = scaler.transform(X_test.values)
    X_evaluate = scaler.transform(df_test.values)

    train_dataloader = load_data(X_train_norm, y_train)
    test_dataloader = load_data(X_test_norm, y_test)

    print(next(iter(train_dataloader)))


    model = SimpleClassifier([27, 180, 180, 180], [180, 180, 180, 180], 3)
    optimizer = torch.optim.SGD(model.parameters(), lr = 0.15)
    for name, param in model.named_parameters():
        print(f"Parameter {name}, shape {param.shape} dtype: {param.dtype}")
    model.to(device)
    lossCalc = nn.CrossEntropyLoss()
    model.train()
    # Training loop
    for epoch in range(500):
        for data_inputs, data_labels in train_dataloader:
            # Before calculating the gradients, we need to ensure that they are all zero.
            # The gradients would not be overwritten, but actually added to the existing ones.
            optimizer.zero_grad()
            ## Step 1: Move input data to device (only strictly necessary if we use GPU)
            data_inputs = data_inputs.to(device)
            data_labels = data_labels.to(device)
            ## Step 2: Run the model on the input data
            preds = model(data_inputs)
            # preds = torch.softmax(preds, 1, torch.float64)
            # preds = preds.squeeze(dim=1) # Output is [Batch size, 1], but we want [Batch size]
            ## Step 3: Calculate the loss
            loss = lossCalc(preds, data_labels.long())
            ## Step 4: Perform backpropagation
            # Perform backpropagation
            loss.backward()
            ## Step 5: Update the parameters
            optimizer.step()
        print(f"Epoch: {epoch}, loss: {loss.item():.3}")
    model.eval() # Set model to eval mode
    true_preds, num_preds = 0., 0.

    with torch.no_grad(): # Deactivate gradients for the following code
        for data_inputs, data_labels in test_dataloader:

            # Determine prediction of model on dev set
            data_inputs, data_labels = data_inputs.to(device), data_labels.to(device)
            preds = model(data_inputs)
            # preds = preds.squeeze(dim=1)
            preds = torch.softmax(preds, dim=1)
            maxValues, pred_labels = torch.max(preds, 1)

            # Keep records of predictions for the accuracy metric (true_preds=TP+TN, num_preds=TP+TN+FP+FN)
            true_preds += (pred_labels == data_labels).sum()
            num_preds += data_labels.shape[0]

    acc = true_preds / num_preds
    print(f"Accuracy of the model: {100.0*acc:4.2f}%")

    # model = model.to(device)
    # state_dict = model.state_dict()
    # print(state_dict)
    # torch.save(state_dict, "simple_model.tar")
    # torch.save(model,"test")

    # # Load state dict from the disk (make sure it is the same name as above)
    # state_dict = torch.load("simple_model.tar")

    # # Create a new model and load the state
    # new_model = SimpleClassifier(num_inputs=27, num_hidden=180, num_outputs=3).to(device)
    # new_model.load_state_dict(state_dict)
    # # Verify that the parameters are the same
    # print("Original model\n", model.state_dict())
    # print("\nLoaded model\n", new_model.state_dict())


if __name__ == "__main__":
    main()