import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
from sklearn.model_selection import train_test_split
from copy import deepcopy
from sklearn.metrics import precision_score, accuracy_score

class SimpleClassifier(nn.Module):

    def __init__(self, inputsList: list[int], neuronsList: list[int], outputs: int, activationFunc: nn.Module = nn.ReLU()):
        super().__init__()
        self._layers = nn.Sequential()
        for inputs, neurons in zip(inputsList, neuronsList):
            self._layers.append(nn.Linear(inputs, neurons, dtype=torch.float64))
            self._layers.append(nn.Dropout(0.6))
            self._layers.append(nn.BatchNorm1d(neurons, dtype=torch.float64))
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
    trainDataset = data.TensorDataset(torch.from_numpy(X), Y)
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

def saveToFile(filename: str, text: str, toAppend: bool):
    mode = "w"
    if toAppend:
        mode = "a"
    with open(filename, mode) as plik:
        plik.write(text)

def main():
    saveToFile("structureTestsBatchNormDropout0_6.txt", f'', toAppend=False)
    # prepare cuda
    device = torch.device("cuda")
    prepare_cuda()

    df_train = pd.read_csv("pakiet/train_data.csv")
    df_test = pd.read_csv("pakiet/test_data.csv")


    categorical = ["HallwayType", "HeatingType", "AptManageType", "TimeToBusStop", "TimeToSubway", "SubwayStation"]
    df_train = encode_categorical(df_train, categorical)
    df_test = encode_categorical(df_test, categorical)



    X_train, y_train, X_validate, y_validate, X_test, y_test = split_data(df_train, y_col="SalePrice")


    X_train_norm = deepcopy(X_train)
    X_validate_norm = deepcopy(X_validate)
    X_evaluate = deepcopy(df_test)
    scaler = StandardScaler()
    nonCategoricalColumns = ["YearBuilt", "Size(sqf)", "Floor", "N_Parkinglot(Ground)", "N_Parkinglot(Basement)", "N_manager", "N_elevators", "N_FacilitiesInApt", "N_FacilitiesNearBy(Total)", "N_SchoolNearBy(Total)"]
    X_train_norm[nonCategoricalColumns] = scaler.fit_transform(X_train_norm[nonCategoricalColumns])
    X_validate_norm[nonCategoricalColumns] = scaler.transform(X_validate[nonCategoricalColumns])
    X_evaluate[nonCategoricalColumns] = scaler.transform(X_evaluate[nonCategoricalColumns])
    X_train_norm = X_train_norm.values
    X_validate_norm = X_validate_norm.values
    X_evaluate = X_evaluate.values

    trainDataLoader = load_data(X_train_norm, y_train)
    validateDataLoader = load_data(X_validate_norm, y_validate)

    # print(next(iter(trainDataLoader)))

    # Train model
    modelArgs = [[[27, 50], [50, 50], 1, nn.ReLU()], [[27, 100], [100, 100], 1, nn.ReLU()], [[27, 270], [270, 270], 1, nn.ReLU()],
                 [[27, 50, 50], [50, 50, 50], 1, nn.ReLU()], [[27, 100, 100], [100, 100, 100], 1, nn.ReLU()], [[27, 270, 270], [270, 270, 270], 1, nn.ReLU()],
                 [[27, 50, 50, 50], [50, 50, 50, 50], 1, nn.ReLU()], [[27, 100, 100, 100], [100, 100, 100, 100], 1, nn.ReLU()], [[27, 270, 270, 270], [270, 270, 270, 270], 1, nn.ReLU()],
                 [[27, 50, 80, 80, 40], [50, 80, 80, 40, 40], 1, nn.ReLU()]]
    epochs = [200, 200, 200, 300, 300, 300, 600, 600, 600, 700]
    for args, nEpochs in zip(modelArgs, epochs):
        model = SimpleClassifier(*args)
        optimizer = torch.optim.SGD(model.parameters(), lr = 0.001)
        model.to(device)
        lossCalc = nn.L1Loss()
        model.train()
        for epoch in range(nEpochs):
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
            predsTrain = model(torch.from_numpy(X_train_norm).to(device))
            maeTrainError = mae(torch.from_numpy(y_train.values).to("cpu"), predsTrain.to("cpu"))
        print(f"Error of the model on validation set: {meanError}")
        print(f"Error of the model on training set: {maeTrainError}")
        predsArray = preds.to("cpu").numpy()
        predsArray = predsArray.reshape((predsArray.shape[0], ))
        predsTrainArray = predsTrain.to("cpu").numpy()
        predsTrainArray = predsTrainArray.reshape((predsTrainArray.shape[0],))
        transform_func(predsArray)
        transform_func(predsTrainArray)
        trueLabels = deepcopy(y_validate.values)
        trueTrainLabels = deepcopy(y_train.values)
        transform_func(trueLabels)
        transform_func(trueTrainLabels)
        accuracy = accuracy_score(trueLabels, predsArray)
        precisions = precision_score(trueLabels, predsArray, labels=[0, 1, 2], average=None)
        trainAccuracy = accuracy_score(trueTrainLabels, predsTrainArray)
        trainPrecisions = precision_score(trueTrainLabels, predsTrainArray, labels=[0, 1, 2], average=None)
        saveToFile("structureTestsBatchNormDropout0_6.txt", f'Struktura sieci: {args} Całkowita dokładność (walidacyjny): {100*accuracy:4.2f}% precyzje dla klas 0, 1 i 2 (walidacyjny): {precisions} Całkowita dokładność (treningowy): {100*trainAccuracy:4.2f}% precyzje dla klas 0, 1 i 2 (treningowy): {trainPrecisions} Obciążenie: {abs(100 - 100*trainAccuracy):4.2f} Wariancja: {100*abs(trainAccuracy-accuracy):4.2f}\n', toAppend=True)
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