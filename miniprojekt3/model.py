from typing import NamedTuple
import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim as optim
import pandas as pd
import numpy as np
import torch.nn.functional as F


TRAIN_PATH = "./data/train/"
VALIDATE_PATH = "./data/validate"
TEST_PATH = "./data/test"


class LinearParams(NamedTuple):
    inputs: int
    outputs: int
    dropout: float = 0.0


class ConvParams(NamedTuple):
    in_channels: int
    out_channels: int
    conv_kernel_size: int
    stride: int = 1
    padding: int = 0
    pool_kernel_size: int = 2
    pool_stride: int = 2


class ConvNet(nn.Module):
    def __init__(self, conv_layers: list[ConvParams], linear_layers: list[LinearParams]):
        super().__init__()
        # Convolution
        self.convolutional_layers = nn.Sequential()
        self.linear_layers = nn.Sequential()
        for layer in conv_layers:
            self.convolutional_layers.append(nn.Conv2d(in_channels=layer.in_channels, out_channels=layer.out_channels, kernel_size=layer.conv_kernel_size, padding=layer.padding))
            self.convolutional_layers.append(nn.BatchNorm2d(layer.out_channels))
            self.convolutional_layers.append(nn.ReLU())
            self.convolutional_layers.append(nn.MaxPool2d(kernel_size=layer.pool_kernel_size, stride=layer.pool_stride))

        for layer in linear_layers[:-1]:
            self.linear_layers.append(nn.Dropout(layer.dropout))
            self.linear_layers.append(nn.Linear(layer.inputs, layer.outputs))
            self.linear_layers.append(nn.ReLU())

        self.linear_layers.append(nn.Linear(linear_layers[-1].inputs, linear_layers[-1].outputs))

    def forward(self, x):
        x = self.convolutional_layers(x)
        x = torch.flatten(x, 1)
        x = self.linear_layers(x)
        return x


def prepare_cuda() -> None:
    # GPU operations have a separate seed we also want to set
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)

    # Additionally, some operations on a GPU are implemented stochastic for efficiency
    # We want to ensure that all operations are deterministic on GPU (if used) for reproducibility
    torch.backends.cudnn.determinstic = True
    torch.backends.cudnn.benchmark = False

def load_image_dataset(path, transform):
    dataset = torchvision.datasets.ImageFolder(path, transform=transform)
    return dataset


def main():
    device = torch.device("cuda")
    prepare_cuda()

    train_transform = transforms.Compose([transforms.ToTensor(), transforms.CenterCrop(24), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    test_transform = transforms.Compose([transforms.ToTensor(), transforms.CenterCrop(24), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    batch_size = 32
    num_workers = 8
    trainset = load_image_dataset(TRAIN_PATH, transform=train_transform)
    validset = load_image_dataset(VALIDATE_PATH, transform=test_transform)
    testset = load_image_dataset(TEST_PATH, transform=test_transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    validloader = torch.utils.data.DataLoader(validset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)


    linear_layers = [
        LinearParams(inputs=144, outputs=100, dropout=0.4),
        LinearParams(inputs=100, outputs=70, dropout=0.4),
        LinearParams(inputs=70, outputs=50),
    ]
    conv_layers = [
        ConvParams(in_channels=3, out_channels=6, conv_kernel_size=5),
        ConvParams(in_channels=6, out_channels=16, conv_kernel_size=5),
    ]

    model = ConvNet(conv_layers=conv_layers, linear_layers=linear_layers).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    epochs = 5 # 5 for testing purposes because training very slow for some reason (possibly out of my control)

    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(trainloader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print('[%d/100] loss: %.3f' %
            (epoch+1 ,  running_loss / 2000))
    running_loss = 0.0

    # Test model
    total = 0
    correct = 0
    model.eval()
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images = images.to(device)
            outputs = model(images).cpu()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the test images: %d %%' % (
        100 * correct / total))


if __name__ == "__main__":
    main()