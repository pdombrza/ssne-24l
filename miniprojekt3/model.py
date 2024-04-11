import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
import pandas as pd
import numpy as np
import torch.nn.functional as F


TRAIN_PATH = "./data/train/"
VALIDATE_PATH = "./data/validate"
TEST_PATH = "./data/validate"


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

    train_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    batch_size = 32
    num_workers = 8
    trainset = load_image_dataset(TRAIN_PATH, transform=train_transform)
    validset = load_image_dataset(VALIDATE_PATH, transform=test_transform)
    testset = load_image_dataset(TEST_PATH, transform=test_transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    validloader = torch.utils.data.DataLoader(validset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)


    images, labels = next(iter(trainloader))


if __name__ == "__main__":
    main()