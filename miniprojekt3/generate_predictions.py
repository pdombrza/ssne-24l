from typing import NamedTuple
import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim as optim
import pandas as pd
import numpy as np
import torch.nn.functional as F
from PIL import Image
import os
import csv
from model_structure_test12_batchNorm1D_more_linears import ConvNet, LinearParams, ConvParams


MODEL_PATH = "./model/finalModel"
DATA_PATH = "./data/test_all"
PREDICTIONS_PATH = "./model/predictions.csv"

def main():
    device = torch.device("cuda")
    model = torch.load(MODEL_PATH).to(device)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    predictions = {}
    model.eval()
    with torch.no_grad():
        for fname in os.listdir(DATA_PATH):
            try:
                img = Image.open(os.path.join(DATA_PATH, fname))
                img = img.convert('RGB')
                img_tensor = transform(img)
                img_tensor = img_tensor.unsqueeze(0).to(device) # unsqueeze because only 1 image at a time
                outputs = model(img_tensor).to("cpu")
                _, predicted = torch.max(outputs.data, 1)
                predictions[fname] = int(predicted)
            except Exception:
                print(fname)

    with open(PREDICTIONS_PATH, "w") as fh:
        writer = csv.writer(fh, delimiter=',')
        for img, pred in predictions.items():
            writer.writerow([img, pred])


if __name__ == "__main__":
    main()