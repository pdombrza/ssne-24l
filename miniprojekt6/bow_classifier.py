import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import numpy as np
import re
from util import augment_data, prepare_cuda, get_class_weights


class BoWClassifier(nn.Module):
    def __init__(self):
        super(BoWClassifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(1000, 800),
            nn.LeakyReLU(),
            nn.Linear(800, 600),
            nn.LeakyReLU(),
            nn.Linear(600, 400),
            nn.LeakyReLU(),
            nn.Linear(400, 200),
            nn.LeakyReLU(),
            nn.Linear(200, 50),
            nn.LeakyReLU(),
            nn.Linear(50, 5),
        )

    def forward(self, x):
        return self.classifier(x)


def review_to_words(raw_review):
    review_text = BeautifulSoup(raw_review, 'lxml').get_text()
    letters_only = re.sub("[^a-zA-Z]", " ", review_text)
    words = letters_only.lower().split()
    stops = set(stopwords.words('english'))
    meaningful_words = [word for word in words if not word in stops]
    return " ".join(meaningful_words)


def vectorize(data):
    num_reviews = data['review'].size

    clean_reviews = []
    for review in range(0, num_reviews):
        if (review+1) % 1000 == 0:
            print('Review {} of {}'.format(review+1, num_reviews))
        clean_reviews.append(review_to_words(data['review'][review]))

    vectorizer = CountVectorizer(analyzer='word', tokenizer=None, preprocessor=None, stop_words=None, max_features=1000)
    data_features = vectorizer.fit_transform(clean_reviews)
    data_features = data_features.toarray()
    return data_features


def get_accuracy(model, data_loader, device):
    correct = 0
    total = 0
    model.eval()
    for imgs, labels in data_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        output = model(imgs)
        pred = output.max(1, keepdim=True)[1]
        correct += pred.eq(labels.view_as(pred)).sum().item()
        total += imgs.shape[0]
    return correct / total



def main():
    device = torch.device("cuda")
    prepare_cuda()
    train_data = "data/train_data.csv"
    test_data = "data/test_data.csv"
    original_data = pd.read_csv(train_data)
    train, valid = train_test_split(original_data, test_size=0.15, random_state=42)
    train = train.reset_index(drop=True)
    valid = valid.reset_index(drop=True)
    train = augment_data(train)

    class_weigths = get_class_weights(train).to(device)

    train_data_features = vectorize(train)
    valid_data_features = vectorize(valid)

    train_data = torch.from_numpy(train_data_features).float()
    train_targets = torch.from_numpy(train["rating"].values).long()
    valid_data = torch.from_numpy(valid_data_features).float()
    valid_targets = torch.from_numpy(valid["rating"].values).long()

    train_dataset = data.TensorDataset(train_data, train_targets)
    valid_dataset = data.TensorDataset(valid_data, valid_targets)

    train_loader = data.DataLoader(train_dataset, batch_size=64, shuffle=True, drop_last=True)
    test_loader = data.DataLoader(valid_dataset, batch_size=64, shuffle=False)

    bow_classifier = BoWClassifier().to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weigths)
    optimizer = optim.Adam(bow_classifier.parameters(), lr=0.001)

    iters = []
    losses = []
    train_acc = []
    val_acc = []
    num_epochs = 100
    for epoch in range(num_epochs):
        epoch_losses = []
        for x, labels in iter(train_loader):
            x, labels = x.to(device), labels.to(device)
            bow_classifier.train()
            out = bow_classifier(x).squeeze()

            loss = criterion(out, labels)
            loss.backward()
            epoch_losses.append(loss.item())
            optimizer.step()
            optimizer.zero_grad()

        loss_mean = np.array(epoch_losses).mean()
        iters.append(epoch)
        losses.append(loss_mean)
        test_acc = get_accuracy(bow_classifier, test_loader, device)
        print(f"Epoch [{epoch}/{num_epochs}] loss {loss_mean:.3} test_acc: {test_acc:.3}")
        train_acc.append(get_accuracy(bow_classifier, train_loader, device))
        val_acc.append(test_acc)

    print("Final Training Accuracy: {}".format(train_acc[-1]))
    print("Final Validation Accuracy: {}".format(val_acc[-1]))


if __name__ == "__main__":
    main()
