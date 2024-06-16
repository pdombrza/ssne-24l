import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch import nn
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import numpy as np
from tqdm import tqdm
import pandas as pd

from util import prepare_cuda, prep_data_nosplit, prep_data_test


def main():
    device = torch.device("cuda")
    prepare_cuda()
    train_data = "data/train_data.csv"
    test_data = "data/test_data.csv"
    train, class_weights = prep_data_nosplit(train_data, augment=True)
    class_weights = class_weights.to(device)

    test = prep_data_test(test_data)

    model_name = "distilbert-base-uncased"
    tokenizer = DistilBertTokenizer.from_pretrained(model_name)
    model = DistilBertForSequenceClassification.from_pretrained(model_name, num_labels=5)

    def preprocess_function(examples):
        return tokenizer(examples['review'],  padding="max_length", truncation=True)
    
    tokenized_train = train.map(preprocess_function, batched=True)
    tokenized_train = tokenized_train.remove_columns(["review"])
    tokenized_train.set_format("torch")

    tokenized_test = test.map(preprocess_function, batched=True)
    tokenized_test = tokenized_test.remove_columns(["review"])
    tokenized_test.set_format("torch")


    batch_size = 16
    train_loader = DataLoader(tokenized_train, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(tokenized_test, batch_size=1, shuffle=False)
    optimizer = Adam(model.parameters(), lr=0.00002)
    model.to(device)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    num_epochs = 5

    for epoch in range(num_epochs):
        losses = []
        for batch in tqdm(train_loader):
            labels = batch["rating"].to(device)
            batch = {"attention_mask": batch['attention_mask'].to(device), "input_ids": batch['input_ids'].to(device)}
            outputs = model(**batch)
            loss = loss_fn(outputs.logits, labels)
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()
            losses.append(loss.item())

        print(f"Epoch [{epoch}/{num_epochs}] loss: {np.mean(losses):.4f}")

    torch.save(model.state_dict(), "distil_bert_state_dict")

    preds = torch.tensor([], dtype=torch.int16)
    for batch in tqdm(test_loader):
        batch = {"attention_mask": batch['attention_mask'].to(device), "input_ids": batch['input_ids'].to(device)}#, "token_type_ids":batch['token_type_ids'].to(device)}

        with torch.no_grad():
            outputs = model(**batch)

        logits = outputs.logits
        predictions_tmp = torch.argmax(logits, dim=-1)
        preds = torch.cat((preds, predictions_tmp.to("cpu")), dim=0)

    preds_array = preds.numpy()
    dataframe = pd.DataFrame(preds_array)
    dataframe.to_csv("piatek_Dombrzalski_Kiełbus.csv", header=False, index=False)



if __name__ == "__main__":
    main()
