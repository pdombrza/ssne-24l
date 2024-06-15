import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch import nn
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import numpy as np
import evaluate
from tqdm import tqdm

from util import prepare_cuda, prep_data, get_class_weights


def main():
    device = torch.device("cuda")
    prepare_cuda()
    train_data = "data/train_data.csv"
    test_data = "data/test_data.csv"
    train, valid = prep_data(train_data, 0.15)

    model_name = "distilbert-base-uncased"
    tokenizer = DistilBertTokenizer.from_pretrained(model_name)
    model = DistilBertForSequenceClassification.from_pretrained(model_name, num_labels=5)

    def preprocess_function(examples):
        return tokenizer(examples['review'],  padding="max_length", truncation=True)
    
    tokenized_train = train.map(preprocess_function, batched=True)
    tokenized_train = tokenized_train.remove_columns(["review", "__index_level_0__"])
    tokenized_val = valid.map(preprocess_function, batched=True)
    tokenized_val = tokenized_val.remove_columns(["review", "__index_level_0__"])
    tokenized_train.set_format("torch")
    tokenized_val.set_format("torch")

    batch_size = 16
    train_loader = DataLoader(tokenized_train, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(tokenized_train, batch_size=batch_size)
    optimizer = Adam(model.parameters(), lr=0.0002)
    model.to(device)
    loss_fn = nn.CrossEntropyLoss()

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

    metric = evaluate.load("accuracy")

    model.eval()
    for batch in val_loader:
        labels = batch["rating"].to(device)
        batch = {"attention_mask": batch['attention_mask'].to(device), "input_ids": batch['input_ids'].to(device)}#, "token_type_ids":batch['token_type_ids'].to(device)}

        with torch.no_grad():
            outputs = model(**batch)

        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        metric.add_batch(predictions=predictions, references=labels)

    score = metric.compute()
    print(score)



if __name__ == "__main__":
    main()
