import fasttext
from util import load_train_valid, augment_data, augment_data_full


def preprocess(data):
    data['formatted'] = '__label__' + data['rating'].astype(str) + ' ' + data['review']
    return data['formatted']


def format_data():
    train_data = "data/train_data.csv"
    train, valid = load_train_valid(train_data, 0.15)
    train = augment_data_full(train)
    formatted_train = preprocess(train)
    formatted_valid = preprocess(valid)
    formatted_train.to_csv("fasttext_data/train.csv", index=False, header=False, quoting=3, escapechar='/')
    formatted_valid.to_csv("fasttext_data/valid.csv", index=False, header=False, quoting=3, escapechar='/')

def main():
    # format_data()
    model = fasttext.train_supervised(input="fasttext_data/train.csv", epoch=700, lr=0.5)
    result = model.test("fasttext_data/valid.csv")
    print("Number of examples:", result[0])
    print("Accuracy:", result[1])

if __name__ == "__main__":
    main()
