import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset
import nlpaug.augmenter.word as naw


def prepare_cuda():
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)

    torch.backends.cudnn.determinstic = True
    torch.backends.cudnn.benchmark = False


def load_train_valid(path, test_size):
    data = pd.read_csv(path)
    train, valid = train_test_split(data, test_size=test_size, random_state=42)
    return train, valid


def prep_data(path, test_size, augment=True):
    train, valid = load_train_valid(path, test_size)
    if augment:
        train = augment_data(train)
    class_weights = get_class_weights(train)
    train = Dataset.from_pandas(train)
    valid = Dataset.from_pandas(valid)
    return train, valid, class_weights


def prep_data_nosplit(path, augment=True):
    train = pd.read_csv(path)
    if augment:
        train = augment_data(train)
    class_weights = get_class_weights(train)
    train = Dataset.from_pandas(train)
    return train, class_weights


def prep_data_test(path):
    test = pd.read_csv(path)
    test = Dataset.from_pandas(test)
    return test


def augment_data(trainset):
    syn_aug = naw.SynonymAug()
    for i in range(4):
        to_augment = trainset.loc[trainset['rating'] == i, 'review']
        to_augment = to_augment.to_list()
        augmented_data = syn_aug.augment(to_augment)
        if i == 3:
            val_counts = trainset['rating'].value_counts()
            len_to_aug = val_counts[4] - val_counts[3]
            augmented_data = augmented_data[:len_to_aug]
        augmented_data = {
            'review': augmented_data,
            'rating': [i] * len(augmented_data),
        }
    trainset = pd.concat([trainset, pd.DataFrame(augmented_data)], ignore_index=True)
    return trainset
            

def augment_data_full(trainset):
    syn_aug = naw.SynonymAug()
    spell_aug = naw.SpellingAug()
    ant_aug = naw.AntonymAug()
    rand_aug = naw.RandomWordAug()
    for i in range(4):
        to_augment = trainset.loc[trainset['rating'] == i, 'review']
        to_augment = to_augment.to_list()
        augmented_data = syn_aug.augment(to_augment)
        if i in (0, 1):
            rand_augmented = rand_aug.augment(to_augment)
            augmented_data += rand_augmented
        if i != 3:
            con_augmented = spell_aug.augment(to_augment)
            ant_augmented = ant_aug.augment(to_augment)
            augmented_data += con_augmented
            augmented_data += ant_augmented
        if i == 3:
            val_counts = trainset['rating'].value_counts()
            len_to_aug = val_counts[4] - val_counts[3]
            augmented_data = augmented_data[:len_to_aug]
        augmented_data = {
            'review': augmented_data,
            'rating': [i] * len(augmented_data),
        }
    trainset = pd.concat([trainset, pd.DataFrame(augmented_data)], ignore_index=True)
    return trainset


def get_class_weights(data):
    class_counts = dict(data["rating"].value_counts())
    labels, counts = zip(*class_counts.items())
    class_weights = torch.tensor([1.0 / count for count in counts], dtype=torch.float32)
    return class_weights
