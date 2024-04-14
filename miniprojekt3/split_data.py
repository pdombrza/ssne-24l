import os
import numpy as np
import shutil


def main():
    data_path = "data/"
    train_all_path = "data/train_all"
    train_folder = os.path.join(data_path, "train")
    valid_folder = os.path.join(data_path, "validate")
    test_folder = os.path.join(data_path, "test")

    for folder_path in [train_folder, valid_folder, test_folder]:
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        for subdir in os.listdir(train_all_path):
            if not os.path.exists(os.path.join(folder_path, subdir)):
                os.makedirs(os.path.join(folder_path, subdir))

    for subdir in os.listdir(train_all_path):
        images = list(os.listdir(os.path.join(train_all_path, subdir)))
        np.random.shuffle(images)
        train_size = round(len(images) * 0.7)
        valid_size = round(len(images) * 0.2)
        test_size = round(len(images) * 0.10)
        for i, image in enumerate(images):
            if i < train_size:
                dest_folder = train_folder
            elif i < train_size + valid_size:
                dest_folder = valid_folder
            else:
                dest_folder = test_folder
            shutil.copy(os.path.join(train_all_path + f"/{subdir}", image), os.path.join(dest_folder + f"/{subdir}", image))


if __name__ == "__main__":
    main()