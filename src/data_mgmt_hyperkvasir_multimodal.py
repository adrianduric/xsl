# Imported modules
from config import params

import os
from PIL import Image
import numpy as np
import random
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split

# Dictionary for enumerating categories (0-indexed)
LABEL_TO_INDEX = {
    "cecum": 0,
    "ileum": 1,
    "retroflex-rectum": 2,
    "hemorrhoids": 3,
    "polyps": 4,
    "ulcerative-colitis-grade-0-1": 5,
    "ulcerative-colitis-grade-1": 6,
    "ulcerative-colitis-grade-1-2": 7,
    "ulcerative-colitis-grade-2": 8,
    "ulcerative-colitis-grade-2-3": 9,
    "ulcerative-colitis-grade-3": 10,
    "bbps-0-1": 11,
    "bbps-2-3": 12,
    "impacted-stool": 13,
    "dyed-lifted-polyps": 14,
    "dyed-resection-margins": 15,
    "pylorus": 16,
    "retroflex-stomach": 17,
    "z-line": 18,
    "barretts": 19,
    "barretts-short-segment": 20,
    "esophagitis-a": 21,
    "esophagitis-b-d": 22
}

# Image transforms to be used for training and evaluation of models
image_transforms = {
    "train": transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.Resize(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(90),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ]),
    
    "eval": transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
}


def generate_random_sentence(length=20):
    # Read the file containing words
    with open(os.path.join(os.getcwd(), "res/hyper-kvasir_multimodal/common_words.txt")) as f:
        words = f.read().splitlines()  # Split the file into a list of words
    
    # Sample random words from the list
    random_words = random.sample(words, length)  # Get a sample of the specified length
    
    # Concatenate the list of words into a single string with spaces
    return ' '.join(random_words).capitalize() + '.'  # Capitalize the first word and add a period


class HyperKvasirMultimodalDataset(Dataset):

    def __init__(self, img_paths, txt_paths, labels, transform_type, augmented_data):
        self.img_paths = img_paths
        self.txt_paths = txt_paths
        self.labels = labels

        assert transform_type == "train" or transform_type == "eval"
        self.data_transforms = image_transforms[transform_type]

        self.augmented_data = augmented_data

    def __len__(self):

        return len(self.labels)

    def __getitem__(self, idx):

        img_path = self.img_paths[idx]
        txt_path = self.txt_paths[idx]
        label = self.labels[idx]
        
        image = Image.open(img_path)
        
        if self.augmented_data:
            text = open(txt_path).read()
        else:
            text = generate_random_sentence()
        
        image = self.data_transforms(image)
               
        return image, text, label, img_path, txt_path


def prepare_data_hyperkvasir_multimodal(seed, augmented_data, split, batch_size=params["batch_size"]):

    # Setting seed as specified (if none is specified, a random seed is sampled)
    if seed:
        torch.manual_seed(seed=seed)
    else:
        seed = random.randint(0, 10000)
        torch.manual_seed(seed=seed)

    # Storing paths to images, text, and corresponding labels
    
    img_dir_path = os.path.join(os.getcwd(), "res/hyper-kvasir")
    txt_dir_path = os.path.join(os.getcwd(), "res/hyper-kvasir_multimodal")
        

    # File handling for HyperKvasir data
    img_paths = []
    txt_paths = []
    labels = []

    file = open(os.path.join(img_dir_path, "image-labels.csv"), "r")
    file.readline()

    for line in file:

        img_filename, organ, finding, classification = line.strip().split(",")
        txt_filename = img_filename

        img_filename += ".jpg"
        txt_filename += ".txt"
        organ = "lower-gi-tract" if organ == "Lower GI" else "upper-gi-tract"

        path_to_img_file = os.path.join(
            img_dir_path, organ, classification, finding, img_filename
        )
        path_to_txt_file = os.path.join(
            txt_dir_path, organ, classification, finding, txt_filename
        )

        enumerated_label = LABEL_TO_INDEX[finding]

        # If binary labels: modify the enumeration of labels
        if params["num_classes"] == 2:

            # If anatomical landmark: label as y=0
            if enumerated_label <= 2 or (enumerated_label >= 16 and enumerated_label <= 18):
                enumerated_label = 0

            # Else if pathological finding: label as y=1
            elif (enumerated_label >= 3 and enumerated_label <= 10) or enumerated_label >= 19:
                enumerated_label = 1

            # Else if in other categories: discard
            else:
                continue

        # Add image and label to stored paths and labels
        img_paths.append(path_to_img_file)
        txt_paths.append(path_to_txt_file)
        labels.append(enumerated_label)

    file.close()

    # If not splitting data in sets, Dataset and DataLoader objects are created and returned containing all images in the dataset
    if not split:

        complete_dataset = HyperKvasirMultimodalDataset(img_paths, txt_paths, labels, transform_type="eval", augmented_data=False)
        complete_dataloader = DataLoader(complete_dataset, batch_size=batch_size, shuffle=False)
        
        return complete_dataset, complete_dataloader

    # Performing split of dataset

    # First splitting test set and the rest
    temp_img_paths, test_img_paths, temp_img_labels, test_img_labels = train_test_split(img_paths, labels, test_size=params["test_size"], stratify=labels if params["stratify"] else None, random_state=seed)

    temp_txt_paths, test_txt_paths, temp_txt_labels, test_txt_labels = train_test_split(txt_paths, labels, test_size=params["test_size"], stratify=labels if params["stratify"] else None, random_state=seed)

    # Then splitting training and validation sets
    train_img_paths, val_img_paths, train_img_labels, val_img_labels = train_test_split(temp_img_paths, temp_img_labels, test_size=params["val_size"], stratify=temp_img_labels if params["stratify"] else None, random_state=seed)

    train_txt_paths, val_txt_paths, train_txt_labels, val_txt_labels = train_test_split(temp_txt_paths, temp_txt_labels, test_size=params["val_size"], stratify=temp_txt_labels if params["stratify"] else None, random_state=seed)

    # Asserting that img and txt labels were split correctly
    assert train_img_labels == train_txt_labels
    assert val_img_labels == val_txt_labels
    assert test_img_labels == test_txt_labels

    # Asserting that sets are disjoint
    assert set(train_img_paths).isdisjoint(set(val_img_paths))
    assert set(train_img_paths).isdisjoint(set(test_img_paths))
    assert set(val_img_paths).isdisjoint(set(test_img_paths))

    # Creating Datasets and DataLoaders
    train_dataset = HyperKvasirMultimodalDataset(train_img_paths, train_txt_paths, train_img_labels, transform_type="train", augmented_data=augmented_data)
    val_dataset = HyperKvasirMultimodalDataset(val_img_paths, val_txt_paths, val_img_labels, transform_type="eval", augmented_data=augmented_data)
    test_dataset = HyperKvasirMultimodalDataset(test_img_paths, test_txt_paths, test_img_labels, transform_type="eval", augmented_data=augmented_data)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_dataset, val_dataset, test_dataset, train_dataloader, val_dataloader, test_dataloader