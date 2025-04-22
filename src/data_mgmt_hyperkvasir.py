# Imported modules
from config import params

import os
from PIL import Image
import numpy as np
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

# Data transforms to be used for training and evaluation of models
data_transforms = {
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

    "train_aug": transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.Resize(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(90),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406, *[np.mean([0.485, 0.456, 0.406]).item()] * 1],
            std=[0.229, 0.224, 0.225, *[np.mean([0.485, 0.456, 0.406]).item()] * 1]
            # mean=[0.485, 0.456, 0.406, *[np.mean([0.485, 0.456, 0.406]).item()] * 4],
            # std=[0.229, 0.224, 0.225, *[np.mean([0.485, 0.456, 0.406]).item()] * 4]
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
    ]),

    "eval_aug": transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.Resize(224),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406, *[np.mean([0.485, 0.456, 0.406]).item()] * 1],
            std=[0.229, 0.224, 0.225, *[np.mean([0.485, 0.456, 0.406]).item()] * 1]
            # mean=[0.485, 0.456, 0.406, *[np.mean([0.485, 0.456, 0.406]).item()] * 4],
            # std=[0.229, 0.224, 0.225, *[np.mean([0.485, 0.456, 0.406]).item()] * 4]
        )
    ])
}


class HyperKvasirDataset(Dataset):
    """
    Custom PyTorch Dataset object for the HyperKvasir dataset.

    Attributes:
        paths (list): Stores file paths to each image in the dataset.
        labels (list): Stores the numerical label (an integer corresponding to a class) of each image in the dataset. 
        data_transforms (string): Specifies whether data transforms should be for training or evaluation.
        augmented_data (bool): Indicates whether the dataset consists of standard images or augmented images with explanations.
    """

    def __init__(self, paths, labels, transform_type, augmented_data):
        """
        Initializes the HyperKvasirDataset instance.

        Args:
            paths (list): List of file paths to the images.
            labels (list): List of numerical labels corresponding to each image.
            transform_type (str): Specifies whether data transforms should be for training ('train') or evaluation ('eval').
            augmented_data (bool): Indicates whether the dataset consists of augmented images.
        """

        self.paths = paths
        self.labels = labels

        assert transform_type == "train" or transform_type == "eval"
        if augmented_data:
            transform_type += "_aug"
        self.data_transforms = data_transforms[transform_type]

        self.augmented_data = augmented_data

    def __len__(self):
        """
        Returns the number of samples in the dataset.
        
        Returns:
            int: Number of samples in the dataset.
        """

        return len(self.labels)

    def __getitem__(self, idx):
        """
        Retrieves the image and its label at the specified index.

        Args:
            idx (int): Index of the image to retrieve.

        Returns:
            tuple: Tuple containing the image tensor, its label, the file path, and the index.
        """

        path = self.paths[idx]
        label = self.labels[idx]
        
        # Load image from image or tensor file depending on whether data is augmented or not
        if self.augmented_data:
            image = torch.load(path)
        else:
            image = Image.open(path)
        
        image = self.data_transforms(image)
               
        return image, label, path, idx


def prepare_data_hyperkvasir(seed, augmented_data, model_explanation, split, batch_size=params["batch_size"]):
    """
    Processes the HyperKvasir dataset to return PyTorch Dataset and DataLoader objects.

    Args:
        seed (int): Seed for random number generation to ensure reproducibility.
        augmented_data (bool): Indicates whether the dataset consists of augmented images with explanations.
        model_explanation (str): Name of the model providing explanations, e.g., 'densenet161'.
        split (bool): If True, splits Datasets and DataLoader objects into training, validation, and test sets. 
                      If False, returns one Dataset and one DataLoader object containing all images from HyperKvasir.
        batch_size (int): Batch size for the DataLoader.

    Returns:
        If `split` is True:
            Tuple[HyperKvasirDataset, HyperKvasirDataset, HyperKvasirDataset, DataLoader, DataLoader, DataLoader]:
            train_dataset, val_dataset, test_dataset, train_dataloader, val_dataloader, test_dataloader
        
        If `split` is False:
            Tuple[HyperKvasirDataset, DataLoader]: complete_dataset, complete_dataloader
    """

    # Setting seed as specified
    if seed:
        torch.manual_seed(seed=seed)

    # Storing paths to images and corresponding labels
    if augmented_data:
        img_dir_path = os.path.join(os.getcwd(), f"res/augmented_images/hyper-kvasir/{model_explanation}")
        csv_file_path = os.path.join(os.getcwd(), "res/augmented_images/hyper-kvasir")
    else:
        img_dir_path = os.path.join(os.getcwd(), "res/hyper-kvasir")

    # File handling for HyperKvasir data
    paths = []
    labels = []

    if augmented_data:
        file = open(os.path.join(csv_file_path, "image-labels.csv"), "r")
    else:
        file = open(os.path.join(img_dir_path, "image-labels.csv"), "r")
    file.readline()

    for line in file:

        img_filename, organ, finding, classification = line.strip().split(",")

        img_filename += ".pt" if augmented_data else ".jpg"
        organ = "lower-gi-tract" if organ == "Lower GI" else "upper-gi-tract"

        path_to_img_file = os.path.join(
            img_dir_path, organ, classification, finding, img_filename
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
        paths.append(path_to_img_file)
        labels.append(enumerated_label)

    file.close()

    # If not splitting data in sets, Dataset and DataLoader objects are created and returned containing all images in the dataset (used for creating CAMs)
    if not split:

        complete_dataset = HyperKvasirDataset(paths, labels, transform_type="eval", augmented_data=False)
        complete_dataloader = DataLoader(complete_dataset, batch_size=batch_size, shuffle=False)
        
        return complete_dataset, complete_dataloader

    # Performing split of dataset

    # First splitting test set and the rest
    temp_paths, test_paths, temp_labels, test_labels = train_test_split(paths, labels, test_size=params["test_size"], stratify=labels if params["stratify"] else None, random_state=seed)

    # Then splitting training and validation sets
    train_paths, val_paths, train_labels, val_labels = train_test_split(temp_paths, temp_labels, test_size=params["val_size"], stratify=temp_labels if params["stratify"] else None, random_state=seed)

    # Asserting that sets are disjoint
    assert set(train_paths).isdisjoint(set(val_paths))
    assert set(train_paths).isdisjoint(set(test_paths))
    assert set(val_paths).isdisjoint(set(test_paths))

    # Creating Datasets and DataLoaders
    train_dataset = HyperKvasirDataset(train_paths, train_labels, transform_type="train", augmented_data=augmented_data)
    val_dataset = HyperKvasirDataset(val_paths, val_labels, transform_type="eval", augmented_data=augmented_data)
    test_dataset = HyperKvasirDataset(test_paths, test_labels, transform_type="eval", augmented_data=augmented_data)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_dataset, val_dataset, test_dataset, train_dataloader, val_dataloader, test_dataloader