# Imported modules
import torch

# Configuration parameters
params = {
    "seed": 77,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "num_classes": 23,
    "epochs": 50,

    # Dataset and DataLoader parameters
    "train_size": 0.7,
    "val_size": 0.15,
    "test_size": 0.15,
    "stratify": True,

    # Model specifications
    "model_names": [
        "resnet152",
        "densenet161",
        "vit_b_16",
        "swin_v2_t"
    ],

    # Hyperparameters
    "batch_size": 16,
    "lr": 1e-4,
    "momentum": 0.9
}
