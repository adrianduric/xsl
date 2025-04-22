# Imported modules
from config import params
from compute_metrics import update_metrics, compute_metrics

import os
import torch
import torch.nn as nn
from tqdm import tqdm


def train_one_epoch(dataloader, model, loss_fn, optimizer):
    """
    Train a given model for one epoch (passing through all images in the DataLoader).

    Args:
        dataloader (DataLoader): DataLoader object for loading data with labels in batches.
        model (torchvision.models): Model to be trained.
        loss_fn (torch.nn): Loss function to be used.
        optimizer (torch.optim): Optimizer to be used.

    Returns:
        dict: Dictionary with metrics as labels and their corresponding measurements as values.
    """

    # Set model to training mode
    model.train()

    # Initialising loss counter
    total_loss = 0
    num_samples = 0
    
    # Iterating through batches
    for X, y, paths, idxs in tqdm(dataloader):

        # Loading batch to device
        X = X.to(params["device"])
        y = y.to(params["device"])
            
        # Forward pass and loss calculation
        preds = torch.squeeze(model(X))
        if params["num_classes"] > 2:
                loss = loss_fn(preds, y)
        else:
            loss = loss_fn(preds, y.float())
        total_loss += loss.item()
        num_samples += len(idxs)

        # Add current batch to metric calculations
        update_metrics(preds=preds, y=y)
        
        # Backpropagation and updating weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Compute final metrics after epoch, and reset metric objects
    metric_measurements = compute_metrics(
        total_loss=total_loss,
        num_samples=num_samples
    )
    
    return metric_measurements


def test_one_epoch(dataloader, model, loss_fn):
    """
    Perform inference with a given model for one epoch (passing through all images in the DataLoader).

    Args:
        dataloader (DataLoader): DataLoader object for loading data with labels in batches.
        model (torchvision.models): Model to be used.
        loss_fn (torch.nn): Loss function to be used.

    Returns:
        tuple: A tuple containing:
            - dict: Dictionary with metrics as labels and their corresponding measurements as values.
            - float: Total loss over the epoch.
            - Tensor: All predictions concatenated as a single tensor.
            - Tensor: All targets concatenated as a single tensor.
    """

    # Set model to evaluation mode
    model.eval()

    # Initialising loss counter
    total_loss = 0
    num_samples = 0

    # Saving all predictions and targets for use in ensemble model
    all_preds = []
    all_targets = []
    
    # Iterating through batches
    for X, y, path, idxs in tqdm(dataloader):

        with torch.no_grad(): 
            # Loading batch to device
            X = X.to(params["device"])
            y = y.to(params["device"])

            # Forward pass and loss calculation
            preds = torch.squeeze(model(X))
            if params["num_classes"] > 2:
                loss = loss_fn(preds, y)
            else:
                loss = loss_fn(preds, y.float())
            total_loss += loss.item()
            num_samples += len(idxs)

            # Saving batch predictions and targets
            all_preds.append(preds)
            all_targets.append(y)

            # Add current batch to metric calculations
            update_metrics(preds=preds, y=y)

    # Compute final metrics after epoch, and reset metric objects
    metric_measurements = compute_metrics(
        total_loss=total_loss,
        num_samples=num_samples
    )

    # Returning all predictions and targets as concatenated tensor
    all_preds = torch.cat(all_preds, dim=0) # [num_samples, num_classes], e.g. [n, 23]
    all_targets = torch.cat(all_targets, dim=0)

    return metric_measurements, total_loss, all_preds, all_targets


def train_model(seed, dataset_name, model, model_name, train_dataloader, val_dataloader, augmented_data, save_model):
    """
    Train the model for a specified number of epochs, and optionally save the model state.

    Args:
        seed (int): Random seed for reproducibility.
        dataset_name (str): The name of the dataset to be used.
        model (torch.nn.Module): The model to be trained.
        model_name (str): Name of the model.
        train_dataloader (DataLoader): DataLoader for the training set.
        val_dataloader (DataLoader): DataLoader for the validation set.
        augmented_data (bool): Indicates whether to use augmented data.
        save_model (bool): Flag indicating whether to save the trained model state.

    Returns:
        tuple: A tuple containing:
            - list: List of dictionaries, each containing training metrics for an epoch.
            - list: List of dictionaries, each containing validation metrics for an epoch.
    """

    # Setting seed as specified
    if seed:
        torch.manual_seed(seed=seed)

    # Initializing loss function
    loss_fn = nn.CrossEntropyLoss() if params["num_classes"] > 2 else nn.BCEWithLogitsLoss()

    # Initializing optimizer
    optimizer = torch.optim.SGD(
        params=model.parameters(),
        lr=params["lr"],
        momentum=params["momentum"]
    )

    # Setting params for early stopping
    patience = 5     # Number of epochs with no improvement after which training will be stopped
    min_delta = 0.001  # Minimum change to be considered as an improvement

    # Storing data from runs
    train_metrics = []
    val_metrics = []

    best_loss = float('inf')
    best_model_state = None
    epochs_without_improvement = 0

    # Train model for specified amount of epochs
    for e in range(params["epochs"]):

        # Performing training for one epoch
        train_metrics_epoch = train_one_epoch(train_dataloader, model, loss_fn, optimizer)

        # Tracking metrics on validation sets during training
        val_metrics_epoch, val_loss, all_predictions, all_targets = test_one_epoch(val_dataloader, model, loss_fn)

        # Saving metrics and predictions per epoch
        train_metrics.append(train_metrics_epoch)
        val_metrics.append(val_metrics_epoch)

        # Check for improvement
        print(f'Epoch {e + 1}/{params["epochs"]}, Validation Loss: {val_loss:.4f}')

        if best_loss - val_loss > min_delta:
            best_loss = val_loss
            best_model_state = model.state_dict()  # Save the current best model state
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        # Early stopping condition
        if epochs_without_improvement >= patience:
            print('Early stopping!')
            break

    # Save model
    if save_model:
        
        # Set save path
        save_path = os.path.join(os.getcwd(), f"res/models/{dataset_name}/{'augmented' if augmented_data else 'non-augmented'}/{model_name}")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # Save the best model state
        torch.save(best_model_state, save_path)

    return train_metrics, val_metrics