# Imported modules
from config import params
from data_mgmt_hyperkvasir_multimodal import prepare_data_hyperkvasir_multimodal
from create_language_explanations import create_language_explanations
from model_testing import compute_confidence_interval
from compute_metrics import update_metrics, compute_metrics
from create_graphs import create_graphs

import os
import datetime
import random
from tqdm import tqdm
import torch
from torch import nn
from transformers import ViltProcessor, ViltModel


################################################################################
# 1) Dataset Definition
################################################################################

# Performed in data_mgmt_hyperkvasir_multimodal

################################################################################
# 2) Data Preparation and Loaders
################################################################################

# Performed in data_mgmt_hyperkvasir_multimodal

################################################################################
# 3) Model Definition
################################################################################


# Define a custom model that includes the ViLT model and a classification head
class ViltClassifier(nn.Module):

    def __init__(self):

        super(ViltClassifier, self).__init__()
        self.vilt_model = ViltModel.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
        self.classification_head = nn.Linear(self.vilt_model.config.hidden_size, params["num_classes"])

    def forward(self, inputs):

        # Get model outputs
        outputs = self.vilt_model(**inputs)

        # Use the output from the [CLS] token
        cls_output = outputs.last_hidden_state[:, 0, :]  # Get the representation corresponding to the [CLS] token

        # Pass through the classifier
        logits = self.classification_head(cls_output)
        # logits = self.classification_head(outputs)

        return logits


################################################################################
# 4) Training Function
################################################################################


def train_one_epoch(dataloader, model, loss_fn, optimizer):

    # Set model to training mode
    model.train()

    # Initialising loss counter
    total_loss = 0
    num_samples = 0
    
    # Initializing ViLT Processor
    processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa", do_resize=False, do_rescale=False, do_center_crop=False, do_normalize=False)

    # Iterating through batches
    for images, texts, labels, img_paths, txt_paths in tqdm(dataloader):
        
        # Preprocess inputs and move to device
        inputs = processor(images, texts, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(params["device"]) for k, v in inputs.items()}
        labels = labels.to(params["device"])
            
        # Forward pass and loss calculation
        preds = torch.squeeze(model(inputs))

        if params["num_classes"] > 2:
                loss = loss_fn(preds, labels)
        else:
            loss = loss_fn(preds, labels.float())
        total_loss += loss.item()
        num_samples += len(labels)

        # Add current batch to metric calculations
        update_metrics(preds=preds, y=labels)
        
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

    # Set model to evaluation mode
    model.eval()

    # Initialising loss counter
    total_loss = 0
    num_samples = 0
    
    # Initializing ViLT Processor
    processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa", do_resize=False, do_rescale=False, do_center_crop=False, do_normalize=False)

    # Iterating through batches
    for images, texts, labels, img_paths, txt_paths in tqdm(dataloader):

        with torch.no_grad(): 
            # Preprocess inputs and move to device
            inputs = processor(images, texts, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(params["device"]) for k, v in inputs.items()}
            labels = labels.to(params["device"])

            # Forward pass and loss calculation
            preds = torch.squeeze(model(inputs))

            if params["num_classes"] > 2:
                loss = loss_fn(preds, labels)
            else:
                loss = loss_fn(preds, labels.float())

            total_loss += loss.item()
            num_samples += len(labels)

            # Add current batch to metric calculations
            update_metrics(preds=preds, y=labels)

    # Compute final metrics after epoch, and reset metric objects
    metric_measurements = compute_metrics(
        total_loss=total_loss,
        num_samples=num_samples
    )

    return metric_measurements, total_loss


def train_model(seed, model, train_dataloader, val_dataloader, augmented_data, save_model):

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
        val_metrics_epoch, val_loss = test_one_epoch(val_dataloader, model, loss_fn)

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
        save_path = os.path.join(os.getcwd(), f"res/models/hyper-kvasir_multimodal/{'augmented' if augmented_data else 'non-augmented'}/vilt")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # Save the best model state
        torch.save(best_model_state, save_path)

    return train_metrics, val_metrics


################################################################################
# 5) Formal testing
################################################################################


def test_model_single_run(model, test_dataloader):
    
    model = model.to(params["device"])

    loss_fn = nn.CrossEntropyLoss() if params["num_classes"] > 2 else nn.BCEWithLogitsLoss()

    # Test model on test set, and saving metrics
    test_metrics, total_loss = test_one_epoch(test_dataloader, model, loss_fn)

    return test_metrics


def test_model(augmented_data, num_runs):

    # To store metrics of each run
    all_run_metrics = []

    print(f"Evaluating model: ViLT")

    for run in range(num_runs):
        print(f"Run {run + 1}/{num_runs}")

        # Sample a random seed for this run
        run_seed = random.randint(0, 10000)
        print(f"Using seed: {run_seed} for this run")

        # Obtaining textual explanations from the Gemma 3 teacher model if testing on XSL data 
        if augmented_data:
            create_language_explanations()

        # Initializing model to be trained for this run (student model if using XSL data)
        student_model = ViltClassifier()
        student_model = student_model.to(params["device"])
        
        # Initializing dataset for this run
        train_dataset, val_dataset, test_dataset, train_dataloader, val_dataloader, test_dataloader = prepare_data_hyperkvasir_multimodal(seed=run_seed, augmented_data=augmented_data, split=True)

        # Performing model training for this run
        train_metrics, val_metrics = train_model(
            seed=None,
            model=student_model,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            augmented_data=augmented_data,
            save_model=True
        )

        # Create graphs of metrics
        create_graphs(
            dataset_name="hyper-kvasir_multimodal",
            model_name="vilt",
            augmented_data=augmented_data,
            train_metrics=train_metrics,
            val_metrics=val_metrics
        )

        # Initializing model again before test so that the saved weights are used
        student_model = ViltClassifier()

        model_path = os.path.join(os.getcwd(), f"res/models/hyper-kvasir_multimodal/{'augmented' if augmented_data else 'non-augmented'}/vilt")
        student_model.load_state_dict(torch.load(model_path, weights_only=True))

        student_model = student_model.to(params["device"])

        # Perform model testing for this run
        test_metrics = test_model_single_run(model=student_model, test_dataloader=test_dataloader)

        # Store test metrics from this run
        all_run_metrics.append(test_metrics)

    # Calculate 95% confidence intervals for metrics
    confidence_intervals = {}
    for metric in all_run_metrics[0].keys():

        data = [run_metrics[metric] for run_metrics in all_run_metrics]
        mean, lower_bound, upper_bound = compute_confidence_interval(data)
        confidence_intervals[metric] = (mean, lower_bound, upper_bound)

    # Get current date for naming files (YYYY-MM-DD)
    current_date = datetime.date.today()
    formatted_date = current_date.strftime("%Y-%m-%d")

    # Saving metrics to text file
    file_save_path = os.path.join(os.getcwd(), f"doc/result_metrics/hyper-kvasir_multimodal/vilt/{'augmented' if augmented_data else 'non-augmented'}/{formatted_date}/test_metrics_vilt.txt")
    os.makedirs(os.path.dirname(file_save_path), exist_ok=True)

    with open(file_save_path, "w") as text_file:
        for metric, values in confidence_intervals.items():
            mean, lower_bound, upper_bound = confidence_intervals[metric]
            text_file.write(f"{metric}: mean={mean}, 95% CI=({lower_bound}, {upper_bound})\n")


if __name__ == "__main__":
    test_model(augmented_data=True, num_runs=5)
