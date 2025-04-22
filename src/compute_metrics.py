# Imported modules
from config import params

from torchmetrics.classification import (
    Accuracy, Precision, Recall, F1Score, MatthewsCorrCoef
)


# Initialising metrics
metric_objects = {
    "accuracy_micro_obj": Accuracy(
        task="multiclass" if params["num_classes"] > 2 else "binary",
        num_classes = params["num_classes"],
        average="micro"
    ).to(device=params["device"]),
    "accuracy_macro_obj": Accuracy(
        task="multiclass" if params["num_classes"] > 2 else "binary",
        num_classes = params["num_classes"],
        average="macro"
    ).to(device=params["device"]),
    "precision_micro_obj": Precision(
        task="multiclass" if params["num_classes"] > 2 else "binary",
        num_classes = params["num_classes"],
        average="micro"
    ).to(device=params["device"]),
    "precision_macro_obj": Precision(
        task="multiclass" if params["num_classes"] > 2 else "binary",
        num_classes = params["num_classes"],
        average="macro"
    ).to(device=params["device"]),
    "recall_micro_obj": Recall(
        task="multiclass" if params["num_classes"] > 2 else "binary",
        num_classes = params["num_classes"],
        average="micro"
    ).to(device=params["device"]),
    "recall_macro_obj": Recall(
        task="multiclass" if params["num_classes"] > 2 else "binary",
        num_classes = params["num_classes"],
        average="macro"
    ).to(device=params["device"]),
    "f1_micro_obj": F1Score(
        task="multiclass" if params["num_classes"] > 2 else "binary",
        num_classes = params["num_classes"],
        average="micro"
    ).to(device=params["device"]),
    "f1_macro_obj": F1Score(
        task="multiclass" if params["num_classes"] > 2 else "binary",
        num_classes = params["num_classes"],
        average="macro"
    ).to(device=params["device"]),
    "mcc_obj": MatthewsCorrCoef(
        task="multiclass" if params["num_classes"] > 2 else "binary",
        num_classes = params["num_classes"]
    ).to(device=params["device"])
}

def update_metrics(preds, y):
    """
    Updates all metric objects with new batch of predictions and targets.

    Parameters:
    preds (Tensor): Predicted labels.
    y (Tensor): True labels.
    """

    for metric in metric_objects.values():
        metric.update(preds, y)

def compute_metrics(total_loss, num_samples):
    """
    Computes and returns current values for all metrics, then resets the metric objects.
    
    Parameters:
    total_loss (float): Total loss accumulated over all batches.
    num_samples (int): Number of samples over which the loss was accumulated.

    Returns:
    dict: A dictionary containing the average loss and computed values of all metrics.
    """

    # Computing metrics
    metric_measurements = {
        "avg_loss": total_loss / num_samples,
        "accuracy_micro": metric_objects["accuracy_micro_obj"].compute().item(),
        "accuracy_macro": metric_objects["accuracy_macro_obj"].compute().item(),
        "precision_micro": metric_objects["precision_micro_obj"].compute().item(),
        "precision_macro": metric_objects["precision_macro_obj"].compute().item(),
        "recall_micro": metric_objects["recall_micro_obj"].compute().item(),
        "recall_macro": metric_objects["recall_macro_obj"].compute().item(),
        "f1_micro": metric_objects["f1_micro_obj"].compute().item(),
        "f1_macro": metric_objects["f1_macro_obj"].compute().item(),
        "mcc": metric_objects["mcc_obj"].compute().item()
    }

    # Resetting metric objects
    for metric in metric_objects.values():
        metric.reset()

    return metric_measurements