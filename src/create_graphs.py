# Imported modules
from config import params

import os
import datetime
import matplotlib.pyplot as plt


def create_graphs(dataset_name, model_name, augmented_data, train_metrics, val_metrics):
    """
    Creates and saves graphs for various metrics calculated on the training and validation sets.
    
    Args:
        dataset_name (str): The name of the dataset on which training and inference are performed.
        model_name (str): The name of the model for which the graphs are being created.
        augmented_data (bool): Flag indicating whether the data is augmented.
        train_metrics (list): List containing dictionaries (one per epoch) with keys being names of various metrics, 
                              and values being the value of a given metric. Values in train_metrics are calculated on the training set.
        val_metrics (list): Same as train_metrics, except the values in val_metrics are calculated on the validation set.
    
    The function plots the following metrics:
        - Accuracy (micro and macro)
        - Precision (micro and macro)
        - Recall (micro and macro)
        - F1 Score (micro and macro)
        - Matthews Correlation Coefficient (MCC)
        - Average Loss
        
    The generated graphs are saved to files in a directory structure that indicates whether the data is augmented and the model name.
    """

    # Creating string for save location to specify if data is augmented
    if augmented_data:
        augmented_data_string = "augmented"
    else:
        augmented_data_string = "non-augmented"

    # Get current date for naming files (YYYY-MM-DD)
    current_date = datetime.date.today()
    formatted_date = current_date.strftime("%Y-%m-%d")

    # Get number of epochs for this run
    num_epochs = len(train_metrics)

    # Plotting accuracy
    plt.plot(
        range(num_epochs),
        [metric["accuracy_micro"] for metric in train_metrics],
        label="Training (micro)"
    )
    plt.plot(
        range(num_epochs),
        [metric["accuracy_macro"] for metric in train_metrics],
        label="Training (macro)"
    )
    plt.plot(
        range(num_epochs),
        [metric["accuracy_micro"] for metric in val_metrics],
        label="Validation (micro)"
    )
    plt.plot(
        range(num_epochs),
        [metric["accuracy_macro"] for metric in val_metrics],
        label="Validation (macro)"
    )
    plt.title("Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Value")
    plt.legend()

    plot_save_path = os.path.join(os.getcwd(), f"doc/graphs/{dataset_name}/{model_name}/{augmented_data_string}/{formatted_date}/accuracy.png")
    os.makedirs(os.path.dirname(plot_save_path), exist_ok=True)
    plt.savefig(plot_save_path)
    plt.clf()

    # Plotting precision
    plt.plot(
        range(num_epochs),
        [metric["precision_micro"] for metric in train_metrics], 
        label="Training (micro)"
    )
    plt.plot(
        range(num_epochs),
        [metric["precision_macro"] for metric in train_metrics], 
        label="Training (macro)"
    )
    plt.plot(
        range(num_epochs),
        [metric["precision_micro"] for metric in val_metrics], 
        label="Validation (micro)"
    )
    plt.plot(
        range(num_epochs),
        [metric["precision_macro"] for metric in val_metrics], 
        label="Validation (macro)"
    )
    plt.title("Precision")
    plt.xlabel("Epochs")
    plt.ylabel("Value")
    plt.legend()

    plot_save_path = os.path.join(os.getcwd(), f"doc/graphs/{dataset_name}/{model_name}/{augmented_data_string}/{formatted_date}/precision.png")
    os.makedirs(os.path.dirname(plot_save_path), exist_ok=True)
    plt.savefig(plot_save_path)
    plt.clf()

    # Plotting recall
    plt.plot(
        range(num_epochs),
        [metric["recall_micro"] for metric in train_metrics], 
        label="Training (micro)"
    )
    plt.plot(
        range(num_epochs),
        [metric["recall_macro"] for metric in train_metrics], 
        label="Training (macro)"
    )
    plt.plot(
        range(num_epochs),
        [metric["recall_micro"] for metric in val_metrics], 
        label="Validation (micro)"
    )
    plt.plot(
        range(num_epochs),
        [metric["recall_macro"] for metric in val_metrics], 
        label="Validation (macro)"
    )
    plt.title("Recall")
    plt.xlabel("Epochs")
    plt.ylabel("Value")
    plt.legend()

    plot_save_path = os.path.join(os.getcwd(), f"doc/graphs/{dataset_name}/{model_name}/{augmented_data_string}/{formatted_date}/recall.png")
    os.makedirs(os.path.dirname(plot_save_path), exist_ok=True)
    plt.savefig(plot_save_path)
    plt.clf()

    # Plotting F1 score
    plt.plot(
        range(num_epochs),
        [metric["f1_micro"] for metric in train_metrics], 
        label="Training (micro)"
    )
    plt.plot(
        range(num_epochs),
        [metric["f1_macro"] for metric in train_metrics], 
        label="Training (macro)"
    )
    plt.plot(
        range(num_epochs),
        [metric["f1_micro"] for metric in val_metrics], 
        label="Validation (micro)"
    )
    plt.plot(
        range(num_epochs),
        [metric["f1_macro"] for metric in val_metrics], 
        label="Validation (macro)"
    )
    plt.title("F1 Score")
    plt.xlabel("Epochs")
    plt.ylabel("Value")
    plt.legend()

    plot_save_path = os.path.join(os.getcwd(), f"doc/graphs/{dataset_name}/{model_name}/{augmented_data_string}/{formatted_date}/f1_score.png")
    os.makedirs(os.path.dirname(plot_save_path), exist_ok=True)
    plt.savefig(plot_save_path)
    plt.clf()

    # Plotting MCC
    plt.plot(
        range(num_epochs),
        [metric["mcc"] for metric in train_metrics], 
        label="Training"
    )
    plt.plot(
        range(num_epochs),
        [metric["mcc"] for metric in val_metrics], 
        label="Validation"
    )
    plt.title("MCC")
    plt.xlabel("Epochs")
    plt.ylabel("Value")
    plt.legend()

    plot_save_path = os.path.join(os.getcwd(), f"doc/graphs/{dataset_name}/{model_name}/{augmented_data_string}/{formatted_date}/mcc.png")
    os.makedirs(os.path.dirname(plot_save_path), exist_ok=True)
    plt.savefig(plot_save_path)
    plt.clf()

    # Plotting loss
    plt.plot(
        range(num_epochs),
        [metric["avg_loss"] for metric in train_metrics], 
        label="Training"
    )
    plt.plot(
        range(num_epochs),
        [metric["avg_loss"] for metric in val_metrics], 
        label="Validation"
    )
    plt.title("Average loss")
    plt.xlabel("Epochs")
    plt.ylabel("Value")
    plt.legend()

    plot_save_path = os.path.join(os.getcwd(), f"doc/graphs/{dataset_name}/{model_name}/{augmented_data_string}/{formatted_date}/avg_loss.png")
    os.makedirs(os.path.dirname(plot_save_path), exist_ok=True)
    plt.savefig(plot_save_path)
    plt.clf()