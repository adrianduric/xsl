# Imported modules
from config import params
from data_mgmt_hyperkvasir import prepare_data_hyperkvasir
from data_mgmt_cifar import prepare_data_cifar
from init_models import init_model
from model_training import test_one_epoch, train_model
from create_explanation import create_cam, create_ensemble_cams, create_average_cam, concat_all_cams
from create_graphs import create_graphs

import os
import datetime
import random
import torch.nn as nn
import numpy as np
import scipy.stats as stats


def compute_confidence_interval(data, confidence=0.95):
    """
    Compute the confidence interval for given data.
    
    Args:
        data (list): List of values for which the confidence interval is computed.
        confidence (float): The confidence level for the interval.
    
    Returns:
        (float, float, float): The mean, lower, and upper bounds of the confidence interval.
    """
    mean = np.mean(data)
    se = stats.sem(data)  # Standard error of the mean
    margin_of_error = se * stats.t.ppf((1 + confidence) / 2.0, len(data) - 1)
    return mean, mean - margin_of_error, mean + margin_of_error


def test_model_single_run(model, test_dataloader):
    """
    Perform a single test run with the specified model and return metrics, total loss, predictions, and targets.

    Args:
        model (torch.nn.Module): The model to be tested.
        test_dataloader (torch.utils.data.DataLoader): DataLoader for the test set.

    Returns:
        tuple: Contains test_metrics (dict), total_loss (float), all_predictions (Tensor), and all_targets (Tensor).
    """
    
    model = model.to(params["device"])

    loss_fn = nn.CrossEntropyLoss() if params["num_classes"] > 2 else nn.BCEWithLogitsLoss()

    # Test model on test set, and saving metrics
    test_metrics, total_loss, all_predictions, all_targets = test_one_epoch(test_dataloader, model, loss_fn)

    return test_metrics, total_loss, all_predictions, all_targets


def test_model(dataset_name, model_name, augmented_data, num_runs, explanation_type=None, peer_model_name=None):
    """
    Run the test for a specific dataset and model multiple times, compute average metrics and their confidence intervals.

    Args:
        dataset_name (str): The name of the dataset to test with (e.g., "hyper-kvasir", "cifar-100-python").
        model_name (str): The name of the model to be tested.
        augmented_data (bool): Whether the data is augmented or not.
        num_runs (int): The number of times to run the test.
        explanation_type (str): The type of explanation to be used. Default is None.
        peer_model_name (str): The name of the model from which to obtain explanations if peer explanation is chosen. Default is None.
    """

    # If testing XAug data, set explanation from correct model(s) to be used
    if not augmented_data:
        model_explanation = None
    elif explanation_type == "self":
        model_explanation = model_name
    elif explanation_type == "peer":
        assert peer_model_name is not None
        model_explanation = peer_model_name
    elif explanation_type == "average":
        model_explanation = "average"
    elif explanation_type == "multi":
        model_explanation = "concatenated"
    else:
        raise ValueError(f"Invalid explanation type (received: {explanation_type}).")

    # To store metrics of each run
    all_run_metrics = []

    print(f"Evaluating model: {model_name}")

    for run in range(num_runs):
        print(f"Run {run + 1}/{num_runs}")

        # Sample a random seed for this run
        run_seed = random.randint(0, 10000)
        print(f"Using seed: {run_seed} for this run")

        # Training teacher model if testing models on XAug data 
        if augmented_data:

            # Initializing non-augmented dataset for training teacher model
            if dataset_name == "hyper-kvasir":
                train_dataset, val_dataset, test_dataset, train_dataloader, val_dataloader, test_dataloader = prepare_data_hyperkvasir(seed=run_seed, augmented_data=False, model_explanation=None, split=True)
            elif dataset_name == "cifar-100-python":
                train_dataset, val_dataset, test_dataset, train_dataloader, val_dataloader, test_dataloader = prepare_data_cifar(seed=run_seed, augmented_data=False, model_explanation=None, split=True, cifar_version=100)
            elif dataset_name == "cifar-10-python":
                train_dataset, val_dataset, test_dataset, train_dataloader, val_dataloader, test_dataloader = prepare_data_cifar(seed=run_seed, augmented_data=False, model_explanation=None, split=True, cifar_version=10)
            else:
                raise ValueError(f"Invalid dataset (received: {dataset_name})")

            # Use one teacher model if self or peer explanation
            if explanation_type in {"self", "peer"}:

                # Initializing teacher model for this run
                teacher_model = init_model(
                    dataset_name=dataset_name,
                    model_name=peer_model_name if explanation_type == "peer" else model_name,
                    augmented_data=False,
                    load_models=False,
                    num_extra_channels=None
                )
                teacher_model = teacher_model.to(params["device"])

                # Performing training of teacher model
                train_metrics, val_metrics = train_model(
                    seed=None,
                    dataset_name=dataset_name,
                    model=teacher_model,
                    model_name=peer_model_name if explanation_type == "peer" else model_name,
                    train_dataloader=train_dataloader,
                    val_dataloader=val_dataloader,
                    augmented_data=False,
                    save_model=True
                )

                # Create graphs of metrics
                create_graphs(
                    dataset_name=dataset_name,
                    model_name=peer_model_name if explanation_type == "peer" else model_name,
                    augmented_data=False,
                    train_metrics=train_metrics,
                    val_metrics=val_metrics
                )

                # Generating explanations from teacher model
                create_cam(dataset_name=dataset_name, model_name=peer_model_name if explanation_type == "peer" else model_name)

            # Use one of each model as teacher if average or multi-explanation
            elif explanation_type in {"average", "multi"}:

                for ensemble_model_name in params["model_names"]:
                
                    # Initializing teacher model for this run
                    ensemble_teacher_model = init_model(
                        dataset_name=dataset_name,
                        model_name=ensemble_model_name,
                        augmented_data=False,
                        load_models=False,
                        num_extra_channels=None
                    )
                    ensemble_teacher_model = ensemble_teacher_model.to(params["device"])

                    # Performing training of teacher model
                    train_metrics, val_metrics = train_model(
                        seed=None,
                        dataset_name=dataset_name,
                        model=ensemble_teacher_model,
                        model_name=ensemble_model_name,
                        train_dataloader=train_dataloader,
                        val_dataloader=val_dataloader,
                        augmented_data=False,
                        save_model=True
                    )

                    # Create graphs of metrics
                    create_graphs(
                        dataset_name=dataset_name,
                        model_name=ensemble_model_name,
                        augmented_data=False,
                        train_metrics=train_metrics,
                        val_metrics=val_metrics
                    )

                # Generating explanations from teacher models with ensemble prediction label
                create_ensemble_cams(dataset_name=dataset_name)

                if explanation_type == "average":
                    create_average_cam(dataset_name=dataset_name)
                elif explanation_type == "multi":
                    concat_all_cams(dataset_name=dataset_name)
                else:
                    raise ValueError(f"Invalid explanation type (received: {explanation_type}).")
                
            else:
                raise ValueError(f"Invalid explanation type (received: {explanation_type}).")


        # Initializing model to be trained for this run (student model if using XAug data)
        student_model = init_model(
            dataset_name=dataset_name,
            model_name=model_name,
            augmented_data=augmented_data,
            load_models=False,
            num_extra_channels=len(params["model_names"]) if explanation_type == "multi" else 1
        )
        student_model = student_model.to(params["device"])
        
        # Initializing dataset for this run
        if dataset_name == "hyper-kvasir":
            train_dataset, val_dataset, test_dataset, train_dataloader, val_dataloader, test_dataloader = prepare_data_hyperkvasir(seed=run_seed, augmented_data=augmented_data, model_explanation=model_explanation, split=True)
        elif dataset_name == "cifar-100-python":
            train_dataset, val_dataset, test_dataset, train_dataloader, val_dataloader, test_dataloader = prepare_data_cifar(seed=run_seed, augmented_data=augmented_data, model_explanation=model_explanation, split=True, cifar_version=100)
        elif dataset_name == "cifar-10-python":
            train_dataset, val_dataset, test_dataset, train_dataloader, val_dataloader, test_dataloader = prepare_data_cifar(seed=run_seed, augmented_data=augmented_data, model_explanation=model_explanation, split=True, cifar_version=10)
        else:
            raise ValueError(f"Invalid dataset (received: {dataset_name})")

        # Performing model training for this run
        train_metrics, val_metrics = train_model(
            seed=None,
            dataset_name=dataset_name,
            model=student_model,
            model_name=model_name,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            augmented_data=augmented_data,
            save_model=True
        )

        # Create graphs of metrics
        create_graphs(
            dataset_name=dataset_name,
            model_name=model_name,
            augmented_data=augmented_data,
            train_metrics=train_metrics,
            val_metrics=val_metrics
        )

        # Initializing model again before test so that the saved weights are used
        student_model = init_model(
            dataset_name=dataset_name,
            model_name=model_name,
            augmented_data=augmented_data,
            load_models=True,
            num_extra_channels=len(params["model_names"]) if explanation_type == "multi" else 1
        )
        student_model = student_model.to(params["device"])

        # Perform model testing for this run
        test_metrics, total_loss, all_predictions, all_targets = test_model_single_run(
            model=student_model,
            test_dataloader=test_dataloader
        )

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
    file_save_path = os.path.join(os.getcwd(), f"doc/result_metrics/{dataset_name}/{model_name}/{explanation_type if augmented_data else 'non-augmented'}/{formatted_date}/test_metrics_{model_name}.txt")
    os.makedirs(os.path.dirname(file_save_path), exist_ok=True)

    with open(file_save_path, "w") as text_file:
        for metric, values in confidence_intervals.items():
            mean, lower_bound, upper_bound = confidence_intervals[metric]
            text_file.write(f"{metric}: mean={mean}, 95% CI=({lower_bound}, {upper_bound})\n")