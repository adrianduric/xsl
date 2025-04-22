# Imported modules
from config import params
from init_models import init_model
from data_mgmt_hyperkvasir import prepare_data_hyperkvasir
from data_mgmt_cifar import prepare_data_cifar
from model_training import train_model
from model_testing import test_model
from create_explanation import create_cam, create_average_cam, concat_all_cams
from create_graphs import create_graphs


def run(
        dataset_name,
        seed=params["seed"],
        model_name=None,
        model_explanation=None,
        augmented_data=False,
        load_models=False,
        train_models=False,
        create_cams=False,
        average_cams=False,
        concat_cams=False,
        test_models=False,
        num_runs=5,
        explanation_type=None
    ):
    """
    Main script to execute various actions based on specified parameters, including training, testing,
    creating Grad-CAM activation maps, and more.

    Args:
        dataset_name (str): The name of the dataset to be used.
        seed (int): Random seed for reproducibility. Default is the seed set in config.py.
        model_name (str, optional): The name of the model to initialize or load. Default is None.
        model_explanation (str, optional): The name of the model providing explanations, e.g., 'densenet161'. Default is None.
        augmented_data (bool): Indicates whether to use augmented data. Default is False.
        load_models (bool): Flag indicating whether to load existing model parameters from a saved state dictionary. Default is False.
        train_models (bool): Flag indicating whether to train the models. Default is False.
        create_cams (bool): Flag indicating whether to create Grad-CAMs for the images. Default is False.
        average_cams (bool): Flag indicating whether to average multiple Grad-CAMs to create an average Grad-CAM. Default is False.
        concat_cams (bool): Flag indicating whether to concatenate all Grad-CAMs into one tensor. Default is False.
        test_models (bool): Flag indicating whether to test a single model. Default is False.
        num_runs (int): Number of runs from which to measure performance metrics and create confidence intervals if model testing is specified. Default is 10.
        explanation_type (str): The type of explanation to be used if model testing is specified. Default is None.
    """


    # Perform training and validation (multiple epochs) if specified
    if train_models:

        # Initializing model
        model = init_model(
            dataset_name=dataset_name,
            model_name=model_name,
            augmented_data=augmented_data,
            load_models=load_models,
            num_extra_channels=5 if model_explanation == "concatenated" else 1
        )
        model = model.to(params["device"])
        
        # Initializing dataset
        if dataset_name == "hyper-kvasir":
            _, _, _, train_dataloader, val_dataloader, _ = prepare_data_hyperkvasir(
                seed=seed,
                augmented_data=augmented_data,
                model_explanation=model_explanation,
                split=True
            )
        elif dataset_name == "cifar-100-python":
            _, _, _, train_dataloader, val_dataloader, _ = prepare_data_cifar(
                seed=seed,
                augmented_data=augmented_data,
                model_explanation=model_explanation,
                split=True,
                cifar_version=100
            )
        elif dataset_name == "cifar-10-python":
            _, _, _, train_dataloader, val_dataloader, _ = prepare_data_cifar(
                seed=seed,
                augmented_data=augmented_data,
                model_explanation=model_explanation,
                split=True,
                cifar_version=10
            )
        else:
            raise ValueError(f"Invalid dataset (received: {dataset_name})")

        # Perform model training
        train_metrics, val_metrics = train_model(
            seed=seed,
            dataset_name=dataset_name,
            model=model,
            model_name=model_name,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            augmented_data=augmented_data,
            save_model=True
        )
        
        # Create graphs of training and validation metrics
        create_graphs(
            dataset_name=dataset_name,
            model_name=model_name,
            augmented_data=augmented_data,
            train_metrics=train_metrics,
            val_metrics=val_metrics
        )

    # If specified, create CAMs from specified model
    if create_cams:
        create_cam(dataset_name=dataset_name, model_name=model_name)

    # Average over all created CAMs to create average CAM if specified
    if average_cams:
        create_average_cam(dataset_name=dataset_name)

    # Concatenate all created CAMs into one tensor if specified
    if concat_cams:
        concat_all_cams(dataset_name=dataset_name)

    # Perform model testing if specified
    if test_models:

        test_model(
            dataset_name=dataset_name,
            model_name=model_name,
            augmented_data=augmented_data,
            num_runs=num_runs,
            explanation_type=explanation_type,
            peer_model_name=model_explanation
        )

if __name__ == "__main__":

    #run(dataset_name="hyper-kvasir", model_name="resnet152", test_models=True, augmented_data=True, explanation_type="average")
    #run(dataset_name="hyper-kvasir", model_name="densenet161", test_models=True, augmented_data=True, explanation_type="average")
    #run(dataset_name="hyper-kvasir", model_name="vit_b_16", test_models=True, augmented_data=True, explanation_type="average")
    #run(dataset_name="hyper-kvasir", model_name="swin_v2_t", test_models=True, augmented_data=True, explanation_type="average")

    pass
