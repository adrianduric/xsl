# Imported modules
from config import params
from init_models import init_model
from data_mgmt_hyperkvasir import prepare_data_hyperkvasir
from data_mgmt_cifar import prepare_data_cifar

import os
import torch
from torchvision import models
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget, BinaryClassifierOutputTarget
from tqdm import tqdm


def get_last_non_classification_layer(model):
    """
    Returns the last non-classification layer of the given model.
    
    Args:
        model (torch.nn.Module): The pre-trained model whose last non-classification layer is to be returned.
    
    Returns:
        torch.nn.Module: The last non-classification layer of the model.
    
    Raises:
        ValueError: If the provided model architecture is unsupported.
    """
    if isinstance(model, models.ResNet):
        return model.layer4[-1]
    elif isinstance(model, models.DenseNet):
        return model.features[-1]
    elif isinstance(model, models.VisionTransformer):
        return model.encoder.layers[-1].ln_1
    elif isinstance(model, models.SwinTransformer):
        return model.features[-1][-1].norm1
    else:
        raise ValueError("Unsupported model architecture!")


# Defining reshape_transform for ViT
def reshape_transform_vit(tensor, height=14, width=14):
    result = tensor[:, 1 :  , :].reshape(tensor.size(0),
        height, width, tensor.size(2))

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result


# Defining reshape_transform for Swin and Swin_V2
def reshape_transform_swin(tensor, height=7, width=7):
    # Bring the channels to the first dimension,
    # like in CNNs.
    result = tensor.transpose(2, 3).transpose(1, 2)
    return result


def create_cam(dataset_name, model_name):
    """
    Generates Grad-CAM activation maps for all images in the specified dataset using a specified model.
    The CAMs are appended to the original images as an additional channel and saved as tensors.

    Args:
        dataset_name (str): The name of the dataset to process (e.g., "hyper-kvasir", "cifar-100-python").
        model_name (str): The name of the model to be used for generating Grad-CAM activation maps.

    Raises:
        ValueError: If the provided model architecture or dataset name is unsupported.
    """
    
    # Initializing model
    model = init_model(
        dataset_name=dataset_name,
        model_name=model_name,
        augmented_data=False,
        load_models=True,
        num_extra_channels=None
    )
    model = model.to(params["device"])
    model.eval()

    # Initializing dataset
    if dataset_name == "hyper-kvasir":
        _, complete_dataloader = prepare_data_hyperkvasir(
            seed=None,
            augmented_data=False,
            model_explanation=None,
            split=False,
            batch_size=8
        )
    elif dataset_name in ["cifar-100-python", "cifar-10-python"]:
        cifar_version = 100 if "100" in dataset_name else 10
        _, complete_dataloader = prepare_data_cifar(
            seed=None,
            augmented_data=False,
            model_explanation=None,
            split=False,
            cifar_version=cifar_version,
            batch_size=128
        )
    else:
        raise ValueError(f"Invalid dataset (received: {dataset_name})")
    

    # Getting last non-classification layer to create CAM with
    target_layers = [get_last_non_classification_layer(model=model)]

    # Choosing reshape_transform
    if isinstance(model, models.VisionTransformer):
        transform = reshape_transform_vit

    elif isinstance(model, models.SwinTransformer):
        transform = reshape_transform_swin
    else:
        transform = None

    # Initialize Grad-CAM
    cam = GradCAM(model=model, target_layers=target_layers, reshape_transform=transform)

    print("Creating CAM explanations for all images...")

    for X, y, paths, idxs in tqdm(complete_dataloader):

        # Loading batch to device
        X = X.to(params["device"])
        
        # Making predictions
        preds = torch.squeeze(model(X))

        # Determine target labels based on predictions
        if params["num_classes"] > 2:  # Multi-class case
            predicted_labels = torch.argmax(preds, dim=1)
        else:  # Binary classification case
            predicted_labels = (preds > 0.5).long()

        if params["num_classes"] > 2:
            targets = [ClassifierOutputTarget(pred_label.item()) for pred_label in predicted_labels]
        else:
            targets = [BinaryClassifierOutputTarget(pred_label.item()) for pred_label in predicted_labels]

        grayscale_cam = cam(input_tensor=X, targets=targets)

        # Append CAM to original image as additional channel
        cam_tensor = torch.tensor(grayscale_cam).to(params["device"]).unsqueeze(dim=1)
        X_with_cam = torch.cat((X, cam_tensor), dim=1)

        # Saving each image tensor to a separate file
        for i in range(X.shape[0]):
            
            image_sample = X_with_cam[i].clone()

            # Changing original file path to separate folder for augmented images
            new_path = paths[i].replace(f"/{dataset_name}", f"/augmented_images/{dataset_name}/{model_name}")

            # Changing file format to indicate PyTorch tensor, not RGB image
            new_path = new_path.replace(".jpg", ".pt")

            os.makedirs(os.path.dirname(new_path), exist_ok=True)
            torch.save(image_sample, new_path)


def create_ensemble_cams(dataset_name):
    """
    Creates Grad-CAM activation maps for all images in the specified dataset using all models.
    The chosen label to make Grad-CAM from is the average predicted label of all models.
    The CAMs are appended to the original images as an additional channel and saved as tensors.

    Args:
        dataset_name (str): The name of the dataset to process (e.g., "hyper-kvasir", "cifar-100-python").
    """

    # Initializing dataset
    if dataset_name == "hyper-kvasir":
        _, complete_dataloader = prepare_data_hyperkvasir(
            seed=None,
            augmented_data=False,
            model_explanation=None,
            split=False,
            batch_size=4
        )
    elif dataset_name in ["cifar-100-python", "cifar-10-python"]:
        cifar_version = 100 if "100" in dataset_name else 10
        _, complete_dataloader = prepare_data_cifar(
            seed=None,
            augmented_data=False,
            model_explanation=None,
            split=False,
            cifar_version=cifar_version,
            batch_size=128
        )
    else:
        raise ValueError(f"Invalid dataset (received: {dataset_name})")

    # List to store all models' predictions
    all_models_preds = []

    # Process each model individually
    for model_name in params["model_names"]:

        # Initialize a model
        model = init_model(
            dataset_name=dataset_name,
            model_name=model_name,
            augmented_data=False,
            load_models=True,
            num_extra_channels=None
        )
        model = model.to(params["device"])
        model.eval()

        # Collect predictions for this model
        model_preds = []

        with torch.no_grad():
            for X, y, paths, idxs in tqdm(complete_dataloader, desc=f"Obtaining predictied labels from {model_name}"):
                X = X.to(params["device"])
                preds = torch.squeeze(model(X))
                model_preds.append(preds)
        
        # Stack all the predictions for this model and store in list
        all_models_preds.append(torch.cat(model_preds, dim=0))

    # Compute the average predictions over all models
    stacked_preds = torch.stack(all_models_preds, dim=0)
    average_preds = torch.mean(stacked_preds, dim=0)

    # Determine target labels based on average predictions
    if average_preds.dim() == 1:
        average_preds = average_preds.unsqueeze(1)

    if params["num_classes"] > 2:  # Multi-class case
        predicted_labels = torch.argmax(average_preds, dim=1)
    else:  # Binary classification case
        predicted_labels = (average_preds > 0.5).long()

    targets = [ClassifierOutputTarget(pred_label.item()) if params["num_classes"] > 2 else BinaryClassifierOutputTarget(pred_label.item()) for pred_label in predicted_labels]

    # After targets are created, empty cache before CAM generation
    torch.cuda.empty_cache()

    # Generate CAM for each model
    for model_name in params["model_names"]:

        # Load model
        model = init_model(
            dataset_name=dataset_name,
            model_name=model_name,
            augmented_data=False,
            load_models=True,
            num_extra_channels=None
        )
        model = model.to(params["device"])
        model.eval()

        # Getting last non-classification layer to create CAM with
        target_layers = [get_last_non_classification_layer(model=model)]

        # Choosing reshape_transform
        transform = None
        if isinstance(model, models.VisionTransformer):
            transform = reshape_transform_vit
        elif isinstance(model, models.SwinTransformer):
            transform = reshape_transform_swin

        #Initialize CAM object
        cam = GradCAM(model=model, target_layers=target_layers, reshape_transform=transform)

        start_idx = 0

        for X, y, paths, idxs in tqdm(complete_dataloader, desc=f"Generating CAMs for {model_name}"):
            
            end_idx = start_idx + X.size(0)

            # Get the appropriate targets for this batch
            batch_targets = targets[start_idx:end_idx]

            # Loading batch to device
            X = X.to(params["device"])

            # Generate CAM
            grayscale_cam = cam(input_tensor=X, targets=batch_targets)

            # Append CAM to original image as additional channel
            cam_tensor = torch.tensor(grayscale_cam).to(params["device"]).unsqueeze(dim=1)
            X_with_cam = torch.cat((X, cam_tensor), dim=1)

            # Saving each image tensor to a separate file
            for i in range(X.shape[0]):
                image_sample = X_with_cam[i].clone()
                new_path = paths[i].replace(f"/{dataset_name}", f"/augmented_images/{dataset_name}/{model_name}")
                new_path = new_path.replace(".jpg", ".pt")
                os.makedirs(os.path.dirname(new_path), exist_ok=True)
                torch.save(image_sample, new_path)

            # Update index to the next batch
            start_idx = end_idx


def create_average_cam(dataset_name):
    """
    Averages over all created Grad-CAM activation maps to generate an average Grad-CAM for each image in the specified dataset and saves the results.

    Args:
        dataset_name (str): The name of the dataset to process (e.g., "hyper-kvasir", "cifar-100-python").
    """

    # Names of all models to be included
    model_names = params["model_names"]

    # Iterate over all images
    csv_file_path = os.path.join(os.getcwd(), f"res/augmented_images/{dataset_name}")

    with open(os.path.join(csv_file_path, "image-labels.csv"), "r") as file:
        
        file.readline()

        for line in file:
            
            if dataset_name == "hyper-kvasir":
                img_filename, organ, finding, classification = line.strip().split(",")

                img_filename += ".pt"
                organ = "lower-gi-tract" if organ == "Lower GI" else "upper-gi-tract"

                relative_file_path = os.path.join(
                    organ, classification, finding, img_filename
                )

            elif dataset_name == "cifar-100-python":
                set_type, coarse_label, fine_label, file_name = line.strip().split(",")

                file_name += ".pt"
                set_type = f"cifar-100_{set_type}"

                relative_file_path = os.path.join(
                    img_dir_path, set_type, coarse_label, fine_label, file_name
                )

            else:
                raise ValueError(f"Invalid dataset (received: {dataset_name})")

            # List to collect all model explanations for current image
            all_cams = []

            # Iterate over all model explanations
            for model_name in model_names:
                
                img_dir_path = os.path.join(os.getcwd(), f"res/augmented_images/{dataset_name}/{model_name}")

                path_to_img_file = os.path.join(
                    img_dir_path, relative_file_path
                )

                # Load CAM produced from current model, and store in list
                current_cam = torch.load(path_to_img_file)
                all_cams.append(current_cam)

            # Copy the RGB channels from the original image
            rgb_channels = all_cams[0][:3] # Shape [3, 224, 224]
                
            # Stack the 4th channels (CAMs) of all tensors along the channel dimension
            all_cams_stacked = torch.stack([cam[3] for cam in all_cams], dim=0)  # Shape [#cams, 224, 224]

            # Average the stacked CAMs along the first dimension
            average_cam = torch.mean(all_cams_stacked, dim=0)  # Shape [224, 224]

            # Combine the first 3 channels with the averaged 4th channel
            average_cam_image = torch.cat((rgb_channels, average_cam.unsqueeze(0)), dim=0)  # Shape [4, 224, 224]

            # Asserting that the first 3 channels are the same as in the original RGB image
            for cam_image in all_cams:
                assert torch.equal(cam_image[:3], average_cam_image[:3]), "The first 3 channels are not the same"
            
            # Saving average CAM
            save_path = os.path.join(os.getcwd(), f"res/augmented_images/{dataset_name}/average/", relative_file_path)
                                         
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(average_cam_image, save_path)


def concat_all_cams(dataset_name):
    """
    Concatenates all created Grad-CAMs for the specified dataset with the original RGB channels for each image and saves the results.

    Args:
        dataset_name (str): The name of the dataset to process (e.g., "hyper-kvasir", "cifar-100-python").
    """

    # Names of all models to be included
    model_names = params["model_names"]
    
    # Iterate over all images
    csv_file_path = os.path.join(os.getcwd(), f"res/augmented_images/{dataset_name}")

    with open(os.path.join(csv_file_path, "image-labels.csv"), "r") as file:
        
        file.readline()

        for line in file:

            if dataset_name == "hyper-kvasir":
                img_filename, organ, finding, classification = line.strip().split(",")

                img_filename += ".pt"
                organ = "lower-gi-tract" if organ == "Lower GI" else "upper-gi-tract"

                relative_file_path = os.path.join(
                    organ, classification, finding, img_filename
                )

            elif dataset_name == "cifar-100-python":
                set_type, coarse_label, fine_label, file_name = line.strip().split(",")

                file_name += ".pt"
                set_type = f"cifar-100_{set_type}"

                relative_file_path = os.path.join(
                    img_dir_path, set_type, coarse_label, fine_label, file_name
                )

            else:
                raise ValueError(f"Invalid dataset (received: {dataset_name})")

            # List to collect all model explanations for current image
            all_cams = []

            # Iterate over all model explanations
            for model_name in model_names:
                
                img_dir_path = os.path.join(os.getcwd(), f"res/augmented_images/{dataset_name}/{model_name}")

                path_to_img_file = os.path.join(
                    img_dir_path, relative_file_path
                )

                # Load CAM produced from current model, and store in list
                current_cam = torch.load(path_to_img_file)
                all_cams.append(current_cam)

            # Copy the RGB channels from the original image
            rgb_channels = all_cams[0][:3]  # Shape [3, 224, 224]

            # Stack the 4th channels (CAMs) of all tensors along the channel dimension
            concatenated_cams = torch.stack([cam[3] for cam in all_cams], dim=0)  # Shape [n, 224, 224]

            # Combine the original RGB channels with the concatenated CAMs
            concatenated_cam_image = torch.cat((rgb_channels, concatenated_cams), dim=0)  # Shape [3 + n, 224, 224]

            # Asserting that the first 3 channels are the same as in the original RGB image
            for cam_image in all_cams:
                assert torch.equal(cam_image[:3], concatenated_cam_image[:3]), "The first 3 channels are not the same"

            # Saving concatenated CAM
            save_path = os.path.join(os.getcwd(), f"res/augmented_images/{dataset_name}/concatenated/", relative_file_path)
                                         
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(concatenated_cam_image, save_path)


# Test program
if __name__ == "__main__":
    create_cam("hyper-kvasir", "vit_b_16")

