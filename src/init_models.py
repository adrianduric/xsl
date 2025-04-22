# Imported modules
from config import params

import os
import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import (
    resnet152, ResNet152_Weights,
    densenet161, DenseNet161_Weights,
    vit_b_16, ViT_B_16_Weights,
    swin_v2_t, Swin_V2_T_Weights
)

def modify_model_for_n_channels(model, num_extra_channels):
    """
    Modify the input model to accept n-channel input images.

    Args:
        model (torch.nn.Module): The model to modify.
        num_extra_channels (int): The number of additional channels to add.

    Returns:
        torch.nn.Module: The modified model with a new first convolutional layer that can accept n-channel input images.
    """
    num_channels = 3 + num_extra_channels

    def get_new_weights(original_weights):
        mean_weights = original_weights.mean(dim=1, keepdim=True)
        extended_weights = [original_weights] + [mean_weights] * num_extra_channels
        return torch.cat(extended_weights, dim=1)

    if isinstance(model, models.ResNet):
        num_filters = model.conv1.out_channels
        kernel_size = model.conv1.kernel_size
        stride = model.conv1.stride
        padding = model.conv1.padding

        new_conv1 = nn.Conv2d(num_channels, num_filters, kernel_size=kernel_size, stride=stride, padding=padding)
        original_weights = model.conv1.weight.data
        new_conv1.weight.data = get_new_weights(original_weights)
        model.conv1 = new_conv1

    elif isinstance(model, models.DenseNet):
        num_filters = model.features.conv0.out_channels
        kernel_size = model.features.conv0.kernel_size
        stride = model.features.conv0.stride
        padding = model.features.conv0.padding

        new_conv0 = nn.Conv2d(num_channels, num_filters, kernel_size=kernel_size, stride=stride, padding=padding)
        original_weights = model.features.conv0.weight.data
        new_conv0.weight.data = get_new_weights(original_weights)
        model.features.conv0 = new_conv0

    elif isinstance(model, models.VisionTransformer):
        num_filters = model.conv_proj.out_channels
        kernel_size = model.conv_proj.kernel_size
        stride = model.conv_proj.stride
        padding = model.conv_proj.padding

        new_conv_proj = nn.Conv2d(num_channels, num_filters, kernel_size=kernel_size, stride=stride, padding=padding)
        original_weights = model.conv_proj.weight.data
        new_conv_proj.weight.data = get_new_weights(original_weights)
        model.conv_proj = new_conv_proj

    elif isinstance(model, models.SwinTransformer):
        num_filters = model.features[0][0].out_channels
        kernel_size = model.features[0][0].kernel_size
        stride = model.features[0][0].stride
        padding = model.features[0][0].padding

        new_conv_layer = nn.Conv2d(num_channels, num_filters, kernel_size=kernel_size, stride=stride, padding=padding)
        original_weights = model.features[0][0].weight.data
        new_conv_layer.weight.data = get_new_weights(original_weights)
        model.features[0][0] = new_conv_layer


def adapt_last_layer(model, num_classes):
    """
    Adapts the last layer of the given model to match the specified number of output classes.

    Args:
        model (torch.nn.Module): The pre-trained model whose last layer is to be adapted.
        num_classes (int): The number of output classes (presumed to be 2 or larger).
    
    Raises:
        ValueError: If the provided model architecture is unsupported.
    """
    if isinstance(model, models.ResNet):
        in_features = model.fc.in_features
        model.fc = nn.Linear(
            in_features=in_features,
            out_features=num_classes if num_classes > 2 else 1,
            bias=True
        )
    elif isinstance(model, models.DenseNet):
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(
            in_features=in_features,
            out_features=num_classes if num_classes > 2 else 1,
            bias=True
        )
    elif isinstance(model, models.VisionTransformer):
        in_features = model.heads.head.in_features
        model.heads.head = nn.Linear(
            in_features=in_features,
            out_features=num_classes if num_classes > 2 else 1,
            bias=True
        )
    elif isinstance(model, models.SwinTransformer):
        in_features = model.head.in_features
        model.head = nn.Linear(
            in_features=in_features,
            out_features=num_classes if num_classes > 2 else 1,
            bias=True
        )
    else:
        raise ValueError("Unsupported model architecture!")


def init_model(dataset_name, model_name, augmented_data, load_models, num_extra_channels):
    """
    Initializes the specified model for the specified dataset. The model is initialized either with pretrained weights, 
    or loaded from a saved state dictionary, as specified by load_models.

    Args:
        dataset_name (str): The name of the dataset to use the model with.
        model_name (str): The name of the model to initialize.
        augmented_data (bool): Indicates whether to load the model with additional channels for augmented data.
        load_models (bool): Flag indicating whether to load existing model parameters from a saved state dictionary.
        num_extra_channels (int): The number of additional channels in the input if using augmented data.

    Returns:
        torch.nn.Module: The initialized model.

    Raises:
        ValueError: If the specified model name is not recognized.
    """
    
    # Initializing models
    model_dict = {
        "resnet152": lambda: resnet152(weights=ResNet152_Weights.DEFAULT),
        "densenet161": lambda: densenet161(weights=DenseNet161_Weights.DEFAULT),
        "vit_b_16": lambda: vit_b_16(weights=ViT_B_16_Weights.DEFAULT),
        "swin_v2_t": lambda: swin_v2_t(weights=Swin_V2_T_Weights.DEFAULT)
    }

    if model_name in model_dict:
        model = model_dict[model_name]()
    else:
        raise ValueError(f"Model {model_name} not recognized. Available options are: {list(model_dict.keys())}")
    
    # If specified, modify model to accept 4 channel inputs for augmented data
    if augmented_data:
        modify_model_for_n_channels(model=model, num_extra_channels=num_extra_channels)
    
    # Swapping last layer for classification of all labels in HyperKvasir
    adapt_last_layer(model=model, num_classes=params["num_classes"])
    
    # Loading model state dicts if specified
    if load_models:
        
        base_path = os.path.join(os.getcwd(), f"res/models/{dataset_name}/{'augmented' if augmented_data else 'non-augmented'}")
        model_path = os.path.join(base_path, model_name)
        model.load_state_dict(torch.load(model_path, weights_only=True))

    return model