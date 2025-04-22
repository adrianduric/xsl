from init_models import init_model

from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import torch
from torchvision import models, transforms
import numpy as np
import matplotlib.pyplot as plt


model = init_model("hyperkvasir", "resnet152", augmented_data=False, load_models=True, num_extra_channels=0)
model.eval()

target_layers = [model.layer4[-1]] #ResNet
#target_layers = [model.features[-1]] #DenseNet
#target_layers = [model.encoder.layers[-1].ln_1] #ViT
#target_layers = [model.features[-1][-1].norm1] #Swin

input_image = Image.open("/home/adrian/xai4iml/res/hyper-kvasir/lower-gi-tract/pathological-findings/polyps/8b5edcf8-c2d3-4ee8-a587-83940161dd0b.jpg").convert('RGB')
resized_img = input_image.resize((224, 224))
rgb_img = np.array(resized_img) / 255.0  # Normalize to 0-1 range

# Define the transformation for the input tensor
transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

input_tensor = transform(input_image).unsqueeze(0).to("cuda")

# We have to specify the target we want to generate the CAM for.
targets = [ClassifierOutputTarget(4)]

""" def reshape_transform(tensor, height=14, width=14):
    result = tensor[:, 1 :  , :].reshape(tensor.size(0),
        height, width, tensor.size(2))

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result """

def reshape_transform(tensor, height=7, width=7):
    # Bring the channels to the first dimension,
    # like in CNNs.
    result = tensor.transpose(2, 3).transpose(1, 2)
    return result

reshape_transform = reshape_transform

# Construct the CAM object once, and then re-use it on many images.
with GradCAM(model=model, target_layers=target_layers, reshape_transform=None) as cam:
  # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
  grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
  # In this example grayscale_cam has only one image in the batch:
  grayscale_cam = grayscale_cam[0, :]
  visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
  
# Visualize the heatmap on the image
#plt.imshow(torch.squeeze(input_tensor, 0).transpose(2, 0).cpu())
#plt.imshow(resized_img)
plt.imshow(visualization)
plt.axis('off')
plt.show()