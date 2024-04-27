import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms


# Define the data transformations, this way all the images have the same size.
# This allows us to run the images through the same neural network.
# Otherwise, the input layer would need to change.
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Resize the images to a fixed size of 256x256.
    transforms.ToTensor(),  # Convert the images to PyTorch tensors.
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize the image tensors.
])