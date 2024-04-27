import torch  # Needed for the neural networks
import scipy  # Needed to process the downloads from torchvision
import torch.nn as nn  # Needed for the neural networks
import torch.optim as optim  # Needed to optimise the neural network
from torchvision import datasets, transforms  # Needed to download and transform/process the images

# Define the data transformations, this way all the images have the same size.
# This allows us to run the images through the same neural network.
# Otherwise, the input layer would need to change.
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Resize the images to a fixed size of 256x256.
    transforms.ToTensor(),  # Convert the images to PyTorch tensors.
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize the image tensors.
])

# Load the Flowers-102 dataset from torchvision.
# Torchvision already has the Oxford Flowers-102 dataset in it, so all we have to do is download it.
train_dataset = datasets.Flowers102(root='./data', split='train', transform=transform, download=True)
val_dataset = datasets.Flowers102(root='./data', split='val', transform=transform, download=True)
test_dataset = datasets.Flowers102(root='./data', split='test', transform=transform, download=True)

# Create data loaders. These allow you to load the data for the neural network. The batch size allows you to train
# the model in batches rather than per image, allowing better and more complex learning. We shuffle the training set
# so that in each epoch, the neural network is opposed to different orders of images, strengthening the hidden layers.
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)


# Define the neural network architecture
class FlowerClassifier(nn.Module):
    def __init__(self, num_classes):
        super(FlowerClassifier, self).__init__()
        self.fc1 = nn.Linear(256 * 256 * 3, 512)  # Hidden layer 1 with 256*256*3 inputs.
        # We have that many inputs as our images are 256x256 with RGB color scheme.
        # 512 nodes in the hidden layer, the number is arbitrary, not calculated.
        self.relu = nn.ReLU()  # ReLU activation function. Allows for more complex non-linear hidden layer functions.
        self.fc2 = nn.Linear(512, num_classes)  # 512 input nodes, the same as the first hidden layer.
        # The outputs are the different types of flowers.

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Reshape the input tensor to a 2D tensor, automatically calculate channels.
        x = self.fc1(x)  # Pass through the first calculation/hidden layer.
        x = self.relu(x)  # Apply ReLU activation
        x = self.fc2(x)  # Pass through the second fully connected layer
        return x
