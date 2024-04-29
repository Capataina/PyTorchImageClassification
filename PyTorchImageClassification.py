import torch  # Needed for the neural networks
import scipy  # Needed to process the downloads from torchvision
import torch.nn as nn  # Needed for the neural networks
from torch import save, load  # Needed to save/load the pt file.
import torch.optim as optim  # Needed to optimise the neural network
from torchvision import datasets, transforms  # Needed to download and transform/process the images

# Define the data transformations, this way all the images have the same size.
# This allows us to run the images through the same neural network.
# Otherwise, the input layer would need to change.
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(25),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
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
# Best loss was 0.04 and best accuracy was 0.2
# Done with random horizontal flips and rotations over 50 epochs.
# class FlowerClassifier(nn.Module):
#    def __init__(self, num_classes):
#        super(FlowerClassifier, self).__init__()
#        self.fc1 = nn.Linear(256 * 256 * 3, 512)  # Hidden layer 1 with 256*256*3 inputs.
#        # We have that many inputs as our images are 256x256 with RGB color scheme.
#        # 512 nodes in the hidden layer, the number is arbitrary, not calculated.
#        self.relu1 = nn.ReLU()  # ReLU activation function. Allows for more complex non-linear hidden layer functions.
#        self.fc2 = nn.Linear(512, 256)  # Quick change added for possible increased accuracy. Now there is 1 more
#        # fully connected layer.
#        self.relu2 = nn.ReLU()  # Another ReLU activation layer after the second hidden layer.
#        self.fc3 = nn.Linear(256, num_classes)
#        # 512 input nodes, the same as the first hidden layer.
#        # The outputs are the different types of flowers.
#
#    def forward(self, x):
#        x = x.view(x.size(0), -1)  # Reshape the input tensor to a 2D tensor, automatically calculate channels.
#        x = self.fc1(x)  # Pass through the first calculation/hidden layer.
#        x = self.relu1(x)  # Apply rectified linear unit activation.
#        x = self.fc2(x)  # Pass through the second calculation/hidden layer.
#        x = self.relu2(x)  # Added second ReLU layer to possibly increase accuracy
#        x = self.fc3(x)  # Last fully connected layer.
#        return x


# Convolutional neural network to better process images. CNN's are better at image processing as they also learn the
# relations of pixels with the nearby pixels rather than simply matching pixels to each other. This gives the neural
# network a type of "spatial awareness" that allows it to recognise patterns.
# Best run parameters:
# Epochs: 50
# Random Horizontal Flips
# Random Rotations: 25
# 256x256 Images
# Test Accuracy: 0.21
# Learning Rate: 0.0005
# Final Loss: 0.9
# Final Notes: The network kept going back and forth 1.3 and 0.6 loss,
# but if we take the average of each 10 epochs, the loss was decreasing,
# meaning the model needs more epochs to be better fitted, potentially.
class FlowerClassifier(nn.Module):
    def __init__(self, num_classes):
        super(FlowerClassifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(32 * 64 * 64, 128)
        self.relu3 = nn.ReLU()

        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu3(x)

        x = self.fc2(x)
        return x


# Initialize the model, loss function, and optimizer.
# The number of classes is 102, as there are 102 different types of flowers, hence the name Flowers-102.
model = FlowerClassifier(num_classes=102)

# We will be using Cross Entropy Loss. There are many reasons for this, for example cross entropy loss encourages the
# model to assign high probabilities to the correct class and low probabilities to the incorrect classes. It is also
# very common in multi-class classification neural networks. It focuses on the overall correctness of the model
# rather than focusing on small details which is important for a dataset this size and most importantly,
# it works very well with the optimising algorithm we will be using, stochastic gradient descent.
criterion = nn.CrossEntropyLoss()

# The optimiser we will be using is stochastic gradient descent. One of the main reasons why I used stochastic
# gradient descent is because we've done gradient descent in our practical. Also, SGD processes small batches at a time,
# making it computationally efficient. It reaches conclusions relatively faster than other optimising algorithms and
# the randomness allows for easier generation of more complex algorithms rather than a linear convergence.
optimizer = optim.SGD(model.parameters(), lr=0.0005, momentum=0.9)

# Training loop.
for epoch in range(50):
    for images, labels in train_loader:
        optimizer.zero_grad()  # Reset the gradients. This allows for no bias at the start.
        outputs = model(images)  # Pass the images to our model.
        loss = criterion(outputs, labels)  # Calculate the loss between the predictions and the true labels.
        loss.backward()  # Apply backpropagation to compute the gradient of the loss.
        optimizer.step()  # We update the model's weight based on the gradient from our loss algorithm.

    # For debug
    print('Epoch {}, Loss: {}'.format(epoch + 1, loss.item()))

    # TODO - Add a validation loop if needed!

# Save the trained model
# torch.save(model.state_dict(), 'flower_classifier.pt')

# Evaluation on the test set
model.eval()  # Set the model to evaluation mode
accuracy = 0.0
with torch.no_grad():  # Disable gradient computation
    for images, labels in test_loader:
        outputs = model(images)  # Forward pass
        _, predicted = torch.max(outputs.data, 1)  # Get the predicted classes
        accuracy += (predicted == labels).sum().item()  # Calculate the accuracy

accuracy /= len(test_dataset)
print(f'Test Accuracy: {accuracy:.2f}')
