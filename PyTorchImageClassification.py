"""
Epoch 1, Training Loss: 0.14396, Validation Loss: 0.13438, Validation Accuracy: 0.07
Epoch 2, Training Loss: 0.13035, Validation Loss: 0.12272, Validation Accuracy: 0.08
Epoch 3, Training Loss: 0.12196, Validation Loss: 0.11611, Validation Accuracy: 0.12
Epoch 4, Training Loss: 0.11423, Validation Loss: 0.11049, Validation Accuracy: 0.15
Epoch 5, Training Loss: 0.11095, Validation Loss: 0.10915, Validation Accuracy: 0.14
Epoch 6, Training Loss: 0.10795, Validation Loss: 0.10808, Validation Accuracy: 0.15
Epoch 7, Training Loss: 0.10415, Validation Loss: 0.10715, Validation Accuracy: 0.16
Epoch 8, Training Loss: 0.10225, Validation Loss: 0.10311, Validation Accuracy: 0.19
Epoch 9, Training Loss: 0.10037, Validation Loss: 0.10423, Validation Accuracy: 0.18
Epoch 10, Training Loss: 0.09769, Validation Loss: 0.10014, Validation Accuracy: 0.20
Epoch 11, Training Loss: 0.09915, Validation Loss: 0.10161, Validation Accuracy: 0.19
Epoch 12, Training Loss: 0.09277, Validation Loss: 0.10156, Validation Accuracy: 0.22
Epoch 13, Training Loss: 0.09385, Validation Loss: 0.09839, Validation Accuracy: 0.21
Epoch 14, Training Loss: 0.09142, Validation Loss: 0.09640, Validation Accuracy: 0.23
Epoch 15, Training Loss: 0.09017, Validation Loss: 0.09519, Validation Accuracy: 0.26
Epoch 16, Training Loss: 0.08757, Validation Loss: 0.09430, Validation Accuracy: 0.25
Epoch 17, Training Loss: 0.08693, Validation Loss: 0.09186, Validation Accuracy: 0.28
Epoch 18, Training Loss: 0.08555, Validation Loss: 0.09538, Validation Accuracy: 0.25
Epoch 19, Training Loss: 0.08251, Validation Loss: 0.09153, Validation Accuracy: 0.27
Epoch 20, Training Loss: 0.08181, Validation Loss: 0.09525, Validation Accuracy: 0.25
Epoch 21, Training Loss: 0.08124, Validation Loss: 0.09210, Validation Accuracy: 0.27
Epoch 22, Training Loss: 0.07836, Validation Loss: 0.09053, Validation Accuracy: 0.29
Epoch 23, Training Loss: 0.07857, Validation Loss: 0.08856, Validation Accuracy: 0.31
Epoch 24, Training Loss: 0.07696, Validation Loss: 0.08966, Validation Accuracy: 0.30
Epoch 25, Training Loss: 0.07551, Validation Loss: 0.08829, Validation Accuracy: 0.31
Epoch 26, Training Loss: 0.07355, Validation Loss: 0.09020, Validation Accuracy: 0.30
Epoch 27, Training Loss: 0.07234, Validation Loss: 0.08902, Validation Accuracy: 0.30
Epoch 28, Training Loss: 0.07185, Validation Loss: 0.09177, Validation Accuracy: 0.29
Epoch 29, Training Loss: 0.06951, Validation Loss: 0.08948, Validation Accuracy: 0.30
Epoch 30, Training Loss: 0.06931, Validation Loss: 0.08793, Validation Accuracy: 0.32
Epoch 31, Training Loss: 0.06681, Validation Loss: 0.09368, Validation Accuracy: 0.28
Epoch 32, Training Loss: 0.06809, Validation Loss: 0.09705, Validation Accuracy: 0.28
Epoch 33, Training Loss: 0.06731, Validation Loss: 0.09019, Validation Accuracy: 0.31
Epoch 34, Training Loss: 0.06352, Validation Loss: 0.08842, Validation Accuracy: 0.33
Epoch 35, Training Loss: 0.06433, Validation Loss: 0.08754, Validation Accuracy: 0.33
Epoch 36, Training Loss: 0.06572, Validation Loss: 0.09378, Validation Accuracy: 0.30
Epoch 37, Training Loss: 0.06394, Validation Loss: 0.09263, Validation Accuracy: 0.31
Epoch 38, Training Loss: 0.06359, Validation Loss: 0.08908, Validation Accuracy: 0.33
Epoch 39, Training Loss: 0.06266, Validation Loss: 0.09367, Validation Accuracy: 0.30
Epoch 40, Training Loss: 0.06110, Validation Loss: 0.08795, Validation Accuracy: 0.33
Epoch 41, Training Loss: 0.06093, Validation Loss: 0.09345, Validation Accuracy: 0.33
Epoch 42, Training Loss: 0.05649, Validation Loss: 0.08687, Validation Accuracy: 0.34
Epoch 43, Training Loss: 0.05459, Validation Loss: 0.08877, Validation Accuracy: 0.35
Epoch 44, Training Loss: 0.05283, Validation Loss: 0.08791, Validation Accuracy: 0.35
Epoch 45, Training Loss: 0.05307, Validation Loss: 0.08765, Validation Accuracy: 0.36
Epoch 46, Training Loss: 0.05180, Validation Loss: 0.08763, Validation Accuracy: 0.36
Epoch 47, Training Loss: 0.04997, Validation Loss: 0.08913, Validation Accuracy: 0.37
Epoch 48, Training Loss: 0.05165, Validation Loss: 0.08864, Validation Accuracy: 0.37
Epoch 49, Training Loss: 0.04994, Validation Loss: 0.08676, Validation Accuracy: 0.37
Epoch 50, Training Loss: 0.05049, Validation Loss: 0.08695, Validation Accuracy: 0.37
Test Accuracy: 0.33
"""

import torch  # Needed for the neural networks
import scipy  # Needed to process the downloads from torchvision
import torch.nn as nn  # Needed for the neural networks
from torch import save, load  # Needed to save/load the pt file.
import torch.optim as optim  # Needed to optimise the neural network
from torch.optim.lr_scheduler import StepLR
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import datasets, transforms  # Needed to download and transform/process the images

# Define the data transformations, this way all the images have the same size.
# This allows us to run the images through the same neural network.
# Otherwise, the input layer would need to change.
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(256, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
    transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05), shear=5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load the Flowers-102 dataset from torchvision.
# Torchvision already has the Oxford Flowers-102 dataset in it, so all we have to do is download it.
train_dataset = datasets.Flowers102(root='./data', split='train', transform=train_transform, download=True)
val_dataset = datasets.Flowers102(root='./data', split='val', transform=test_transform, download=True)
test_dataset = datasets.Flowers102(root='./data', split='test', transform=test_transform, download=True)

# Create data loaders. These allow you to load the data for the neural network. The batch size allows you to train
# the model in batches rather than per image, allowing better and more complex learning. We shuffle the training set
# so that in each epoch, the neural network is opposed to different orders of images, strengthening the hidden layers.
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class FlowerClassifier(nn.Module):
    def __init__(self, num_classes):
        super(FlowerClassifier, self).__init__()
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv_block4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


# Initialize the model, loss function, and optimizer.
# The number of classes is 102, as there are 102 different types of flowers, hence the name Flowers-102.
model = FlowerClassifier(num_classes=102).to(device)

# We will be using Cross Entropy Loss. There are many reasons for this, for example cross entropy loss encourages the
# model to assign high probabilities to the correct class and low probabilities to the incorrect classes. It is also
# very common in multi-class classification neural networks. It focuses on the overall correctness of the model
# rather than focusing on small details which is important for a dataset this size and most importantly,
# it works very well with the optimising algorithm we will be using, stochastic gradient descent.
criterion = nn.CrossEntropyLoss().to(device)

# The optimiser we will be using is stochastic gradient descent. One of the main reasons why I used stochastic
# gradient descent is because we've done gradient descent in our practical. Also, SGD processes small batches at a time,
# making it computationally efficient. It reaches conclusions relatively faster than other optimising algorithms and
# the randomness allows for easier generation of more complex algorithms rather than a linear convergence.

# Adam
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ReduceLROnPlateau
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

best_val_loss = float('inf')
counter = 0
best_model_state = None

for epoch in range(50):  # increase to 50 epochs
    model.train()
    train_loss = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    val_loss = 0
    val_accuracy = 0
    model.eval()
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            val_loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            val_accuracy += (predicted == labels).sum().item()

    val_loss /= len(val_loader.dataset)
    val_accuracy /= len(val_loader.dataset)
    scheduler.step(val_loss)  # Scheduler steps based on validation loss

    print(
        f'Epoch {epoch + 1}, Training Loss: {train_loss / len(train_loader.dataset):.5f}, Validation Loss: {val_loss:.5f}, Validation Accuracy: {val_accuracy:.2f}')

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model_state = model.state_dict()  # Save the best model's state dictionary
        counter = 0
    else:
        counter += 1
        if counter >= 10:
            print(f"Early stopping at epoch {epoch + 1}")
            break

# Save the trained model
# torch.save(model.state_dict(), 'flower_classifier.pt')

model.load_state_dict(best_model_state)

# Evaluation on the test set
model.eval()  # Set the model to evaluation mode
accuracy = 0.0
with torch.no_grad():  # Disable gradient computation
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)  # Forward pass
        _, predicted = torch.max(outputs.data, 1)  # Get the predicted classes
        accuracy += (predicted == labels).sum().item()  # Calculate the accuracy

accuracy /= len(test_dataset)
print(f'Test Accuracy: {accuracy:.2f}')
