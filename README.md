# PyTorch Flower Classification

This project implements a deep learning approach for classifying flower species using the Oxford Flowers-102 dataset. It utilizes a custom Convolutional Neural Network (CNN) architecture built with PyTorch.

## Features

- **Custom CNN Architecture**: A sophisticated deep learning model specifically designed for flower species classification, featuring 7 convolutional blocks with increasing complexity.

- **Advanced Data Augmentation**: Utilizes a range of image transformation techniques including random resized crop, horizontal flips, rotations, and colour jitters to enhance model robustness and generalization.

- **Label Smoothing**: Implements an advanced regularization technique to prevent overfitting and improve model generalization by introducing a small amount of noise in the labels.

- **Dynamic Learning Rate Adjustment**: Employs the ReduceLROnPlateau scheduler to automatically adjust the learning rate based on validation performance, optimizing the training process.

- **Early Stopping**: Implements an early stopping mechanism to prevent overfitting by halting training when the validation loss stops improving.

- **Efficient Data Loading**: Uses PyTorch's DataLoader with multi-processing support for efficient batch loading and preprocessing of images.

- **GPU Acceleration**: Automatically utilizes CUDA-enabled GPUs if available for faster training and inference.

- **Model Persistence**: Saves the best-performing model during training, allowing for easy resumption of training or deployment.

- **Comprehensive Evaluation**: Provides detailed training progress logs and final test set evaluation to assess model performance.

## Requirements

- **Python**: Version 3.x (3.6 or higher recommended)

- **PyTorch**: Version 1.7.0 or higher. This deep learning framework is the core of our model implementation.

- **torchvision**: Version compatible with the installed PyTorch. Used for dataset loading, image transformations, and data augmentation.

- **scipy**: Required for processing the downloads from torchvision.

- **CUDA Toolkit**: (Optional but recommended) Version compatible with PyTorch for GPU acceleration.

- **Hardware**: 
  - CPU: Any x86-64 processor (Intel or AMD) supporting SSE4.2 and AVX2 instructions.
  - RAM: Minimum 8GB, 16GB or more recommended for faster processing.
  - GPU: NVIDIA GPU with CUDA support (optional but highly recommended for faster training).

- **Operating System**: 
  - Linux (Ubuntu 18.04 or later recommended)
  - macOS (10.15 Catalina or later)
  - Windows 10

- **Storage**: At least 10GB of free disk space for the dataset and model checkpoints.

- **Internet Connection**: Required for downloading the dataset and any additional dependencies.

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/pytorch-flower-classification.git
   cd pytorch-flower-classification
   ```

2. Install the required dependencies:
   ```
   pip install torch torchvision scipy
   ```

## Usage

To train and evaluate the model, run the main Python script:

```
python PyTorchImageClassification.py
```

This script will:
1. Download and prepare the Flowers-102 dataset
2. Train the model for 175 epochs (or until early stopping is triggered)
3. Save the best model as 'flower_classifier.pt'
4. Evaluate the model on the test set and print the test accuracy

## Model Architecture

The FlowerClassifier model consists of:
- 7 convolutional blocks, each containing:
  - Convolutional layer
  - Batch normalization
  - ReLU activation
  - Max pooling
- Global average pooling
- Two fully connected layers with dropout

## Training Process

- Optimizer: Adam (learning rate: 0.0001, weight decay: 0.0075)
- Loss function: Cross-Entropy Loss
- Learning rate scheduler: ReduceLROnPlateau
- Data augmentation: Random resized crop, horizontal flip, rotation, and colour jitter
- Label smoothing (smoothing factor: 0.1)
- Early stopping (patience: 20 epochs)

## Results

The model achieves a test accuracy of 72% on the Flowers-102 dataset.

## Dataset

This project uses the Oxford Flowers-102 dataset, which contains 102 flower categories. The dataset is automatically downloaded and processed using Torchvision.

## License

This project is open-source and available under the MIT License.
