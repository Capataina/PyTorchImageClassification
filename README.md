# Flower Classification Model

This project implements a convolutional neural network (CNN) for classifying images of flowers from the Oxford Flowers-102 dataset.

## Requirements

- Python 3.x
- PyTorch
- torchvision
- scipy

## Installation

1. Clone the repository:

   ```
   Hidden for submission anonymity
   ```

2. Install the required packages:
   ```
   pip install torch torchvision scipy
   ```

## Usage

1. Run the script:

   ```
   python flower_classifier.py
   ```

   The script will automatically download the Oxford Flowers-102 dataset, train the CNN model, and evaluate its performance on the test set.

2. The trained model will be saved as `flower_classifier.pt` in the current directory.

## Model Architecture

The CNN model consists of the following components:

- 7 convolutional blocks, each containing:
  - Convolutional layer
  - Batch normalization
  - ReLU activation
  - Max pooling
- Adaptive average pooling layer
- Fully connected layers with ReLU activation and dropout

## Training Details

- The model is trained for 175 epochs using the Adam optimizer with a learning rate of 0.0001 and weight decay of 0.0075.
- Label smoothing is applied to prevent overfitting and provide advanced regularization.
- The learning rate is dynamically adjusted using the `ReduceLROnPlateau` scheduler based on the validation loss.
- Early stopping is implemented to prevent overfitting by stopping the training if the validation loss stops improving for 20 consecutive epochs.

## Dataset

The Oxford Flowers-102 dataset is used for training and evaluation. It consists of 102 different types of flowers. The dataset is automatically downloaded and preprocessed using data transformations.

## Results

The highest accuracy achieved by the model is 72% and the average accuracy is around 70% after training.

## License

This project is licensed under the [MIT License](LICENSE).

---
