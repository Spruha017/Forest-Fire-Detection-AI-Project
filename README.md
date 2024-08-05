# Forest Fire Detection Using Deep Learning

## Overview

This project leverages deep learning techniques to build a model that detects forest fires in images. It involves training a Convolutional Neural Network (CNN) using TensorFlow and Keras to classify images into two categories: "fire" and "nofire."

## Requirements

- Python 3.x
- TensorFlow 2.x
- Keras
- NumPy
- Matplotlib
- Seaborn
- scikit-learn
- Kaggle API

## Setup

1. **Clone the Repository**

   ```bash
   git clone https://github.com/your-username/forest-fire-detection.git
   cd forest-fire-detection
   ```

2. **Install Dependencies**

   You can use `pip` to install the required libraries:

   ```bash
   pip install tensorflow numpy matplotlib seaborn scikit-learn kaggle
   ```

3. **Kaggle API Setup**

   - Create a Kaggle account if you don’t already have one.
   - Go to your Kaggle account settings and create a new API token. Download the `kaggle.json` file.
   - Place the `kaggle.json` file in the root directory of this project.

## Usage

1. **Download and Prepare Dataset**

   The dataset can be downloaded directly from Kaggle and extracted. Run the following commands to set up the dataset:

   ```python
   !mkdir ~/.kaggle
   !cp kaggle.json ~/.kaggle/kaggle.json
   !kaggle datasets download -d alik05/forest-fire-dataset
   !unzip forest-fire-dataset.zip -d /content
   ```

2. **Train the Model**

   To train the model, execute the training script. This will preprocess the images, train a CNN model, and save the trained model.

   ```python
   python train_model.py
   ```

   The model will be saved as `forestfire_detection_model.h5`.

3. **Evaluate the Model**

   You can evaluate the performance of the model by generating confusion matrices and accuracy scores:

   ```python
   python evaluate_model.py
   ```

4. **Test the Model**

   To test the model with new images, use the provided testing script:

   ```python
   python test_model.py
   ```

   You can replace the `img_path` variable with the path to the image you want to classify.

## Code Description

- **Data Preparation and Augmentation**: Uses TensorFlow's `ImageDataGenerator` to preprocess and augment training data.
- **Model Definition**: Builds a CNN model with Conv2D, MaxPooling2D, Flatten, Dense, and Dropout layers.
- **Model Training**: Trains the model using binary cross-entropy loss and Adam optimizer.
- **Evaluation**: Computes accuracy, precision, recall, and F1 score. Plots confusion matrices.
- **Testing**: Predicts fire status for new images.

## Example Usage

You can test the model with the following example code:

```python
img_path = '/path/to/your/image.jpg'
result = predict_fire_status(img_path)
print(f'The image is classified as: {result}')
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- The dataset used is sourced from Kaggle.
- Thanks to TensorFlow and Keras for providing powerful tools for deep learning.

---
