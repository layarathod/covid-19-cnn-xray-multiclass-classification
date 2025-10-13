# Multi-Class CNN Image Classification

This repository contains a **TensorFlow/Keras-based Convolutional Neural Network (CNN)** built to perform **multi-class image classification**.  
The notebook demonstrates a complete deep-learning pipeline — from data loading and augmentation to model training, evaluation, and activation-map visualization.

## Dataset used for this can be found here: 
https://drive.google.com/drive/folders/1lxKlpZRHBKUE0LgJZESjNhRfEnENC0kA?usp=drive_link

---

### Project Workflow

### 1. Environment Setup
- Mounts Google Drive to access datasets stored under `/content/drive/MyDrive/…`.
- Defines the training and testing dataset directories (`COVID_DATASET/Train` and `COVID_DATASET/Test`).

### 2. Import Dependencies
Imports all required Python libraries:
- **Numerical & visualization:** `numpy`, `matplotlib`
- **Deep learning:** `tensorflow.keras` (`Sequential`, `Conv2D`, `MaxPool2D`, `BatchNormalization`, `Dropout`, `Flatten`, `Dense`, `ImageDataGenerator`, `plot_model`)

### 3. Data Preparation & Augmentation
Creates data generators for preprocessing and augmentation:
- **Training generator:** rescales images by 1/255, applies random shear, zoom, and horizontal flips.
- **Validation generator:** only rescales images by 1/255.
- Loads data using `flow_from_directory`, automatically labeling each class and resizing images to **224×224** with batch size 16.

---

## CNN Model Architecture

The model is built using Keras’ **Sequential API**:

| Layer Type | Filters | Kernel | Extras |
|-------------|----------|---------|--------|
| Conv2D + BN | 16 | 3×3 | ReLU activation |
| Conv2D + BN | 16 | 3×3 | MaxPool2D |
| Conv2D + BN | 32 | 3×3 | MaxPool2D + Dropout(0.25) |
| Conv2D + BN | 64 | 5×5 | MaxPool2D + Dropout(0.25) |
| Conv2D + BN | 128 | 5×5 | MaxPool2D + Dropout(0.25) |
| Flatten | – | – | – |
| Dense(1024) | – | – | ReLU |
| Dense(512) | – | – | ReLU |
| Dense(3) | – | – | Softmax output (3 classes) |

- **Activation Function:** ReLU (except final softmax)  
- **Loss:** `categorical_crossentropy`  
- **Optimizer:** RMSprop  
- **Metric:** `categorical_accuracy`  
- **Diagram:** generated with `plot_model(model, to_file='model.png')`

---

## Training

Model trained via:

```python
model.fit_generator(
    train_generator,
    steps_per_epoch=16,
    epochs=100,
    validation_data=validation_generator,
    validation_steps=16
)

