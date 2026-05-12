# Flower Classifier using MobileNet

A Deep Learning Computer Vision project for flower image classification using Transfer Learning with MobileNet, TensorFlow, and Keras.

---

## Project Overview

This project classifies flower images into different categories using a pre-trained MobileNet architecture.

Instead of training a CNN from scratch, Transfer Learning was used to leverage MobileNet’s powerful feature extraction capabilities, resulting in faster training and improved performance.

---

## Technologies Used

- Python
- TensorFlow
- Keras
- MobileNet
- Transfer Learning
- NumPy
- Matplotlib
- Jupyter Notebook

---

## Features

- Flower image classification
- Transfer Learning using MobileNet
- Custom prediction pipeline
- Image preprocessing
- Label mapping using JSON
- Saved trained model (`.h5`)
- Test image predictions

---

## Project Structure

```text
flower-classifier-mobilenet/
│
├── assets/
├── test_images/
├── final_flower_classifier_model.h5
├── label_map.json
├── predict.py
├── workspace_utils.py
├── flower_classifier_project.html
├── README.md
```

---

## Model

The project uses:

```text
MobileNet
```

as the base pre-trained CNN model for feature extraction.

Custom classification layers were added on top of MobileNet to classify flower categories.

---

## Saved Model

The trained model is saved as:

```text
final_flower_classifier_model.h5
```

---

## Prediction Script

The file:

```text
predict.py
```

is used to run predictions on custom flower images.

---

## Goal

The purpose of this project is to understand:

- Computer Vision fundamentals
- Image Classification
- Transfer Learning
- Deep Learning workflows
- TensorFlow/Keras model deployment

---

## Author

Muthanna Abubaha