
### Plant-Disease-Classifier

## Overview

This repository contains a project for classifying potato plant diseases using a fine-tuned ResNet18 architecture. The classifier can identify three categories of potato plant conditions:

1. **Potato_healthy**
2. **Potato_late_blight**
3. **Potato_early_blight**

The model has been trained using the `fastai` library, leveraging a dataset of potato plant images. The goal is to provide a reliable tool for farmers and agricultural researchers to quickly diagnose and address potato plant diseases.

### Project Links

- [**Kaggle Notebook:**](https://www.kaggle.com/code/hillol10/potato-disease-cnn)
- [**Hugging Face Space:**](https://huggingface.co/spaces/hillol7/potatoes)

## Features

- **ResNet18 Architecture:** The model is based on ResNet18, a deep convolutional neural network of 18-layers that is fine-tuned for this specific classification task.
- **Three Classifications:** The classifier can identify if a potato plant is healthy, affected by early blight, or suffering from late blight.
- **User-Friendly Interface:** The classifier is deployed on Hugging Face Spaces with an interactive Gradio interface.

## Dataset

The model is trained on a dataset of potato plant images. The dataset includes labeled images of healthy potato plants, as well as plants affected by early blight and late blight.

## Installation

To run this project locally, follow these steps:

1. **Clone the Repository:**

    ```bash
    git clone https://github.com/ringerH/Potato-Disease-Classifier.git
    cd Potato-Disease-Classifier
    ```

2. **Create a Virtual Environment:**

    ```bash
    python -m venv venv
    source venv/bin/activate # On Windows, use `venv\Scripts\activate`
    ```

3. **Install Dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

## Usage

To classify an image of a potato plant, use the following command:

1. **Run the Gradio App:**

    ```bash
    python app.py
    ```

2. **Upload an Image:** Access the local URL provided by Gradio to upload an image of a potato plant.

3. **View the Classification:** The model will classify the image into one of the three categories and display the result.

## How It Works

### Model Architecture

- **ResNet18:** The ResNet18 architecture is used, which is known for its ability to effectively handle image classification tasks by using skip connections to overcome the vanishing gradient problem.
- **Fine-Tuning:** The model is pre-trained on ImageNet and fine-tuned on the potato plant dataset to specialize in recognizing the specific classes of potato plant conditions.

### Training

- **Fastai Library:** The `fastai` library simplifies the training process with powerful data augmentation techniques, transfer learning, and model evaluation metrics.
- **Data Augmentation:** Various data augmentation techniques such as flipping, rotation, and zooming were applied to enhance the robustness of the model.
