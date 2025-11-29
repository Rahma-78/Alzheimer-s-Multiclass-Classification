# Alzheimer's Multiclass Classification using VGG16

![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://drive.google.com/file/d/1ioNGO6ffP547x1rHVcVb2Aaj_IAjav60/view?usp=sharing)


## 📋 Project Overview

This project implements a Deep Learning approach for the multiclass classification of Alzheimer's disease based on brain MRI images. It utilizes **Transfer Learning** with the pre-trained **VGG16** model for feature extraction and a custom classification head to predict different stages of Alzheimer's disease.

### 🎯 Key Features
- **Transfer Learning:** Uses pre-trained VGG16 model for efficient feature extraction.
- **Data Augmentation:** Implements rotation, zooming, and shifting.
- **Comprehensive Evaluation:** accuracy, precision, recall, F1-score, and confusion matrix analysis.
- **Model Checkpointing:** Automatically saves the best-performing model during training.
- **Visual Analysis:** Provides detailed visualizations of training progress and prediction results.

---

## 💻 Run on Google Colab

You can run the training notebook and view the analysis directly in your browser:

| Platform | Link |
| :--- | :--- |
| **Google Colab** | [**Click here to open the Notebook**](https://drive.google.com/file/d/1ioNGO6ffP547x1rHVcVb2Aaj_IAjav60/view?usp=sharing) |

---

## 📊 Dataset

The project uses the **Alzheimer's Multiclass Dataset (Equal and Augmented)**, containing brain MRI images categorized into four classes:

| Class | Description |
| :--- | :--- |
| **NonDemented** | No signs of dementia (healthy control) |
| **VeryMildDemented** | Very mild stage of Alzheimer's disease |
| **MildDemented** | Mild stage of Alzheimer's disease |
| **ModerateDemented** | Moderate stage of Alzheimer's disease |

---

## 📈 Experimental Results

The model achieves high accuracy in distinguishing between the four classes. Below is a sample of the prediction results on the test set.

**Sample Predictions:**
The green text indicates a correct prediction where the `Predicted` label matches the `Actual` label.

![Model Prediction Results](PASTE_YOUR_IMAGE_LINK_HERE)

*(Note: Drag and drop your image here in the GitHub editor to generate the link)*

---

## 🔧 Model Architecture

### Base Model
- **VGG16**: Pre-trained on ImageNet for feature extraction.
- **Input Shape**: (224, 224, 3).
- **Transfer Learning**: Frozen convolutional layers.

### Custom Classification Head
- **Flatten Layer**: Converts features to 1D.
- **Batch Normalization**: Normalizes and optimizes input data.
- **Dense Layers**: 1024 neurons with ReLU activation.
- **Output Layer**: 4 neurons with softmax activation.

---

## 🏗️ Project Structure

```text
├── src/
│   ├── model/
│   │   ├── best_model.keras      # Trained model weights
│   
├── notebooks/
│   └── training_notebook.ipynb   # Jupyter notebook for training
├── requirements.txt              # Python dependencies
└── README.md                     # Project documentation


### Setup Instructions

1. Clone the repository:
```bash
git clone https://github.com/your-username/alzheimers-classification.git
cd alzheimers-classification
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```


### Using the Pre-trained Model

To use the pre-trained model for inference:

```python
from src.model.inference import predict_image

# Load the model and make predictions
model_path = "src/model/best_model.keras"
image_path = "path/to/test/image.jpg"
prediction = predict_image(model_path, image_path)

print(f"Predicted class: {prediction}")
```
