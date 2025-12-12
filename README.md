# 🧠 Brain Tumor Classification using Convolutional Neural Networks (CNN)

This project uses a **Convolutional Neural Network (CNN)** to classify MRI images into **tumor** and **no tumor** categories.  
The model is built using **PyTorch** and trained on a dataset of brain MRI scans.

---

## 📌 Project Overview

Brain tumor detection from MRI images is a critical task in medical imaging.  
This project builds a deep learning model that:

- Loads MRI images  
- Applies data transformations  
- Trains a CNN from scratch  
- Evaluates the model  
- Predicts tumor vs. no-tumor for new images  

---

## 🧠 Model Architecture

A custom CNN is implemented with:

- Convolutional layers  
- ReLU activation  
- MaxPooling  
- Dropout  
- Fully connected layers  

The architecture is optimized for image classification tasks.

---

## 🧹 Dataset Information

- Classes: **Yes (Tumor)**, **No (No Tumor)**  
- Each class contains MRI images  
- The dataset is typically divided into:
  - **Train set**
  - **Validation set**
  - **Test set**

Image preprocessing includes:

- Resizing  
- Normalization  
- Converting to PyTorch tensors  

---

## 📂 Project Structure
**Data folder will be generated using the code**

-Brain_Tumor_CNN/
│── Brain_tumor_CNN.ipynb
│── data/
│   ├── yes/
│   ├── no/
│── models/
│── results/
│── README.md

## How to Run the Project

### 1️⃣ Install Dependencies
pip install torch torchvision matplotlib numpy pillow

### 2️⃣ Run the Notebook
Open Jupyter Notebook:
Then run:

Brain_tumor_CNN.ipynb
---

## 🏋️ Training Details

The notebook handles:

- Data loading using **ImageFolder + DataLoader**
- Data augmentation (transforms)
- Loss function: **CrossEntropyLoss**
- Optimizer: **Adam**
- Training loop with:
  - forward pass  
  - backward pass  
  - weight updates  

---

## 📊 Model Evaluation

The notebook prints:

- Training accuracy  
- Validation accuracy  
- Loss curves  
- Confusion matrix (optional)

Example expected result:

Train Accuracy: 98%
Validation Accuracy: 96%
Test Accuracy: 95%

(*Your actual results may vary depending on the dataset.*)

---

## 🔍 Predictions on New Images

The notebook includes code to:

- Load a single MRI image  
- Apply transforms  
- Run the model  
- Output: **Tumor / No Tumor**

---

## 🛠 Technologies Used

- Python  
- PyTorch  
- Torchvision  
- NumPy  
- Matplotlib  
- PIL (Pillow)
---

## 🙌 Author

**Kuldeep Patel**  
Machine Learning & Deep Learning Engineer

If you like this project, consider ⭐ starring the repository!
