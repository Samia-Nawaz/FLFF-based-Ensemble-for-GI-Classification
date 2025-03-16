# FLFF-based Ensemble for Gastrointestinal Disease Classification

## Overview

This repository implements a **Federated Learning Flower Framework (FLFF)**-based **ensemble deep learning** model for **gastrointestinal disease classification**. The approach integrates:
- **Secure Image Encryption** using an **Improved Lorenz Chaos System**
- **Federated Learning** for privacy-preserving model training
- **Deep Feature Extraction** with **Variational Autoencoders (VAE), Dual Conditional Autoencoders (DCAE), and Attention-Based Bi-GRU (ABiGRU)**
- **Chimp Optimization Algorithm** for fine-tuning hyperparameters
- **Ensemble Learning** for robust classification performance

## Features
✔ Secure **medical image encryption** for privacy protection  
✔ **Federated learning** for decentralized model training  
✔ **Deep feature extraction** using **autoencoders and recurrent networks**  
✔ **Chimp Optimization Algorithm (CO)** for hyperparameter tuning  
✔ **High accuracy** of **98.59%** on the **Kvasir dataset**  

## Project Structure

```
📂 FLFF-based-Ensemble-for-GI-Classification
│── data_preprocessing.py           # Preprocessing of medical images
│── image_encryption.py             # Improved Lorenz Chaos encryption
│── federated_learning.py           # Implementation of Flower FL framework
│── feature_extraction.py           # Deep feature extraction (VAE, DCAE, ABiGRU)
│── ensemble_model.py               # Ensemble learning model
│── chimp_optimization.py           # Chimp optimization for hyperparameter tuning
│── model_training.py               # Model training and evaluation
│── security_analysis.py            # Encryption & decryption performance analysis
│── model_deployment.py             # Preparing model for deployment
│── README.md                       # Project documentation
│── requirements.txt                 # Required dependencies
```

## Installation

### **1. Clone the Repository**
```bash
git clone https://github.com/Samia-Nawaz/FLFF-based-Ensemble-for-GI-Classification.git
cd FLFF-based-Ensemble-for-GI-Classification
```

### **2. Install Dependencies**
Ensure you have **Python 3.8+** installed. Then, run:
```bash
pip install -r requirements.txt
```

### **3. Dataset Setup**
- Download the **Kvasir dataset**: [Link](https://datasets.simula.no/kvasir/)
- Place it in a folder named **data/** inside the project directory.

## Usage

### **1. Data Preprocessing**
```bash
python data_preprocessing.py
```
- Resizes images, normalizes pixel values, and applies augmentation.

### **2. Image Encryption**
```bash
python image_encryption.py
```
- Encrypts images using **Lorenz Chaos System**.

### **3. Federated Learning Training**
```bash
python federated_learning.py
```
- Trains the model using **Flower Framework for Federated Learning**.

### **4. Feature Extraction**
```bash
python feature_extraction.py
```
- Extracts deep features using **VAE, DCAE, ABiGRU**.

### **5. Train the Ensemble Model**
```bash
python ensemble_model.py
```
- Combines extracted features for **classification**.

### **6. Hyperparameter Optimization**
```bash
python chimp_optimization.py
```
- Optimizes model parameters using **Chimp Optimization Algorithm**.

### **7. Model Training & Evaluation**
```bash
python model_training.py
```
- Trains the model and evaluates accuracy, precision, and recall.

### **8. Security Analysis**
```bash
python security_analysis.py
```
- Evaluates **encryption and decryption performance**.

### **9. Model Deployment**
```bash
python model_deployment.py
```
- Converts the trained model for **real-time clinical deployment**.

## Results

| Model  | Accuracy | Precision | Recall | F1-score | Kappa |
|--------|---------|----------|--------|---------|--------|
| **Proposed Ensemble** | **98.59%** | **98.01%** | **98.14%** | **98.09%** | **0.97** |
| VAE | 94.78% | 95.76% | 94.10% | 94.18% | 0.93 |
| ABiGRU | 93.32% | 92.65% | 94.20% | 93.06% | 0.90 |
| CNN | 86.59% | 85.67% | 87.52% | 86.19% | 0.82 |
| SVM | 75.81% | 74.98% | 76.12% | 75.48% | 0.70 |

 
