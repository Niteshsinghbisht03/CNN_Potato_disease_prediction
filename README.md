# Potato Disease Classification Using CNN

# Overview

This project aims to classify potato leaf diseases using a Convolutional Neural Network (CNN). The model is trained on a dataset of potato leaves affected by different diseases, enabling accurate detection and classification.




# Dataset

The dataset consists of images of potato leaves categorized into different classes:    
1. Healthy   
2. Early Blight                                     
3. Late Blight

The images are preprocessed and augmented to enhance model performance.


# Model Architecture

The CNN model consists of:
1. Convolutional layers with ReLU activation
2. MaxPooling layers
3. Fully connected dense layers
4. Softmax activation for classification

# Training

The model is trained using categorical cross-entropy loss.
Adam optimizer is used for better convergence.
The dataset is split into training, validation, and test sets.

# Results
The trained model achieves high accuracy on validation data.
Performance is evaluated using precision, recall, and F1-score.

# Future Improvements
Use a larger dataset for better generalization.
Implement transfer learning for improved accuracy.
Deploy the model as a web application for real-time predictions.



