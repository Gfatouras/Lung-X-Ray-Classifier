# Lung Disease X-ray Classification using Convolutional Neural Networks
![COVID-28](https://github.com/user-attachments/assets/63ae7808-6d32-49b6-9bb9-ca552bea93da)
![Viral Pneumonia-12](https://github.com/user-attachments/assets/e6fd44d9-4239-4978-9dd4-e7c5c38e3a69)

## Overview

This repository contains the code and data used for classifying lung X-ray images to identify COVID-19, pneumonia, and normal cases. The [dataset](https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database) was obtained from Kaggle and includes thousands of X-ray images along with corresponding mask images. A baseline model using a sequential neural network was created first, followed by a Convolutional Neural Network (CNN) that achieved a test accuracy of 86%.

A separate test notebook is also included `test.ipynb`. This notebook is a deployment that uses samples of X-ray images along with their masks located in `test/`, and overlays a prediction for each image, which is stored in `test_results/`. Some output examples are shown above.
# Important Note:
To install the required packages, run the following command in the local project folder:

+ pip install -r requirements.txt

If you get OOM errors, or tensorflow errors, simply restart the program. This is most likley due to insufficient memory. The models and dataframes are saved locally after they are created.

## Business Problem

By analyzing these images with machine learning models, we aim to support diagnostic processes, improve the accuracy of medical image classification, and provide diagnosis support.

### Table of Contents

- [Overview](#overview)
- [Business Problem](#business-problem)
- [Data Sources](#data-sources)
  - [Feature Extraction](#feature-extraction)
  - [Model Creation](#model-creation)
  - [Evaluation](#evaluation)
- [Conclusion](#conclusion)
- [Contact](#contact)

## Data Sources

The dataset used is the X-ray radiology [dataset](https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database) available on Kaggle, which includes thousands of lung X-ray images categorized into normal, COVID-19, and viral pneumonia, along with corresponding mask images to reduce noise.

Image counts:
- Covid-19: 3,616 
- Normal: 10,192
- Viral Pneumonia: 1,345

## **Image Processing and Data Handling**

The initial steps involved ensuring that the image dimensions were 256x256 pixels. Then, the images were converted to pixel intensity arrays. Masks were applied to include only the target lung areas before feeding the data into the models.

For the dataset:
- **Normal and COVID-19 X-ray Images**: Kept at 2,000 images each. This was done intentionally to address the CNN model’s difficulty in differentiating between normal and COVID-19 images. Using more than 2,000 images did not improve the accuracy and only increased processing time.
- **Pneumonia Images**: Retained at 1,345 images. This was the maximum number of pneumonia images, and was left undersampled since the model had a harder time classifying normal and COVID-19 images compared to pneumonia.

The masks helped in focusing on the relevant lung areas, enhancing the model's ability to learn and classify the images effectively.

## Model Creation

Five models were developed:

- **Baseline Model:** A sequential neural network (NN) used as a starting point, which achieved an accuracy of 76%.

- **Model with ReduceLROnPlateau:** A Sequential NN model with the ReduceLROnPlateau callback, which improved the accuracy to 81%.

- **CNN Model:** A Convolutional Neural Network (CNN) that also achieved an accuracy of 81%.

- **CNN + Data Augmentation:** A CNN model combined with data augmentation techniques, which improved the accuracy to 86%.

- **CNN + L2 Regularization:** A CNN model with L2 regularization, which achieved an accuracy of 83%.


## Evaluation

The final CNN model was evaluated using the following metrics:

- **Accuracy Metrics**
- **Confusion Matrices**
- **ROC-AUC Scores**
![CNN_Datagen_Optimal_Loss](https://github.com/user-attachments/assets/f5de1822-17dd-4dde-a868-0df58035b46e)
![CNN_Datagen_Optimal_Accuracy](https://github.com/user-attachments/assets/33b6dd3f-ec36-48b5-b40b-f875ed82990f)
![CNN_Datagen_Optimal_Confusion](https://github.com/user-attachments/assets/501db939-d61a-4704-a0b2-9cc3406486c3)
![CNN_Datagen_Optimal_ROC](https://github.com/user-attachments/assets/f74fe00d-c62d-49c0-af12-34243a80b3a4)
CNN layer structure:

![Sequential_CNN_Datagen_Optimal_Structure](https://github.com/user-attachments/assets/f736a020-4042-4009-beb7-fe796526e8e9)

## **Conclusion**

### **CNN Model Performance**

The CNN model with data augmentation performed the best, achieving a test accuracy of 86%. The model's similar F1 scores and smooth convergence of loss values indicate that there is no overfitting present. Among the different classifications, viral pneumonia was the easiest to identify, with the best classification rate of 97%. This model used MaxPooling, and Conv2D layers which were set to 3x3 pixels. These small grids were able to scan through each image, and identify features in a 2D manner, rather than the 1D method for the baseline model. This added more complex layers to the model, which greatly aided in the accuracy. Additional information for how MaxPooling and Conv2D can be utilized in machine learning algorithms can be found below:

Gholamalinezhad, H., & Khosravi, H. (2020). Pooling methods in deep neural networks, a review. https://www.semanticscholar.org/paper/Pooling-Methods-in-Deep-Neural-Networks%2C-a-Review-Gholamalinezhad-Khosravi/8f66ed7f0e2089b4f5219c782687bea368c7f4ee

Full pdf version: https://arxiv.org/pdf/2009.07485

This model can be utilized by healthcare professionals to assist in:

- **Decision support**
- **Early detection**
- **Disease progression monitoring**
- **Reducing human error**
- **Providing rapid diagnosis through X-ray images alone**

---

## **Future Improvements**

### **Enhancing the Model**

While the CNN model performed well, there are still ways to improve it:

- **Additional Training Data**: More normal and COVID-19 images should be included in training to improve the accuracy between thest two classes.
- **Pre-trained Models**: Utilizing pre-trained models could increase the accuracy rate to around 91%.
- **Broader Disease Detection**: Include different lung diseases to allow the model to detect additional illnesses, as current diseases like bronchitis might slip detection.


## Contact

- **Greg Fatouras**
- **LinkedIn Profile**: https://www.linkedin.com/in/gfatouras/
- **Email**: fatourasg@gmail.com
