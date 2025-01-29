# **Age, Gender, and Ethnicity Prediction Using Functional API**  

## **Overview**  
This project utilizes the Functional API in TensorFlow to build a multi-output deep learning model capable of predicting age, gender, and ethnicity from facial images. The project employs data augmentation and transfer learning using XceptionNet as the convolutional base to enhance performance and achieve robust predictions.  

---

## **Dataset**  
The dataset used for this project is the [UTKFace Dataset](https://www.kaggle.com/jangedoo/utkface-new).  
- **Description**: The dataset contains over 20,000 facial images labeled with age, gender, and ethnicity.  
- **Target Outputs**:  
  - **Age**: Continuous variable indicating the person's age.  
  - **Gender**: Binary classification (`0`: Female, `1`: Male).  
  - **Ethnicity**: Multi-class classification (`0-4` representing different ethnic groups).  

---

## **Key Features**  

1. **Functional API**:  
   - Designed a multi-output neural network for simultaneous prediction of age, gender, and ethnicity.  
   - Ensured flexibility in handling multiple inputs and outputs.  

2. **Transfer Learning**:  
   - Leveraged **XceptionNet** as a pre-trained convolutional base for feature extraction.  
   - Fine-tuned specific layers to adapt to the dataset for enhanced performance.  

3. **Data Augmentation**:  
   - Applied transformations such as rotation, flipping, zooming, and shifting to increase data diversity and prevent overfitting.  

4. **Multi-Output Model**:  
   - Age: Regression output (mean squared error loss).  
   - Gender: Binary classification output (binary cross-entropy loss).  
   - Ethnicity: Multi-class classification output (categorical cross-entropy loss).  

---

## **Model Architecture**  

- **Convolutional Base**: XceptionNet for feature extraction.  
- **Custom Heads**: Separate dense layers for predicting age, gender, and ethnicity.  
- **Loss Functions**: Combined loss for all outputs to optimize the multi-task learning process.  
- **Optimizer**: Adam optimizer with a learning rate scheduler for efficient training.  

---

## **Project Workflow**  

1. **Data Preprocessing**:  
   - Loaded and cleaned the UTKFace dataset.  
   - Normalized pixel values to the range `[0, 1]`.  
   - Split the data into training, validation, and test sets.  

2. **Data Augmentation**:  
   - Applied techniques such as random rotation, width/height shifting, and horizontal flipping using TensorFlow's `ImageDataGenerator`.  

3. **Model Development**:  
   - Used the Functional API to construct the model with shared convolutional layers and task-specific dense heads.  
   - Incorporated XceptionNet as a pre-trained base with frozen weights during initial training.  

4. **Training**:  
   - Trained the model on the augmented dataset with a combined loss function.  
   - Evaluated the model on unseen test data for accuracy and performance metrics.  

---
### **Prerequisites**  
Ensure the following libraries are installed:  
- Python 3.8+  
- TensorFlow 2.7+  
- NumPy  
- Matplotlib  
- Pandas  

