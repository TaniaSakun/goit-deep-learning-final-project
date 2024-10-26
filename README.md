# goit-deep-learning-final-project
Pet Adoption Speed Prediction

## 1. Project Overview

This project aims to classify pet adoption speeds based on images and textual descriptions.

- **Image Data:** ResNet50, pre-trained on ImageNet, is used to process pet images to extract features that may correlate with adoption speed.
- **Text Data:** Pet descriptions are analyzed for specific features that may impact adoption likelihood, such as age, breed, health status, and sentiment.
- **Adoption Speed Categories:**

  - 1 to 7 days (1)
  - 8 to 30 days (2)
  - 31 to 90 days (3)
  - Over 100 days (4)
  
- **Model Evaluation:** The Quadratic Weighted Kappa (QWK) metric evaluates model performance by comparing predicted adoption speeds to actual data, giving a robust measurement of prediction consistency.

## 2. Data Preparation

### 2.1 Image Processing

**Feature Extraction:** Using ResNet50 allows us to leverage pre-trained knowledge to identify high-level visual features, which may help predict how quickly an animal might be adopted based on its visual appeal.

### 2.2 Text Data Processing

- **Feature Engineering:** Key features extracted from descriptions include:
- **Age:** Converted to numeric values for modeling.
- **Breed:** Recognized from a predefined list, which may relate to adoption desirability.
- **Health:** Information regarding vaccinations, deworming, and sterilization.
- **Sentiment Analysis:** VADER sentiment analysis extracts positive, neutral, and negative sentiment scores from descriptions.
- **Normalization & TF-IDF Transformation:** Text data is standardized by lemmatization, removing stop words, and then converted to a numerical format using TF-IDF for efficient modeling.

## 3. Model Architecture and Training

### 3.1 Model Architecture

The model is structured as an ensemble that processes both image and text features.

- **Text Feature Layers:** Text features, including TF-IDF vectors and manually extracted features, pass through dense layers with ReLU activation, normalization, and dropout to prevent overfitting.
- **Image Feature Layers:** Image features from ResNet50 are similarly processed through dense layers.
- **Output Layer:** The final layer is a classification layer predicting adoption speed classes based on the combined image and text features.

### 3.2 Hyperparameter Tuning

Learning rate, dropout, and batch size were manually tuned based on experimentation, focusing on model stability and generalization.

## 4. Evaluation and Results

The project is evaluated using QWK, with the following results:

- **Training Data Results: High accuracy on training data suggests the model captures meaningful patterns within this dataset.
- **Test Data Results: Lower QWK scores on test data suggest potential overfitting or inadequate generalization to new data.
- **Result Analysis:**

  - **Text Features:** While features like sentiment and health status diversify the data, their impact on overall accuracy was limited.
  - **Image Features:** ResNet50 helped capture visual characteristics, though image quality and pet appearance variability introduced inconsistencies.

## 5. Conclusions and Suggestions for Improvement

Despite substantial data preparation and tuning, the results fell short of targets. Here are some potential improvements:

### 5.1 Additional Feature Engineering

Interactive Features: Unfortunately, examining relationships between features (age, breed, gender, vaccination, dewormed status, health, sterilization status, and updated method for extracting the number of pictures) hasn't enhanced model understanding.
Advanced NLP: Leveraging transformers like BERT to encode text may capture more nuanced meanings in descriptions.

### 5.2 Alternative Architectures and Approaches

- **Different CNN Models:** Exploring EfficientNet for image extraction could yield better results, as it has shown strong performance on diverse image sets.
- **Ensemble Models:** Combining predictions from multiple models may improve stability and accuracy.

### 5.3 Improved Regularization

- **Increased Dropout:** Experimenting with higher dropout or L2 regularization could reduce overfitting.
- **Data Augmentation:** Applying transformations to images may improve model resilience against variations in image quality.

### 5.4 Error Analysis

**Misclassification Review:** Analyzing common misclassifications could highlight where the model systematically errs, helping refine preprocessing or model layers.
