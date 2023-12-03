# Hate-Speech-Detection-System
A model that is used to detect whether a comment is a hate speech or not.

# Detecting Hate Speech in Text - AI Project

## Overview
This project dives into the challenging task of identifying hate speech within textual data using machine learning. With over 24,000 records and 7 columns, the primary aim is to predict hate speech leveraging the 'hate_speech' and 'tweet' columns.

**Reason**: Understanding the dataset's structure and defining objectives based on research insights was essential to set the project's direction.

## Preprocessing
- **Initial Exploration**: Grasping the dataset's dimensions and columns, guiding our focus based on research insights.
  
**Reason**: Lowercasing and handling URLs, non-words, and punctuations ensured uniformity and effective text processing.

## Word Vocabulary
- **Bag of Words Creation**: Constructing word frequency representations to grasp common words in the dataset.

**Reason**: Creating a bag of words gives a quick overview of prevalent terms across hate and non-hate comments.

## Data Visualization
- Generating Word Clouds: Visualization of common words in hate and non-hate comments.

**Reason**: Visualizing words helps in understanding the vocabulary trends within different comment categories.

## Statistical Analysis
- Conducting T-tests: Comparing text lengths between hate speech and non-hate speech comments.

**Reason**: Statistical tests provide insights into differences, aiding feature selection or engineering.

## Dataset Splitting
- Segregating data into training, validation, and test sets.

**Reason**: Proper data partitioning ensures unbiased model evaluation and performance estimation.

## LSTM Model Building
- Creating an LSTM-based hate speech detection model.
- Hyperparameter tuning using GridSearchCV.

**Reason**: LSTM architectures are suitable for sequence data; tuning ensures optimal model performance.

## Evaluation Metrics
- Calculating accuracy, mean absolute error, and root mean squared error.

**Reason**: These metrics offer varied insights into model performance.

## GRU Model Building
- Implementing a GRU-based hate speech detection model.
- Hyperparameter tuning using GridSearchCV.

**Reason**: Exploring different architectures to compare performance against LSTM.

## BERT Model Implementation
- Leveraging BERT models for hate speech classification.
- Training with 'distilbert-base-cased' checkpoint for sequence classification.
- Utilizing 'ktrain' for BERT model training and prediction.

**Reason**: BERT's state-of-the-art performance and fine-tuning capabilities for text classification.

### Model Saving
- Storing the best-performing BERT model post-training.

**Reason**: Saving the model ensures reusability and avoids repetitive training.

## Conclusion
- Highlighting the complete workflow of hate speech detection, from data preprocessing to multiple model architectures and evaluation metrics.
- Encouraging further exploration and customizations for improving hate speech detection models.

This journey was a profound exploration of hate speech detection, encompassing various methodologies and models, aiming to contribute to a more respectful online discourse.

Link to the demo video :https://youtu.be/OiZP3NLfZqA
