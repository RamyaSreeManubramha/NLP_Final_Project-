# Sentiment Analysis for Marketing

## Project Overview

This project implements a sentiment analysis model using **Long Short-Term Memory (LSTM)** networks to classify customer reviews as **positive**, **negative**, or **neutral**. The approach leverages **deep learning techniques**, **word embeddings**, and **data preprocessing** to enhance model performance and extract meaningful insights from textual data. The study aims to provide a robust solution for analyzing customer sentiment in applications such as **product feedback analysis**, **social media monitoring**, and **market research**.

## Key Features
- **Dataset**: Amazon Earphones Reviews from Kaggle.
- **Model Architecture**:
  - Embedding Layer with pre-trained word embeddings (GloVe).
  - Two LSTM layers for capturing long-term dependencies in text.
  - Dense layers for feature transformation and classification.
- **Performance Metrics**: Accuracy, F1-Score, and a confusion matrix were used to evaluate the model.
- **Applications**: Suitable for real-world sentiment analysis tasks like **customer feedback analysis** and **social media sentiment monitoring**.

## Dataset
The dataset used in this project contains reviews of earphone products with the following columns:
- **ReviewBody**: Detailed text of the review (input for the model).
- **ReviewStar**: Rating (1–5) mapped to sentiment labels:
  - 1–2 stars: Negative sentiment
  - 3 stars: Neutral sentiment
  - 4–5 stars: Positive sentiment

Sentiment distribution in the dataset:
- **Positive**: 65.6%
- **Negative**: 23.9%
- **Neutral**: 10.5%

[Dataset Source](https://www.kaggle.com/datasets/shitalkat/amazonearphonesreviews)

## Methodology
1. **Data Cleaning**:
   - Removed missing values.
   - Performed text preprocessing (lowercasing, removing special characters, stopword removal).
   - Balanced the dataset to address class imbalance.
2. **Model Training**:
   - Utilized **categorical cross-entropy** loss and **Adam optimizer**.
   - Evaluated the model using accuracy and F1-score.
3. **Evaluation**:
   - Achieved an overall accuracy of **73%**.
   - Strong performance for positive sentiments (F1-score: 0.83).
   - Highlighted challenges with neutral sentiment classification due to class imbalance.

## Results
- **Accuracy**: 73%
- **F1-Scores**:
  - Positive: 0.83
  - Negative: 0.58
  - Neutral: 0.00 (reflecting class imbalance and low representation)
- The LSTM model proved effective for positive and negative sentiment but struggled with neutral sentiment.

## Limitations
1. **Class Imbalance**: Neutral sentiment was underrepresented.
2. **Model Instability**: Observed dips in validation accuracy during training.

## Future Work
- Address class imbalance using techniques like SMOTE or weighted loss functions.
- Experiment with **transformer-based models** (e.g., BERT, RoBERTa) for improved performance.
- Incorporate data augmentation techniques to enhance the diversity of examples.
- Explore explainable AI techniques for better model interpretability.

## Dependencies
- Python (>=3.7)
- TensorFlow/Keras
- Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn
- NLTK or SpaCy (for text preprocessing)
