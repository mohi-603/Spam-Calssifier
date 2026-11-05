# Spam Message Classifier

## Project Overview
A machine learning project that classifies SMS messages as **Spam** or **Ham (Legitimate)** using Natural Language Processing (NLP).

The project uses TF-IDF vectorization and compares Multinomial Naive Bayes with Logistic Regression.

## Dataset
- **Source:** [UCI SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection)
- Original messages: 5,572
- Messages after removing duplicates: 5,169
- Ham messages: 4,516
- Spam messages: 653

## Technologies Used
- Python
- Pandas and NumPy
- Matplotlib and Seaborn
- Scikit-learn
- NLTK
- Joblib
- Jupyter Notebook

## Project Workflow
1. Data loading and exploration
2. Duplicate removal and label analysis
3. Text preprocessing and stop-word handling
4. Train/test splitting
5. TF-IDF feature extraction using unigrams and bigrams
6. Model training and comparison
7. Threshold tuning using validation data
8. Final evaluation on held-out test data
9. Interactive message prediction
10. Model serialization using Joblib

## Model Comparison

| Metric | Naive Bayes (Default) | Logistic Regression |
|---|---:|---:|
| Accuracy | 95.45% | 94.39% |
| Spam Precision | 100.00% | 92.94% |
| Spam Recall | 64.12% | 60.31% |
| Spam F1-Score | 78.14% | 73.15% |

## Final Model Performance

**Selected Model:** Multinomial Naive Bayes  
**Selected Spam Threshold:** 0.20

| Metric | Score |
|---|---:|
| Accuracy | 97.29% |
| Spam Precision | 98.13% |
| Spam Recall | 80.15% |
| Spam F1-Score | 88.24% |

### Confusion Matrix

| Actual / Predicted | Ham | Spam |
|---|---:|---:|
| Ham | 901 | 2 |
| Spam | 26 | 105 |

The model correctly identified 105 spam messages and incorrectly classified 2 legitimate messages as spam.

## Sample Predictions

| Message | Prediction |
|---|---|
| Hey, are we meeting tomorrow at university? | Ham |
| Congratulations! You have won a free prize. Call now to claim your reward! | Spam |
| URGENT! You have won $1000 cash. Click here to claim your free reward! | Spam |

## Saved Model

The trained model and its TF-IDF vectorizer are saved together:

`models/spam_classifier.joblib`

The saved artifact also includes the selected classification threshold.

## How to Run

1. Clone or download this repository.
2. Install the required dependencies.
3. Open `Spam_Classifier.ipynb` in Jupyter Notebook or JupyterLab.
4. Run the notebook cells sequentially.
5. Use the interactive prediction function to test messages.

Install dependencies:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn nltk joblib jupyter
```

## Limitations
- The model was trained on the UCI SMS Spam Collection dataset.
- Performance on newer or different types of messages may vary.
- Spam probability is an estimate, not a guarantee.

## Author
Muhammad Mohsin Khan