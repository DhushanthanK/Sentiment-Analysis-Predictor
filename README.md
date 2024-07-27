# Sentiment-Based-Binary-Classification-Model-for-Customer-Reviews 

## Overview
This repository contains a comprehensive sentiment analysis project that classifies restaurant reviews as positive or negative. The project involves data preprocessing, model training, and sentiment prediction using a Bag of Words (BoW) model and a Naive Bayes classifier. Additionally, it demonstrates how to apply the trained model to a fresh dataset of reviews to predict sentiments.

## Project Structure
- `a1_RestaurantReviews_HistoricDump.tsv`: Historic dataset used for training the sentiment analysis model.
- `a2_RestaurantReviews_FreshDump.tsv`: Fresh dataset used for predicting sentiments.
- `b1_Sentiment_Analysis_Model.ipynb`: Jupyter notebook containing the code for training the sentiment analysis model.
- `b2_Sentiment_Predictor.ipynb`: Jupyter notebook containing the code for predicting sentiments on a fresh dataset.
- `c1_BoW_Sentiment_Model.pkl`: Pickled Bag of Words (BoW) dictionary.
- `c2_Classifier_Sentiment_Model`: Pickled Naive Bayes classifier model.
- `c3_Predicted_Sentiments_Fresh_Dump.tsv`: TSV file with predicted sentiments for the fresh dataset.
- `d_Business_Deck_Sentiment_Analysis.pdf`: Business deck summarizing the sentiment analysis results.

## Requirements
- Python 3.x
- NumPy
- Pandas
- NLTK
- Scikit-learn
- Pickle
- Joblib

## Installation
1. Clone the repository:
    ```bash
    git clone https://github.com/your-username/Sentiment-Analysis-Predictor.git
    ```
2. Navigate to the project directory:
    ```bash
    cd Sentiment-Analysis-Predictor
    ```
3. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Usage
1. **Training the Model:**
    - Open `b1_Sentiment_Analysis_Model.ipynb` in Jupyter Notebook.
    - Run the notebook cells to preprocess the data, train the Naive Bayes model, and save the BoW dictionary and classifier.

2. **Predicting Sentiments:**
    - Open `b2_Sentiment_Predictor.ipynb` in Jupyter Notebook.
    - Run the notebook cells to preprocess the fresh dataset, load the saved BoW dictionary and classifier, predict sentiments, and save the results.

## Results
The results of the sentiment predictions on the fresh dataset can be found in `c3_Predicted_Sentiments_Fresh_Dump.tsv`.

## Contributing
If you would like to contribute to this project, please fork the repository and submit a pull request.

## Contact
For any questions or feedback, please contact [dhushanthankumararatnam@gmail.com](mailto:dhushanthankumararatnam@gmail.com]).
