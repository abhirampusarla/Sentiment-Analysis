# 🛍️ Amazon Product Review Sentiment Analysis

This project performs sentiment analysis on Amazon product reviews using Natural Language Processing (NLP) techniques. It leverages the VADER (Valence Aware Dictionary and sEntiment Reasoner) sentiment analyzer from NLTK to quantify the sentiment polarity of customer feedback.

## 📁 Dataset
The dataset used is Reviews.csv, sourced from Amazon product reviews.
For efficiency and demonstration, only the first 500 reviews are considered in this analysis.

## 🔍 Features & Workflow
Data Loading & Preprocessing: Load and preview Amazon reviews, filter the first 500 rows for faster processing.

## Exploratory Analysis:
Visualize distribution of star ratings.
Examine individual reviews and perform tokenization, POS tagging, and named entity recognition (NER).

## Sentiment Analysis:
Apply VADER sentiment analysis to each review.
Extract positive, neutral, negative, and compound sentiment scores.

## Visualizations:
Sentiment distribution by star rating using bar plots.
Compare sentiment scores (positive, neutral, negative) across different review ratings.

## 📊 Sample Insights
Bar charts indicate how sentiment scores align with star ratings.
Compound sentiment scores are generally higher for 4- and 5-star reviews and lower for 1-star reviews, validating the sentiment analysis.

## 🛠️ Tools & Libraries
pandas, numpy – Data handling
matplotlib, seaborn – Data visualization
nltk – Tokenization, POS tagging, named entity recognition, sentiment analysis
tqdm – Progress tracking for loop operations

## 📌 How to Use
### Clone the repository and install required packages:
pip install pandas numpy matplotlib seaborn nltk tqdm
Download the dataset and place Reviews.csv in your working directory.
Run the Jupyter notebook or Python script to explore insights from Amazon customer reviews.

## 📈 Future Enhancements
Analyze a larger sample or full dataset.
Add classification models to predict star ratings based on text.
Perform time-based sentiment trend analysis.
