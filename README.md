# Sentiment analysis from steam reviews

This project implements a machine learning model for predicting sentiment (voted up/down) from steam reviews text.

**notebooks** folder contains Jupyter Notebook files that are used for data exploration and wrangling, and experimentation with model generation.

**data** folder contains the data and word dictionary used in the model training.

## Data

The data is downloaded from https://www.kaggle.com/datasets/kieranpoc/steam-reviews, and not provided in this repository. If you want to try to run the code in this repository, you need to download the data and extract it in the data folder.

It contains more than 100 million steam reviews. The dataset contains texts in many languages, but only english language reviews are used in the model.

## Model

The model is implemented as a Recurrent Neural Network, which processes the review texts token by token (word by word).