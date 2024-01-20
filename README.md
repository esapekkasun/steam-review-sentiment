# Sentiment analysis from steam reviews

This project implements a machine learning model for predicting sentiment (voted up/down) from steam reviews text.

**notebooks** folder contains Jupyter Notebook files that are used for data exploration and wrangling, and experimentation with model generation.

**data** folder contains the data and word dictionary used in the model training.

**scripts** folder contains Python scripts for processing data, generating dictionary, and training the model.

## Data description

The data is downloaded from https://www.kaggle.com/datasets/kieranpoc/steam-reviews, and not provided in this repository. If you want to try to run the code in this repository, you need to download the data and extract it in the **data** folder.

It contains more than 100 million steam reviews. The dataset contains texts in many languages, but only english language reviews are used in the model.

## Model description

The model is implemented as a Recurrent Neural Network, which processes the review texts token by token (word by word).

## Data processing and model generation

1. Download data and extract to **data** folder.
2. Execute script **build_dataset.py** in **scripts** folder, this generates filtered and splitted datasets in the **data** folder.
3. Execute script **build_dictionary.py**, this generates files of most used tokens in the reviews in the **data** folder.
4. Execute script **train_rnn_model.py**, this trains the model and saves model weights to a file.