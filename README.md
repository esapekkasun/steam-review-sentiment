# Sentiment analysis from steam reviews

This project implements a machine learning model for predicting sentiment (voted up/down) from steam reviews text.

**data** folder contains the data and word dictionary used in the model training.

**model** folder contains the trained weights of the RNN model.

**notebooks** folder contains Jupyter Notebook files that are used for data exploration and wrangling, and experimentation with model generation.

**scripts** folder contains Python scripts for processing data, generating dictionary, and training the model.

## Data description

The data is downloaded from https://www.kaggle.com/datasets/kieranpoc/steam-reviews, and not provided in this repository. If you want to try to run the code in this repository, you need to download the data and extract it in the **data** folder.

The dataset contains more than 100 million steam reviews. It contains texts in many languages, but only english language reviews are used in the model.

## Model description

The model is implemented as a Recurrent Neural Network (RNN) with pytorch and torchtext libraries. The RNN processes the review texts token by token (word by word) and outputs a single value for classifying the text sentiment (voted up/down).

## Data processing and model generation

1. Download data and extract to **data** folder.
2. Execute script **build_dataset.py** in **scripts** folder, this generates filtered and splitted datasets in the **data** folder.
3. Execute script **build_dictionary.py**, this generates files of most used tokens in the reviews in the **data** folder.
4. Execute script **train_rnn_model.py**, this trains the model and saves model weights to a file.

![image](assets/training.png)

5. Execute script **testing.py**, this runs inference on the model with testing data and displays accuracy.

![image](assets/testing.png)

## Model usage

The model can be tested with running the script **inference.py** in **scripts** folder.

![image](assets/inference.png)

## Todo

Further assess model performance with confusion matrix and other metrics.

Implement API for using the model and deploy model for public access.

Create a website that uses the API to show outputs from the model.

## Acknowledgements and resources

<a href="https://www.learnpytorch.io">Learn PyTorch for Deep Learning</a>

<a href="https://coderzcolumn.com">CoderzColumn</a>

<a href="http://www.freepik.com">Preview picture designed by vectorjuice / Freepik</a>