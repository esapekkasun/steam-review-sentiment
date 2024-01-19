#!/usr/bin/env python

"""
This script reads the dataset and processes it in the following ways:
 - filter to leave only english language reviews
 - exclude some voted_up data to balance voted_up and voted_down values
 - split data into training, validation, and test sets
"""

import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# Define variables
filename = "../data/all_reviews/all_reviews.csv"
filename_output = "../data/all_reviews/all_reviews_english_balanced.csv"
chunksize = 10 ** 4
data_size = 113883717
random_state = 11
filename_train = "../data/reviews_train.csv"
filename_validation = "../data/reviews_validation.csv"
filename_test = "../data/reviews_test.csv"

def process_data():
    """
    Process the data to only contain english language
    and balance voted_up and voted_down values  
    """
    print("Generating filtered and balanced dataset")
    with pd.read_csv(filename, chunksize=chunksize) as reader:
        first_chunk = True
        for chunk in tqdm(reader, total=int(data_size / chunksize)):
            chunk_eng = chunk[chunk["language"] == "english"][["review", "voted_up"]]
            len_down = len(chunk_eng[chunk_eng["voted_up"] == 0])
            chunk_eng_bal = chunk_eng.sort_values("voted_up")[:len_down * 2]
            if first_chunk:
                chunk_eng_bal.to_csv(filename_output)
                first_chunk = False
            else:
                chunk_eng_bal.to_csv(filename_output, mode='a', header=False) 

def split_data():
    """
    Read filtered and balanced data and split into training, validation,
    and test sets  
    """
    print("Loading data")
    data = pd.read_csv(filename_output)
    print("Splitting training data")
    train, rest = train_test_split(data, test_size=0.4, random_state=random_state)
    train.to_csv(filename_train)
    del data, train
    print("Splitting validation and test data")
    validation, test = train_test_split(rest, test_size=0.5, random_state=random_state)
    validation.to_csv(filename_validation)
    test.to_csv(filename_test)

if __name__ == "__main__":
    process_data()
    split_data()
    print("DONE")