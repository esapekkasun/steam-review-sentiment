#!/usr/bin/env python

""" 
This script generates dictionaries of 1k and 10k most used words in the data
"""

import pandas as pd
from tqdm import tqdm
from torchtext.data import get_tokenizer

# Define variables
filename = "data/all_reviews/all_reviews_english_balanced.csv"
chunksize = 10 ** 4
filename_1k = "data/tokens_list_1k.csv"
filename_10k = "data/tokens_list_10k.csv"

def append_with_frequency(dict, tokens):
    for t in tokens:
        if t in dict.keys():
            dict[t] += 1
        else:
            dict[t] = 1

def generate_dictionary():
    """
    Read all data to create dict of token frequency
    """

    print("Loading data")
    data = pd.read_csv(filename)
    
    print("Generating dictionary")
    tokenizer = get_tokenizer("basic_english")
    tokens = {}
    for r in tqdm(data["review"]):
        try:
            t = tokenizer(r.lower())
            append_with_frequency(tokens, t)
        except:
            pass
    
    print("Save 10k and 1k most frequent tokens into files")
    tokens_sorted = sorted(tokens.items(), key=lambda item: item[1], reverse=True)
    tokens_list_1k = pd.DataFrame(tokens_sorted[:1000], columns=["token", "frequency"])
    tokens_list_10k = pd.DataFrame(tokens_sorted[:10000], columns=["token", "frequency"])
    tokens_list_1k.to_csv(filename_1k)
    tokens_list_10k.to_csv(filename_10k)

if __name__ == "__main__":
    generate_dictionary()
    print("DONE")