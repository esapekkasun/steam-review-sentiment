#!/usr/bin/env python

"""
This script is used for testing the model inference manually
"""

import pandas as pd
import torch
from torch import nn
from tqdm import tqdm
from train_rnn_model import RNN, batch_to_tensors, accuracy_fn,\
                            embedding_size, hidden_size, num_layers, \
                            device, batch_size, model_save_path

# Define variables
filename_test = "../data/reviews_test.csv"

# Create model
model = RNN(embedding_size, hidden_size, num_layers).to(device)
model.load_state_dict(torch.load(model_save_path))

# Model testing
while True:
    sentence = input("Input (write 'quit' to end):")
    if sentence == "quit":
        break
    
    print(f"\n> {sentence}")
    with torch.no_grad():
        X, Y = batch_to_tensors([sentence], [0])
        
        # Do forward pass and predict
        output = model(X)
        predictions = torch.round(torch.sigmoid(output))

        # Print prediction
        guess = "Voted up" if predictions[0] else "Voted down"
        print(guess, "\n")