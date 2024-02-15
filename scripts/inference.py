#!/usr/bin/env python

"""
This script is used for testing the model inference manually
"""

import torch
from scripts.train_rnn_model import RNN, batch_to_tensors, \
                            embedding_size, hidden_size, num_layers, \
                            device, model_save_path

# Define variables
filename_test = "data/reviews_test.csv"

# Create model
model = RNN(embedding_size, hidden_size, num_layers).to(device)
model.load_state_dict(torch.load(model_save_path))

# Define model pipeline
def model_pipeline(text: str):
    with torch.no_grad():
        X, Y = batch_to_tensors([text], [0])
        
        # Do forward pass and predict
        output = model(X)
        predictions = torch.round(torch.sigmoid(output))
        guess = "Voted up" if predictions[0] else "Voted down"
        return guess

if __name__ == "__main__":

    # Model testing
    while True:
        text = input("Input (write 'quit' to end):")
        if text == "quit":
            break

        # Print text and prediction
        print(f"\n> {text}")
        guess = model_pipeline(text)
        print(guess, "\n")
