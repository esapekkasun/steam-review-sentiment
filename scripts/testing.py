#!/usr/bin/env python

"""
This script is used for testing the model with test data
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

# Loss and optimizer
criterion = nn.BCEWithLogitsLoss()

# Read data
print("Reading data")
data_test = pd.read_csv(filename_test)

# Model testing
print("Testing model")
model.eval()
sum_loss = 0
sum_acc = 0
n_chunks = 0
with torch.inference_mode():
    for c_start in tqdm(range(0, len(data_test), batch_size)):
        c_slice = slice(c_start, c_start + batch_size)
        X, Y = batch_to_tensors(data_test.loc[c_slice, "review"],
                                data_test.loc[c_slice, "voted_up"])
        # Do forward pass
        output = model(X)
        predictions = torch.round(torch.sigmoid(output))
        # Calculate loss and accuracy
        sum_loss += criterion(output, Y)
        sum_acc += accuracy_fn(y_true=Y, y_pred=predictions)
        n_chunks += 1
loss = sum_loss / n_chunks
acc = sum_acc / n_chunks

# Print information
print(f"loss: {loss:.5f}, accuracy: {acc:.2f}")
