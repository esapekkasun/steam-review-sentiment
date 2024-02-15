#!/usr/bin/env python

""" 
This script trains the RNN model and stores it into a file
"""

import torch
import torch.nn as nn
import pandas as pd
from torchtext.data import get_tokenizer
from torchtext.vocab import vocab
from collections import OrderedDict
from tqdm import tqdm

# Define variables
tokens_filename = "data/tokens_list_10k.csv"
filename_train = "data/reviews_train.csv"
filename_validation = "data/reviews_validation.csv"
model_save_path = "model/rnn_sentiment_model.pth"

# Initialize pytorch device and tokenizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = get_tokenizer("basic_english")

# Initialize vocabulary
tokens = pd.read_csv(tokens_filename).drop("Unnamed: 0", axis=1)
tokens_dict = tokens.set_index("token")["frequency"].to_dict(OrderedDict)
vocabulary = vocab(tokens_dict, specials=["<UNKNOWN>"])
vocabulary.set_default_index(vocabulary["<UNKNOWN>"])

# Model Hyper-parameters
max_words = 25
batch_size = 10000
embedding_size = 50
learning_rate = 0.001
hidden_size = 50
num_layers = 1

def batch_to_tensors(reviews, sentiments):
    """
    Create tensors from batch of data
    """
    X = [vocabulary(tokenizer(text)) if type(text) == str else [0] * max_words for text in reviews]
    X = [tokens + ([0] * (max_words-len(tokens))) if len(tokens) < max_words else tokens[:max_words] for tokens in X]
    Y = [[s] for s in sentiments]
    return torch.tensor(X, dtype=torch.int32, device=device), torch.tensor(Y, dtype=torch.float, device=device)

def accuracy_fn(y_true, y_pred):
    """
    Calculate accuracy (a classification metric)
    """
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100 
    return acc

class RNN(nn.Module):
    """
    Recurrent neural network with one hidden layer
    """

    def __init__(self, embedding_size, hidden_size, num_layers):
        super(RNN, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(len(vocabulary), embedding_size)
        self.rnn = nn.RNN(embedding_size, hidden_size, num_layers,
                          batch_first=True, nonlinearity="relu")
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, X_batch):
        # Set initial hidden states
        h0 = torch.zeros(self.num_layers, len(X_batch), self.hidden_size).to(device)

        # Calculate embeddings
        embeddings = self.embedding(X_batch)

        # Forward propagate RNN
        output, hidden = self.rnn(embeddings, h0)

        # Decode the hidden state of the last step
        return self.linear(output[:,-1])

if __name__ == "__main__":

    # Create model
    model = RNN(embedding_size, hidden_size, num_layers).to(device)

    # Loss and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    def train(X, Y):
        """
        Training with a single batch
        """
        # Do forward pass
        output = model(X)
        predictions = torch.round(torch.sigmoid(output))
        # Calculate loss
        loss = criterion(output, Y)
        # Reset gradients and do backward propagation
        optimizer.zero_grad()
        loss.backward()
        # Update weights
        optimizer.step()
        return output, loss.item(), predictions

    # Read data
    print("Reading data")
    data_train = pd.read_csv(filename_train)
    data_validation = pd.read_csv(filename_validation)

    # Train model
    print("Training model")
    n_iters = 10
    for i in range(n_iters):

        # Training
        model.train()
        sum_loss = 0
        sum_acc = 0
        n_chunks = 0
        for c_start in tqdm(range(0, len(data_train), batch_size)):
            c_slice = slice(c_start, c_start + batch_size)
            X, Y = batch_to_tensors(data_train.loc[c_slice, "review"],
                                    data_train.loc[c_slice, "voted_up"])
            output, loss, predictions = train(X, Y)
            sum_loss += loss
            sum_acc += accuracy_fn(y_true=Y, y_pred=predictions)
            n_chunks += 1
        train_loss = sum_loss / n_chunks
        train_acc = sum_acc / n_chunks

        # Validation            
        model.eval()
        sum_loss = 0
        sum_acc = 0
        n_chunks = 0
        with torch.inference_mode():
            for c_start in tqdm(range(0, len(data_validation), batch_size)):
                c_slice = slice(c_start, c_start + batch_size)
                X, Y = batch_to_tensors(data_validation.loc[c_slice, "review"],
                                        data_validation.loc[c_slice, "voted_up"])
                # Do forward pass
                output = model(X)
                predictions = torch.round(torch.sigmoid(output))
                # Calculate loss and accuracy
                sum_loss += criterion(output, Y)
                sum_acc += accuracy_fn(y_true=Y, y_pred=predictions)
                n_chunks += 1
        validation_loss = sum_loss / n_chunks
        validation_acc = sum_acc / n_chunks

        # Print information
        print(f"{(i+1)/n_iters*100:2.0f} % done | loss: {train_loss:.5f}, accuracy: {train_acc:.2f}% | val loss: {validation_loss:.5f}, val acc: {validation_acc:.2f}%")

    # Save the model state dict
    print(f"Saving model to: {model_save_path}")
    torch.save(obj=model.state_dict(), f=model_save_path)