#!/usr/bin/env python

""" 
This script runs the model API
"""

from fastapi import FastAPI
from scripts.inference import model_pipeline

# Initialise app
app = FastAPI()

# Implement GET response
@app.get("/")
def read_root():
    return {"message": "post text to /review"}

# Implement POST response
@app.post("/review")
def ask(text: str):
    sentiment = model_pipeline(text)
    return {"sentiment": sentiment}