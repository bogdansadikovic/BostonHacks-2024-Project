import torch
import numpy as np
import torchvision
import os
import sys
import streamlit

# Load the model(Once it is finished):
def load_model(model_path):
    model = torch.load(model_path)
    model.eval()
    return model




