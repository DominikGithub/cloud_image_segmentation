'''
Evaluation of model predictions on validation set.
'''

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM


# Load model
file_path = './segformer_cloud.pth'
model = AutoModelForCausalLM.from_pretrained("./segformer_cloud.pt")
# model.load_state_dict(torch.load(file_path))
model.eval()