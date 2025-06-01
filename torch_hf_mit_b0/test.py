'''
Make test prediction using the fine tuned model.
'''

from Dataloader import CloudSegDataloader
from train import SegformerWithUpsample

import os
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from transformers import SegformerForSemanticSegmentation, SegformerFeatureExtractor
from transformers.modeling_outputs import SemanticSegmenterOutput
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback, TrainerCallback
from visualization import visualize_dataset_samples
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.nn as nn
import torch
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from PIL import Image
import time

TIME_SEC = 0                 # NOTE set according to training scripts model timestamp to be loaded
EVAL_BATCH_SIZE = 10


def plot_testset():
    '''
    Load last trained model and plot validation set sample prediction masks vs true masks. 
    '''
    # hf segformer data preprocessor
    preprocessor = SegformerFeatureExtractor.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512", use_fast=False)
    preprocessor.do_resize = False
    preprocessor.do_rescale = False
    preprocessor.do_normalize = True  

    # load data
    val_ds = CloudSegDataloader('validation', preprocessor)

    # load model
    file_path = f'./segformer_cloud_{TIME_SEC}.pth'
    base_model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/segformer-b0-finetuned-ade-512-512",
        num_labels=1,
        ignore_mismatched_sizes=True
    )
    model = SegformerWithUpsample(base_model)
    model.load_state_dict(torch.load(file_path))
    model.eval()
    # freeze model params for inference
    for param in model.parameters():
        param.requires_grad = False


    ## test inference on validation dataset
    test_time_path = f'./tests/{TIME_SEC}'
    if not os.path.exists(test_time_path):
        os.makedirs(test_time_path)


    # make test prediction and plot
    loader = DataLoader(val_ds, batch_size=EVAL_BATCH_SIZE)
    map = next(iter(loader))
    X = map['pixel_values']
    y_true = (map['labels'].cpu().numpy() * 255).astype(np.uint8)

    with torch.no_grad():
        y_pred = model(X)
        logits = y_pred.logits 
        preds = logits * 255 # torch.sigmoid(logits) > 0.5                 # NOTE discrete?
        preds_np_lst = preds.detach().cpu().numpy()

    # plot prediction vs true mask
    for i in range(len(preds_np_lst)):
        eval_fig = make_subplots(rows=2, cols=1,  vertical_spacing=0.05, subplot_titles=['Prediction', 'Ground truth'])
        # Prediction 
        pred_fig = px.imshow(preds_np_lst[i, 0, :, :], color_continuous_scale='gray', binary_string=False)
        pred_trace = pred_fig.data[0]
        pred_trace.update(coloraxis='coloraxis')
        eval_fig.add_trace(pred_trace, row=1, col=1)
        # Ground truth
        gt_img = y_true[i, 0]   # (y_true[i, 0].cpu().numpy() * 255).astype(np.uint8)
        gt_fig = px.imshow(gt_img, color_continuous_scale='gray', binary_string=False)
        gt_trace = gt_fig.data[0]
        gt_trace.update(coloraxis='coloraxis')
        eval_fig.add_trace(gt_trace, row=2, col=1)
        # style
        eval_fig.update_layout(
            autosize=False,
            coloraxis=dict(colorscale='gray'),
            coloraxis_showscale=True,
            width=1000,
            height=1600,
            annotations=[
                dict(text='Prediction', x=0.5, xref='paper', y=1, yref='paper',
                    xanchor='center', yanchor='bottom',
                    showarrow=False, font=dict(size=20)),
                dict(text='Ground truth', x=0.5, xref='paper', y=0.5, yref='paper',
                    xanchor='center', yanchor='bottom',
                    showarrow=False, font=dict(size=20)),
            ]
        )
        eval_fig.write_html(f'{test_time_path}/eval_mask_{i}.html')
        eval_fig.write_image(f"{test_time_path}/eval_mask_{i}.png")


if __name__ == "__main__":
    plot_testset()