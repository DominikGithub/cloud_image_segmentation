'''
Make test predictions loading the trained model.
'''

import os
import glob
import json
from pathlib import Path
from tqdm import tqdm
import  multiprocessing as mp
import numpy as np
import pandas as pd
import keras
from keras.layers import Input, Conv2D, UpSampling2D, Concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.applications import VGG19
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from PIL import Image

from utils import *

BATCH_SIZE = 12

TIME_SEC = 0                # TODO set to load the right model

# load dataset
val_dataset_tf = load_tfrecords_set('val').batch(BATCH_SIZE)
for x, y in val_dataset_tf.take(1):
    print("Image shape:", x.shape)
    print("Mask shape:", y.shape)


# load trained model
seg_model = keras.saving.load_model("model.keras", custom_objects={'f1_seg_score': f1_seg_score})
seg_model.compile()


test_time_path = f'./tests/{TIME_SEC}'
if not os.path.exists(test_time_path):
    os.makedirs(test_time_path)


# make test prediction and plot
for x, y_true in val_dataset_tf.take(1):
    eval_seg_mask_lst = seg_model.predict(x)
    # plot file per test sample
    for i, y_pred in enumerate(eval_seg_mask_lst):
        eval_fig = make_subplots(rows=2, cols=1,  vertical_spacing=0.05, subplot_titles=['Prediction', 'Ground truth'])
        # Prediction 
        pred_fig = px.imshow(y_pred[:, :, 0], color_continuous_scale='gray', binary_string=False)
        pred_trace = pred_fig.data[0]
        pred_trace.update(coloraxis='coloraxis')
        eval_fig.add_trace(pred_trace, row=1, col=1)
        # Ground truth
        gt_fig = px.imshow(y_true[i, :, :], color_continuous_scale='gray', binary_string=False)
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
