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

BATCH_SIZE = 4


# load dataset
def parse_proto(example_proto):
    '''
    Parse single sample pair from TF sample format.
    '''
    feat_shp = [1024, 1024, 3]
    targ_shp = [1024, 1024]
    feature_dict = {
        'X': tf.io.FixedLenSequenceFeature(feat_shp, tf.float32, allow_missing=True, default_value=[0.0]),
        'y': tf.io.FixedLenSequenceFeature(targ_shp, tf.int64, allow_missing=True, default_value=[0]),
    }
    parsed_features = tf.io.parse_single_example(example_proto, feature_dict)
    feat  = tf.cast(parsed_features['X'], tf.float32)
    label = tf.cast(parsed_features['y'], tf.int64)
    feat = tf.reshape(feat, feat_shp)
    feat.set_shape(feat_shp)
    
    label = tf.reshape(label, targ_shp)
    label.set_shape(targ_shp)
    return feat, label


def load_tfrecords_set(set_name='dummy'):
    '''
    Load dataset from serialized TFRecord files.
    '''
    tfrecord_train_files = glob.glob(f'./tfdataset/{set_name}/*.tfrecords')
    dataset_tf = tf.data.TFRecordDataset(tfrecord_train_files, compression_type='ZLIB', num_parallel_reads=os.cpu_count())
    dataset_tf = dataset_tf.map(parse_proto, num_parallel_calls=tf.data.AUTOTUNE)
    return dataset_tf


val_dataset_tf = load_tfrecords_set('val').batch(BATCH_SIZE)
for x, y in val_dataset_tf.take(1):
    print("Image shape:", x.shape)
    print("Mask shape:", y.shape)


# load trained model
seg_model = keras.saving.load_model("model.keras", custom_objects={'f1_seg_score': f1_seg_score})
seg_model.compile()

# make test prediction
for x, y_true in val_dataset_tf.take(1):
    eval_seg_mask_lst = seg_model.predict(x)
    # plot file per test sample
    for i, y_pred in enumerate(eval_seg_mask_lst):
        eval_fig = make_subplots(rows=2, cols=1, subplot_titles = ['Prediction', 'Ground truth'])
        # prediction
        eval_fig.add_trace(px.imshow(y_pred[:,:,0]).data[0], row=1, col=1)
        # ground truth
        eval_fig.add_trace(px.imshow(y_true[i,:,:]).data[0], row=2, col=1)
        eval_fig.update_layout(
            autosize=False,
            width=1000,
            height=1600,
        )
        # eval_fig.write_html(f'./tests/eval_mask_{i}.html')
        eval_fig.write_image(f"./tests/eval_mask_{i}.png")
