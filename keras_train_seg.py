'''
Transfer learning of VGG19 model for segmentation tasks of clouds in LWIR images.
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
# import tensorflow as tf
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image

from utils import *

BATCH_SIZE = 12

# load preprocessed datasets
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

train_dataset_tf = load_tfrecords_set('train').batch(BATCH_SIZE)
val_dataset_tf = load_tfrecords_set('val').batch(BATCH_SIZE)

for x, y in train_dataset_tf.take(1):
    print("Image shape:", x.shape)
    print("Mask shape:", y.shape)


# define model
# pretrained VGG feature extraction base model
inp = Input(shape=(1024, 1024, 3))
vgg_model = VGG19(weights='imagenet', include_top=False, input_tensor=inp)
vgg_model.trainable = False

x = vgg_model(inp, training=False)

# trainable segmentation layers
x = keras.layers.Conv2D(1, (1, 1), activation='sigmoid', name='segmentation_head')(x)
outputs = keras.layers.UpSampling2D(size=(32, 32), interpolation='bilinear')(x)


#--------------------------------
inp = Input(shape=(1024, 1024, 3))
base = VGG19(include_top=False, weights='imagenet', input_tensor=inp)
base.trainable = False  # später teilweise freigeben

x = base.output  # (32x32x512)

# Kompakter Decoder (wenige Filter + BatchNorm)
for filters in [128, 64, 32]:
    x = UpSampling2D()(x)
    x = Conv2D(filters, 3, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)

x = UpSampling2D()(x)  # (256x256)
x = Conv2D(16, 3, padding='same', activation='relu')(x)

x = UpSampling2D(size=(4,4))(x)  # (1024x1024)
output = Conv2D(1, 1, activation='sigmoid')(x)

model = Model(inp, output)
#-----------------------------------



# # add trainable upsampling layers to expected output size
# # skip tensors
# b1, b2 = vgg_model.get_layer("block1_pool").output, vgg_model.get_layer("block2_pool").output
# b3, b4 = vgg_model.get_layer("block3_pool").output, vgg_model.get_layer("block4_pool").output
# b5 = vgg_model.get_layer("block5_pool").output       # 32×32×512

# # decoder (FCN‑8s style)
# d = Conv2D(512, 3, padding="same", activation="relu")(b5)
# d = UpSampling2D()(d)                                # 64×64
# d = Concatenate()([d, b4])
# d = Conv2D(256, 3, padding="same", activation="relu")(d)

# d = UpSampling2D()(d)                                # 128×128
# d = Concatenate()([d, b3])
# d = Conv2D(128, 3, padding="same", activation="relu")(d)

# d = UpSampling2D()(d)                                # 256×256
# d = Concatenate()([d, b2])
# d = Conv2D(64, 3, padding="same", activation="relu")(d)

# d = UpSampling2D()(d)                                # 512×512
# d = Concatenate()([d, b1])
# d = Conv2D(32, 3, padding="same", activation="relu")(d)

# d = UpSampling2D()(d)                                # 1024×1024
# outputs = Conv2D(1, 1, activation="sigmoid")(d)


seg_model = keras.models.Model(inp, outputs)


# train model
# def f1_seg_score(y_true, y_pred, threshold=0.5):
#     '''
#     Flatten output for 4D image segmentation task.
#     '''
#     y_pred = tf.cast(y_pred > threshold, tf.float32)
#     y_true = tf.cast(y_true, tf.float32)

#     y_pred_flat = tf.reshape(y_pred, [-1])
#     y_true_flat = tf.reshape(y_true, [-1])

#     tp = tf.reduce_sum(y_true_flat * y_pred_flat)
#     fp = tf.reduce_sum(y_pred_flat) - tp
#     fn = tf.reduce_sum(y_true_flat) - tp

#     f1 = (2 * tp + 1e-8) / (2 * tp + fp + fn + 1e-8)
#     return f1


seg_model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.01),                                # TODO 0.001
                loss=keras.losses.BinaryCrossentropy(from_logits=False),
                metrics=[
                        keras.metrics.BinaryIoU(target_class_ids=[1],threshold=0.5),                          # TODO change to one class only!
                        f1_seg_score
                ]
)
print(seg_model.summary())

es_cb = EarlyStopping(monitor='val_f1_seg_score', patience=10, verbose=0, mode='max')
save_cb = ModelCheckpoint('./model.keras', save_best_only=True, monitor='val_f1_seg_score', mode='max')
lr_cb = ReduceLROnPlateau(monitor='val_f1_seg_score', factor=0.1, patience=5, verbose=1, epsilon=1e-4, mode='max')
json_log = open('./loss_log.json', mode='wt', buffering=1)
log_cb = keras.callbacks.LambdaCallback(
    on_epoch_end=lambda epoch, logs: json_log.write(json.dumps({'epoch': epoch, 
                                                                'loss': logs['loss'],
                                                                'val_loss': logs['val_loss'], 
                                                                'binary_io_u': logs['binary_io_u'], 
                                                                'val_binary_io_u': logs['val_binary_io_u'], 
                                                                'f1_seg_score': logs['f1_seg_score'], 
                                                                'val_f1_seg_score': logs['val_f1_seg_score']})),
)
try: history_fine = seg_model.fit(train_dataset_tf, 
                                    validation_data=val_dataset_tf, 
                                    callbacks=[es_cb, save_cb, lr_cb, log_cb],
                                    epochs=1)
except KeyboardInterrupt: print("\r\nTraining interrupted")

seg_model.save("./segmentation_model.keras")


# evaluation
loss, accuracy, fscore = seg_model.evaluate(val_dataset_tf)
print('Test binary_io_u & F1 :', accuracy, fscore)


# plot evaluation 
acc = history_fine.history['binary_io_u']
f1 = history_fine.history['f1_seg_score']
val_acc = history_fine.history['val_binary_io_u']
val_f1 = history_fine.history['val_f1_seg_score']
loss = history_fine.history['loss']
val_loss = history_fine.history['val_loss']


df = pd.DataFrame(dict(
    acc = history_fine.history['binary_io_u'],
    f1 = history_fine.history['f1_seg_score'],
    val_acc = history_fine.history['val_binary_io_u'],
    val_f1 = history_fine.history['val_f1_seg_score'],
    loss = history_fine.history['loss'],
    val_loss = history_fine.history['val_loss']
))
fig = px.line(df, title="Accuracy and Loss over Epochs", markers=True)
fig.add_trace(go.Scatter(x=[len(val_loss)+1], y=[fscore], name="Eval F1"))
fig.add_trace(go.Scatter(x=[len(val_loss)+1], y=[accuracy], name="Eval BIoU"))
# fig.show()
fig.write_html("./metrics_loss.html")
