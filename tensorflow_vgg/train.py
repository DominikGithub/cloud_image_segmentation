'''
Transfer learning of VGG19 model for segmentation tasks of clouds in LWIR images.
'''

import json
import time
import pandas as pd
import keras
from keras.layers import Input, Conv2D, UpSampling2D, Concatenate, BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, LambdaCallback, TensorBoard
from keras.applications import VGG19
import plotly.express as px
import plotly.graph_objects as go

from utils import *

TIME_SEC = int(time.time())
BATCH_SIZE = 10

# load preprocessed datasets
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

# trainable decoder layer
x = keras.layers.Conv2D(1, (1, 1), activation='sigmoid', name='segmentation_head')(x)
outputs = keras.layers.UpSampling2D(size=(32, 32), interpolation='bilinear')(x)



# #--------------------------------
# # intermediate sized upscaling model
# x = vgg_model.output  # (32x32x512)

# # compact decoder (fewer filter + BatchNorm)
# x = UpSampling2D()(x)  # 64x64
# x = Conv2D(128, 3, padding='same', activation='relu')(x)
# x = BatchNormalization()(x)

# x = UpSampling2D()(x)  # 128x128
# x = Conv2D(64, 3, padding='same', activation='relu')(x)
# x = BatchNormalization()(x)

# x = UpSampling2D()(x)  # 256x256
# x = Conv2D(32, 3, padding='same', activation='relu')(x)
# x = BatchNormalization()(x)

# x = UpSampling2D()(x)  # 512x512
# x = Conv2D(16, 3, padding='same', activation='relu')(x)

# x = UpSampling2D()(x)  # 1024x1024
# outputs = Conv2D(1, 1, activation='sigmoid')(x)
# #-----------------------------------


# # overfitting
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
seg_model.compile(optimizer=keras.optimizers.Adam(learning_rate=2e-3),
                loss=keras.losses.BinaryCrossentropy(from_logits=False),
                metrics=[
                        keras.metrics.BinaryIoU(target_class_ids=[1],threshold=0.5),
                        f1_seg_score
                ]
)
print(seg_model.summary())

es_cb = EarlyStopping(monitor='val_f1_seg_score', patience=20, verbose=0, mode='max')
save_cb = ModelCheckpoint(f'./model_{TIME_SEC}.keras', save_best_only=True, monitor='val_f1_seg_score', mode='max')
lr_cb = ReduceLROnPlateau(monitor='val_f1_seg_score', factor=0.1, patience=5, verbose=1, min_delta=1e-4, mode='max')
json_log = open(f'./log_{TIME_SEC}.json', mode='wt', buffering=1)
log_cb = LambdaCallback(
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
                                    callbacks=[es_cb, save_cb, lr_cb, log_cb],  #fine_tune_cb
                                    epochs=50)
except KeyboardInterrupt: print("\r\nTraining interrupted")

seg_model.save(f"./segmentation_model_{TIME_SEC}.keras")


# evaluation
loss, accuracy, fscore = seg_model.evaluate(val_dataset_tf)
print('Test binary_io_u & F1 :', accuracy, fscore)


# plot training metrics 
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
fig.write_html(f"./metrics_loss_{TIME_SEC}.html")
