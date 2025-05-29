'''
Transfer learning of VGG19 for segmentation of clouds in LWIR images.
'''

import glob
from tqdm import tqdm
import numpy as np
import pandas as pd
import keras
from keras.layers import Input, Conv2D, UpSampling2D, Concatenate
from keras.applications import VGG19
import tensorflow as tf
# import matplotlib.pyplot as plt
import plotly.express as px
from PIL import Image

# load datasets
train_dir = "./dataset_clouds_from_lwir/training/"
val_dir = "./dataset_clouds_from_lwir/validation/"

N_TEST_SAMPLES = 50                                             # TODO remove filter

def create_dataset_from_image_path(path):
    '''
    Load TIFF images as TF datasets.
    '''
    img_mask_lst  = glob.glob(path+"clouds/*.tif")
    img_cloud_lst = glob.glob(path+"lwir/*.tif")

    cloud_arr = np.ndarray((0, 1024, 1024, 3))
    mask_arr = np.ndarray((0, 1024, 1024))
    # image
    for img in tqdm(img_cloud_lst[:N_TEST_SAMPLES]):                                                     # TODO remove filter
        image = keras.utils.load_img(img, target_size=(1024, 1024, 3))
        input_arr = keras.utils.img_to_array(image)

        input_arr = tf.keras.applications.vgg19.preprocess_input(
            input_arr#, data_format='channels_last'                 # TODO channel last??
        )
        input_arr = np.array([input_arr])
        cloud_arr = np.concatenate((cloud_arr, input_arr), axis=0)
    # segmentation mask
    for mask in tqdm(img_mask_lst[:N_TEST_SAMPLES]):                                                    # TODO remove filter   
        # msk_arr = keras.utils.load_img(mask, target_size=(1024,1024), color_mode='grayscale')
        # msk_arr = keras.utils.img_to_array(msk_arr) #/ 255.0
        msk_arr = Image.open(mask)

        msk_arr = np.array([msk_arr]) / 255.0
        mask_arr = np.concatenate((mask_arr, msk_arr), axis=0)
    
    print(path, cloud_arr.shape, mask_arr.shape)
    features_dataset = tf.data.Dataset.from_tensor_slices(cloud_arr)
    labels_dataset = tf.data.Dataset.from_tensor_slices(mask_arr)
    return tf.data.Dataset.zip((features_dataset, labels_dataset))

train_dataset_tf = create_dataset_from_image_path(train_dir).batch(16)

val_dataset_tf = create_dataset_from_image_path(val_dir).batch(16)


# transfer learning pretrained VGG19 model
inp = Input(shape=(1024, 1024, 3))

# pretrained fixed layers
vgg_model = VGG19(weights='imagenet', include_top=False, input_tensor=inp)
vgg_model.trainable = False

x = vgg_model(inp, training=False)

# trainable segmentation layers
x = keras.layers.Conv2D(1, (1, 1), activation='sigmoid', name='segmentation_head')(x)
outputs = keras.layers.UpSampling2D(size=(32, 32), interpolation='bilinear')(x)

seg_model = keras.models.Model(inp, outputs)

seg_model.compile(optimizer=keras.optimizers.Adam(),
                loss=keras.losses.BinaryCrossentropy(from_logits=False),
                metrics=[
                        keras.metrics.BinaryIoU(target_class_ids=[0,1],threshold=0.5)
                        # keras.metrics.BinaryAccuracy()
                ]
)
print(seg_model.summary())
try: history_fine = seg_model.fit(train_dataset_tf, validation_data=val_dataset_tf, epochs=10, verbose=True)
except KeyboardInterrupt: print("\r\nTraining interrupted")



# evaluation
loss, accuracy = seg_model.evaluate(val_dataset_tf)
print('Test binary_io_u :', accuracy)

print(history_fine.history)

acc = history_fine.history['binary_io_u']
val_acc = history_fine.history['val_binary_io_u']

loss = history_fine.history['loss']
val_loss = history_fine.history['val_loss']



# plot evaluation 
df = pd.DataFrame(dict(
    acc = history_fine.history['binary_io_u'],
    val_acc = history_fine.history['val_binary_io_u'],
    loss = history_fine.history['loss'],
    val_loss = history_fine.history['val_loss']
))
fig = px.line(df, title="Accuracy and Loss over Epochs", markers=True)
fig.show()
