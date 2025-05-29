'''
Transfer learning of VGG19 for segmentation of clouds in LWIR images.
'''

import glob
from pathlib import Path
from tqdm import tqdm
import numpy as np
import pandas as pd
import keras
from keras.layers import Input, Conv2D, UpSampling2D, Concatenate
from keras.applications import VGG19
import tensorflow as tf
import plotly.express as px
from PIL import Image

# load datasets
train_dir = "./dataset_clouds_from_lwir/training/"
val_dir = "./dataset_clouds_from_lwir/validation/"

BATCH_SIZE = 16

N_TEST_SAMPLES = 200                                             # TODO remove filter

def create_dataset_from_image_path(path, set_name='dummy'):
    '''
    Load TIFF images as TF datasets.
    '''
    img_mask_lst  = glob.glob(path+"clouds/*.tif")
    img_cloud_lst = glob.glob(path+"lwir/*.tif")

    # cloud images
    serialized_img_npy_path = f'./image_{set_name}.npy'
    if Path(serialized_img_npy_path).is_file():
        with open(serialized_img_npy_path, 'rb') as f:
            cloud_arr = np.load(f)
    else:
        cloud_arr = np.ndarray((0, 1024, 1024, 3))
        for img in tqdm(img_cloud_lst):   # [:N_TEST_SAMPLES]                                                  # TODO remove filter
            image = keras.utils.load_img(img, target_size=(1024, 1024, 3))
            input_arr = keras.utils.img_to_array(image)
            input_arr = tf.keras.applications.vgg19.preprocess_input(input_arr)
            input_arr = np.array([input_arr])
            cloud_arr = np.concatenate((cloud_arr, input_arr), axis=0)
        
        # persist numpy array to file
        with open(serialized_img_npy_path, 'wb') as f:
            np.save(f, cloud_arr)
        
    # segmentation mask
    serialized_mask_npy_path = f'./mask_{set_name}.npy'
    if Path(serialized_mask_npy_path).is_file():
        with open(serialized_mask_npy_path, 'rb') as f:
            mask_arr = np.load(f)
    else:
        mask_arr = np.ndarray((0, 1024, 1024))
        for mask in tqdm(img_mask_lst):   # [:N_TEST_SAMPLES]                                                 # TODO remove filter   
            msk_arr = Image.open(mask)
            msk_arr = np.array([msk_arr]) / 255.0
            mask_arr = np.concatenate((mask_arr, msk_arr), axis=0)
        
        # persist numpy array to file
        with open(serialized_mask_npy_path, 'wb') as f:
            np.save(f, mask_arr)

    print(path, cloud_arr.shape, mask_arr.shape)
    features_dataset = tf.data.Dataset.from_tensor_slices(cloud_arr)
    labels_dataset = tf.data.Dataset.from_tensor_slices(mask_arr)
    return tf.data.Dataset.zip((features_dataset, labels_dataset))

train_dataset_tf = create_dataset_from_image_path(train_dir, 'train').batch(BATCH_SIZE)
val_dataset_tf = create_dataset_from_image_path(val_dir, 'val').batch(BATCH_SIZE)


exit(0)


# pretrained VGG19
inp = Input(shape=(1024, 1024, 3))
vgg_model = VGG19(weights='imagenet', include_top=False, input_tensor=inp)
vgg_model.trainable = False

x = vgg_model(inp, training=False)

# trainable segmentation layers
x = keras.layers.Conv2D(1, (1, 1), activation='sigmoid', name='segmentation_head')(x)
outputs = keras.layers.UpSampling2D(size=(32, 32), interpolation='bilinear')(x)


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


def f1_score_segmentation(y_true, y_pred, threshold=0.5):
    '''
    Flatten output for 4D image segmentation task.
    '''
    y_pred = tf.cast(y_pred > threshold, tf.float32)
    y_true = tf.cast(y_true, tf.float32)

    y_pred_flat = tf.reshape(y_pred, [-1])
    y_true_flat = tf.reshape(y_true, [-1])

    tp = tf.reduce_sum(y_true_flat * y_pred_flat)
    fp = tf.reduce_sum(y_pred_flat) - tp
    fn = tf.reduce_sum(y_true_flat) - tp

    f1 = (2 * tp + 1e-8) / (2 * tp + fp + fn + 1e-8)
    return f1


seg_model.compile(optimizer=keras.optimizers.Adam(),
                loss=keras.losses.BinaryCrossentropy(from_logits=False),
                metrics=[
                        keras.metrics.BinaryIoU(target_class_ids=[0,1],threshold=0.5), 
                        f1_score_segmentation
                ]
)
print(seg_model.summary())
try: history_fine = seg_model.fit(train_dataset_tf, validation_data=val_dataset_tf, epochs=4)
except KeyboardInterrupt: print("\r\nTraining interrupted")

seg_model.save("./segmentation_model.keras")


# evaluation
loss, accuracy, fscore = seg_model.evaluate(val_dataset_tf)
print('Test binary_io_u/F1 :', accuracy, fscore)

print(history_fine.history)

acc = history_fine.history['binary_io_u']
f1 = history_fine.history['f1_score_segmentation']
val_acc = history_fine.history['val_binary_io_u']
val_f1 = history_fine.history['val_f1_score_segmentation']
loss = history_fine.history['loss']
val_loss = history_fine.history['val_loss']



# plot evaluation 
df = pd.DataFrame(dict(
    acc = history_fine.history['binary_io_u'],
    f1 = history_fine.history['f1_score_segmentation'],
    val_acc = history_fine.history['val_binary_io_u'],
    val_f1 = history_fine.history['val_f1_score_segmentation'],
    loss = history_fine.history['loss'],
    val_loss = history_fine.history['val_loss']
))
fig = px.line(df, title="Accuracy and Loss over Epochs", markers=True)
fig.show()
fig.write_html("./metrics_loss.html")
