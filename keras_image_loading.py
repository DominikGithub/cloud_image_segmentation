'''
Transfer learning of VGG19 for segmentation of clouds in LWIR images.
'''

import glob
from tqdm import tqdm
import numpy as np
import keras
from keras.layers import Input
from keras.applications import VGG19
import tensorflow as tf
import matplotlib.pyplot as plt

# create cross validation TF datasets
train_dir = "./dataset_clouds_from_lwir/training/"
val_dir = "./dataset_clouds_from_lwir/validation/"

def create_dataset_from_image_path(path):
    '''
    Load TIFF images as TF datasets.
    '''
    img_cloud_lst= glob.glob(path+"clouds/*.tif")
    img_mask_lst = glob.glob(path+"lwir/*.tif")

    cloud_arr = np.ndarray((0, 1024, 1024, 3))
    mask_arr = np.ndarray((0, 1024, 1024, 1))
    # cloud image
    for img in tqdm(img_cloud_lst[:100]):                                                     # TODO remove filter
        image = keras.utils.load_img(img, target_size=(1024, 1024, 3))
        input_arr = keras.utils.img_to_array(image)

        input_arr = tf.keras.applications.vgg19.preprocess_input(
            input_arr, data_format='channels_last'
        )
        
        input_arr = np.array([input_arr])
        cloud_arr = np.concatenate((cloud_arr, input_arr), axis=0)
    # segmentation mask
    for mask in tqdm(img_mask_lst[:100]):                                                     # TODO remove filter   
        image = keras.utils.load_img(mask, color_mode='grayscale', target_size=(1024,1024))
        msk_arr = keras.utils.img_to_array(image) / 255.0
        msk_arr = np.array([msk_arr])
        mask_arr = np.concatenate((mask_arr, msk_arr), axis=0)
    
    print(path, cloud_arr.shape, mask_arr.shape)
    # return tf.data.Dataset.from_tensor_slices((cloud_arr, mask_arr))
    features_dataset = tf.data.Dataset.from_tensor_slices(cloud_arr)
    labels_dataset = tf.data.Dataset.from_tensor_slices(mask_arr)
    return tf.data.Dataset.zip((features_dataset, labels_dataset))

train_dataset_tf = create_dataset_from_image_path(train_dir).batch(2)
val_dataset_tf = create_dataset_from_image_path(val_dir).batch(2)

# transfer learning pretrained VGG19 model
inp = Input(shape=(1024, 1024,3))

# pretrained fixed layers
vgg_model = VGG19(weights='imagenet', include_top=False)
vgg_model.trainable = False

x = vgg_model(inp, training=False)

# # new trainable segmentation layers
x = keras.layers.Conv2D(1, (1, 1), activation='sigmoid', name='segmentation_head')(x)
x = keras.layers.UpSampling2D(size=(32, 32), interpolation='bilinear')(x)  # 7*32=1024
seg_model = keras.models.Model(inp, x)

seg_model.compile(optimizer=keras.optimizers.Adam(),
              loss=keras.losses.BinaryCrossentropy(from_logits=False),
              metrics=[keras.metrics.BinaryAccuracy()])

print(seg_model.summary())

history_fine = seg_model.fit(train_dataset_tf, 
                                epochs=20,
                                validation_data=val_dataset_tf)

# evaluation
loss, accuracy = seg_model.evaluate(val_dataset_tf)
print('Test BinaryCrossentropy :', accuracy)


acc = history_fine.history['BinaryCrossentropy']
val_acc = history_fine.history['val_BinaryCrossentropy']

loss = history_fine.history['loss']
val_loss = history_fine.history['val_loss']


## plot evaluation 
plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training BinaryCrossentropy')
plt.plot(val_acc, label='Validation BinaryCrossentropy')
plt.ylim([0.8, 1])
plt.legend(loc='lower right')
plt.title('Training and Validation BinaryCrossentropy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.ylim([0, 1.0])
# plt.plot([initial_epochs-1,initial_epochs-1],plt.ylim(), label='Start Fine Tuning')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show()