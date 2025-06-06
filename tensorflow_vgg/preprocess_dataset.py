'''
Convert raw image data into batched TF datasets.
'''

import multiprocessing as mp
from pprint import pprint
import glob
from tqdm import tqdm
import tensorflow as tf
import numpy as np
from PIL import Image
import plotly.express as px

train_dir = "./dataset_clouds_from_lwir/training/"
val_dir = "./dataset_clouds_from_lwir/validation/"

def _serialize_sample_as_tfrecord(X, y):
    '''
    Serialize single TF sample pair of image and mask.
    '''
    feature_dict = {
        'X': tf.train.Feature(float_list=tf.train.FloatList(value=X.flatten())),
        'y': tf.train.Feature(int64_list=tf.train.Int64List(value=y.flatten()))
    }
    example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
    return example.SerializeToString()


def serialize_batch_pair_from_file_list(img_cloud_lst, img_mask_lst, batch_idx, set_name='dummy'):
    '''
    Load TIFF images pairs of image and mask as numpy and serialize to TFRecord file per batch of samples.
    '''
    # validate order and completness of features and label files
    img_id_lst = [p.split('/')[-1].split('.')[0] for p in img_cloud_lst]
    msk_id_lst = [p.split('/')[-1].split('.')[0] for p in img_mask_lst]
    for idx, id in enumerate(img_id_lst):
        assert msk_id_lst[idx] == id, f'{idx=} {id=}'
    # ----------------------------------------------------------

    # cloud images
    cloud_arr = np.ndarray((0, 1024, 1024, 3))
    for img in tqdm(img_cloud_lst):
        image = Image.open(img)                                                                      # TODO validate image orientation
        
        # TODO add EXIF settings
         
        input_arr = np.array(image) / 255.0
        # turn grayscale into VGG 3-channel image
        input_arr = np.stack((input_arr,)*3, axis=-1)
        # apply VGG specific preparation
        input_arr = tf.keras.applications.vgg19.preprocess_input(input_arr)
        input_arr = np.array([input_arr])
        cloud_arr = np.concatenate((cloud_arr, input_arr), axis=0)
        
    # segmentation mask
    mask_arr = np.ndarray((0, 1024, 1024))
    for mask in tqdm(img_mask_lst):
        msk_arr = Image.open(mask)                                                                      # TODO validate image orientation
        
        # TODO add EXIF settings
        
        msk_arr = np.array([msk_arr]) / 255.0
        mask_arr = np.concatenate((mask_arr, msk_arr), axis=0)

    ptfd = f'./tfdataset/{set_name}/{batch_idx}.tfrecords'
    with tf.io.TFRecordWriter(ptfd, options=tf.io.TFRecordOptions(compression_type='ZLIB', compression_level=9)) as writer:
        for x, y in tqdm(zip(cloud_arr, mask_arr)):
            x = x.reshape((1024, 1024, 3))              # TODO reshape again?
            # cast numpy default precicision
            x = x.astype(np.float32)
            y = y.astype(np.int64)

            serialized = _serialize_sample_as_tfrecord(x, y)
            writer.write(serialized)


def _chunks(lst, s):
    '''
    Split list into batches.
    '''
    for i in range(0, len(lst), s):
        yield lst[i:i + s]


def serialize_batches(path, set_name, BATCH_SIZE=100):
    '''
    Parallel batch serialization.
    '''
    img_cloud_lst = glob.glob(path+"lwir/*.tif")
    img_mask_lst  = glob.glob(path+"clouds/*.tif")
    
    batches = list(_chunks(img_cloud_lst, BATCH_SIZE))

    tup_lst = [(b, img_mask_lst[i*BATCH_SIZE:i*BATCH_SIZE+BATCH_SIZE], i, set_name) for i, b in enumerate(batches)]
    with mp.Pool() as p:
        p.starmap(serialize_batch_pair_from_file_list, tup_lst)

serialize_batches(train_dir, 'train')
serialize_batches(val_dir, 'val')
