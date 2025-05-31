'''
Util functions.
'''

import os
import tensorflow as tf
import glob

def f1_seg_score(y_true, y_pred, threshold=0.5):
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


def load_tfrecords_set(set_name='dummy'):
    '''
    Load dataset from serialized TFRecord files.
    '''
    tfrecord_train_files = glob.glob(f'./tfdataset/{set_name}/*.tfrecords')
    dataset_tf = tf.data.TFRecordDataset(tfrecord_train_files, compression_type='ZLIB', num_parallel_reads=os.cpu_count())
    dataset_tf = dataset_tf.map(parse_proto, num_parallel_calls=tf.data.AUTOTUNE)
    return dataset_tf
    

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