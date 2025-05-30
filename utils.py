'''
Util functions.
'''

import tensorflow as tf

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