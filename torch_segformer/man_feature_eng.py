'''
Image feature engineering test. 
'''

import os
from glob2 import glob
import math
import numpy as np
import rasterio
import plotly.express as px
from plotly.subplots import make_subplots
from skimage.filters import sobel
from skimage.feature import local_binary_pattern
from scipy.ndimage import uniform_filter, variance
from transformers import SegformerFeatureExtractor


base_path = "../dataset_clouds_from_lwir"
img_files = glob(os.path.join(base_path, 'training', 'lwir', '*.TIF'))
msk_files = glob(os.path.join(base_path, 'training', 'clouds', '*.TIF'))


# segformer preprocessor
preprocessor = SegformerFeatureExtractor.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512", use_fast=False)
preprocessor.do_resize = False
preprocessor.do_rescale = False
preprocessor.do_normalize = True  


def load_tif_image(path):
    with rasterio.open(path) as src:
        image = src.read(1)
    return image


def extract_features(image):
    # add raw image for segformer
    features = [image]
    # # Normalized
    # norm = (image - np.mean(image)) / (np.std(image) + 1e-8)
    # features.append(norm)
    # Sobel edges
    sobel_edges = sobel(image)
    features.append(sobel_edges)
    # LBP
    lbp = local_binary_pattern(image, P=8, R=1, method='uniform')
    features.append(lbp)
    # # Local mean and variance
    # local_mean = uniform_filter(image, size=5)
    # # local_var = variance(image)
    # # features.append([local_mean, local_var])
    # features.append(np.array(local_mean))
    return np.stack(features, axis=0)


sample_idx = 1
mask = load_tif_image(msk_files[sample_idx])
img = load_tif_image(img_files[sample_idx])
sample_file_id = msk_files[sample_idx].split('/')[-1].split('.')[0]
print(msk_files[sample_idx])


feat_arr = extract_features(img)
sh = feat_arr.shape
n_ch = sh[0]


# hf preprocessing
enc = preprocessor(images=feat_arr, return_tensors="pt")
feat_arr = enc["pixel_values"].squeeze(0)
    

fig = make_subplots(rows=n_ch+1, cols=1)

# mask
feat_fig = px.imshow(mask)
trace = feat_fig.data[0]
fig.add_trace(trace, row=1, col=1)


for idx in range(1, n_ch+1):
    f = feat_arr[idx-1]
    feat_fig = px.imshow(f)
    trace = feat_fig.data[0]
    fig.add_trace(trace, row=idx+1, col=1)

fig.update_layout(
    autosize=False,
    coloraxis=dict(colorscale='gray'),
    coloraxis_showscale=True,
    width=1000,
    height=2000,
)
fig.write_image(f"./features_sa{sample_file_id}.png")
# fig.show()