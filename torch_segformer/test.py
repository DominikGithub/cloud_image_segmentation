'''
Make test prediction using the fine tuned model.
'''

from Dataloader import CloudSegDataloader
from train import SegformerWithUpsample

import os
import numpy as np
from tqdm import tqdm
from glob2 import glob
from PIL import Image, ImageOps
import plotly.express as px
from plotly import graph_objects as go
from plotly.subplots import make_subplots
from transformers import SegformerForSemanticSegmentation, SegformerFeatureExtractor
from torch.utils.data import DataLoader
import torch

TIME_SEC = 0                 # NOTE set according to training scripts model timestamp to be loaded

image_path = "../dataset_clouds_from_lwir"


def plot_testset():
    '''
    Load last trained model and plot validation set sample prediction masks vs true masks. 
    '''
    # hf segformer data preprocessor
    preprocessor = SegformerFeatureExtractor.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512", use_fast=False)
    preprocessor.do_resize = False
    preprocessor.do_rescale = False
    preprocessor.do_normalize = True  

    # load model
    file_path = f'./segformer_cloud_{TIME_SEC}.pth'
    base_model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/segformer-b0-finetuned-ade-512-512",
        num_labels=1,
        ignore_mismatched_sizes=True
    )
    model = SegformerWithUpsample(base_model)
    model.load_state_dict(torch.load(file_path), strict=False)
    model.eval()
    # freeze model params for inference
    for param in model.parameters():
        param.requires_grad = False

    # test inference on dataset
    test_time_path = f'./tests/{TIME_SEC}_3fold'
    if not os.path.exists(test_time_path):
        os.makedirs(test_time_path)

    # load data
    # img_files = glob(os.path.join(image_path, 'test', 'lwir', '*.TIF'))
    test_ds = CloudSegDataloader('test', preprocessor)
    EVAL_BATCH_SIZE = len(test_ds)
    
    loader = DataLoader(test_ds, batch_size=EVAL_BATCH_SIZE)
    map = next(iter(loader))
    X = map['pixel_values']
    y_true_lst = (map['labels'].cpu().numpy() * 255).astype(np.uint8)

    # make predictions on test dataset and plot
    with torch.no_grad():
        y_pred = model(X)
        logits = y_pred.logits 
        preds = logits * 255
        preds_bin = torch.sigmoid(preds) > 0.5
        
        y_preds_arr = preds.detach().cpu().numpy()
        y_preds_bin_arr = preds_bin.detach().cpu().numpy()

        # plot
        for i in tqdm(range(y_preds_arr.shape[0])):
            # image
            # img = Image.open(img_files[i]).convert('F')
            # img = ImageOps.exif_transpose(img)
            # img_np = np.array(img, dtype=np.float32) / 255.0
            fig_img = px.imshow(X[i,0], color_continuous_scale='gray')
            fig_img.update_layout(width=1000, height=1200, autosize=False)
            fig_img.write_image(test_time_path+f"/image_{i}.png")
            # mask
            gt_fig = px.imshow(y_true_lst[i,0], color_continuous_scale='gray', zmin=0, zmax=1)
            gt_fig.write_image(test_time_path+f"/mask_{i}.png")
            # pred logits
            pred_fig = px.imshow(y_preds_arr[i,0,:,:], color_continuous_scale='gray')
            pred_fig.write_image(test_time_path+f"/y_{i}.png")
            # pred bin
            pred_bin_fig = px.imshow(y_preds_bin_arr[i,0,:,:], color_continuous_scale='gray')
            pred_bin_fig.write_image(test_time_path+f"/y_bin_{i}.png")
        

if __name__ == "__main__":
    plot_testset()