'''
Make test prediction using the fine tuned model.
'''

from Dataloader import CloudSegDataloader
from train import SegformerWithUpsample

import os
import numpy as np
from tqdm import tqdm
from glob2 import glob
from PIL import Image
import plotly.express as px
from plotly import graph_objects as go
from plotly.subplots import make_subplots
from transformers import SegformerForSemanticSegmentation, SegformerFeatureExtractor
from torch.utils.data import DataLoader
import torch

TIME_SEC = 1748854305                 # NOTE set according to training scripts model timestamp to be loaded

image_path = "../../dataset_clouds_from_lwir"


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
    model.load_state_dict(torch.load(file_path))
    model.eval()
    # freeze model params for inference
    for param in model.parameters():
        param.requires_grad = False

    # test inference on dataset
    test_time_path = f'./tests/{TIME_SEC}_3fold'
    if not os.path.exists(test_time_path):
        os.makedirs(test_time_path)

    # load data
    img_files = glob(os.path.join(image_path, 'test', 'lwir', '*.TIF'))
    
    test_ds = CloudSegDataloader('test', preprocessor)
    EVAL_BATCH_SIZE = len(test_ds)
    
    # make test prediction and plot
    loader = DataLoader(test_ds, batch_size=EVAL_BATCH_SIZE)
    map = next(iter(loader))
    X = map['pixel_values']
    y_true_lst = (map['labels'].cpu().numpy() * 255).astype(np.uint8)

    with torch.no_grad():
        y_pred = model(X)
        logits = y_pred.logits 
        preds = logits * 255
        preds_bin = torch.sigmoid(preds) > 0.5
        
        preds_np_lst = preds.detach().cpu().numpy()
        preds_bin_lst = preds_bin.detach().cpu().numpy()

    # plot prediction mask  logits and discrete vs ground truth mask and cloud image
    for i in tqdm(range(len(preds_np_lst))):
        eval_fig = make_subplots(rows=2, cols=2, vertical_spacing=0.05, 
                                subplot_titles=['Prediction', 'Ground truth', 'bin(y_pred)', 'LWIR'])
        # Prediction 
        pred_fig = px.imshow(preds_np_lst[i, 0, :, :], color_continuous_scale='gray')
        pred_trace = pred_fig.data[0]
        pred_trace.update(coloraxis='coloraxis')
        eval_fig.add_trace(pred_trace, row=1, col=1)
        # Prediction binary
        pred_fig = px.imshow(preds_bin_lst[i, 0, :, :], color_continuous_scale='gray')
        pred_trace = pred_fig.data[0]
        pred_trace.update(coloraxis='coloraxis2')
        eval_fig.add_trace(pred_trace, row=2, col=1)
        # Ground truth
        gt_fig = px.imshow(y_true_lst[i, 0], color_continuous_scale='gray', zmin=0, zmax=1)
        gt_trace = gt_fig.data[0]
        gt_trace.update(coloraxis='coloraxis3')
        eval_fig.add_trace(gt_trace, row=1, col=2)
        # Image
        img = Image.open(img_files[i]).convert('F')
        img_np = np.array(img, dtype=np.float32) / 255.0
        img_np = np.stack([img_np]*3, axis=-1)
        img_trace = go.Heatmap(z=img_np[:, :, 0], coloraxis='coloraxis4')
        eval_fig.add_trace(img_trace, row=2, col=2)
        # style
        eval_fig.update_layout(
            width=1600,
            height=1600,
            autosize=False,
            coloraxis=dict(colorscale='gray'),
            coloraxis2=dict(colorscale='gray'),
            coloraxis3=dict(colorscale='gray'),
            coloraxis4=dict(colorscale='gray'),
            coloraxis_showscale=False,
            coloraxis2_showscale=False,
            coloraxis3_showscale=False,
            coloraxis4_showscale=False,
            annotations=[
                dict(text='Prediction', x=0.25, xref='paper', y=1, yref='paper',
                    xanchor='center', yanchor='bottom',
                    showarrow=False, font=dict(size=20)),
                dict(text='Ground truth', x=0.75, xref='paper', y=1, yref='paper',
                    xanchor='center', yanchor='bottom',
                    showarrow=False, font=dict(size=20)),
                dict(text='bin(y_pred)', x=0.25, xref='paper', y=0.48, yref='paper',
                    xanchor='center', yanchor='bottom',
                    showarrow=False, font=dict(size=20)),
                dict(text='LWIR', x=0.75, xref='paper', y=0.48, yref='paper',
                    xanchor='center', yanchor='bottom',
                    showarrow=False, font=dict(size=20)),
            ]
        )
        # eval_fig.write_html(f'{test_time_path}/eval_{i}.html')
        eval_fig.write_image(f"{test_time_path}/eval_{i}.png")


if __name__ == "__main__":
    plot_testset()