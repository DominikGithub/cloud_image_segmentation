'''
Utils function for visualization.
'''

import numpy as np
import glob
from PIL import Image
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from Dataloader import CloudSegDataloader
from transformers import SegformerFeatureExtractor


# hf segformer data preprocessor
preprocessor = SegformerFeatureExtractor.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512", use_fast=False)
preprocessor.do_resize = False
preprocessor.do_rescale = False
preprocessor.do_normalize = True  

# load data
train_ds = CloudSegDataloader('training', preprocessor)
val_ds = CloudSegDataloader('validation', preprocessor)
test_ds = CloudSegDataloader('test', preprocessor)
n_steps_per_epoch = len(train_ds)
print('# batches (train, val, test):', n_steps_per_epoch, len(val_ds), len(test_ds))


def visualize_dataset_samples(ds, num=3):
    '''
    Plot some image and mask samples.
    '''
    loader = DataLoader(ds, batch_size=num)
    map = next(iter(loader))
    X = map['pixel_values']
    mask_lst = map['labels'].cpu().numpy()
    
    for idx in range(num):
        img = X[idx].numpy()
        msk =  mask_lst[idx]
        print(img.shape, msk.shape)
        
        fig1 = px.imshow(msk[0], color_continuous_scale='gray')
        fig1.update_layout(width=1200, height=1200, autosize=False, coloraxis_showscale=False,
                            annotations=[
                                dict(text='Mask', x=0.5, xref='paper', y=1, yref='paper',
                                    xanchor='center', yanchor='bottom',
                                    showarrow=False, font=dict(size=20))
                                ]
                            )
        # fig1.write_image(f"./mask_{idx}.png")
        fig1.show()
        
        fig2 = px.imshow(img[0], color_continuous_scale='gray')
        fig2.update_layout(width=1200, height=1200, autosize=False, coloraxis_showscale=False,
                            annotations=[
                                dict(text='Image', x=0.5, xref='paper', y=1, yref='paper',
                                    xanchor='center', yanchor='bottom',
                                    showarrow=False, font=dict(size=20))
                                ]
                            )
        # fig2.write_image(f"./image_{idx}.png")
        fig2.show()
        

def visualize_attention_maps(model, image, file_name):
    '''
    Plot model attention maps to highlight important image regions.
    '''
    pass


def plot_pixel_hist():
    '''
    Plot sample image pixel intensity histogram.
    '''
    # training data set samples
    # img_path = glob.glob("../dataset_clouds_from_lwir/training/lwir/*.tif")[1]
    img_id = 11
    img_path = f'../dataset_clouds_from_lwir/training/lwir/{img_id}.tif'
    print(img_path)

    img = Image.open(img_path)
    np_img_arr = np.array(img) / 255.0
    
    # histogram
    counts, bins = np.histogram(np_img_arr, bins=range(0,255))
    # print(counts, bins)
    plt.bar(bins[:-1] - 0.5, counts, width=1, edgecolor='none')
    plt.title("LWIR Pixel Intensity Histogram")
    plt.savefig(f'./histogram_{img_id}_bar.png')


        
if __name__ == '__main__':
    # visualize_dataset_samples(test_ds, num=2)
    plot_pixel_hist()