'''
Utils function for visualization.
'''

import numpy as np
import plotly.express as px
from plotly.subplots import make_subplots
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
        
        
if __name__ == '__main__':
    visualize_dataset_samples(test_ds, num=2)