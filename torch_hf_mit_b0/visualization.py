'''
Utils function for visualization.
'''

import plotly.express as px
from plotly.subplots import make_subplots


def visualize_dataset_samples(ds, num=3):
    '''
    Plot some image and mask samples.
    '''
    ds_iter = iter(ds)
    for _ in range(num):
        map = next(ds_iter)
        i = map['pixel_values']
        m = map['labels']
        # print(i.numpy().shape, m.numpy().shape)
        # print(i)
        # print(m)

        # plot image and mask
        eval_fig = make_subplots(rows=2, cols=1,  vertical_spacing=0.05, subplot_titles=['IR image', 'Mask'])
        # Image 
        pred_fig = px.imshow(i.numpy()[0,:,:], color_continuous_scale='gray', binary_string=False)
        pred_trace = pred_fig.data[0]
        pred_trace.update(coloraxis='coloraxis')
        eval_fig.add_trace(pred_trace, row=1, col=1)
        # Mask
        gt_fig = px.imshow(m.numpy(), color_continuous_scale='gray', binary_string=False)
        gt_trace = gt_fig.data[0]
        gt_trace.update(coloraxis='coloraxis')
        eval_fig.add_trace(gt_trace, row=2, col=1)
        # style
        eval_fig.update_layout(
            autosize=False,
            coloraxis=dict(colorscale='gray'),
            coloraxis_showscale=True,
            width=1000,
            height=1600,
            annotations=[
                dict(text='Image', x=0.5, xref='paper', y=1, yref='paper',
                    xanchor='center', yanchor='bottom',
                    showarrow=False, font=dict(size=20)),
                dict(text='Mask', x=0.5, xref='paper', y=0.5, yref='paper',
                    xanchor='center', yanchor='bottom',
                    showarrow=False, font=dict(size=20)),
            ]
        )
        eval_fig.show()
        eval_fig.write_html(f'./torch_input_{i}.html')
        eval_fig.write_image(f"./torch_input_{i}.png")