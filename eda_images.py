'''
EDA .TIF image files.

- file format meta data
- shape/channels
- sample visualization

TIF meta data field explanation -> https://exiv2.org/tags.html
'''

import numpy as np
from pprint import pprint
import glob
from PIL import Image
from PIL.TiffTags import TAGS

# training data set samples
mask_train_lst  = glob.glob("./dataset_clouds_from_lwir/training/clouds/*.tif")
img_train_lst = glob.glob("./dataset_clouds_from_lwir/training/lwir/*.tif")
pprint(img_train_lst[:5])

# cloud image 
img = Image.open(img_train_lst[0])
img.show()
# Shape (H x W x C)
print('Image shape: ', np.array(img).shape)


# -------------- TIFF file metadata --------------
# Satellite image
meta_dict = {TAGS[key] : img.tag[key] for key in img.tag_v2}
pprint(meta_dict)
'''
Sample TIF meta data 
------------------------------
{'BitsPerSample': (16,),
 'Compression': (5,),
 'ImageLength': (1024,),
 'ImageWidth': (1024,),
 'PhotometricInterpretation': (1,),
 'PlanarConfiguration': (1,),
 'Predictor': (2,),
 'RowsPerStrip': (4,),
 'SampleFormat': (1,),
 'SamplesPerPixel': (1,),
 'StripByteCounts': (8539,
                    ......
  )
}   
'''

# Segmentation mask
mask_img = Image.open(mask_train_lst[0])
mask_img.show()
# Shape (H x W)
print('Mask shape: ', np.array(mask_img).shape)



meta_dict = {TAGS[key] : mask_img.tag[key] for key in mask_img.tag_v2}
pprint(meta_dict)
'''
Sample TIF meta data
---------------------------
{'BitsPerSample': (16,),
 'Compression': (5,),
 'ImageLength': (1024,),
 'ImageWidth': (1024,),
 'PhotometricInterpretation': (1,),
 'PlanarConfiguration': (1,),
 'Predictor': (2,),                     # compression algorithm
 'RowsPerStrip': (4,),                  # compression
 'SampleFormat': (1,),
 'SamplesPerPixel': (1,),
 'StripByteCounts': (8539, 
                    ......
  )
}                
'''
