'''
EDA .TIF image files meta data.
'''

import numpy as np
from pprint import pprint
import glob
from PIL import Image
from PIL.TiffTags import TAGS

# training data set samples
img_cloud_train_lst= glob.glob("./dataset_clouds_from_lwir/training/clouds/*.tif")
img_mask_train_lst = glob.glob("./dataset_clouds_from_lwir/training/lwir/*.tif")
pprint(img_cloud_train_lst[:5])

# cloud image 
img = Image.open(img_cloud_train_lst[0])
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

# segmentation mask
img = Image.open(img_mask_train_lst[0])
meta_dict = {TAGS[key] : img.tag[key] for key in img.tag_v2}
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
# TIF meta data field explanation -> https://exiv2.org/tags.html


# visualize sample image
'''
$ sudo apt-get install libtiff-tools
'''
im = Image.open(img_mask_train_lst[0])
# im.show()

print(im.n_frames)
na = np.array(im)
print(len(na), len(na[0]), na)