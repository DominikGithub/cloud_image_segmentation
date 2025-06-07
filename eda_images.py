'''
EDA .TIF image files meta data.
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



# --------------------------------
import rasterio
with rasterio.open(img_train_lst[0]) as src:
    print("Metadata:", src.meta)
    print("Tags:", src.tags())
    print("Band descriptions:", src.descriptions)
    print("CRS:", src.crs)
    print("Transform (Geo info):", src.transform)
# -------------------------------------



# cloud image 
img = Image.open(img_train_lst[0])

img.show()
np_img_arr = np.array(img)
print('image shape: ', np_img_arr.shape)


meta_dict = {TAGS[key] : img.tag[key] for key in img.tag_v2}
# pprint(meta_dict)
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
mask_img = Image.open(mask_train_lst[0])

mask_img.show()
np_mask_arr = np.array(mask_img)
print('mask shape: ', np_mask_arr.shape)

meta_dict = {TAGS[key] : mask_img.tag[key] for key in mask_img.tag_v2}
# pprint(meta_dict)
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


# # visualize sample image
# '''
# $ sudo apt-get install libtiff-tools
# '''
# print('images: ', len(img_cloud_train_lst))
# print('masks:  ', len(img_mask_train_lst))
# im = Image.open(img_cloud_train_lst[0])
# im.show()

# na = np.array(im)
# print('mask shape: ', na.shape)
# print(na)