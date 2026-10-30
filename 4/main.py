import numpy as np
from skimage import img_as_float, img_as_ubyte
import skimage.io as skio

import matplotlib.pyplot as plt
import skimage.transform as sktr
import argparse
import json
from scipy.ndimage import distance_transform_edt
from math import floor, ceil
import time
from cv2 import GaussianBlur

from utils import *


parser = argparse.ArgumentParser()
parser.add_argument("--get_points", action="store_true")
parser.add_argument("--image", choices=["cliff", "building", "gateway", "painting", "table"], default="cliff")
args = parser.parse_args()
image_name = args.image
rectify_dims = {'painting': [400, 150], 'table': [400, 600]}

resize_factor = 0.25
if image_name in ['painting', 'table']:
    image_large = img_as_float(skio.imread(f'images/{image_name}_large.jpg'))
    image_large = np.rot90(image_large, k=-1) # correcting a portrait quirk
    image = sktr.rescale(image_large, resize_factor, anti_aliasing=True, channel_axis=-1)
    skio.imsave(f"images/{image_name}.jpg", img_as_ubyte(image))
    height, width = rectify_dims[image_name]
    rectify(image, image_name, height, width, args.get_points)
else:
    # left_large = skio.imread(f'images/{image}1_large.jpg')
    # ref_large = skio.imread(f'images/{image}2_large.jpg')
    # right_large = skio.imread(f'images/{image}3_large.jpg')
    # left = sktr.rescale(left_large, resize_factor, anti_aliasing=True, channel_axis=-1)
    # ref = sktr.rescale(ref_large, resize_factor, anti_aliasing=True, channel_axis=-1)
    # right  = sktr.rescale(right_large, resize_factor, anti_aliasing=True, channel_axis=-1)
    # skio.imsave(f"images/{image}1.jpg", img_as_ubyte(left))
    # skio.imsave(f"images/{image}2.jpg", img_as_ubyte(ref))
    # skio.imsave(f"images/{image}3.jpg", img_as_ubyte(right))
    left = img_as_float(skio.imread(f'images/{image_name}1.jpg'))
    ref = img_as_float(skio.imread(f'images/{image_name}2.jpg'))
    right = img_as_float(skio.imread(f'images/{image_name}3.jpg'))
    mosaic(ref, left, right, image_name, args.get_points)