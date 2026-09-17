# CS180 (CS280A): Project 1 starter Python code
import numpy as np
import skimage as sk
import skimage.io as skio
import matplotlib.pyplot as plt
from skimage.util import img_as_ubyte
from utils import align, l2, ncc, align_recursive
import os

import argparse
parser = argparse.ArgumentParser(description="Pass a specific filepath to analyze a specific image; else script runs for all of them")
parser.add_argument("filepath", type=str, nargs="?", help="specific image")


def process(filename, multiple=True):

    imname, imext = os.path.splitext(filename)
    impath = f'data/{imname}{imext}'

    # read in the image
    original_im = skio.imread(impath)
    height = original_im.shape[0]
    im = original_im[int(original_im.shape[0]*0.005): int(original_im.shape[0]*0.995), :]
    print("Processing", imname, "with shape", im.shape, "after crop")

    # convert to double (might want to do this later on to save memory)    
    im = sk.img_as_float(im)
        
    # compute the height of each part (just 1/3 of total)
    height = np.floor(im.shape[0] / 3.0).astype(np.uint16)

    # separate color channels
    b = im[:height]
    g = im[height: 2*height]
    r = im[2*height: 3*height]

    # choose starting displacement
    disp_x = 15 if b.shape[0] < 500 else 2;
    disp_y = 15 if b.shape[0] < 500 else 2;

    # align green image with blue
    green_disp = align_recursive(b, g, disp_x, disp_y, ncc)
    ag = np.roll(g, shift=green_disp, axis=(1, 0))
    print("Green displacement", green_disp)

    # align red image with aligned green image
    red_disp = align_recursive(ag, r, disp_x, disp_y, ncc)
    ar = np.roll(r, shift=red_disp, axis=(1, 0))
    print("Red displacement", red_disp)

    # create a color image
    im_out = np.dstack([ar, ag, b])

    # save the image
    fname = f'out/out_{imname}.jpg'
    skio.imsave(fname, img_as_ubyte(im_out))

    # display the image
    if not multiple:
        plt.imshow(im_out)
        plt.axis('off')
        plt.show()

args = parser.parse_args()
if args.filepath is not None:
    process(args.filepath, multiple=False) 
else:
    for filename in sorted(os.listdir('data')):
        process(filename)