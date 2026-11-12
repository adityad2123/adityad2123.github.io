from skimage import img_as_float
import skimage.io as skio

from harris import *
from utils import *

parser = argparse.ArgumentParser()
parser.add_argument("--image", choices=["cliff", "building", "gateway"], default="cliff")
args = parser.parse_args()
image_name = args.image

left = img_as_float(skio.imread(f'images/{image_name}1.png'))
ref = img_as_float(skio.imread(f'images/{image_name}2.png'))
right = img_as_float(skio.imread(f'images/{image_name}3.png'))

ims = {"left": [left], "ref": [ref], "right": [right]}
ims = anms_and_finds_descriptors(ims, image_name)
mosaic(ims, image_name)
        
