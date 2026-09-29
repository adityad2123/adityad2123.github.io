import numpy as np
from scipy.signal import convolve2d
import time
import matplotlib.pyplot as plt
from skimage import io, img_as_float
from skimage.transform import rescale
import cv2

# per lecture, we can think of it as f * ((1+alpha)e - alpha*g)
def unsharp_then_mask_kernel(alpha, size):
    gaussian_1d = cv2.getGaussianKernel(ksize=size, sigma=(size - 1) / 6)
    gaussian_2d = np.outer(gaussian_1d, gaussian_1d)
    identity = np.zeros((size, size))
    identity[size // 2, size // 2] = 1
    return (1+alpha)*identity - (alpha)*gaussian_2d

def convolve_channels(img, kernel):
    # This is a 3D convolution, we convolve for each color channel separately
    conv = lambda ch: convolve2d(ch, kernel, mode="same", boundary="symm")
    return np.stack([conv(img[:, :, c]) for c in range(img.shape[2])], axis=-1)

images = {"taj": "inputs/taj.jpg", "pumpkin": "inputs/pumpkin.jpg"}
alphas = [0, 0.5, 1, 2, 5, 10]

for name, path in images.items():
    img = img_as_float(io.imread(path))
    for alpha in alphas:
        k = unsharp_then_mask_kernel(alpha, 9)
        out = np.clip(convolve_channels(img, k), 0, 1)
        plt.imsave(f"out_p2/{name}_alpha_{alpha}.jpg", out)

# we also need to show the blurred and high freq versions for an alpha that I like
for name, path in images.items():
    img = img_as_float(io.imread(path))
    size = 9
    gaussian_1d = cv2.getGaussianKernel(ksize=size, sigma=(size - 1) / 6)
    gaussian_2d = np.outer(gaussian_1d, gaussian_1d)
    blurred = convolve_channels(img, gaussian_2d)
    high = img - blurred
    plt.imsave(f"out_p2/{name}_blurred.jpg", np.clip(blurred, 0, 1))
    high_grayscale = high.mean(axis=-1)
    plt.imsave(f"out_p2/{name}_high.jpg", high_grayscale, cmap="gray", vmin=np.abs(high_grayscale).min(), vmax=np.abs(high_grayscale).max())