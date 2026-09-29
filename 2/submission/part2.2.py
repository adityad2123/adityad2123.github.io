import matplotlib.pyplot as plt
from align_image_code import align_images
from skimage import io, img_as_float
from scipy.signal import convolve2d
import cv2
import numpy as np


# First load images
telly = img_as_float(io.imread('inputs/telephone.jpg', as_gray=True))
banana = img_as_float(io.imread('inputs/banana.jpg', as_gray=True))
# Next align images (this code is provided, but may be improved)

puffin = img_as_float(io.imread('inputs/puffin.jpg', as_gray=True))
baseball = img_as_float(io.imread('inputs/baseball.jpg', as_gray=True))

derek = img_as_float(io.imread('inputs/derek.jpg', as_gray=True))
nutmeg = img_as_float(io.imread('inputs/nutmeg.jpg', as_gray=True))

# Next align images (this code is provided, but may be improved)
telly_aligned, banana_aligned = align_images(telly, banana)
plt.imsave(f"out_hybrid/telephone_aligned.jpg", telly_aligned, cmap='gray', vmin=0, vmax=1)
plt.imsave(f"out_hybrid/banana_aligned.jpg", banana_aligned, cmap='gray', vmin=0, vmax=1)
# puffin_aligned, baseball_aligned = align_images(puffin, baseball)
# derek_aligned, nutmeg_aligned = align_images(derek, nutmeg)


## You will provide the code below. Sigma1 and sigma2 are arbitrary 
## cutoff values for the high and low frequencies
def hybrid_image(im1_aligned, im2_aligned, sigma1, sigma2, fourier=False, name=""):
    # Produces hybrid images
    ksize1 = 2 * int(3 * sigma1) + 1
    g1 = cv2.getGaussianKernel(ksize=ksize1, sigma=sigma1)
    low_pass_gauss = np.outer(g1, g1)

    ksize2 = 2 * int(3 * sigma2) + 1
    g2 = cv2.getGaussianKernel(ksize=ksize2, sigma=sigma2)
    high_pass_gauss = np.outer(g2, g2)

    identity = np.zeros((ksize2, ksize2))
    identity[ksize2 // 2, ksize2 // 2] = 1
    low = convolve2d(im1_aligned, low_pass_gauss, mode="same", boundary="fill", fillvalue=0)
    high = convolve2d(im2_aligned, identity - high_pass_gauss, mode="same", boundary="fill", fillvalue=0)

    hybrid = low + high

    if fourier:
        images = {"im1": im1_aligned, "im2": im2_aligned, "im1_low": low, "im2_high": high, "hybrid": hybrid}
        for label, im in images.items():
            spectrum = np.log(np.abs(np.fft.fftshift(np.fft.fft2(im))))
            plt.imsave(f"out_hybrid/{name}_fourier_{label}.png", spectrum, cmap="gray")

    plt.imsave(f"out_hybrid/{name}_hybrid.jpg", hybrid, cmap="gray", vmin=0, vmax=1)
    return hybrid

bt_sigma1 = 7
bt_sigma2 = 4
bt_hybrid = hybrid_image(telly_aligned, banana_aligned, bt_sigma1, bt_sigma2, fourier=True, name='banana_telephone2')
pb_sigma1 = 15
pb_sigma2 = 7
# pb_hybrid = hybrid_image(puffin_aligned, baseball_aligned, pb_sigma1, pb_sigma2, fourier=False, name='puffin_baseball')

dn_sigma1 = 15
dn_sigma2 = 7
# dn_hybrid = hybrid_image(derek_aligned, nutmeg_aligned, dn_sigma1, dn_sigma2, fourier=False, name='derek_nutmeg')
