import numpy as np
from skimage import img_as_float, img_as_ubyte
import skimage.io as skio

import matplotlib.pyplot as plt
import skimage.transform as sktr
from skimage.color import rgb2gray
import argparse
import json
from scipy.ndimage import distance_transform_edt
from math import floor, ceil
import time
from cv2 import GaussianBlur

from harris import *
from utils import *

def plot_harris_coords(im, coords, label, image_name, type=''):
    """Plots Harris coordinates given coords and an image"""
    h, w = im.shape[:2]
    plt.figure(figsize=(w/100, h/100))
    plt.imshow(im)
    plt.scatter(coords[1], coords[0], s=4, c="r", marker="o", linewidths=0)
    plt.axis("off")
    plt.savefig(f'outputs/{image_name}_{label}_harris_corners{type}.png', bbox_inches="tight", pad_inches=0, dpi=100)
    plt.close()

def extract_descriptor(corner, blurred):
    # no out of bounds error since edge discard is 20
    """Extracts feature descriptors by taking in blurred image, looking at 40x40 pixel, and subsamplign"""
    y, x = corner
    patch = blurred[y-20:y+20, x-20:x+20]
    descriptor = patch[2::5, 2::5]
    # normalize descriptor
    mean = descriptor.mean()
    sd = descriptor.std()
    return (descriptor - mean)/sd

def plot_descriptors(descriptors, n, label, image_name):
    """Plots n of the descriptors we extract"""
    fig, ax = plt.subplots(1, n, figsize=(2*n, 2))
    for i in range(n):
        ax[i].imshow(descriptors[i], cmap="gray")
        ax[i].axis("off")
    fig.savefig(f"outputs/{image_name}_{label}_descriptors.png", dpi=100, bbox_inches="tight")
    plt.close(fig)

def match_features(ims, ref_label, other_label):
    """Finds matches for each reference descriptor; returns the pair with the first-nearest neighbor if we pass the Lowe threshold"""
    _, _, ref_descriptor = ims[ref_label]
    _, _, other_descriptor = ims[other_label]
    # flatten 8x8 descriptors to 64 dim for euclidean distance calculation
    ref_descriptor = ref_descriptor.reshape(ref_descriptor.shape[0], -1)
    other_descriptor = other_descriptor.reshape(other_descriptor.shape[0], -1)
    desc_distances = dist2(ref_descriptor, other_descriptor)
    nearest_neighbor = np.argmin(desc_distances, axis=1)
    neighbor_dists = np.sort(desc_distances, axis=1)[:, :2]
    # Calculate lowe ratio; error of nn1/error of nn2
    lowe_ratios = neighbor_dists[:, 0] / neighbor_dists[:, 1]
    threshold = 0.5
    pairs_to_keep = lowe_ratios < threshold
    feature_matches = np.stack([np.arange(len(ref_descriptor))[pairs_to_keep], nearest_neighbor[pairs_to_keep]], axis=1)
    return feature_matches

def plot_feature_matches(ims, matches, n, ref_label, other_label, image_name):
    """Plots n of the feature matches"""
    _, _, ref_descriptor = ims[ref_label]
    _, _, other_descriptor = ims[other_label]
    fig, ax = plt.subplots(2, n, figsize=(2*n, 4))
    for i in range(n):
        ref_match, other_match = matches[i]
        ax[0, i].imshow(ref_descriptor[ref_match], cmap="gray")
        ax[1, i].imshow(other_descriptor[other_match], cmap="gray")
        ax[0, i].axis("off")
        ax[1, i].axis("off")
    fig.savefig(f"outputs/{image_name}_{ref_label}_{other_label}_descriptor_matches.png", dpi=100, bbox_inches="tight")
    plt.close(fig)

def plot_feature_matches_on_image(ims, matches, n, ref_label, other_label, image_name):
    """Plots n of the feature matches as they are on the image"""
    ref, ref_coords, _ = ims[ref_label]
    other, other_coords, _ = ims[other_label]
    both_images = np.concatenate([ref, other], axis=1)
    subset_matches = matches[:n]
    ref_matches = ref_coords[:, subset_matches[:, 0]]
    other_matches = other_coords[:, subset_matches[:, 1]]
    colors = plt.cm.hsv(np.linspace(0, 1, n, endpoint=False))

    fig, ax = plt.subplots(figsize=(16, 6))
    ax.imshow(both_images)
    for i in range(ref_matches.shape[1]):
        y_ref, x_ref = ref_matches[:, i]
        y_other, x_other = other_matches[:, i]
        ax.add_patch(plt.Rectangle((x_ref - 20, y_ref - 20), 40, 40, fill=False, edgecolor=colors[i], linewidth=1))
        ax.add_patch(plt.Rectangle((x_other - 20 + ref.shape[1], y_other - 20), 40, 40, fill=False, edgecolor=colors[i], linewidth=1))
    
    ax.axis("off")
    fig.savefig(f"outputs/{image_name}_{ref_label}_{other_label}_correspondences.png", dpi=100, bbox_inches="tight")
    plt.close(fig)

def anms_and_finds_descriptors(ims, image_name):
    """Implements ANMS algorithm; Also finds the descriptors for each image"""
    for label, im in ims.items():
        im = im[0]
        show_output = image_name == 'cliff' and label == 'left'
        gray_im = rgb2gray(im)

        if show_output:
            h, coords = get_harris_corners(gray_im)
            plot_harris_coords(im, coords, label, image_name)

        h, coords = get_harris_corners(gray_im, min_distance=3)

        if show_output:
            plot_harris_coords(im, coords, label, image_name, "_candidate")

        # find all distances between the candidate coordinates
        cand_h_vals = h[coords[0], coords[1]]
        distances = dist2(coords.T, coords.T)

        c_robust = 0.9
        num_coords = 500
        # we want to find the "closest" larger h
        # minimum of all of our distances where h is "larger"
        # thus if the point is not larger dist should be inf
        # for i in range(coords.shape[1]):
        #     for j in range(coords.shape[1]):
        #         if h_vals[i] >= c_robust * h_vals[j]:
        #             distances[i, j] = np.inf
        # super vectorized!
        distances[cand_h_vals[:, np.newaxis] >= c_robust * cand_h_vals[np.newaxis, :]] = np.inf
        # find minimum distance to "larger" h
        r_vals = distances.min(axis=1)
        # find the num_coords corners with the highest r values
        anms_corners = np.argsort(r_vals)[::-1][:num_coords]
        anms_coords = coords[:, anms_corners]
        ims[label].append(anms_coords)
        if show_output:
            plot_harris_coords(im, anms_coords, label, image_name, '_anms')
            plot_harris_coords(im, coords[:, np.argsort(cand_h_vals)[::-1][:num_coords]], label, image_name, "_non_anms")

        blurred_im = GaussianBlur(gray_im, (0, 0), 2)
        descriptors = np.array([extract_descriptor(corner, blurred_im) for corner in anms_coords.T])
        ims[label].append(descriptors)
        if show_output:
            plot_descriptors(descriptors, 10, label, image_name)
    return ims

def match(ims, image_name):
    show_output = image_name == "cliff"
    ref_left_matches = match_features(ims, "ref", "left")
    if show_output:
        plot_feature_matches(ims, ref_left_matches, 10, "ref", "left", image_name)
        plot_feature_matches_on_image(ims, ref_left_matches, 10, "ref", "left", image_name)
    ref_right_matches = match_features(ims, "ref", "right")
    return ref_left_matches, ref_right_matches

def mosaic(ims, name):
    """Params from 4a are slightly changed; instead of taking in ref, left, right -> we just give whole list ims
    and get_points is not needed because we are finding our own points! Code after Warping left print statement is identical"""
    ref_left_matches, ref_right_matches = match(ims, name)

    ref, ref_coords, _ = ims["ref"]
    left, left_coords, _ = ims["left"]
    right, right_coords, _ = ims["right"]

    # this is because our coordinates have been (2, N) with (y, x), but we want (N, 2) and also (x, y) for ransac/compute_H
    ref_match_to_left = ref_coords[:, ref_left_matches[:, 0]][::-1].T
    left_match_to_ref = left_coords[:, ref_left_matches[:, 1]][::-1].T

    ref_match_to_right = ref_coords[:, ref_right_matches[:, 0]][::-1].T
    right_match_to_ref = right_coords[:, ref_right_matches[:, 1]][::-1].T

    H_1_to_ref = ransac(ref_match_to_left, left_match_to_ref, 2)
    H_3_to_ref = ransac(ref_match_to_right, right_match_to_ref, 2)
    
    print("Warping left")
    left_warped, left_alpha, left_xmin, left_ymin = warpImageBilinear(left, H_1_to_ref)
    # skio.imsave(f'outputs/{name}1_warped.jpg', img_as_ubyte(left_warped))

    print("Warping right")
    right_warped, right_alpha, right_xmin, right_ymin = warpImageBilinear(right, H_3_to_ref)
    # skio.imsave(f'outputs/{name}3_warped.jpg', img_as_ubyte(right_warped))

    x_min = min(left_xmin, right_xmin, 0)
    y_min = min(left_ymin, right_ymin, 0)
    x_max = max(left_xmin + left_warped.shape[1], right_xmin + right_warped.shape[1], ref.shape[1])
    y_max = max(left_ymin + left_warped.shape[0], right_ymin + right_warped.shape[0], ref.shape[0])

    # create a global canvas for each of the images
    left_global = np.zeros((y_max-y_min, x_max-x_min, ref.shape[2]))
    left_a_global = np.zeros((y_max-y_min, x_max-x_min))
    right_global = left_global.copy()
    right_a_global = left_a_global.copy()
    ref_global = left_global.copy()
    ref_a_global = left_a_global.copy()

    # put each of the images onto the global canvas, with the shifted bounds
    left_global[left_ymin - y_min : left_ymin - y_min + left_warped.shape[0], left_xmin - x_min : left_xmin - x_min + left_warped.shape[1]] = left_warped
    left_a_global[left_ymin - y_min : left_ymin - y_min + left_warped.shape[0], left_xmin - x_min : left_xmin - x_min + left_warped.shape[1]] = left_alpha

    right_global[right_ymin - y_min : right_ymin - y_min + right_warped.shape[0], right_xmin - x_min : right_xmin - x_min + right_warped.shape[1]] = right_warped
    right_a_global[right_ymin - y_min : right_ymin - y_min + right_warped.shape[0], right_xmin - x_min : right_xmin - x_min + right_warped.shape[1]] = right_alpha

    ref_global[-1 * y_min : -1 * y_min + ref.shape[0], -1 * x_min : -1 * x_min + ref.shape[1]] = ref
    ref_a_global[-1 * y_min : -1 * y_min + ref.shape[0], -1 * x_min : -1 * x_min + ref.shape[1]] = 1

    # create the full image by averaging the alpha values at each place, and black if nothing there/all alphas are 0
    def get_result(images, alphas):
        full_img = sum(im * alpha for im, alpha in zip(images, alphas))
        result_img = np.zeros_like(full_img)
        np.divide(full_img, sum(alphas), out=result_img, where=(sum(alphas) != 0))
        return result_img

    images = [left_global, ref_global, right_global]
    alphas = [left_a_global, ref_a_global, right_a_global]

    dist_alphas = list(map(transform_alpha, alphas))

    high_freq_mask_to_choose = np.argmax(np.stack(dist_alphas), axis=0)
    valid = np.stack(dist_alphas).max(axis=0) > 0
    high_freq_masks = [(high_freq_mask_to_choose == i) & valid for i in range(len(dist_alphas))]

    kernel_size = 11
    low_freq = [GaussianBlur(im, (kernel_size, kernel_size), 0) for im in images]
    high_freq = [im - lo for im, lo in zip(images, low_freq)]
    low_freq_result = get_result(low_freq, [a[:, :, np.newaxis] for a in dist_alphas])
    high_freq_result = get_result(high_freq, [a[:, :, np.newaxis] for a in high_freq_masks])

    skio.imsave(f'outputs/{name}_result_automatic.jpg', img_as_ubyte(np.clip(low_freq_result + high_freq_result, 0, 1)))

def compute_H(im1_pts, im2_pts):
    """From 4a, unchanged"""
    A = []
    b = []
    for (x1, y1), (x2, y2) in zip(im1_pts, im2_pts):
        A.append([x1, y1, 1, 0, 0, 0, -1*x1*x2, -1*y1*x2])
        A.append([0, 0, 0, x1, y1, 1, -1*x1*y2, -1*y1*y2])
        b.append(x2)
        b.append(y2)
    A = np.array(A)
    b = np.array(b)
    x, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
    H = np.append(x, 1).reshape(3, 3)
    return H

def ransac(ref_to_other, other_to_ref, epsilon, seed=23):
    rng = np.random.default_rng(seed)
    # Convert our points to project to homogenous coords
    other_to_ref_homog = np.concatenate([other_to_ref, np.ones((other_to_ref.shape[0], 1))], axis=1)
    best = 0
    best_H = None
    # Iterate for 1000 times (can tune this hyperparam)
    for _ in range(1000):
        # choose 4 feature pairs at random
        random_matches = rng.choice(other_to_ref.shape[0], 4, replace=False)
        # find exact H
        candidate_H = compute_H(other_to_ref[random_matches], ref_to_other[random_matches])
        # find how these coordinates would look like projected (tranposed, row row corresponds to point)
        other_to_ref_projected_homog = other_to_ref_homog @ candidate_H.T
        # convert to normal coords + find distance
        other_to_ref_projected = other_to_ref_projected_homog[:, :2] / other_to_ref_projected_homog[:, 2:3]
        errors = np.linalg.norm(other_to_ref_projected - ref_to_other, axis=1)
        inliers = errors < epsilon
        if inliers.sum() > best:
            best = inliers.sum()
            best_H = candidate_H
    print(f"We find {best.sum()} inliers")
    return best_H

def transform_alpha(alpha):
    """From 4a, unchanged"""
    padded = np.pad(alpha, 1)
    return distance_transform_edt(padded)[1:-1, 1:-1]

def warpImageBilinear(im, H):
    """From 4a, unchanged"""
    # define boundary points for original image
    boundaries = np.array([[0, 0, im.shape[1]-1, im.shape[1]-1],
                              [0, im.shape[0]-1, 0, im.shape[0]-1],
                              [1, 1, 1, 1],])
    # find corresponding output boundaries in homogenous coords
    output_boundaries = H @ boundaries
    x_y_output = output_boundaries[:2] / output_boundaries[2]
    # find max and min x
    x_min, x_max = x_y_output[0].min(), x_y_output[0].max()
    y_min, y_max = x_y_output[1].min(), x_y_output[1].max()

    # get the discrete coordinates that match x_min and x_max, put them in a matrix
    Xs, Ys = np.meshgrid(np.arange(floor(x_min), ceil(x_max) + 1), np.arange(floor(y_min), ceil(y_max) + 1))
    all_points = np.stack([Xs.flatten(), Ys.flatten(), np.ones_like(Xs).flatten()])

    # get the corresponding points in the input image
    transformed_points = np.linalg.inv(H) @ all_points
    # create empty array for output image
    output_image = np.zeros((ceil(y_max) - floor(y_min) + 1, ceil(x_max) - floor(x_min) + 1, im.shape[2]))
    output_alpha_mask = np.zeros((ceil(y_max) - floor(y_min) + 1, ceil(x_max) - floor(x_min) + 1))

    # vectorized implementation
    wx, wy, w = transformed_points[0], transformed_points[1], transformed_points[2]
    x, y = all_points[0], all_points[1]
    x_input = wx / w
    y_input = wy / w
    valid_mask = (x_input >= 0) & (y_input >= 0) & (y_input < im.shape[0]) & (x_input < im.shape[1])

    x0 = np.floor(x_input).astype(int)[valid_mask]
    x1 = np.clip(np.floor(x_input).astype(int) + 1, 0, im.shape[1]-1)[valid_mask]
    y0 = np.floor(y_input).astype(int)[valid_mask]
    y1 = np.clip(np.floor(y_input).astype(int) + 1, 0, im.shape[0]-1)[valid_mask]

    dx = (x_input - np.floor(x_input).astype(int))[valid_mask][:, np.newaxis]
    dy = (y_input - np.floor(y_input).astype(int))[valid_mask][:, np.newaxis]

    top = im[y0, x0] * (1-dx) + im[y0, x1] * dx
    bottom = im[y1, x0] * (1-dx) + im[y1, x1] * dx
    interpolated = top * (1-dy) + bottom * dy
    output_alpha_mask[(y - floor(y_min))[valid_mask], (x - floor(x_min))[valid_mask]] = 1
    output_image[(y - floor(y_min))[valid_mask], (x - floor(x_min))[valid_mask], :] = interpolated

    # # iterative implementation
    # for (wx, wy, w), (x, y, _) in zip(transformed_points.T, all_points.T):
    #     # for each of the transformed points, find the nearest neighbor in the original image, and set to value in output_image
    #     x_input = wx / w
    #     y_input = wy / w
    #     if x_input < 0 or y_input < 0 or y_input >= im.shape[0] or x_input >= im.shape[1]:
    #         pass
    #     else:
    #         x0 = floor(x_input)
    #         x1 = np.clip(floor(x_input) + 1, 0, im.shape[1]-1)
    #         y0 = floor(y_input)
    #         y1 = np.clip(floor(y_input) + 1, 0, im.shape[0]-1)
    #         dx = (x_input - floor(x_input))
    #         dy = (y_input - floor(y_input))
            
    #         top = im[y0, x0] * (1-dx) + im[y0, x1] * dx
    #         bottom = im[y1, x0] * (1-dx) + im[y1, x1] * dx
    #         interpolated = top * (1-dy) + bottom * dy
    #         output_alpha_mask[y - floor(y_min), x - floor(x_min)] = 1
    #         output_image[y - floor(y_min), x - floor(x_min), :] = interpolated

    return output_image, output_alpha_mask, floor(x_min), floor(y_min)
