import numpy as np
from skimage import img_as_float, img_as_ubyte
import skimage.io as skio

import matplotlib.pyplot as plt
import skimage.transform as sktr
import json
from scipy.ndimage import distance_transform_edt
from math import floor, ceil
import time
from cv2 import GaussianBlur

def get_points(im1, im2, im1_name, im2_name) -> tuple:
    im1_points = []
    im2_points = []
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.imshow(np.concatenate([im1, im2], axis=1))
    while True:
        pt1 = plt.ginput(n=1, timeout=0)
        if not pt1:
            break
        im1_points.append(pt1[0])
        pt2 = plt.ginput(n=1, timeout=0)
        if not pt2:
            break
        im2_points.append((pt2[0][0] - im1.shape[1], pt2[0][1]))
        ax.plot((pt1[0][0], pt2[0][0]), (pt1[0][1], pt2[0][1]), color='yellow', linestyle='-', marker='o', linewidth=0.5, markersize=3)
    plt.close(fig)
    points = {f"{im1_name}": im1_points, f"{im2_name}": im2_points}
    with open(f"points/{im1_name}_{im2_name}_points.json", "w") as json_file:
        json.dump(points, json_file, indent=4)
    return points

def load_points(im1_name, im2_name):
    with open(f"points/{im1_name}_{im2_name}_points.json", "r") as json_file:
        points = json.load(json_file)
    return points

def plot_correspondences(im1, im2, im1_name, im2_name, points):
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.imshow(np.concatenate([im1, im2], axis=1))
    offset = im1.shape[1]

    for (x1, y1), (x2, y2) in zip(points[f"{im1_name}"], points[f"{im2_name}"]):
        ax.plot([x1, x2 + offset], [y1, y2], color='yellow',
                linewidth=0.5, marker='o', markersize=3)
    ax.axis('off')
    fig.savefig(f'outputs/{im1_name}_{im2_name}_correspondences.jpg', dpi=200, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

# We want to warp 1 onto 2, 3 onto 2
def compute_H(im1_pts, im2_pts):
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
    print("H matrix", H)
    return H

def warpImageNearestNeighbor(im, H):
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
    # iterate through all sets of points

    # vectorized implementation
    wx, wy, w = transformed_points[0], transformed_points[1], transformed_points[2]
    x, y = all_points[0], all_points[1]
    x_input = np.round(wx / w).astype(int)
    y_input = np.round(wy / w).astype(int)
    valid_mask = (x_input >= 0) & (y_input >= 0) & (y_input < im.shape[0]) & (x_input < im.shape[1])
    output_alpha_mask[(y - floor(y_min))[valid_mask], (x - floor(x_min))[valid_mask]] = 1
    output_image[(y - floor(y_min))[valid_mask], (x - floor(x_min))[valid_mask], :] = im[y_input[valid_mask], x_input[valid_mask], :]

    # # iterative implementation
    # for (wx, wy, w), (x, y, _) in zip(transformed_points.T, all_points.T):
    #     # for each of the transformed points, find the nearest neighbor in the original image, and set to value in output_image
    #     x_input = np.round(wx / w).astype(int)
    #     y_input = np.round(wy / w).astype(int)
    #     # if we are out of bounds, do nothing
    #     if x_input < 0 or y_input < 0 or y_input >= im.shape[0] or x_input >= im.shape[1]:
    #         pass
    #     else:
    #         output_alpha_mask[y - floor(y_min), x - floor(x_min)] = 1
    #         output_image[y - floor(y_min), x - floor(x_min), :] = im[y_input, x_input, :]

    return output_image, output_alpha_mask, floor(x_min), floor(y_min)

def warpImageBilinear(im, H):
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

def transform_alpha(alpha):
    padded = np.pad(alpha, 1)
    return distance_transform_edt(padded)[1:-1, 1:-1]

def mosaic(ref, left, right, name, get_points):
    if get_points:
        im_1_2 = get_points(left, ref, f"{name}1", f"{name}2")
        im_3_2 = get_points(right, ref, f"{name}3", f"{name}2")
    else:
        im_1_2 = load_points(f"{name}1", f"{name}2")
        im_3_2 = load_points(f"{name}3", f"{name}2")

    plot_correspondences(left, ref, f"{name}1", f"{name}2", im_1_2)
    plot_correspondences(right, ref, f"{name}3", f"{name}2", im_3_2)
    H_1_to_ref = compute_H(im_1_2[f"{name}1"], im_1_2[f"{name}2"])
    H_3_to_ref = compute_H(im_3_2[f"{name}3"], im_3_2[f"{name}2"])
    
    print("Warping left")
    left_warped, left_alpha, left_xmin, left_ymin = warpImageBilinear(left, H_1_to_ref)
    skio.imsave(f'outputs/{name}1_warped.jpg', img_as_ubyte(left_warped))

    print("Warping right")
    right_warped, right_alpha, right_xmin, right_ymin = warpImageBilinear(right, H_3_to_ref)
    skio.imsave(f'outputs/{name}3_warped.jpg', img_as_ubyte(right_warped))

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
    skio.imsave(f'outputs/{name}_result_naive.jpg', img_as_ubyte(get_result(images, [a[:, :, np.newaxis] for a in alphas])))

    dist_alphas = list(map(transform_alpha, alphas))
    skio.imsave(f'outputs/{name}_result_alpha_mask.jpg', img_as_ubyte(get_result(images, [a[:, :, np.newaxis] for a in dist_alphas])))

    high_freq_mask_to_choose = np.argmax(np.stack(dist_alphas), axis=0)
    valid = np.stack(dist_alphas).max(axis=0) > 0
    high_freq_masks = [(high_freq_mask_to_choose == i) & valid for i in range(len(dist_alphas))]

    kernel_size = 11
    low_freq = [GaussianBlur(im, (kernel_size, kernel_size), 0) for im in images]
    high_freq = [im - lo for im, lo in zip(images, low_freq)]
    low_freq_result = get_result(low_freq, [a[:, :, np.newaxis] for a in dist_alphas])
    high_freq_result = get_result(high_freq, [a[:, :, np.newaxis] for a in high_freq_masks])

    skio.imsave(f'outputs/{name}_result_laplacian.jpg', img_as_ubyte(np.clip(low_freq_result + high_freq_result, 0, 1)))

def rectify(im, name, height, width, get_points):
    # get points for the one image
    if get_points:
        plt.figure(figsize=(12, 8)); plt.imshow(im)
        points = plt.ginput(4, timeout=0)
        plt.close()
        with open(f'points/{name}_rectify.json', 'w') as f:
            json.dump(points, f)
    else:
        with open(f'points/{name}_rectify.json', 'r') as f:
            points = np.array(json.load(f))

    # create our array of target points
    target = np.array([[0, 0], [width, 0], [width, height], [0, height]])
    H_rectify = compute_H(points, target)

    # do our warping
    for warp_function, function_name in ((warpImageNearestNeighbor, 'nearest_neighbor'), (warpImageBilinear, 'bilinear')):
        t = time.perf_counter()
        out, _, _, _ = warp_function(im, H_rectify)
        print(f'{function_name}: {time.perf_counter() - t:.2f}s')
        skio.imsave(f'outputs/{name}_rectify_{function_name}.jpg', img_as_ubyte(np.clip(out, 0, 1)))