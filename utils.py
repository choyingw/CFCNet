import os
import torch
import shutil
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2

def colored_depthmap(depth, d_min=None, d_max=None):
    if d_min is None:
        d_min = np.min(depth)
    if d_max is None:
        d_max = np.max(depth)
    depth_relative = (depth - d_min) / (d_max - d_min)
    return 255 * plt.cm.jet(depth_relative)[:,:,:3] # H, W, C

cmap = plt.cm.jet
def depth_colorize_16(depth):
    depth = (depth - np.min(depth)) / (np.max(depth) - np.min(depth))
    depth = 255* 256 * cmap(depth)[:,:,:3] # H, W, C
    return depth.astype('uint16')

def depth_colorize_8(depth):
    depth = (depth - np.min(depth)) / (np.max(depth) - np.min(depth))
    depth = 255* cmap(depth)[:,:,:3] # H, W, C
    return depth.astype('uint8')

def Enlarge_pixel(sparse_depth):
    for i in range(2,sparse_depth.shape[0]-2):
        for j in range(2,sparse_depth.shape[1]-2):
            if np.sum(sparse_depth[i][j]) > 0:
                for w in range(-2,2):
                    for h in range(-2,2):
                        sparse_depth[i+w][j+h] = sparse_depth[i][j]

    return sparse_depth

def merge_into_row_with_pred_visualize(input, depth_input, rgb_sparse, depth_target, depth_est):
    rgb = 255 * np.transpose(np.squeeze(input.cpu().numpy()), (1,2,0))[:,:,(2,1,0)] # H, W, C
    rgb_sparse = 255 * np.transpose(np.squeeze(rgb_sparse.cpu().numpy()), (1,2,0))[:,:,(2,1,0)]
    depth_input_cpu = np.squeeze(depth_input.cpu().numpy())
    depth_target_cpu = np.squeeze(depth_target.cpu().numpy())
    depth_pred_cpu = np.squeeze(depth_est.cpu().numpy())

    d_min = min(np.min(depth_input_cpu), np.min(depth_target_cpu), np.min(depth_pred_cpu))
    d_max = max(np.max(depth_input_cpu), np.max(depth_target_cpu), np.min(depth_pred_cpu))
    depth_input_col = colored_depthmap(depth_input_cpu, d_min, d_max)
    # depth_target_col = colored_depthmap(depth_target_cpu, d_min, d_max)
    # depth_pred_col = colored_depthmap(depth_pred_cpu, d_min, d_max)
    depth_input_col = (depth_colorize_8(depth_input_cpu))
    depth_target_col = (depth_colorize_8(depth_target_cpu))
    depth_pred_col = depth_colorize_8(depth_pred_cpu)

    img_merge = np.hstack([rgb, depth_pred_col])
    #img_merge = np.hstack([rgb,depth_input_col])
    #depth_merge = np.hstack([depth_pred_col,depth_target_col])
    #img_merge = np.vstack([img_merge,depth_merge])

    return img_merge

def add_row(img_merge, row):
    return np.vstack([img_merge, row])

def save_image(img_merge, filename):
    img_merge = Image.fromarray(img_merge.astype('uint8'))
    img_merge.save(filename)

def save_image_cv2(image_numpy, image_path):
    #image_pil = Image.fromarray(image_numpy)
    cv2.imwrite(image_path,image_numpy)
    #image_pil.save(image_path)


