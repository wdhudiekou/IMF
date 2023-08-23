import os
import cv2
import kornia
import torch
import numpy as np

def softmax(map1, map2, c):
    exp_x1 = np.exp(map1*c)
    exp_x2 = np.exp(map2*c)
    exp_sum = exp_x1 + exp_x2
    map1 = exp_x1/exp_sum
    map2 = exp_x2/exp_sum
    return map1, map2

def vsm(img):
    his = np.zeros(256, np.float64)
    for i in range(img.shape[0]): # 256
        for j in range(img.shape[1]): # 256
            his[img[i][j]] += 1
    sal = np.zeros(256, np.float64)
    for i in range(256):
        for j in range(256):
            sal[i] += np.abs(j - i) * his[j]
    map = np.zeros_like(img, np.float64)
    for i in range(256):
        map[np.where(img == i)] = sal[i]
    if map.max() == 0:
        return np.zeros_like(img, np.float64)
    return map / (map.max())

if __name__ == '__main__':

    ir_path = "../dataset/raw/ctrain/RoadScene/ir/"
    vi_path = "../dataset/raw/ctrain/RoadScene/vi/"

    ir_file_list = sorted(os.listdir(ir_path))
    vi_file_list = sorted(os.listdir(vi_path))

    ir_map_path = "../dataset/raw/ctrain/RoadScene/ir_map_soft/"
    vi_map_path = "../dataset/raw/ctrain/RoadScene/vi_map_soft/"

    if not os.path.exists(ir_map_path):
        os.makedirs(ir_map_path)
    if not os.path.exists(vi_map_path):
        os.makedirs(vi_map_path)

    for idx, (ir_filename, vi_filename) in enumerate(zip(ir_file_list, vi_file_list)):

        ir_filepath = os.path.join(ir_path, ir_filename)
        vi_filepath = os.path.join(vi_path, vi_filename)

        img_ir = cv2.imread(ir_filepath, cv2.IMREAD_GRAYSCALE)
        img_vi = cv2.imread(vi_filepath, cv2.IMREAD_GRAYSCALE)

        map_ir = vsm(img_ir)
        map_vi = vsm(img_vi)

        w_ir, w_vi = softmax(map_ir, map_vi, c=5)

        img_w_ir = (w_ir * 255).astype(np.uint8)
        img_w_vi = (w_vi * 255).astype(np.uint8)

        ir_save_name = os.path.join(ir_map_path, ir_filename)
        vi_save_name = os.path.join(vi_map_path, vi_filename)

        cv2.imwrite(ir_save_name, img_w_ir)
        cv2.imwrite(vi_save_name, img_w_vi)

