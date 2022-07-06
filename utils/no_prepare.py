import os
import glob
import sys
import numpy as np
from .eval_utils import read_image
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import datasets.create_lmdb as create_lmdb
import datasets.generate_LR as generate_LR

def get_HR_paths(dataroot_HR, list_path):
    lines = create_lmdb.get_list(list_path)
    path_set = {}
    for line in lines:
        p, s = line.split('/')
        p, s = os.path.join(dataroot_HR, p), os.path.join(dataroot_HR, p, s)
        if p in path_set:
            path_set[p].add(s)
        else:
            ss = set()
            ss.add(s)
            path_set[p] = ss
    return path_set

def downsample(image, up_scale):
    return generate_LR.downsample(image, up_scale)

def read_seqseq_images(GT_path, up_scale):
    """Read a sequence of images, both LR and HR.

    Args:
        path (str): The path of the image sequence.

    Returns:
        array: (N, H, W, C) RGB images.
    """
    imgs_GT_path = sorted(glob.glob(os.path.join(GT_path, '*')))
    imgs_GT = [read_image(img_path) for img_path in imgs_GT_path]
    imgs_LR = np.stack([downsample(img.copy(), up_scale) for img in imgs_GT], axis=0)
    imgs_GT = np.stack(imgs_GT, axis=0)
    return imgs_GT, imgs_LR

if __name__ == "__main__":
    dataroot_HR = "./vimeo90k/vimeo_septuplet/sequences"
    list_path = "./vimeo90k/vimeo_septuplet/sep_trainlist.txt"
    print(get_HR_paths(dataroot_HR, list_path))