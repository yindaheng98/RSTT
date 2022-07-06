"""Create lmdb files for Vimeo90K training dataset.
"""
import os
import sys
import glob
import pickle
import numpy as np
import lmdb
import cv2
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import ProgressBar

def create_lmdb(data_path, text_path, save_path):
    """Create lmdb for Vimeo90k dataset.

    Args:
        data_path (str): Path to LR/HR images.
        text_path (str): Path to sep_trainlist.txt.
        save_path (str): Path to save lmdb file.
    """
    batch = 3000

    if os.path.exists(save_path):
        print('Folder [{:s}] already exists.'.format(save_path))
        return

    # Read all image paths to a list
    print('Reading image path list ...')
    with open(text_path) as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]

    imgs, keys = [], []
    for line in lines:
        dir = line.split('/')[0]
        subdir = line.split('/')[1]
        filenames = glob.glob(os.path.join(data_path, dir, subdir) + '/*')
        imgs.extend(filenames)
        for j in range(7):
            keys.append('{}_{}_{}'.format(dir, subdir, j + 1))

    imgs, keys = sorted(imgs), sorted(keys)
    imgs = [v for v in imgs if v.endswith('.png')]
    print('Calculating the total size of images...')
    data_size = sum(os.stat(v).st_size for v in imgs)

    # Create lmdb environment
    env = lmdb.open(save_path, map_size=data_size * 10)
    txn = env.begin(write=True)  # txn is a Transaction object

    # Write data to lmdb
    pbar = ProgressBar(len(imgs))

    i = 0
    for path, key in zip(imgs, keys):
        pbar.update('Write {}'.format(key))
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        key_byte = key.encode('ascii')
        H, W, C = img.shape  # fixed shape
        txn.put(key_byte, img)
        i += 1
        if  i % batch == 1:
            txn.commit()
            txn = env.begin(write=True)

    txn.commit()
    env.close()
    print('Finish reading and writing {} images.'.format(len(imgs)))
    print('Finish writing lmdb.')

    # Create meta information
    print('Creating lmdb meta information.')
    meta_info = {}
    meta_info['name'] = save_path.split('/')[-1][:-4]
    meta_info['resolution'] = '{}_{}_{}'.format(C, H, W)
    key_set = set()
    for key in keys:
        a, b, _ = key.split('_')
        key_set.add('{}_{}'.format(a, b))
    meta_info['keys'] = key_set
    pickle.dump(meta_info, open(os.path.join(save_path, 'Vimeo_keys.pkl'), "wb"))
    print('Finish creating lmdb meta information.')

def get_list(text_path):
    # Read all image paths to a list
    with open(text_path) as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
    return lines

def get_keys(text_path):
    # Read all image paths to a list
    lines = get_list(text_path)
    keys = []
    for line in lines:
        dir = line.split('/')[0]
        subdir = line.split('/')[1]
        for j in range(7):
            keys.append('{}_{}_{}'.format(dir, subdir, j + 1))

    keys = sorted(keys)
    key_set = set()
    for key in keys:
        a, b, _ = key.split('_')
        key_set.add('{}_{}'.format(a, b))
    return key_set

if __name__ == "__main__":
    text_path = "./vimeo90k/vimeo_septuplet/sep_trainlist.txt"
    print(get_keys(text_path))