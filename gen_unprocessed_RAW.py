from unprocess import unprocess
import glob
import cv2
import tensorflow as tf
import glob
from tqdm import tqdm
import os
import numpy as np

import pickle

IMG_DIR='/tmp3/r07922076/coco/coco2017/train2017'
OUT_DIR='/tmp3/r07922076/unprocessed_RAW/train2017'

if not os.path.exists(OUT_DIR):
    os.makedirs(OUT_DIR)


config = tf.ConfigProto()
config.gpu_options.allow_growth = True

sess = tf.Session(config=config)
with sess.as_default():
    for imgpath in tqdm(sorted(glob.glob(os.path.join(IMG_DIR, '*.jpg')))):
        img = cv2.imread(imgpath)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # check if img contain odd height / width
        h, w, _ = img.shape
        if img.shape[0] % 2 == 1:
            h = img.shape[0] + 1
        if img.shape[1] % 2 == 1:
            w = img.shape[1] + 1

        plane = np.zeros((h,w,3))
        plane[:img.shape[0],:img.shape[1],:] = img[:,:,:]
        
        plane = tf.cast(plane, tf.float32) / 255.0
        un, metadata = unprocess(plane)
        
        copy_metadata = dict()
        for key in metadata:
            copy_metadata[key] = metadata[key].eval()
        
        path_raw = os.path.join(OUT_DIR, os.path.basename(imgpath).replace('jpg','pkl'))
        
        with open(path_raw, 'wb') as pf:
            content = dict()
            content['raw'] = un.eval()
            content['metadata'] = copy_metadata
            pickle.dump(content, pf)