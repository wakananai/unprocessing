from unprocess import unprocess
import glob
import cv2
import tensorflow as tf
import glob
from tqdm import tqdm
import os
import numpy as np

import pickle
import argparse

# IMG_DIR='/tmp3/r07922076/coco/coco2017/val2017'
# OUT_DIR='/tmp3/r07922076/unprocessed_COCO2017_RAW/val2017'

parser = argparse.ArgumentParser()
parser.add_argument('--ev', type=int, required=True)
args = parser.parse_args()


subdir = f'decay_{args.ev}'
IMG_DIR= f'/tmp3/r07922076/PASCALRAW_original/{subdir}/jpg'
OUT_DIR= f'/tmp3/r07922076/unprocessed_PASCALRAW_RAW/{subdir}'

if not os.path.exists(OUT_DIR):
    os.makedirs(OUT_DIR)


config = tf.ConfigProto()
config.gpu_options.allow_growth = True


input_image = tf.placeholder(tf.float32, shape=[None, None, 3])

un_raw, meta = unprocess(input_image)

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
        plane = plane.astype(np.float32) / 255.0
        un, metadata = sess.run([un_raw, meta], feed_dict={input_image: plane})

        path_raw = os.path.join(OUT_DIR, os.path.basename(imgpath).replace('jpg','pkl'))

        with open(path_raw, 'wb') as pf:
            content = dict()
            content['raw'] = un
            content['metadata'] = metadata
            pickle.dump(content, pf)
