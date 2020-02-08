from unprocess import unprocess
import glob
import cv2
import tensorflow as tf
img = cv2.imread('./marker_2199.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
print(img.shape)
img = tf.cast(img, tf.float32) / 255.0

un, label = unprocess(img)
print(un.shape)
print(type(un))
print(label)
sess = tf.Session()
with sess.as_default():
    cv2.imwrite('xxximage.png', un[:,:,:3].eval())
    print('****************************')
    print(f'the shape of un:{un.shape}')
    print(type(un.eval()))
    print(label)

