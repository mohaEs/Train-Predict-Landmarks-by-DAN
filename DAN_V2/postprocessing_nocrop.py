from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os 
import sys
import glob
import random
import numpy as np
import cv2
import uuid

import tensorflow as tf
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--input_dir",  help="path to folder containing images")
parser.add_argument("--output_dir",  help="where to p")
parser.add_argument("--img_size",  help="")

# export options
a = parser.parse_args()

input_dir=a.input_dir
img_size=int(a.img_size)
mirror_file=None
output_dir=a.output_dir
repeat=1

BATCH_SIZE = 128

def getAffine(From, To):
    FromMean = np.mean(From, axis=0)
    ToMean = np.mean(To, axis=0)

    FromCentralized = From - FromMean
    ToCentralized = To - ToMean

    FromVector = (FromCentralized).flatten()
    ToVector = (ToCentralized).flatten()

    DotResult = np.dot(FromVector, ToVector)
    NormPow2 = np.linalg.norm(FromCentralized) ** 2

    a = DotResult / NormPow2
    b = np.sum(np.cross(FromCentralized, ToCentralized)) / NormPow2

    R = np.array([[a, b], [-b, a]])
    T = ToMean - np.dot(FromMean, R)

    return R, T


def _load_data(imagepath, ptspath):
    def makerotate(angle):
        rad = angle * np.pi / 180.0
        return np.array([[np.cos(rad), np.sin(rad)], [-np.sin(rad), np.cos(rad)]], dtype=np.float32)

    srcpts = np.genfromtxt(ptspath.decode())#skip_header=0,  skip_footer=1
    # print(srcpts)
    # x, y = np.min(srcpts, axis=0).astype(np.int32)
    # w, h = np.ptp(srcpts, axis=0).astype(np.int32)

    # print('***************** x,y')
    # print(x, y)
    # print('***************** w,h')
    # print(w, h)
    # pts = (srcpts - [x, y]) / [w, h]
    # print('***************** pts')
    # print(pts)

    img = cv2.imread(imagepath.decode(), cv2.IMREAD_UNCHANGED)
    try:
        height, width, channels =img.shape
    except:
        height, width=img.shape
    pts = srcpts/height



    pts = pts * img_size

    R, T = getAffine(srcpts, pts)
    M = np.zeros((2, 3), dtype=np.float32)
    M[0:2, 0:2] = R.T
    M[:, 2] = T
    img = cv2.warpAffine(img, M, (img_size, img_size))


    _,filename = os.path.split(imagepath.decode())
    filename,_ = os.path.splitext(filename)

    uid = str(uuid.uuid1())


    cv2.imwrite(os.path.join(output_dir,filename + '_pred.png'),img)
    np.savetxt(os.path.join(output_dir,filename + '_pred.pts'), pts ,delimiter=', ',fmt='%.1f')

    return img,pts.astype(np.float32)

def _input_fn(img, pts):
    dataset_image = tf.data.Dataset.from_tensor_slices(img)
    dataset_pts = tf.data.Dataset.from_tensor_slices(pts)
    dataset = tf.data.Dataset.zip((dataset_image, dataset_pts))

    dataset = dataset.prefetch(BATCH_SIZE)
    dataset = dataset.repeat(repeat)
    dataset = dataset.map(lambda imagepath, ptspath: tuple(tf.py_func(_load_data, [
                          imagepath, ptspath], [tf.uint8,tf.float32])), num_parallel_calls=8)                     
    dataset = dataset.prefetch(1)

    return dataset

def _get_filenames(data_dir, listext):
    imagelist = []
    for ext in listext:
        p = os.path.join(data_dir, ext)
        imagelist.extend(glob.glob(p))

    ptslist = []
    for image in imagelist:
        ptslist.append(os.path.splitext(image)[0] + ".pts")

    return imagelist, ptslist

def main(argv):
    imagenames, ptsnames = _get_filenames(input_dir, ["*.jpg", "*.png"])
    print('***********************************')
    print(input_dir)
    print('***********************************')
    #print(imagenames)
    mirror_array = np.genfromtxt(mirror_file, dtype=int, delimiter=',') if mirror_file else np.zeros(1)
    
    dataset = _input_fn(imagenames,ptsnames)
    print('***********************************')
    print(dataset)
    next_element = dataset.make_one_shot_iterator().get_next()
    print('***********************************')
    print(dataset)
    
    img_list = []
    pts_list = []

    with tf.Session() as sess:
        count = 0
        while True:
            try:
                
                img,pts = sess.run(next_element)
                img_list.append(img)
                pts_list.append(pts)
            except tf.errors.OutOfRangeError:

                print("end")
                break


if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    print(sys.argv)
    tf.app.run(argv=sys.argv)