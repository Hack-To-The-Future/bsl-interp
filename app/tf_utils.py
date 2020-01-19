import tensorflow as tf

from cv2 import cv2
from typing import List

IMAGE_SIZE = 92


def format_tf(data: List[cv2.VideoCapture], label):

    dataset = tf.data.Dataset.from_tensor_slices(
        (tf.constant(data), tf.constant([label]*len(data)))
    )
    dataset = dataset.map(proc_img)
    return dataset


def proc_img(frame: cv2.VideoCapture, label):
    img = (tf.cast(frame, tf.float32)/127.5) - 1
    img = tf.image.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
    return img
