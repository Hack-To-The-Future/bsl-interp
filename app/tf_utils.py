from typing import List

import numpy as np
import tensorflow as tf
import typer
from cv2 import cv2

IMAGE_SIZE = 92


def save_data(data: List[cv2.VideoCapture], label: str):
    with tf.io.TFRecordWriter(f"{label}.tfrecords") as writer:
        with typer.progressbar(data) as progress:
            for img in progress:
                img = np.asarray(
                    tf.image.resize(img, (IMAGE_SIZE, IMAGE_SIZE)).numpy(),
                    dtype=np.uint8,
                )
                proto = serialise(img.tobytes(), label)
                writer.write(proto.SerializeToString())


def serialise(img_str: str, label: str):
    feature = {
        "label": tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[label.encode()])
        ),
        "image_raw": tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[img_str])
        ),
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))


def load_data(label: str):
    raw_image_dataset = tf.data.TFRecordDataset(f"{label}.tfrecords")
    return raw_image_dataset.map(_parse_image)


def _parse_image(proto):
    image_features = {
        "label": tf.io.FixedLenFeature([], tf.string),
        "image_raw": tf.io.FixedLenFeature([], tf.string),
    }
    image_features = tf.io.parse_single_example(proto, image_features)
    return image_features


def get_image(features):
    image_raw = features["image_raw"].numpy()
    np_img = np.frombuffer(image_raw, dtype=np.uint8)
    return np.resize(np_img, (IMAGE_SIZE, IMAGE_SIZE, 3))


def format_tf(data: List[cv2.VideoCapture], label: str):
    dataset = tf.data.Dataset.from_tensor_slices(
        (tf.constant(data), tf.constant([label] * len(data)))
    )
    dataset = dataset.map(proc_img)
    return dataset


def proc_img(frame: cv2.VideoCapture, label):
    img = (tf.cast(frame, tf.float32) / 127.5) - 1
    img = tf.image.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
    return img
