import os
import pickle
from typing import Dict, List

import numpy as np
import tensorflow as tf
import typer
from cv2 import cv2
from sklearn import neighbors

IMAGE_SIZE = 96  # options: [96, 128, 160, 192, 224]
LABEL_FILE = "labels.pkl"
CLASS_FILE = "class.pkl"
WEIGHTS = "distance"  # or use uniform
NN = 100


def save_data(
    data: List[cv2.VideoCapture],
    label: str,
    model: tf.keras.applications.MobileNetV2
):
    convt_imgs = []
    with typer.progressbar(data) as progress:
        for img in progress:
            convt_imgs.append(process_image(img))

    # Get predictions
    images4d = np.asarray(convt_imgs)
    predictions = model.predict(images4d)
    typer.echo(f"Predictions completed for {len(convt_imgs)} images")
    label_dict = load_label_dict()
    label_num = set_label_number(label, label_dict)

    # Update model
    classifier = load_classifier()
    classifier.fit(predictions, [label_num] * len(convt_imgs))

    # Save model and classifier
    save_label_dict(label_dict)
    save_classifier(classifier)


def process_image(img: cv2.VideoCapture) -> np.ndarray:
    img = (tf.cast(img, tf.float32)/127.5) - 1
    img = tf.image.resize(img, (IMAGE_SIZE, IMAGE_SIZE)).numpy()
    return img


def load_label_dict() -> Dict[str, int]:
    if os.path.isfile(LABEL_FILE):
        with open(LABEL_FILE, "rb") as f:
            return pickle.loads(f.read())
    else:
        return {}


def save_label_dict(label_dict: Dict[str, int]):
    with open(LABEL_FILE, "wb+") as f:
        f.write(pickle.dumps(label_dict))


def load_classifier() -> neighbors.KNeighborsClassifier:
    if os.path.isfile(CLASS_FILE):
        with open(CLASS_FILE, "rb") as f:
            return pickle.loads(f.read())
    else:
        return neighbors.KNeighborsClassifier(NN, weights=WEIGHTS)


def save_classifier(knn: neighbors.KNeighborsClassifier):
    with open(CLASS_FILE, "wb+") as f:
        f.write(pickle.dumps(knn))


def set_label_number(label, label_dict: Dict[str, int]):
    if label in label_dict:
        return label_dict[label]
    else:
        num = len(label_dict)
        label_dict[label] = num
        return num


def get_label(labels: Dict[str, int], num: int) -> str:
    return list(labels.keys())[list(labels.values()).index(num)]


def init_model():
    IMG_SHAPE = (IMAGE_SIZE, IMAGE_SIZE, 3)

    # Pre-trained model with MobileNetV2
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=IMG_SHAPE,
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False
    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
    model = tf.keras.Sequential([
        base_model,
        global_average_layer
    ])
    return model
