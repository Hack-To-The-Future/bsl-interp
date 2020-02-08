import numpy as np
import typer
from cv2 import cv2
import tensorflow as tf

from app.tf_utils import (init_model, load_classifier, load_label_dict,
                          process_image, get_label)
from app.utils import init_webcam, web_img_gen, add_text_to_frame

app = typer.Typer()


@app.command("run")
def run_model(
    flip: bool = typer.Option(
        True,
        help="Flips the webcam horizontally"
    )
):
    """
    Runs the previously trained model
    """

    classifier = load_classifier()
    labels = load_label_dict()
    model = init_model()
    cap = init_webcam()

    for frame in web_img_gen(cap, flip):
        img = process_image(frame)
        prediction = model.predict(img[np.newaxis, ...])
        probs = tf.nn.softmax(prediction).numpy()
        res = classifier.predict(probs)
        prob = classifier.predict_proba(probs)

        typer.echo(res)
        add_text_to_frame(
            frame,
            f"{get_label(labels, res[0])} conf: {prob}"
        )

        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
