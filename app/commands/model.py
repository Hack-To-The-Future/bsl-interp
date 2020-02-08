import numpy as np
import typer
from datetime import datetime
from cv2 import cv2
import tensorflow as tf

from app.tf_utils import (
    init_model,
    load_classifier,
    load_label_dict,
    process_image,
    get_label,
)
from app.utils import init_webcam, web_img_gen, add_text_to_frame

app = typer.Typer()


@app.command("run")
def run_model(
    flip: bool = typer.Option(True, help="Flips the webcam horizontally")
):
    """
    Runs the previously trained model
    """

    classifier = load_classifier()
    labels = load_label_dict()
    model = init_model()
    cap = init_webcam()

    word = " "
    last_time = datetime.utcnow()

    for frame in web_img_gen(cap, flip):
        img_time = datetime.utcnow()
        img = process_image(frame)
        prediction = model.predict(img[np.newaxis, ...])
        probs = tf.nn.softmax(prediction).numpy()
        res = classifier.predict(probs)[0]
        prob = classifier.predict_proba(probs)[0]

        label = get_label(labels, res)
        add_text_to_frame(frame, f"{label} conf: {prob}")

        isnt_empty_label = not (label == "empty")
        is_good_prob = prob[res] > 0.99
        enough_time_passed = (img_time - last_time).seconds > 1
        is_different_letter = not (label == word[-1])

        if (
            isnt_empty_label
            and is_good_prob
            and enough_time_passed
            and is_different_letter
        ):
            word += label
            last_time = datetime.utcnow()
            typer.echo(word)
            cv2.imwrite(f"{label}.jpg", frame)

        cv2.imshow("frame", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    typer.echo(f"Recorded word: {word}")
