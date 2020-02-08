import os
import time

import typer

from app.tf_utils import CLASS_FILE, LABEL_FILE, LABELS_FILE, LOGITS_FILE
from app.tf_utils import init_model, load_label_dict, save_data
from app.utils import init_webcam, webcam_capture

app = typer.Typer()


@app.command("create")
def create_dataset(
    label: str,
    flip: bool = typer.Option(True, help="Flips the webcam horizontally"),
    record_time: int = typer.Option(5, help="Seconds of data to record"),
):
    """
    Records images from the webcam and stores them as a
    tensorflow dataset with the given LABEL
    """
    model = init_model()
    if os.path.exists(f"{label}.tfrecord"):
        os.remove(f"{label}.tfrecord")
    cap = init_webcam()
    timer = 4
    start_time = time.time()
    data = []
    while timer > -(record_time + 1):
        text = f"{timer}"
        if timer < 0:
            text = "Recording!"
            webcam_capture(cap, flip, text, data)
        else:
            webcam_capture(cap, flip, text, None)

        if (time.time() - start_time) > 1:
            start_time = time.time()
            timer -= 1

    save_data(data, label, model)

    typer.echo(f"Recorded images!")
    typer.Exit()


@app.command("list")
def list_labels():
    """
    Lists datasets and labels currently saved
    """
    ld = load_label_dict()
    for label in ld.keys():
        typer.echo(label)
    tl = len(ld)
    typer.echo(f"Found {tl} labels in total")


@app.command("erase")
def remove_model():
    """
    Removes the model
    """
    # TODO: extend to remove individual labels
    files = [LABEL_FILE, CLASS_FILE, LABELS_FILE, LOGITS_FILE]

    for f in files:
        if os.path.isfile(f):
            os.remove(f)

    typer.echo("Model deleted")
