import typer
import time
import os

from app.utils import webcam_capture, init_webcam
from app.tf_utils import save_data, load_data

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
    if os.path.exists(f"{label}.tfrecord"):
        os.remove(f"{label}.tfrecord")
    cap = init_webcam()
    timer = 4
    start_time = time.time()
    data = []
    while timer > -record_time:
        text = f"{timer}"
        if timer < 0:
            text = "Recording!"
            webcam_capture(cap, flip, text, None, data)
        else:
            webcam_capture(cap, flip, text, None, None)

        if (time.time() - start_time) > 1:
            start_time = time.time()
            timer -= 1

    save_data(data, label)

    typer.echo(f"Recorded images!")
    typer.Exit()


@app.command("load")
def load_datasets(label: str):
    """
    Loads dataset with given label
    """
    data = load_data(label)
    typer.echo(f"Loaded {len(data)} images")


@app.command("list")
def list_datasets():
    """
    Lists datasets and labels currently saved
    """
    typer.echo("Not implemented")
