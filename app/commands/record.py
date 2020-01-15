import typer
import time

from app import app
from app.utils import webcam_capture, init_webcam


@app.command("record")
def record_dataset(
    label: str,
    flip: bool = typer.Option(True, help="Flips the webcam horizontally"),
    record_time: int = typer.Option(5, help="Seconds of data to record")
):
    """
    Records images from the webcam and stores them as a
    tensorflow dataset with the given LABEL
    """
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

        if ((time.time() - start_time) > 1):
            start_time = time.time()
            timer -= 1

    typer.echo(f"Recorded {len(data)} images!")
    typer.Exit()
