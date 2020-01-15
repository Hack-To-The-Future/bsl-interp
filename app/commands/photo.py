import typer
import time

from app import app
from app.utils import save_img, webcam_capture, init_webcam


@app.command("photo")
def save_photo(
    name: str,
    flip: bool = typer.Option(True, help="Flips the webcam horizontally")
):
    """
    Takes a photo and saves it with the NAME provided
    """
    cap = init_webcam()
    timer = 4
    start_time = time.time()
    while timer > 0:
        frame = webcam_capture(cap, flip, f"{timer}")
        if (time.time() - start_time) > 1:
            start_time = time.time()
            timer -= 1

    save_img(name, frame)
