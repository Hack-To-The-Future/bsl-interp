import typer
import time

from cv2 import cv2

from app.utils import webcam_capture, init_webcam

app = typer.Typer()


@app.command("test")
def test_webcam(
    flip: bool = typer.Option(True, help="Flips the webcam horizontally")
):
    """
    This command is used to test the webcam,
    press 'q' to exit
    """
    cap = init_webcam()
    while True:
        webcam_capture(cap, flip)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            typer.echo(f"Exiting...")
            break

    typer.Exit()


@app.command("photo")
def save_photo(
    name: str,
    flip: bool = typer.Option(True, help="Flips the webcam horizontally"),
):
    """
    Takes a photo and saves it with the NAME provided
    """
    cap = init_webcam()
    timer = 4
    start_time = time.time()
    while timer > 0:
        webcam_capture(cap, flip, f"{timer}")
        if (time.time() - start_time) > 1:
            start_time = time.time()
            timer -= 1

    webcam_capture(cap, flip, f"{timer}", name)
    typer.Exit()
