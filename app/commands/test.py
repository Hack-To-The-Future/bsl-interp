import typer

from cv2 import cv2

from app import app
from app.utils import webcam_capture, init_webcam


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
        if cv2.waitKey(1) & 0xFF == ord('q'):
            typer.echo(f"Exiting...")
            break

    typer.Exit()
