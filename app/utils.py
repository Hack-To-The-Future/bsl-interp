import typer
from typing import List
from cv2 import cv2


def init_webcam():
    typer.echo(f"Starting webcam...")
    return cv2.VideoCapture(0)


def add_text_to_frame(frame: cv2.VideoCapture, text: str):
    bottomLeftCornerOfText = (10, 40)
    fontScale = 1
    fontColor = (255, 255, 255)
    lineType = 2

    cv2.putText(
        frame,
        text,
        bottomLeftCornerOfText,
        cv2.FONT_HERSHEY_SIMPLEX,
        fontScale,
        fontColor,
        lineType,
    )


def webcam_capture(
    cap: cv2.VideoCapture,
    flip: bool,
    text: str = None,
    data: List[str] = None,
):

    ret, frame = cap.read()
    if flip:
        frame = cv2.flip(frame, 1)

    if text is not None:
        add_text_to_frame(frame, text)

    cv2.imshow("frame", frame)
    cv2.waitKey(1)

    if data is not None:
        data.append(frame)


def web_img_gen(
    cap: cv2.VideoCapture,
    flip: bool,
):

    while True:

        ret, frame = cap.read()
        if flip:
            frame = cv2.flip(frame, 1)

        yield frame
