import typer
from typing import List
from cv2 import cv2


def init_webcam():
    typer.echo(f"Starting webcam...")
    return cv2.VideoCapture(0)


def webcam_capture(
    cap: cv2.VideoCapture,
    flip: bool,
    text: str = None,
    filename: str = None,
    dataset: List[cv2.VideoCapture] = None
):

    bottomLeftCornerOfText = (10, 40)
    fontScale = 1
    fontColor = (255, 255, 255)
    lineType = 2

    ret, frame = cap.read()
    if flip:
        frame = cv2.flip(frame, 1)

    if text is not None:
        cv2.putText(
            frame, text,
            bottomLeftCornerOfText,
            cv2.FONT_HERSHEY_SIMPLEX,
            fontScale,
            fontColor,
            lineType
        )

    cv2.imshow('frame', frame)
    cv2.waitKey(1)

    if filename is not None:
        cv2.imwrite(f"{filename}.jpg", frame)

    if dataset is not None:
        dataset.append(frame)

    return dataset
