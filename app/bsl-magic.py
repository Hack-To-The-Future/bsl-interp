from app import app

from app.commands import model
from app.commands import webcam
from app.commands import datasets

app.add_typer(
    webcam.app,
    name="webcam",
    help="Webcam interaction"
)
app.add_typer(
    datasets.app,
    name="datasets",
    help="Record datasets for training"
)
app.add_typer(
    model.app,
    name="model",
    help="Train and run the model"
)


if __name__ == "__main__":
    app()
