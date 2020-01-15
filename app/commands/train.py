import typer

from app import app


@app.command("train")
def train_model():
    """
    Trains a transfer learning model from the datasets saved
    """
    typer.echo("Not implemented!!")
