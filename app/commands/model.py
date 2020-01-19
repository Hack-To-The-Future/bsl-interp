import typer

app = typer.Typer()


@app.command("train")
def train_model():
    """
    Trains a transfer learning model from the datasets saved
    """
    typer.echo("Not implemented!!")


@app.command("run")
def run_model():
    """
    Runs the previously trained model
    """
    typer.echo("Not implemented!!")
