import typer

from app import app


@app.command("run")
def run_model():
    """
    Runs the previously trained model
    """
    typer.echo("Not implemented!!")
