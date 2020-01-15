from app import app
from app.commands.photo import save_photo # noqa
from app.commands.test import test_webcam  # noqa
from app.commands.record import record_dataset  # noqa
from app.commands.train import train_model  # noqa
from app.commands.run import run_model  # noqa

if __name__ == "__main__":
    app()
