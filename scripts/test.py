import pytorch_lightning as pl
from pathlib import Path
from argparse import ArgumentParser

import sys

#sys.path.append("..")
from src.models import Model


def get_checkpoint(data_folder: Path, version: int) -> str:

    log_folder = data_folder / "lightning_logs" / f"version_{version}" / "checkpoints"
    checkpoint = list(log_folder.glob("*.ckpt"))
    print(log_folder.resolve())
    return str(checkpoint[0])


def test_model():
    parser = ArgumentParser()

    parser.add_argument("--version", type=int, default=19)

    args = parser.parse_args()

    #model_path = get_checkpoint(Path("../data"), args.version)
    model_path = get_checkpoint(Path("data"), args.version)

    print(f"Using model {model_path}")

    model = Model.load_from_checkpoint(model_path)
    model.cuda()

    trainer = pl.Trainer()
    trainer.test(model)


if __name__ == "__main__":
    test_model()
