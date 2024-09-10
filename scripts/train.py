from datetime import datetime
from pathlib import Path
from typing import cast

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
from transformers import (
    DonutProcessor,
    VisionEncoderDecoderConfig,
    VisionEncoderDecoderModel,
)

from src.domain.dataset import Dataset
from src.domain.model import Model

BASE_MODEL = Path("model_output/base_model")
IMAGE_SIZE = (700, 500)
TRAIN_PATH = Path("dataset/train")
VALIDATION_PATH = Path("dataset/validation")


def train() -> None:
    batch_size = 1
    lr = 1e-6
    epoch_num = 3
    model_output_path = Path(
        f"model_output/donut_{batch_size}_{lr}_{round(datetime.now().timestamp())}",
    )

    config = VisionEncoderDecoderConfig.from_pretrained(BASE_MODEL)
    base_model = cast(
        VisionEncoderDecoderModel,
        VisionEncoderDecoderModel.from_pretrained(
            BASE_MODEL,
            config=config,
        ),
    )
    processor = cast(DonutProcessor, DonutProcessor.from_pretrained(BASE_MODEL))
    model = Model(processor, base_model, lr, epoch_num)

    training_dataset = Dataset.load(TRAIN_PATH, model, training=True)

    validation_dataset = Dataset.load(VALIDATION_PATH, model, training=False)

    train_dataloader = DataLoader(
        training_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=16,
    )

    val_dataloader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=16,
    )

    trainer = Trainer(
        accelerator="gpu",
        devices=1,
        max_epochs=epoch_num,
        check_val_every_n_epoch=1,
        gradient_clip_val=1.0,
        precision="16-mixed",
        num_sanity_val_steps=0,
        callbacks=[
            ModelCheckpoint(
                dirpath=model_output_path,
                filename="every_{epoch}_{v_num}",
                every_n_epochs=1,
                save_top_k=-1,
            ),
        ],
    )

    trainer.fit(model, train_dataloader, val_dataloader)

    model.model.save_pretrained(model_output_path)
    model.processor.save_pretrained(model_output_path)


if __name__ == "__main__":
    train()
