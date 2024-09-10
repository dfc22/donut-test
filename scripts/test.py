from pathlib import Path
from typing import cast

from PIL import Image
from torch import Tensor
from torchvision.transforms.v2.functional import pil_to_tensor, to_grayscale, to_pil_image
from transformers import (
    DonutProcessor,
    VisionEncoderDecoderConfig,
    VisionEncoderDecoderModel,
)

from src.domain.model import Model

MODEL_PATH = Path("model_output/donut_1_1e-06_1725948282")
IMAGE_PATH = Path("dummy_business_card.png")


def inference() -> None:
    config = VisionEncoderDecoderConfig.from_pretrained(MODEL_PATH)
    ved_model = cast(
        VisionEncoderDecoderModel,
        VisionEncoderDecoderModel.from_pretrained(
            MODEL_PATH,
            config=config,
        ),
    )
    processor = cast(DonutProcessor, DonutProcessor.from_pretrained(MODEL_PATH))
    model = Model(processor, ved_model)

    image = Image.open(IMAGE_PATH)
    processed_image = to_pil_image(to_grayscale(pil_to_tensor(image)))
    pixel_values = cast(
        Tensor,
        model.processor(
            processed_image.convert("RGB"),
            return_tensors="pt",
        ).pixel_values,
    )

    result = model.inference(pixel_values)

    print(result)


if __name__ == "__main__":
    inference()
