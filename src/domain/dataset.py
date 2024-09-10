import json
from pathlib import Path
from typing import cast

from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset as TorchDataset
from torchvision.transforms.v2.functional import pil_to_tensor, to_grayscale, to_pil_image

from src.domain.business_card import BusinessCard
from src.domain.model import Model


class Dataset(TorchDataset[tuple[Tensor, Tensor, str]]):
    def __init__(
        self,
        data: list[BusinessCard],
        model: Model,
        *,
        training: bool = True,
    ) -> None:
        self.data = data
        self.model = model
        self.training = training

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, str]:
        business_card = self.data[index]

        image = Image.open(business_card.image_path)

        pixel_values = self._image_to_tensor(image, random_padding=self.training)
        labels = self._target_string_to_tensor(business_card.xml)

        return pixel_values, labels, business_card.xml

    def __len__(self) -> int:
        return len(self.data)

    @classmethod
    def load(
        cls,
        path: Path,
        model: Model,
        *,
        training: bool = True,
    ) -> "Dataset":
        with (path / "label.json").open() as f:
            labels_json = cast(list[dict], json.load(f))

        return cls(
            [
                BusinessCard(
                    image_path=label_json["image_path"],
                    company=label_json["company"],
                    name=label_json["name"],
                    email=label_json["email"],
                    phone_number=label_json["phone_number"],
                    address=label_json["address"],
                )
                for label_json in labels_json
            ],
            model,
            training=training,
        )

    def _gray_scaling_image(self, image: Image.Image) -> Image.Image:
        return to_pil_image(to_grayscale(pil_to_tensor(image)))

    def _image_to_tensor(self, image: Image.Image, *, random_padding: bool) -> Tensor:
        preprocess_image = self._gray_scaling_image(image)
        pixel_values = cast(
            Tensor,
            self.model.processor(
                preprocess_image.convert("RGB"),
                random_padding=random_padding,
                return_tensors="pt",
            ).pixel_values,
        )

        return pixel_values.squeeze()

    def _target_string_to_tensor(self, target: str) -> Tensor:
        ignore_id = -100
        input_ids = cast(
            Tensor,
            self.model.tokenizer(
                target,
                add_special_tokens=False,
                max_length=self.model.model.config.decoder.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
                return_special_tokens_mask=True,
            ).input_ids,
        ).squeeze(0)

        labels = input_ids.clone()
        labels[labels == self.model.tokenizer.pad_token_id] = ignore_id

        return labels
