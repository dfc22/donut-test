import re
from typing import cast, override

from nltk import edit_distance
from pytorch_lightning import LightningModule
from torch import Tensor, optim
from transformers import (
    DonutProcessor,
    LogitsProcessorList,
    PreTrainedModel,
    RepetitionPenaltyLogitsProcessor,
    VisionEncoderDecoderModel,
    XLMRobertaTokenizer,
)

from src.domain.business_card import BusinessCard
from src.domain.inference_processor import InferenceLogitsProcessor


class Model(LightningModule):
    def __init__(
        self,
        processor: DonutProcessor,
        model: VisionEncoderDecoderModel,
        lr: float | None = None,
        epochs: int | None = None,
    ) -> None:
        super().__init__()
        self.processor = processor
        self.tokenizer = cast(XLMRobertaTokenizer, processor.tokenizer)

        bos_token_id, eos_token_id = cast(
            list[int],
            self.tokenizer.convert_tokens_to_ids(
                ["<s>", "</s>"],
            ),
        )
        model.config.pad_token_id = self.tokenizer.pad_token_id
        model.config.decoder_start_token_id = bos_token_id
        model.config.eos_token_id = eos_token_id
        model.config.decoder.max_length = 1000
        newly_added_num = self.tokenizer.add_special_tokens(
            {
                "additional_special_tokens": [
                    tags
                    for tags in BusinessCard.get_xml_tags()
                    if tags not in self.tokenizer.all_special_tokens
                ],
            },
        )

        if newly_added_num > 0:
            cast(PreTrainedModel, model.decoder).resize_token_embeddings(len(self.tokenizer))

        self.model = model
        self._lr = lr
        self._epochs = epochs
        self.training_step_losses = []
        self.validation_step_losses = []
        self.validation_step_scores = []

    @property
    def lr(self) -> float:
        if self._lr is None:
            msg = "Learning rate is not set."
            raise ValueError(msg)
        return self._lr

    @property
    def epochs(self) -> int:
        if self._epochs is None:
            msg = "Epochs is not set."
            raise ValueError(msg)
        return self._epochs

    @override
    def configure_optimizers(self) -> optim.Optimizer:
        return optim.Adam(self.parameters(), lr=self.lr)

    @override
    def training_step(
        self,
        batch: tuple[Tensor, Tensor, list[str]],
        _batch_idx: int,
    ) -> Tensor:
        pixel_values, labels, _ = batch

        outputs = self.model(pixel_values, labels=labels[:, 1:])
        loss = cast(Tensor, outputs.loss)

        self.training_step_losses.append(loss.item())

        return loss

    @override
    def validation_step(
        self,
        batch: tuple[Tensor, Tensor, list[str]],
        _batch_idx: int,
    ) -> Tensor:
        pixel_values, labels, targets = batch
        outputs = self.model(pixel_values, labels=labels)
        loss = cast(Tensor, outputs.loss)

        self.validation_step_losses.append(loss.item())

        predictions = self.inference(pixel_values)

        scores = []

        for pred, answer in zip(predictions, targets, strict=True):
            score = edit_distance(pred, answer) / max(len(pred), len(answer))
            scores.append(score)

            self.print(f"Prediction: {pred}")
            self.print(f"    Answer: {answer}")
            self.print(f" Normed ED: {score}")

        average_score = sum(scores) / len(scores)
        self.validation_step_scores.append(average_score)

        self.log(
            "val_edit_distance",
            average_score,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        return loss

    def inference(self, pixel_values: Tensor) -> list[str]:
        outputs = self.model.generate(
            pixel_values,
            max_length=self.model.config.decoder.max_length,
            pad_token_id=self.model.config.pad_token_id,
            eos_token_id=self.model.config.eos_token_id,
            use_cache=True,
            num_beams=1,
            bad_words_ids=[[self.tokenizer.unk_token_id]],
            return_dict_in_generate=True,
            logits_processor=LogitsProcessorList(
                [InferenceLogitsProcessor(self.tokenizer), RepetitionPenaltyLogitsProcessor(1.06)],
            ),
        )

        pattern = re.compile(r"(<s_[a-zA-Z0-9_]+>)\s")

        predictions = []
        for seq in self.tokenizer.batch_decode(outputs.sequences):
            seq_ = seq.replace(
                self.tokenizer.pad_token,
                "",
            )
            seq_ = re.sub(pattern, r"\1", seq_)
            predictions.append(seq_)

        return predictions
