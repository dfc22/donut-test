from typing import cast

from torch import FloatTensor, LongTensor, Tensor
from transformers import (
    LogitsProcessor,
    XLMRobertaTokenizer,
)

from src.domain.business_card import BusinessCard


class InferenceLogitsProcessor(LogitsProcessor):
    def __init__(self, tokenizer: XLMRobertaTokenizer) -> None:
        self.tokenizer = tokenizer
        self.special_tokens = BusinessCard.get_xml_tags()
        self.special_token_ids = cast(
            list[int],
            tokenizer.convert_tokens_to_ids(self.special_tokens),
        )

    def _last_tag(self, ids: Tensor) -> str:
        last_special_token_id = next(
            (token_id for token_id in reversed(ids.tolist()) if token_id in self.special_token_ids),
        )
        return self.tokenizer.convert_ids_to_tokens(last_special_token_id)

    @staticmethod
    def _candidate_tags(last_tag: str) -> list[str]:
        return {
            "<s>": ["<s_company>"],
            "<s_company>": ["</s_company>"],
            "</s_company>": ["<s_name>"],
            "<s_name>": ["</s_name>"],
            "</s_name>": ["<s_email>"],
            "<s_email>": ["</s_email>"],
            "</s_email>": ["<s_phone_number>"],
            "<s_phone_number>": ["</s_phone_number>"],
            "</s_phone_number>": ["<s_address>"],
            "<s_address>": ["</s_address>"],
            "</s_address>": ["</s>"],
        }[last_tag]

    def __call__(self, input_ids: LongTensor, scores: FloatTensor) -> FloatTensor:
        for i_row in range(len(input_ids)):
            ids = input_ids[i_row]

            last_tag_label = self._last_tag(ids)

            candidates = self._candidate_tags(last_tag_label)

            forbidden = [
                token_id
                for token_id in self.special_token_ids
                if self.tokenizer.convert_ids_to_tokens(token_id) not in candidates
            ]

            scores[i_row, forbidden] = -float("inf")

        return scores
