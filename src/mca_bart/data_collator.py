from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
from transformers import BatchEncoding
from transformers.utils import PaddingStrategy

from .encoder_decoder import MCAForConditionalGeneration
from .tokenizer import MCATokenizer


@dataclass
class DataCollatorForMultipleChannelAttention:
    tokenizer: MCATokenizer
    model: Optional[MCAForConditionalGeneration] = None
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    channel_padding: Union[bool, str, PaddingStrategy] = True
    channel_truncation: Optional[bool] = None
    channel_max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"

    def __call__(self, features,
                 return_tensors=None,
                 channel_padding: Union[bool, str, PaddingStrategy] = None,
                 channel_truncation: Optional[bool] = None,
                 channel_max_length: Optional[int] = None) -> BatchEncoding:
        """ mainly copy from DataCollatorForSeq2Seq"""

        input_ids = [feature['input_ids'] for feature in features]

        channel_padding = channel_padding if channel_padding is not None else self.channel_padding
        channel_truncation = channel_truncation if channel_truncation is not None else self.channel_truncation
        channel_max_length = channel_max_length if channel_max_length is not None else self.channel_max_length

        if return_tensors is None:
            return_tensors = self.return_tensors
        # if return_tensors != 'pt':
        #     raise "Non-torch tensor is not supported yet!"
        labels = [feature["labels"] for feature in features] if "labels" in features[0].keys() else None
        # We have to pad the labels before calling `tokenizer.pad` as this method won't pad them and needs them of the
        # same length to return tensors.
        if labels is not None:
            max_label_length = max(len(l) for l in labels)
            if self.pad_to_multiple_of is not None:
                max_label_length = (
                        (max_label_length + self.pad_to_multiple_of - 1)
                        // self.pad_to_multiple_of
                        * self.pad_to_multiple_of
                )

            padding_side = self.tokenizer.padding_side
            for feature in features:
                remainder = [self.label_pad_token_id] * (max_label_length - len(feature["labels"]))
                if isinstance(feature["labels"], list):
                    feature["labels"] = (
                        feature["labels"] + remainder if padding_side == "right" else remainder + feature["labels"]
                    )
                elif padding_side == "right":
                    feature["labels"] = np.concatenate([feature["labels"], remainder]).astype(np.int64)
                else:
                    feature["labels"] = np.concatenate([remainder, feature["labels"]]).astype(np.int64)
        input_ids, mask, channel_weight = self.tokenizer.pad_multi_channel(input_ids,
                                                                           channel_padding=channel_padding,
                                                                           channel_max_length=channel_max_length,
                                                                           channel_truncation=channel_truncation)
        new_features = {
            'input_ids': input_ids,
            'attention_mask': mask,
            'channel_weight': channel_weight
        }
        if labels is not None:
            new_features['labels'] = [feature["labels"] for feature in features]
        # new_features = {k: torch.LongTensor(v) for k, v in new_features.items()}
        new_features = BatchEncoding(new_features, tensor_type=return_tensors)
        # prepare decoder_input_ids
        if (
                labels is not None
                and self.model is not None
                and hasattr(self.model, "prepare_decoder_input_ids_from_labels")
        ):
            decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(labels=new_features["labels"])
            new_features["decoder_input_ids"] = decoder_input_ids

        return new_features
