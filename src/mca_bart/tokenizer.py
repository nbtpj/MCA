import copy
import itertools
from typing import List, Any, Tuple, Union, Optional

from torch import TensorType
from transformers import BartTokenizer, BatchEncoding, BartTokenizerFast
from transformers.tokenization_utils_base import PreTokenizedInput, TextInput, TruncationStrategy
from transformers.utils import PaddingStrategy


def is_multi_context(thing: List[str]):
    assert isinstance(thing, List) and all(
        [isinstance(sub_context, str) for sub_context in thing]), "Not multi-context sample found!"


def pad_to_length(lst: List[List[Any]],
                  pad_factor: Any,
                  to_length: int = None):
    if to_length is None:
        to_length = max([len(each) for each in lst])
    attn_mask = [[
                     1,
                 ] * len(sample) + [0] * (to_length - len(sample))
                 for sample in lst]
    paded = [
        sample + [
            pad_factor,
        ] * (to_length - len(sample)) for sample in lst
    ]
    return paded, attn_mask


class MCATokenizer(BartTokenizer):

    def __call__(
            self,
            text: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]] = None,
            text_pair: Optional[Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]]] = None,
            text_target: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]] = None,
            text_pair_target: Optional[
                Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]]
            ] = None,
            add_special_tokens: bool = True,
            padding: Union[bool, str, PaddingStrategy] = False,
            truncation: Union[bool, str, TruncationStrategy] = None,
            max_length: Optional[int] = None,
            channel_padding: Union[bool, str, PaddingStrategy] = None,
            channel_truncation: Optional[bool] = None,
            channel_max_length: Optional[int] = None,
            stride: int = 0,
            is_split_into_words: bool = False,
            pad_to_multiple_of: Optional[int] = None,
            return_tensors: Optional[Union[str, TensorType]] = None,
            return_token_type_ids: Optional[bool] = None,
            return_attention_mask: Optional[bool] = None,
            return_overflowing_tokens: bool = False,
            return_special_tokens_mask: bool = False,
            return_offsets_mapping: bool = False,
            return_length: bool = False,
            verbose: bool = True,
            **kwargs,
    ) -> BatchEncoding:
        all_kwargs = {
            "add_special_tokens": add_special_tokens,
            "padding": padding,
            "truncation": truncation,
            "max_length": max_length,
            "stride": stride,
            "is_split_into_words": is_split_into_words,
            "pad_to_multiple_of": pad_to_multiple_of,
            "return_tensors": return_tensors,
            "return_token_type_ids": return_token_type_ids,
            "return_attention_mask": return_attention_mask,
            "return_overflowing_tokens": return_overflowing_tokens,
            "return_special_tokens_mask": return_special_tokens_mask,
            "return_offsets_mapping": return_offsets_mapping,
            "return_length": return_length,
            "verbose": verbose,
        }
        all_kwargs.update(kwargs)
        # multi-channel mode
        if isinstance(text[0], List):
            _kwargs = copy.copy(all_kwargs)
            _kwargs['padding'] = 'do_not_pad'
            _kwargs['return_tensors'] = None
            flatten_sig_contexts = list(itertools.chain(*text))
            results = super().__call__(flatten_sig_contexts, **_kwargs)
            batched_encode, batched_mask = copy.deepcopy(results['input_ids']), \
                copy.deepcopy(results['attention_mask'])
            if text_target is not None or text_pair_target is not None:
                output = super().__call__(text_target=text_target,
                                          text_pair_target=text_pair_target,
                                          **all_kwargs)
                results.update({'labels': output['input_ids']})
            # gather every multi-context
            input_ids = []
            masks = []
            last_index = 0
            for ctx in text:
                input_ids.append(batched_encode[last_index:last_index + len(ctx)])
                masks.append(batched_mask[last_index:last_index + len(ctx)])
                last_index += len(ctx)
            results['input_ids'] = input_ids
            results['attention_mask'] = masks

            if ('return_tensors' in all_kwargs and all_kwargs['return_tensors'] is not None) \
                    or (channel_padding is not None or channel_truncation is not None):
                results['input_ids'], results['attention_mask'], results['channel_weight'] = self.pad_multi_channel(
                    results['input_ids'], channel_padding=channel_padding,
                    channel_max_length=channel_max_length,
                    channel_truncation=channel_truncation)

            tensor_type = all_kwargs['return_tensors'] if 'return_tensors' in all_kwargs else None
            return BatchEncoding(results, tensor_type=tensor_type)
        else:
            return super().__call__(text=text,
                                    text_pair=text_pair,
                                    text_target=text_target,
                                    text_pair_target=text_pair_target, **all_kwargs)

    def pad_multi_channel(self, batch_input_ids: List[List[List[int]]],
                          channel_padding: Union[bool, str, PaddingStrategy] = True,
                          channel_truncation: Optional[bool] = None,
                          channel_max_length: Optional[int] = None) -> \
            Tuple[List[List[List[int]]], List[List[List[int]]], List[List[List[int]]]]:
        flat_ids = list(itertools.chain(*batch_input_ids))
        gather_idx = []
        total = 0

        # avgScaler = torch.ones(batch_size, num_channel, device=attn_weights.device
        #                        ) / num_channel  # average scale for all channel
        for each in batch_input_ids:
            gather_idx.append((total, total + len(each)))
            total += len(each)
        flat_ids, flat_mask = pad_to_length(flat_ids, self.pad_token_id, None)
        channel_pad_factor = [self.pad_token_id, ] * len(flat_ids[0])
        channel_mask_factor = [0, ] * len(flat_ids[0])
        pad_to_channel = max([e - b for (b, e) in gather_idx])
        if channel_padding == 'max_length' or channel_padding == PaddingStrategy.MAX_LENGTH:
            pad_to_channel = max([e - b for (b, e) in gather_idx]) if channel_max_length is None else channel_max_length

        if channel_truncation is True:
            pad_to_channel = max([e - b for (b, e) in gather_idx]) if channel_max_length is None else channel_max_length
            gather_idx = [(b, min(e, b + pad_to_channel)) for (b, e) in gather_idx]

        new_batch = []
        new_mask = []
        channel_weight = []
        for (b, e) in gather_idx:
            padded = flat_ids[b:e] + [channel_pad_factor, ] * (pad_to_channel - e + b)
            mask = flat_mask[b:e] + [channel_mask_factor, ] * (pad_to_channel - e + b)
            padded_weight = [1 / float(e - b), ] * (e - b) + [0, ] * (pad_to_channel - e + b)
            new_batch.append(padded)
            new_mask.append(mask)
            channel_weight.append(padded_weight)
        return new_batch, new_mask, channel_weight


class MCATokenizerFast(BartTokenizerFast):

    def __call__(
            self,
            text: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]] = None,
            text_pair: Optional[Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]]] = None,
            text_target: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]] = None,
            text_pair_target: Optional[
                Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]]
            ] = None,
            add_special_tokens: bool = True,
            padding: Union[bool, str, PaddingStrategy] = False,
            truncation: Union[bool, str, TruncationStrategy] = None,
            max_length: Optional[int] = None,
            channel_padding: Union[bool, str, PaddingStrategy] = None,
            channel_truncation: Optional[bool] = None,
            channel_max_length: Optional[int] = None,
            stride: int = 0,
            is_split_into_words: bool = False,
            pad_to_multiple_of: Optional[int] = None,
            return_tensors: Optional[Union[str, TensorType]] = None,
            return_token_type_ids: Optional[bool] = None,
            return_attention_mask: Optional[bool] = None,
            return_overflowing_tokens: bool = False,
            return_special_tokens_mask: bool = False,
            return_offsets_mapping: bool = False,
            return_length: bool = False,
            verbose: bool = True,
            **kwargs,
    ) -> BatchEncoding:
        all_kwargs = {
            "add_special_tokens": add_special_tokens,
            "padding": padding,
            "truncation": truncation,
            "max_length": max_length,
            "stride": stride,
            "is_split_into_words": is_split_into_words,
            "pad_to_multiple_of": pad_to_multiple_of,
            "return_tensors": return_tensors,
            "return_token_type_ids": return_token_type_ids,
            "return_attention_mask": return_attention_mask,
            "return_overflowing_tokens": return_overflowing_tokens,
            "return_special_tokens_mask": return_special_tokens_mask,
            "return_offsets_mapping": return_offsets_mapping,
            "return_length": return_length,
            "verbose": verbose,
        }
        all_kwargs.update(kwargs)
        # multi-channel mode
        if isinstance(text[0], List):
            _kwargs = copy.copy(all_kwargs)
            _kwargs['padding'] = 'do_not_pad'
            _kwargs['return_tensors'] = None
            flatten_sig_contexts = list(itertools.chain(*text))
            results = super().__call__(flatten_sig_contexts, **_kwargs)
            batched_encode, batched_mask = copy.deepcopy(results['input_ids']), \
                copy.deepcopy(results['attention_mask'])
            if text_target is not None or text_pair_target is not None:
                output = super().__call__(text_target=text_target,
                                          text_pair_target=text_pair_target,
                                          **all_kwargs)
                results.update({'labels': output['input_ids']})
            # gather every multi-context
            input_ids = []
            masks = []
            last_index = 0
            for ctx in text:
                input_ids.append(batched_encode[last_index:last_index + len(ctx)])
                masks.append(batched_mask[last_index:last_index + len(ctx)])
                last_index += len(ctx)
            results['input_ids'] = input_ids
            results['attention_mask'] = masks

            if ('return_tensors' in all_kwargs and all_kwargs['return_tensors'] is not None) \
                    or (channel_padding is not None or channel_truncation is not None):
                results['input_ids'], results['attention_mask'], results['channel_weight'] = self.pad_multi_channel(
                    results['input_ids'], channel_padding=channel_padding,
                    channel_max_length=channel_max_length,
                    channel_truncation=channel_truncation)

            tensor_type = all_kwargs['return_tensors'] if 'return_tensors' in all_kwargs else None
            return BatchEncoding(results, tensor_type=tensor_type)
        else:
            return super().__call__(text=text,
                                    text_pair=text_pair,
                                    text_target=text_target,
                                    text_pair_target=text_pair_target, **all_kwargs)

    def pad_multi_channel(self, batch_input_ids: List[List[List[int]]],
                          channel_padding: Union[bool, str, PaddingStrategy] = True,
                          channel_truncation: Optional[bool] = None,
                          channel_max_length: Optional[int] = None) -> \
            Tuple[List[List[List[int]]], List[List[List[int]]], List[List[List[int]]]]:
        flat_ids = list(itertools.chain(*batch_input_ids))
        gather_idx = []
        total = 0

        # avgScaler = torch.ones(batch_size, num_channel, device=attn_weights.device
        #                        ) / num_channel  # average scale for all channel
        for each in batch_input_ids:
            gather_idx.append((total, total + len(each)))
            total += len(each)
        flat_ids, flat_mask = pad_to_length(flat_ids, self.pad_token_id, None)
        channel_pad_factor = [self.pad_token_id, ] * len(flat_ids[0])
        channel_mask_factor = [0, ] * len(flat_ids[0])
        pad_to_channel = max([e - b for (b, e) in gather_idx])
        if channel_padding == 'max_length' or channel_padding == PaddingStrategy.MAX_LENGTH:
            pad_to_channel = max([e - b for (b, e) in gather_idx]) if channel_max_length is None else channel_max_length

        if channel_truncation is True:
            pad_to_channel = max([e - b for (b, e) in gather_idx]) if channel_max_length is None else channel_max_length
            gather_idx = [(b, min(e, b + pad_to_channel)) for (b, e) in gather_idx]

        new_batch = []
        new_mask = []
        channel_weight = []
        for (b, e) in gather_idx:
            padded = flat_ids[b:e] + [channel_pad_factor, ] * (pad_to_channel - e + b)
            mask = flat_mask[b:e] + [channel_mask_factor, ] * (pad_to_channel - e + b)
            padded_weight = [1 / float(e - b), ] * (e - b) + [0, ] * (pad_to_channel - e + b)
            new_batch.append(padded)
            new_mask.append(mask)
            channel_weight.append(padded_weight)
        return new_batch, new_mask, channel_weight


if __name__ == '__main__':
    tok = MCATokenizer.from_pretrained('../../.cache/mc-base')
    a = tok([['this is a text wd', 'this is ', 'sfj ssjb jd'],
             ['this is a text fsnb', 'this  a text']],
            text_target=['hi', 'hello these, is , a'],
            # padding=True,
            # return_tensors='pt',
            # channel_truncation=True,
            # channel_max_length=2,bug
            )
    pass
