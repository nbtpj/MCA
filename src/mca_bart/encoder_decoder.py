import math
import random
from typing import *

import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import BartConfig
from transformers.modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPastAndCrossAttentions,
    Seq2SeqLMOutput,
    Seq2SeqModelOutput,
)
from transformers.modeling_utils import ModelOutput
from transformers.utils import logging

from .modeling_bart import BartDecoder, BartLearnedPositionalEmbedding, BartPretrainedModel
from .modeling_bart import BartEncoder as Encoder
from .multiple_channel_decoder import MultiChannelDecoderLayer

logger = logging.get_logger(__name__)


def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int,
                       decoder_start_token_id: int):
    """
    Shift input ids one token to the right.
    Copy from transformers.models.bart.modeling_bart
    """
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    shifted_input_ids[:, 0] = decoder_start_token_id

    if pad_token_id is None:
        raise ValueError("self.model.config.pad_token_id has to be defined.")
    # replace possible -100 values in labels by `pad_token_id`
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

    return shifted_input_ids


def _make_causal_mask(input_ids_shape: torch.Size,
                      dtype: torch.dtype,
                      past_key_values_length: int = 0):
    """
    Make causal mask used for autoregressive self-attention.
    """
    # past_key_values_length: int = 0
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), torch.tensor(torch.finfo(dtype).min))
    mask_cond = torch.arange(mask.size(-1))
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat(
            [torch.zeros(tgt_len, past_key_values_length, dtype=dtype), mask],
            dim=-1)
    return mask[None, None, :, :].expand(bsz, 1, tgt_len,
                                         tgt_len + past_key_values_length)


def _expand_mask(mask: torch.Tensor,
                 dtype: torch.dtype,
                 tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, num_channel, seq_len]` to `[bsz, 1, num_channel, tgt_seq_len, src_seq_len]`.
    """
    mask = mask.transpose(0, 1)
    num_channel, bsz, kv_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else kv_len

    expanded_mask = mask[:, :, None, :].expand(num_channel, bsz, tgt_len,
                                               kv_len).to(dtype)
    expanded_mask = expanded_mask.transpose(
        0, 1)  # bsz, num_channel tgt_len, kv_len
    expanded_mask = expanded_mask[:, None,
                    ...]  # bsz, 1, num_channel tgt_len, kv_len

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.to(torch.bool),
                                     torch.finfo(dtype).min)


class Decoder(BartDecoder):
    """
    Transformer decoder consisting of *config.decoder_layers* layers. Each layer is a [`BartDecoderLayer`]

    Args:
        config: BartConfig
        embed_tokens (nn.Embedding): output embedding
    """

    def __init__(self, config, embed_tokens: Optional[nn.Embedding] = None):
        super().__init__(config)
        self.dropout = config.dropout
        self.layerdrop = config.decoder_layerdrop
        self.padding_idx = config.pad_token_id
        self.max_target_positions = config.max_position_embeddings
        self.embed_scale = math.sqrt(
            config.d_model) if config.scale_embedding else 1.0

        if embed_tokens is not None:
            self.embed_tokens = embed_tokens
        else:
            self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model,
                                             self.padding_idx)

        self.embed_positions = BartLearnedPositionalEmbedding(
            config.max_position_embeddings,
            config.d_model,
        )
        self.layers = nn.ModuleList([
            MultiChannelDecoderLayer(config)
            for _ in range(config.decoder_layers)
        ])
        self.layernorm_embedding = nn.LayerNorm(config.d_model)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def _prepare_decoder_attention_mask(self, attention_mask, input_shape,
                                        inputs_embeds, past_key_values_length):
        # create self_attention mask
        # [bsz, tgt_seq_len] -> [bsz, 1, tgt_seq_len, tgt_seq_len]
        combined_attention_mask = attention_mask
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape,
                inputs_embeds.dtype,
                past_key_values_length=past_key_values_length).to(
                inputs_embeds.device)

        return combined_attention_mask

    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            channel_weight: Optional[torch.Tensor] = None,
            encoder_hidden_states: Optional[torch.FloatTensor] = None,
            encoder_attention_mask: Optional[torch.LongTensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            cross_attn_head_mask: Optional[torch.Tensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            force_lazy: Optional[bool] = None,
            lazy_batch_size: Optional[int] = 1,
    ) -> Union[Tuple, BaseModelOutputWithPastAndCrossAttentions]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (output_hidden_states
                                if output_hidden_states is not None else
                                self.config.output_hidden_states)
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time"
            )
        elif input_ids is not None:
            input = input_ids
            input_shape = input.shape
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            input = inputs_embeds[:, :, -1]
        else:
            raise ValueError(
                "You have to specify either decoder_input_ids or decoder_inputs_embeds"
            )

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[
            2] if past_key_values is not None else 0

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input) * self.embed_scale

        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, input_shape, inputs_embeds, past_key_values_length)

        # expand encoder attention mask
        if encoder_hidden_states is not None and encoder_attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, num_channel, tgt_seq_len, src_seq_len]
            encoder_attention_mask = _expand_mask(encoder_attention_mask,
                                                  inputs_embeds.dtype,
                                                  tgt_len=input_shape[-1])

        # embed positions
        positions = self.embed_positions(input_shape, past_key_values_length)

        hidden_states = inputs_embeds + positions
        hidden_states = self.layernorm_embedding(hidden_states)

        hidden_states = nn.functional.dropout(hidden_states,
                                              p=self.dropout,
                                              training=self.training)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_cross_attentions = () if (
                output_attentions and encoder_hidden_states is not None) else None
        next_decoder_cache = () if use_cache else None

        # check if head_mask/cross_attn_head_mask has a correct number of layers specified if desired
        for attn_mask, mask_name in zip([head_mask, cross_attn_head_mask],
                                        ["head_mask", "cross_attn_head_mask"]):
            if attn_mask is not None:
                if attn_mask.size()[0] != (len(self.layers)):
                    raise ValueError(
                        f"The `{mask_name}` should be specified for {len(self.layers)} layers, but it is for"
                        f" {head_mask.size()[0]}.")

        for idx, decoder_layer in enumerate(self.layers):
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            dropout_probability = random.uniform(0, 1)
            if self.training and (dropout_probability < self.layerdrop):
                continue

            past_key_value = past_key_values[
                idx] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:

                if use_cache:
                    logger.warning(
                        "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                    )
                    use_cache = False

                def create_custom_forward(module):

                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, output_attentions, use_cache)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(decoder_layer),
                    hidden_states,
                    attention_mask,
                    channel_weight,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    head_mask[idx] if head_mask is not None else None,
                    cross_attn_head_mask[idx]
                    if cross_attn_head_mask is not None else None,
                    None,
                    force_lazy,
                    lazy_batch_size,
                )
            else:
                # print(hidden_states.size(), encoder_hidden_states.size())

                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    channel_weight=channel_weight,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    layer_head_mask=(head_mask[idx]
                                     if head_mask is not None else None),
                    cross_attn_layer_head_mask=(cross_attn_head_mask[idx]
                                                if cross_attn_head_mask
                                                   is not None else None),
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    force_lazy=force_lazy,
                    lazy_batch_size=lazy_batch_size
                )
            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (
                    layer_outputs[3 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

                if encoder_hidden_states is not None:
                    all_cross_attentions += (layer_outputs[2],)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(v for v in [
                hidden_states, next_cache, all_hidden_states, all_self_attns,
                all_cross_attentions
            ] if v is not None)
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            cross_attentions=all_cross_attentions,
        )


class MCAModel(BartPretrainedModel):

    def __init__(self, config):
        super().__init__(config)

        padding_idx, vocab_size = config.pad_token_id, config.vocab_size
        self.shared = nn.Embedding(vocab_size, config.d_model, padding_idx)

        self.encoder = Encoder(config, self.shared)
        self.decoder = Decoder(config, self.shared)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, value):
        self.shared = value
        self.encoder.embed_tokens = self.shared
        self.decoder.embed_tokens = self.shared

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    def encode(self,
               input_ids: torch.LongTensor,
               attention_mask: Optional[torch.Tensor] = None,
               inputs_embeds: Optional[torch.FloatTensor] = None,
               head_mask: Optional[torch.Tensor] = None,
               output_attentions: Optional[bool] = None,
               output_hidden_states: Optional[bool] = None,
               force_lazy: Optional[bool] = None,
               lazy_batch_size: Optional[int] = 1,
               **kwargs):
        """
        Enable lazy encoding for extreme-large input (>tens of channels). This option prevents out-of-mem error.
        force_lazy: whether using lazy encoding or not (default not)
        lazy_batch_size: the size that suitable for executing device (often cuda) (default 1)
        """

        batch_size, num_channel, kv_len = input_ids.size()
        embed_dim = self.shared.embedding_dim
        encoder_input_ids = input_ids.reshape(batch_size * num_channel, kv_len)
        if inputs_embeds is not None:
            inputs_embeds = inputs_embeds.reshape(batch_size * num_channel, *inputs_embeds.shape[2:])
        encoder_attention_mask = attention_mask.reshape(
            batch_size *
            num_channel, kv_len) if attention_mask is not None else None
        if force_lazy is True:
            index_list = list(range(batch_size * num_channel))
            chunks = [index_list[i:i + lazy_batch_size] for i in range(0, len(index_list), lazy_batch_size)]
            encoder_outputs = {
                'last_hidden_state': None,
                'hidden_states': None,
                'attentions': None,
            }
            for chunk in chunks:
                chunk_encoder_outputs = self.encoder(
                    input_ids=encoder_input_ids[chunk],
                    attention_mask=encoder_attention_mask[chunk],
                    head_mask=head_mask,
                    inputs_embeds=inputs_embeds[chunk] if inputs_embeds is not None else None,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=True,
                )
                if encoder_outputs['last_hidden_state'] is None:
                    encoder_outputs['last_hidden_state'] = chunk_encoder_outputs['last_hidden_state']
                    if 'hidden_states' in chunk_encoder_outputs:
                        encoder_outputs['hidden_states'] = chunk_encoder_outputs['hidden_states']
                    if 'attentions' in chunk_encoder_outputs:
                        encoder_outputs['attentions'] = chunk_encoder_outputs['attentions']
                else:
                    encoder_outputs['last_hidden_state'] = torch.cat((encoder_outputs['last_hidden_state'],
                                                                      chunk_encoder_outputs['last_hidden_state']),
                                                                     dim=0)
                    if 'hidden_states' in chunk_encoder_outputs:
                        encoder_outputs['hidden_states'] = [torch.cat((previous,
                                                                       each_hidden),
                                                                      dim=0)
                                                            for each_hidden, previous in
                                                            zip(chunk_encoder_outputs['hidden_states'],
                                                                encoder_outputs['hidden_states'])]
                    if 'attentions' in chunk_encoder_outputs:
                        encoder_outputs['attentions'] = [torch.cat((previous,
                                                                    each_attn),
                                                                   dim=0)
                                                         for each_attn, previous in
                                                         zip(chunk_encoder_outputs['attentions'],
                                                             encoder_outputs['attentions'])]
            encoder_outputs = BaseModelOutput(**encoder_outputs)

        else:
            encoder_outputs = self.encoder(
                input_ids=encoder_input_ids,
                attention_mask=encoder_attention_mask,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=True,
            )
        encoder_outputs['last_hidden_state'] = encoder_outputs[
            'last_hidden_state'].view(batch_size, num_channel, kv_len,
                                      embed_dim)
        if 'hidden_states' in encoder_outputs:
            encoder_outputs['hidden_states'] = [
                hidden_states.view(batch_size, num_channel, *hidden_states.shape[1:])
                for hidden_states in encoder_outputs['hidden_states']
            ]
        if 'attentions' in encoder_outputs:
            encoder_outputs['attentions'] = [
                attention.view(batch_size, num_channel, *attention.shape[1:])
                for attention in encoder_outputs['attentions']
            ]
        return encoder_outputs

    def decode(self,
               encoder_outputs: BaseModelOutput,
               attention_mask: Optional[torch.Tensor] = None,
               channel_weight: Optional[torch.Tensor] = None,
               decoder_input_ids: Optional[torch.LongTensor] = None,
               decoder_attention_mask: Optional[torch.LongTensor] = None,
               decoder_head_mask: Optional[torch.Tensor] = None,
               decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
               use_cache: Optional[bool] = None,
               output_attentions: Optional[bool] = None,
               output_hidden_states: Optional[bool] = None,
               return_dict: Optional[bool] = None,
               cross_attn_head_mask: Optional[torch.Tensor] = None,
               past_key_values: Optional = None,
               force_lazy: Optional[bool] = None,
               lazy_batch_size: Optional[int] = 1,
               **kwargs):

        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            channel_weight=channel_weight,
            encoder_hidden_states=encoder_outputs[0],
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            force_lazy=force_lazy,
            lazy_batch_size=lazy_batch_size,
        )
        return decoder_outputs

    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            channel_weight: Optional[torch.Tensor] = None,
            decoder_input_ids: Optional[torch.LongTensor] = None,
            decoder_attention_mask: Optional[torch.LongTensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            decoder_head_mask: Optional[torch.Tensor] = None,
            cross_attn_head_mask: Optional[torch.Tensor] = None,
            encoder_outputs: Optional[List[torch.FloatTensor]] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            force_lazy: Optional[bool] = None,
            lazy_batch_size: Optional[int] = 1,
    ) -> Union[Tuple, Seq2SeqModelOutput]:

        # different to other models, Bart automatically creates decoder_input_ids from
        # input_ids if no decoder_input_ids are provided
        if decoder_input_ids is None and decoder_inputs_embeds is None:
            if input_ids is None:
                raise ValueError(
                    "If no `decoder_input_ids` or `decoder_inputs_embeds` are "
                    "passed, `input_ids` cannot be `None`. Please pass either "
                    "`input_ids` or `decoder_input_ids` or `decoder_inputs_embeds`."
                )

            decoder_input_ids = shift_tokens_right(
                input_ids[0], self.config.pad_token_id,
                self.config.decoder_start_token_id)

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (output_hidden_states
                                if output_hidden_states is not None else
                                self.config.output_hidden_states)
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        """input_ids in shape of [num_channel, batch_size, seq_len]"""
        #         num_channel, batch_size, kv_len = input_ids.size()
        #         _ , tgt_len = decoder_input_ids.size()
        #         embed_dim = self.shared.embedding_dim
        #         encoder_input_ids = input_ids.reshape(num_channel * batch_size, kv_len)
        #         encoder_attention_mask = attention_mask.reshape(num_channel * batch_size, kv_len) if attention_mask is not None else None

        if encoder_outputs is None:
            encoder_outputs = self.encode(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                force_lazy=force_lazy,
                lazy_batch_size=lazy_batch_size
            )
        # If the user passed a tuple for encoder_outputs, we wrap it in a BaseModelOutput when return_dict=True
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1]
                if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2]
                if len(encoder_outputs) > 2 else None,
            )
        # print(encoder_outputs.last_hidden_state.size())

        # decoder outputs consists of (dec_features, past_key_value, dec_hidden, dec_attn)
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs[0],
            encoder_attention_mask=attention_mask,
            channel_weight=channel_weight,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            force_lazy=force_lazy,
            lazy_batch_size=lazy_batch_size,
        )

        if not return_dict:
            return decoder_outputs + encoder_outputs

        return Seq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )


class MCAForConditionalGeneration(BartPretrainedModel):
    base_model_prefix = "model"
    _keys_to_ignore_on_load_missing = [r"final_logits_bias", r"lm_head.weight"]

    def __init__(self, config: BartConfig):
        super().__init__(config)
        self.model = MCAModel(config)
        self.register_buffer(
            "final_logits_bias",
            torch.zeros((1, self.model.shared.num_embeddings)))
        self.lm_head = nn.Linear(config.d_model,
                                 self.model.shared.num_embeddings,
                                 bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def freeze_encoder(self):
        for param in self.model.get_encoder().parameters():
            param.requires_grad = False

    def get_encoder(self):
        return self.model.get_encoder()

    def get_decoder(self):
        return self.model.get_decoder()

    def resize_token_embeddings(self, new_num_tokens: int) -> nn.Embedding:
        new_embeddings = super().resize_token_embeddings(new_num_tokens)
        self._resize_final_logits_bias(new_num_tokens)
        return new_embeddings

    def _resize_final_logits_bias(self, new_num_tokens: int) -> None:
        old_num_tokens = self.final_logits_bias.shape[-1]
        if new_num_tokens <= old_num_tokens:
            new_bias = self.final_logits_bias[:, :new_num_tokens]
        else:
            extra_bias = torch.zeros((1, new_num_tokens - old_num_tokens),
                                     device=self.final_logits_bias.device)
            new_bias = torch.cat([self.final_logits_bias, extra_bias], dim=1)
        self.register_buffer("final_logits_bias", new_bias)

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            channel_weight: Optional[torch.Tensor] = None,
            decoder_input_ids: Optional[torch.LongTensor] = None,
            decoder_attention_mask: Optional[torch.LongTensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            decoder_head_mask: Optional[torch.Tensor] = None,
            cross_attn_head_mask: Optional[torch.Tensor] = None,
            encoder_outputs: Optional[List[torch.FloatTensor]] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            force_lazy: Optional[bool] = None,
            lazy_batch_size: Optional[int] = 1,
    ) -> Union[Tuple, Seq2SeqLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if labels is not None:
            if use_cache:
                logger.warning(
                    "The `use_cache` argument is changed to `False` since `labels` is provided."
                )
            use_cache = False
            if decoder_input_ids is None and decoder_inputs_embeds is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id,
                    self.config.decoder_start_token_id)

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            channel_weight=channel_weight,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            force_lazy=force_lazy,
            lazy_batch_size=lazy_batch_size,
        )
        lm_logits = self.lm_head(outputs[0]) + self.final_logits_bias

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(
                lm_logits.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((masked_lm_loss,) +
                    output) if masked_lm_loss is not None else output

        return Seq2SeqLMOutput(
            loss=masked_lm_loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )

    def encode(self, *args, **kwargs):
        return self.model.encode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        return self.model.decode(*args, **kwargs)

    def prepare_inputs_for_generation(self,
                                      decoder_input_ids,
                                      attention_mask=None,
                                      head_mask=None,
                                      decoder_head_mask=None,
                                      cross_attn_head_mask=None,
                                      use_cache=None,
                                      encoder_outputs=None,
                                      past_key_values=None,
                                      force_lazy: Optional[bool] = None,
                                      lazy_batch_size: Optional[int] = 1,
                                      **kwargs):
        # cut decoder_input_ids if past is used
        if past_key_values is not None:
            decoder_input_ids = decoder_input_ids[:, -1:]
        # if use_cache:
        #     past_key_values = past_key_values
        # else:
        #     past_key_values = None

        return {
            "input_ids":
                None,  # encoder_outputs is defined. input_ids not needed
            'past_key_values': past_key_values,
            "encoder_outputs": encoder_outputs,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache":
                use_cache,  # change this to avoid caching (presumably for debugging)

            "force_lazy": force_lazy,
            "lazy_batch_size": lazy_batch_size,
        }

    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        return shift_tokens_right(labels, self.config.pad_token_id,
                                  self.config.decoder_start_token_id)

    @staticmethod
    def _reorder_cache(past, beam_idx):
        reordered_past = ()
        for layer_past in past:
            # cached cross_attention states don't have to be reordered -> they are always the same
            reordered_past += (tuple(
                past_state.index_select(0, beam_idx)
                for past_state in layer_past[:2]) + layer_past[2:],)
        return reordered_past

    def _prepare_encoder_decoder_kwargs_for_generation(
            self, inputs_tensor: torch.Tensor, model_kwargs, model_input_name: Optional[str] = None
    ) -> Dict[str, Any]:

        # 2. prepare encoder args and encoder kwargs from model kwargs
        irrelevant_prefix = ["decoder_", "cross_attn", "use_cache"]
        encoder_kwargs = {
            argument: value
            for argument, value in model_kwargs.items()
            if not any(argument.startswith(p) for p in irrelevant_prefix)
        }

        # 3. make sure that encoder returns `ModelOutput`
        model_input_name = model_input_name if model_input_name is not None else self.main_input_name
        encoder_kwargs["return_dict"] = True
        encoder_kwargs[model_input_name] = inputs_tensor
        model_kwargs["encoder_outputs"]: ModelOutput = self.encode(**encoder_kwargs)

        return model_kwargs
