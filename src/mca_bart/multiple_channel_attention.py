from typing import Optional, Tuple

import torch
import torch.utils.checkpoint
from torch import nn

from .utils import lazy_bmm, batch_first_single_output_wrapper


def averagePoolingMultipleChannel(
        attn_weights: torch.Tensor,
        value_states: torch.Tensor,
        dropout: float,
        head_dim: int,
        num_channel: int,
        batch_size: int,
        num_heads: int,
        training: bool,
        channel_weight: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
        force_lazy: Optional[bool] = None,
        lazy_batch_size: Optional[int] = 1, ) -> torch.Tensor:
    """
    attn_weights in shape of [batch_size * num_heads * num_channel, tgt_len, kv_len]
    """
    tgt_len, kv_len = attn_weights.size()[-2:]
    if channel_weight is None:
        avgScaler = torch.ones(batch_size, num_channel, device=attn_weights.device
                               ) / num_channel  # average scale for all channel
    else:
        avgScaler = channel_weight

    if output_attentions:
        # this operation is a bit awkward, but it's required to
        # make sure that attn_weights keeps its gradient.
        # In order to do so, attn_weights have to be reshaped
        # twice and have to be reused in the following
        attn_weights_reshaped = attn_weights.view(batch_size, num_heads,
                                                  num_channel, tgt_len, kv_len)
        attn_weights = attn_weights_reshaped.view(
            batch_size * num_heads * num_channel, tgt_len, kv_len)
    else:
        attn_weights_reshaped = None

    attn_probs = nn.functional.dropout(attn_weights,
                                       p=dropout,
                                       training=training)
    value_states = value_states.view(batch_size * num_heads, kv_len,
                                     num_channel,
                                     head_dim).transpose(1, 2).contiguous()
    value_states = value_states.view(batch_size * num_heads * num_channel,
                                     kv_len, head_dim)
    if force_lazy is True:
        attn_output = lazy_bmm(attn_probs,
                               value_states, lazy_batch_size=lazy_batch_size).view(batch_size, num_heads,
                                                                                   num_channel, tgt_len, head_dim)
    else:
        attn_output = torch.bmm(attn_probs,
                                value_states).view(batch_size, num_heads,
                                                   num_channel, tgt_len, head_dim)

    # channel-wise pooling
    avgScaler.to(attn_output.device)
    attn_output = avgScaler.view(batch_size, 1, num_channel, 1,
                                 1) * attn_output
    attn_output = attn_output.sum(dim=2)
    attn_output = attn_output.view(batch_size, num_heads, tgt_len, head_dim)
    attn_output = attn_output.transpose(1, 2)
    # attention output in shape: batch_size, tgt_len, num_heads, head_dim
    return attn_output, attn_weights, attn_weights_reshaped


class pooolingMultipleChannelAttention(nn.Module):
    """
    Multi-Channel attention, an improve mechanism implementing 
    Multi-headed attention from 'Attention Is All You Need' paper.
    Initialize in the same way as transformers's source code.
    """

    def __init__(
            self,
            embed_dim: int,
            num_heads: int,
            dropout: float = 0.0,
            is_decoder: bool = False,
            bias: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads

        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {num_heads}).")
        self.scaling = self.head_dim ** -0.5
        self.is_decoder = False

        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def forward(
            self,
            hidden_states: torch.Tensor,
            key_value_states: torch.Tensor,
            past_key_value: Optional[Tuple[torch.Tensor]] = None,
            attention_mask: Optional[torch.Tensor] = None,
            layer_head_mask: Optional[torch.Tensor] = None,
            channel_weight: Optional[torch.Tensor] = None,
            output_attentions: bool = False,
            force_lazy: Optional[bool] = None,
            lazy_batch_size: Optional[int] = 1,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
            key_value_states: [batch_size x num_channel x kv_len x embedding_dim]
            hidden_states: [batch_size x tgt_len x embedding_dim]
            past_key_value: pre-calculated projected kay-value states
        """
        batch_size, num_channel, kv_len, embedding_dim = key_value_states.size(
        )
        tgt_len = hidden_states.size()[1]

        # query state: [batch_size x num_head x tgt_len x head_dim]
        query_states = self.q_proj(hidden_states) * self.scaling
        query_states = query_states.view(batch_size, tgt_len, self.num_heads,
                                         self.head_dim).transpose(
            1, 2).contiguous()
        if past_key_value is None:
            # key-value state: [batch_size x num_head x kv_len x num_channel x head_dim]
            key_value_states = key_value_states.contiguous(
            )  # key_value_states: [batch_size x num_channel x embedding_dim]
            if force_lazy is True:
                key_states = batch_first_single_output_wrapper(key_value_states,
                                                               fn=self.k_proj,
                                                               lazy_batch_size=lazy_batch_size).view(
                    batch_size, num_channel, kv_len, self.num_heads,
                    self.head_dim).transpose(1, 3).contiguous()
                value_states = batch_first_single_output_wrapper(key_value_states,
                                                                 fn=self.v_proj,
                                                                 lazy_batch_size=lazy_batch_size).view(
                    batch_size, num_channel, kv_len, self.num_heads,
                    self.head_dim).transpose(1, 3).contiguous()
            else:
                key_states = self.k_proj(key_value_states).view(
                    batch_size, num_channel, kv_len, self.num_heads,
                    self.head_dim).transpose(1, 3).contiguous()
                value_states = self.v_proj(key_value_states).view(
                    batch_size, num_channel, kv_len, self.num_heads,
                    self.head_dim).transpose(1, 3).contiguous()
        else:
            assert past_key_value[0].size() == (batch_size, self.num_heads, kv_len,
                                                num_channel, self.head_dim), "Cached key state not in right shape!"
            assert past_key_value[1].size() == (batch_size, self.num_heads, kv_len,
                                                num_channel, self.head_dim), "Cached value state not in right shape!"
            key_states = past_key_value[0]
            value_states = past_key_value[1]
        past_key_value_to_cache = (key_states, value_states)
        # calculate multi-head attention

        proj_shape = (batch_size * self.num_heads, -1, self.head_dim)
        query_states = query_states.view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)

        src_len = key_states.size(1)  # src_len = kv_len x num_channel
        if force_lazy is True:
            attn_weights = lazy_bmm(
                query_states, key_states.transpose(1, 2),
                lazy_batch_size=lazy_batch_size
            )  # attention_weight: [(batch_size x num_head) x tgt_len x src_len]
        else:
            attn_weights = torch.bmm(
                query_states, key_states.transpose(1, 2)
            )  # attention_weight: [(batch_size x num_head) x tgt_len x src_len]
        attn_weights = attn_weights.view(batch_size * self.num_heads, tgt_len,
                                         kv_len, num_channel).contiguous()
        if attention_mask is not None:
            if attention_mask.size() != (batch_size, 1, tgt_len, kv_len,
                                         num_channel):
                if attention_mask.size() == (batch_size, 1, num_channel,
                                             tgt_len, kv_len):
                    attention_mask = attention_mask.transpose(-2, -3).transpose(-1, -2)
                else:
                    raise ValueError(
                        f"Attention mask should be of size {(batch_size, 1, tgt_len, kv_len, num_channel)} or {(batch_size, 1, num_channel, tgt_len, kv_len)}, but is {attention_mask.size()}"
                    )
            attn_weights = attn_weights.view(batch_size, self.num_heads,
                                             tgt_len, kv_len,
                                             num_channel) + attention_mask
            # if channel_weight is None:
            #     attention_mask = attention_mask.view(batch_size, tgt_len, kv_len,
            #                                          num_channel)
            #     attention_mask = attention_mask.transpose(-2, -3).transpose(-1, -3)
            #     channel_mask = torch.where(attention_mask.count_nonzero(-1) == kv_len, 0,
            #                                1)  # batch_size x num_channel x tgt_len
            #
            #     channel_weight = torch.where(channel_mask.sum(-1) > 0, 1, 0)  # batch_size x num_channel
            #     channel_weight = channel_weight / channel_weight.sum(-1).view(batch_size, -1)

        attn_weights = nn.functional.softmax(attn_weights, dim=-2).transpose(
            -1, -3).contiguous()
        attn_weights = attn_weights.view(
            batch_size * self.num_heads * num_channel, kv_len,
            tgt_len).transpose(-1, -2).contiguous()
        # attention_weight: [(batch_size x num_head x num_channel) x tgt_len x kv_len]

        if layer_head_mask is not None:
            if layer_head_mask.size() != (self.num_heads,):
                raise ValueError(
                    f"Head mask for a single layer should be of size {(self.num_heads,)}, but is"
                    f" {layer_head_mask.size()}")
            attn_weights = layer_head_mask.view(
                1, -1, 1, 1, 1) * attn_weights.view(
                batch_size, self.num_heads, num_channel, tgt_len, kv_len)
            attn_weights = attn_weights.view(
                batch_size * self.num_heads * num_channel, tgt_len, kv_len)
        attn_output, attn_weights, attn_weights_reshaped = averagePoolingMultipleChannel(
            attn_weights, value_states, self.dropout, self.head_dim,
            num_channel, batch_size, self.num_heads, self.training,
            channel_weight,
            output_attentions,
            force_lazy,
            lazy_batch_size, )

        # Use the `embed_dim` from the config (stored in the class) rather than `hidden_state` because `attn_output` can be
        # partitioned aross GPUs when using tensor-parallelism.
        attn_output = attn_output.reshape(batch_size, tgt_len, self.embed_dim)

        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights_reshaped, past_key_value_to_cache
