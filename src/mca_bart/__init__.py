"""
    Attention-level pooling version of mca_bart.
    This is origin idea of multi-channel attention

"""
from .data_collator import DataCollatorForMultipleChannelAttention
from .encoder_decoder import MCAForConditionalGeneration, MCAModel
from .tokenizer import MCATokenizer, MCATokenizerFast