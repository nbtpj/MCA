from datasets import Dataset, DatasetDict
from typing import Optional, Callable, Union
from transformers import PreTrainedTokenizer
import numpy as np


def random_delete_words(text: str, mask_prob: float = 0.15) -> str:
    words = text.split()
    mask = np.random.choice([0, 1], size=(len(words),), p=[mask_prob, 1 - mask_prob])
    for idx in mask:
        if idx == 0:
            words[idx] = ''
    return ' '.join(words)


def random_mask_words(text: str, mask_prob: float = 0.15, mask_tok: str = '--') -> str:
    words = text.split()
    mask = np.random.choice([0, 1], size=(len(words),), p=[mask_prob, 1 - mask_prob])
    for idx in mask:
        if idx == 0:
            words[idx] = mask_tok
    return ' '.join(words)


class PMCAugmentation:
    random_seed: int = 10
    num_channel: int = 3
    gather_column: str = 'context'
    custom_context_combination: Optional[Callable] = None
    drop_last_example: bool = True
    tokenizer: PreTrainedTokenizer = None

    @staticmethod
    def gather_query_with_mul_contexts(samples, query_column: str = 'question',
                                       context_column: str = 'context',
                                       qc_splitter: str = '<||||>',
                                       new_columns: str = 'qcontexts'
                                       ):
        contexts = samples[context_column]
        samples[new_columns] = []
        for q in samples[query_column]:
            mul_ctx = [q + qc_splitter + ctx for ctx in contexts]
            samples[new_columns].append(mul_ctx)
        return samples

    @staticmethod
    def prompt_with_mul_context(samples, query_column: str = 'question',
                                context_column: str = 'context',
                                new_column: str = 'qcontexts',
                                prompt: str = 'Give the detailed answer for the question "{q}" and the information "{v}"'
                                ):
        contexts = samples[context_column]
        samples[new_column] = []
        for q in samples[query_column]:
            ctx = [prompt.format(q=q, v=v) for v in contexts]
            samples[new_column].append(ctx)
        return samples

    @staticmethod
    def prompt_word_filling(samples, text_column: str = 'text',
                            new_column: str = 'qcontexts',
                            label_column: str = 'cleaned_text',
                            prompt: str = '''Some words have been replaced by characters {t}.
                                     Replace them with the most suitable words: "{c}"'''
                            ):
        texts = samples[text_column]
        samples[label_column] = samples[text_column]
        samples[new_column] = [[prompt.format(t='--',
                                              c=random_mask_words(text, mask_tok='--'))] for text in texts]
        return samples

    @staticmethod
    def prompt_word_delete(samples, text_column: str = 'text',
                           new_column: str = 'qcontexts',
                           label_column: str = 'cleaned_text',
                           prompt: str = '''Some words have been removed from the text.
                                 Rewrite the text with the most suitable words: "{c}"'''
                           ):
        texts = samples[text_column]
        samples[label_column] = samples[text_column]
        samples[new_column] = [[prompt.format(c=random_delete_words(text))] for text in texts]
        return samples

    def __call__(self, dataset: Union[Dataset, DatasetDict], **kwargs) \
            -> Union[Dataset, DatasetDict]:
        shuffled_dataset = dataset.shuffle(self.random_seed)

        def gather(samples, gather_column):
            samples[gather_column] = [samples[gather_column], ] * len(samples[gather_column])
            return samples

        if self.custom_context_combination is not None:
            new_dataset = shuffled_dataset.map(self.custom_context_combination, batched=True,
                                               batch_size=self.num_channel,
                                               drop_last_batch=self.drop_last_example,
                                               fn_kwargs=kwargs)
        else:
            new_dataset = shuffled_dataset.map(gather, batched=True,
                                               batch_size=self.num_channel,
                                               drop_last_batch=self.drop_last_example)
        return new_dataset
