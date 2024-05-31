import argparse
import copy
import itertools
import os
import datasets
import pandas as pd

from mca_bart import (
    MCAForConditionalGeneration,
    MCATokenizer,
    DataCollatorForMultipleChannelAttention
)
from mca_eval import AdaptiveLengthTrainer
from tqdm import tqdm
from transformers import (
    BartForConditionalGeneration,
    PegasusForConditionalGeneration,
    BigBirdPegasusForConditionalGeneration,
    DataCollatorForSeq2Seq,
    BartTokenizer,
    AutoTokenizer, PreTrainedModel,
)
if 'CACHE_DIR' not in os.environ:
    os.environ['CACHE_DIR'] = '.cache'
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=None, help="The batch size", type=int, required=False)
args = parser.parse_args()
dirname = os.path.dirname(__file__)

comparative_models = {
    'distil-BART': (
        BartForConditionalGeneration, 'sshleifer/distilbart-cnn-6-6', BartTokenizer, DataCollatorForSeq2Seq),
    'BART-large': (BartForConditionalGeneration, 'facebook/bart-large-cnn', BartTokenizer, DataCollatorForSeq2Seq),
    'distil-BART+MCA': (
        MCAForConditionalGeneration, 'sshleifer/distilbart-cnn-6-6', MCATokenizer,
        DataCollatorForMultipleChannelAttention),
    'BART-large+MCA': (
        MCAForConditionalGeneration, 'facebook/bart-large-cnn', MCATokenizer, DataCollatorForMultipleChannelAttention),
    'PEGASUS': (PegasusForConditionalGeneration, 'google/pegasus-cnn_dailymail', AutoTokenizer,
                DataCollatorForSeq2Seq),
    'BigBird': (BigBirdPegasusForConditionalGeneration, 'google/bigbird-pegasus-large-bigpatent', AutoTokenizer,
                DataCollatorForSeq2Seq),
}

model_map_name = {
    'sshleifer/distilbart-cnn-6-6': 'distil-BART+MCA',
    'facebook/bart-large-cnn': 'BART-large+MCA',
    'sshleifer/distilbart-cnn-6-6-non-mca': 'distil-BART',
    'facebook/bart-large-cnn-non-mca': 'BART-large',
    'google/pegasus-cnn_dailymail':'PEGASUS',
    'google/bigbird-pegasus-large-bigpatent': 'BigBird'
}

print('Loading DUC2004 dataset')
testset = datasets.load_dataset('nbtpj/DUC2004',
                                cache_dir=os.environ['CACHE_DIR'])['train']
print('_______________________________')

print('Loading models and tokenizers')
models = {}
tokenizers = {}
data_collators = {}
for model_name, hyper_params in tqdm(comparative_models.items(), 'Initializing models and tokenizers'):
    architecture, pretrained_parameters_path, tokenizer_type, data_collator = hyper_params
    models[model_name] = lambda: comparative_models[model_name][0].from_pretrained(comparative_models[model_name][1],
                                                                                   cache_dir=os.environ['CACHE_DIR'])
    tokenizers[model_name] = tokenizer_type.from_pretrained(pretrained_parameters_path,
                                                            cache_dir=os.environ['CACHE_DIR'])
    data_collators[model_name] = data_collator(tokenizer=tokenizers[model_name])
print('_______________________________')

remove_columns = testset.column_names


def vertical_scale(example):
    return {'context': example['context'], 'answer': example['summary']}


def horizontal_scale(example):
    return {'context': '<s>'.join(example['context']), 'answer': example['summary']}


ver_set = testset.map(vertical_scale, remove_columns=remove_columns)
hor_set = testset.map(horizontal_scale, remove_columns=remove_columns)

datasets = {
    'distil-BART': hor_set,
    'BART-large': hor_set,
    'distil-BART+MCA': ver_set,
    'BART-large+MCA': ver_set,
    'PEGASUS': hor_set,
    'BigBird': hor_set,
}


def tokenize(example, tokenizer):
    max_length = tokenizer.model_max_length
    channels = example['context']
    example['input_ids'] = tokenizer(channels, truncation=False)['input_ids']
    if isinstance(example['input_ids'][0], list):
        # multiple channel mode
        total_length = sum([len(channel_k) for channel_k in example['input_ids']])
        overflow_length = sum([max(0, len(channel_k) - max_length) for channel_k in example['input_ids']])
    else:
        # casual mode
        total_length = len(example['input_ids'])
        overflow_length = max(0, len(example['input_ids']) - max_length)
    example['input_ids'] = tokenizer(channels, truncation=True)['input_ids']
    example['overflow_ratio'] = overflow_length / total_length

    return example


tokenize_testsets = {}

for model_name in comparative_models:
    print('Tokenizing ' + model_name)
    tokenizer = tokenizers[model_name]
    tokenize_testsets[model_name] = datasets[model_name].map(tokenize, remove_columns=['context', 'answer'],
                                                             fn_kwargs={'tokenizer': tokenizer})

from transformers import Seq2SeqTrainingArguments
from mca_eval import FixedLengthTrainer, compute_metrics
import pickle
import numpy as np

os.environ['TOKENIZERS_PARALLELISM'] = 'true'

if os.path.isfile(f'{dirname}/duc2004-test-predicts-all-iter-results.bin'):
    with open(f'{dirname}/duc2004-test-predicts-all-iter-results.bin', 'rb') as f:
        ttresults = pickle.load(f)
else:
    ttresults = {}

if args.batch_size is not None:
    training_args = Seq2SeqTrainingArguments(
        output_dir="test",
        per_device_eval_batch_size=args.batch_size,
        predict_with_generate=True,
        report_to=None,
    )
else:
    training_args = Seq2SeqTrainingArguments(
        output_dir="test",
        auto_find_batch_size=True,
        predict_with_generate=True,
        report_to=None,
    )

length_ranges = [
    (0, 40),
    (40, 100),
    (100, 160),
    (160, 220),
    (220, 280)
]

for (CONTROL_MIN, CONTROL_MAX) in length_ranges:
    predicts = {}
    results = {}
    decoded_texts = {}
    decoded_labels = {}

    if os.path.isfile(f'{dirname}/duc2004-predicts-length-range-{CONTROL_MIN}-{CONTROL_MAX}.bin'):
        with open(f'{dirname}/duc2004-predicts-length-range-{CONTROL_MIN}-{CONTROL_MAX}.bin', 'rb') as f:
            predicts = pickle.load(f)

    for model_name in tqdm(comparative_models, f'Predicting range {CONTROL_MIN}-{CONTROL_MAX}'):
        print('\n___________________')
        print(model_name)
        print('___________________\n')
        tokenizer = tokenizers[model_name]
        data_collator = data_collators[model_name]
        if model_name not in predicts:
            print(comparative_models[model_name][:-2])
            model: PreTrainedModel = models[model_name]()
            model.config.early_stopping = False
            trainer = FixedLengthTrainer(
                model=model,
                args=training_args,
                data_collator=data_collator,
                tokenizer=tokenizer,
            )
            predicts[model_name] = trainer.predict(tokenize_testsets[model_name].remove_columns(['overflow_ratio']),
                                                   min_length=CONTROL_MIN,
                                                   max_length=CONTROL_MAX,
                                                   num_beams=4)

        result, decoded_preds, decoded_label = compute_metrics(predicts[model_name][:2],
                                                               tokenizer,
                                                               golden=testset['summary'])

        decoded_texts[model_name] = decoded_preds
        results[model_name] = result
        results[model_name]['overflow_ratio'] = round(np.mean(tokenize_testsets[model_name]['overflow_ratio']), 4)
        decoded_labels[model_name] = decoded_label

        print(result)
        print('___________________')
    with open(f'{dirname}/duc2004-predicts-length-range-{CONTROL_MIN}-{CONTROL_MAX}.bin', 'wb') as f:
        pickle.dump(predicts, f)
    ttresults[f'{CONTROL_MIN}-{CONTROL_MAX}'] = results
with open(f'{dirname}/duc2004-test-predicts-all-iter-results.bin', 'wb+') as f:
    pickle.dump(ttresults, f)

print('Writing meta to files')
fixed_rs = ttresults[f'100-160']
df = pd.DataFrame.from_dict({model_name: fixed_rs[model_name] for model_name in fixed_rs})
df = df.transpose()
df.to_csv(f'{dirname}/documents/DUC2004-full-fixed-setting.csv')
tokenizer = tokenizers['distil-BART']
encode_summary = lambda example: {'ids': tokenizer(example['summary'], truncation=False)['input_ids']}
summary_ids = testset.map(encode_summary, remove_columns=testset.column_names)
summary_ids = list(itertools.chain(*summary_ids['ids']))
summary_lens = [len(ids) for ids in summary_ids]
des = pd.Series(summary_lens).describe()
statistics_by_model = {}

for ranges, results in ttresults.items():
    for model_name in results:
        if model_name not in statistics_by_model:
            statistics_by_model[model_name] = []
        rs = copy.copy(results[model_name])
        rs['gen_range'] = ranges
        statistics_by_model[model_name].append(rs)

for model_name in statistics_by_model:
    statistics_by_model[model_name] = pd.DataFrame(statistics_by_model[model_name])

with open(f'{dirname}/documents/DUC2004-scores.pk', 'wb+') as f:
    to_write = (des, statistics_by_model, model_map_name)
    pickle.dump(to_write, f)
print('------------------------------')

print('Testing adaptive setting')
predicts = {}
if os.path.isfile(f'{dirname}/duc2004-perfect-length-predicts.bin'):
    with open(f'{dirname}/duc2004-perfect-length-predicts.bin', 'rb') as f:
        predicts = pickle.load(f)

results = {}
decoded_texts = {}
decoded_labels = {}

for model_name in tqdm(comparative_models, f'Predicting adaptive setting'):
    print('\n___________________')
    print(model_name)
    print('___________________\n')
    tokenizer = tokenizers[model_name]
    data_collator = data_collators[model_name]
    if model_name not in predicts:
        print(comparative_models[model_name][:-2])
        model: PreTrainedModel = models[model_name]()
        model.config.early_stopping = False
        trainer = AdaptiveLengthTrainer(
            model=model,
            args=training_args,
            data_collator=data_collator,
            tokenizer=tokenizer,
        )
        predicts[model_name] = trainer.predict(tokenize_testsets[model_name].remove_columns(['overflow_ratio']),
                                               max_length=160,
                                               length_deviant=10,
                                               num_beams=4)

    result, decoded_preds, decoded_label = compute_metrics(predicts[model_name][:2],
                                                           tokenizer,
                                                           golden=testset['summary'])

    decoded_texts[model_name] = decoded_preds
    results[model_name] = result
    results[model_name]['overflow_ratio'] = round(np.mean(tokenize_testsets[model_name]['overflow_ratio']), 4)
    decoded_labels[model_name] = decoded_label

    print(result)
    print('___________________')
print('------------------------------')
with open(f'{dirname}/duc2004-perfect-length-predicts.bin', 'wb') as f:
    pickle.dump(predicts, f)

print('Saving results')
df = pd.DataFrame.from_dict({model_name: results[model_name] for model_name in results})
df = df.transpose()
df.to_csv(f'{dirname}/documents/DUC2004-full-adaptive-setting.csv')
print('------------------------------')
