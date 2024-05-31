import argparse
import copy
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

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=None, help="The batch size", type=int, required=False)
args = parser.parse_args()
dirname = os.path.dirname(__file__)
if 'CACHE_DIR' not in os.environ:
    os.environ['CACHE_DIR'] = '.cache'
comparative_models = {
    'distil-BART': (
        BartForConditionalGeneration, 'theojolliffe/bart-large-cnn-pubmed1o3-pubmed2o3', BartTokenizer, DataCollatorForSeq2Seq),
    'BART-large': (BartForConditionalGeneration, 'theojolliffe/bart-cnn-science', BartTokenizer, DataCollatorForSeq2Seq),
    'distil-BART+MCA': (
        MCAForConditionalGeneration, 'theojolliffe/bart-large-cnn-pubmed1o3-pubmed2o3', MCATokenizer,
        DataCollatorForMultipleChannelAttention),
    'BART-large+MCA': (
        MCAForConditionalGeneration, 'theojolliffe/bart-cnn-science', MCATokenizer, DataCollatorForMultipleChannelAttention),
    'PEGASUS': (PegasusForConditionalGeneration, 'google/pegasus-pubmed', AutoTokenizer,
                DataCollatorForSeq2Seq),
    'BigBird': (BigBirdPegasusForConditionalGeneration, 'google/bigbird-pegasus-large-pubmed', AutoTokenizer,
                DataCollatorForSeq2Seq),
}

print('Loading MLSR-cochrane dataset')
testset = datasets.load_dataset('allenai/mslr2022', 'cochrane',
                                cache_dir=os.environ['CACHE_DIR'])['validation']
# testset = testset.select(range(5))
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
    if issubclass(data_collator, DataCollatorForMultipleChannelAttention):
        data_collators[model_name] = data_collator(tokenizer=tokenizers[model_name],
                                                   model=models[model_name],
                                                   channel_padding=True,
                                                   channel_truncation=True,
                                                   channel_max_length=20,
                                                   )
    else:
        data_collators[model_name] = data_collator(tokenizer=tokenizers[model_name])
print('_______________________________')

remove_columns = testset.column_names


def vertical_scale(example):
    return {'context': example['abstract'], 'answer': example['target']}


def horizontal_scale(example):
    return {'context': '<s>'.join(example['abstract']), 'answer': example['target']}


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
    tokenize_testsets[model_name] = datasets[model_name].map(tokenize,
                                                             remove_columns=datasets[model_name].column_names,
                                                             fn_kwargs={'tokenizer': tokenizer})

from transformers import Seq2SeqTrainingArguments
from mca_eval import FixedLengthTrainer, compute_metrics
import pickle
import numpy as np

os.environ['TOKENIZERS_PARALLELISM'] = 'true'

if os.path.isfile(f'{dirname}/MLSR-cochrane-test-predicts-all-iter-results.bin'):
    with open(f'{dirname}/MLSR-cochrane-test-predicts-all-iter-results.bin', 'rb') as f:
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

    if os.path.isfile(f'{dirname}/MLSR-cochrane-predicts-length-range-{CONTROL_MIN}-{CONTROL_MAX}.bin'):
        with open(f'{dirname}/MLSR-cochrane-predicts-length-range-{CONTROL_MIN}-{CONTROL_MAX}.bin', 'rb') as f:
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
                                                               golden=testset['target'])

        decoded_texts[model_name] = decoded_preds
        results[model_name] = result
        results[model_name]['overflow_ratio'] = round(np.mean(tokenize_testsets[model_name]['overflow_ratio']), 4)
        decoded_labels[model_name] = decoded_label

        print(result)
        print('___________________')
    with open(f'{dirname}/MLSR-cochrane-predicts-length-range-{CONTROL_MIN}-{CONTROL_MAX}.bin', 'wb') as f:
        pickle.dump(predicts, f)
    ttresults[f'{CONTROL_MIN}-{CONTROL_MAX}'] = results
with open(f'{dirname}/MLSR-cochrane-test-predicts-all-iter-results.bin', 'wb+') as f:
    pickle.dump(ttresults, f)

print('Writing meta to files')
fixed_rs = ttresults[f'100-160']
df = pd.DataFrame.from_dict({model_name: fixed_rs[model_name] for model_name in fixed_rs})
df = df.transpose()
df.to_csv(f'{dirname}/documents/MLSR-cochrane-full-fixed-setting.csv')
tokenizer = tokenizers['distil-BART']
encode_summary = lambda example: {'ids': tokenizer(example['target'], truncation=False)['input_ids']}
summary_ids = testset.map(encode_summary, remove_columns=testset.column_names)
summary_ids = summary_ids['ids']
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

model_name_map = {
    'distil-BART': 'bart-large-cnn-pubmed1o3-pubmed2o3',
    'BART-large': 'bart-cnn-science',
    'distil-BART+MCA': 'bart-large-cnn-pubmed1o3-pubmed2o3+MCA',
    'BART-large+MCA': 'bart-cnn-science+MCA',
    'PEGASUS': 'pegasus-pubmed',
    'BigBird': 'bigbird-pegasus-large-pubmed'
}

with open(f'{dirname}/documents/MLSR-cochrane-scores.pk', 'wb+') as f:
    to_write = (des, statistics_by_model, model_name_map)
    pickle.dump(to_write, f)
print('------------------------------')

print('Testing adaptive setting')
predicts = {}
if os.path.isfile(f'{dirname}/MLSR-cochrane-perfect-length-predicts.bin'):
    with open(f'{dirname}/MLSR-cochrane-perfect-length-predicts.bin', 'rb') as f:
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
                                                           golden=testset['target'])

    decoded_texts[model_name] = decoded_preds
    results[model_name] = result
    results[model_name]['overflow_ratio'] = round(np.mean(tokenize_testsets[model_name]['overflow_ratio']), 4)
    decoded_labels[model_name] = decoded_label

    print(result)
    print('___________________')
print('------------------------------')
with open(f'{dirname}/MLSR-cochrane-perfect-length-predicts.bin', 'wb') as f:
    pickle.dump(predicts, f)

print('Saving results')
df = pd.DataFrame.from_dict({model_name: results[model_name] for model_name in results})
df = df.transpose()
df.to_csv(f'{dirname}/documents/MLSR-cochrane-full-adaptive-setting.csv')
print('------------------------------')
