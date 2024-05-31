import argparse
import copy
import os
import pickle
import numpy as np
import pandas as pd
from datasets import load_dataset
from mca_bart import (
    MCAForConditionalGeneration,
    MCATokenizer,
    DataCollatorForMultipleChannelAttention
)
from mca_eval import FixedLengthTrainer, AdaptiveLengthTrainer, compute_metrics
from transformers import (
    BartForConditionalGeneration,
    PegasusForConditionalGeneration,
    BigBirdPegasusForConditionalGeneration,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    AutoTokenizer, )

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=None, help="The batch size", type=int, required=False)
args = parser.parse_args()
dirname = os.path.dirname(__file__)
os.environ['TOKENIZERS_PARALLELISM'] = 'true'
if 'CACHE_DIR' not in os.environ:
    os.environ['CACHE_DIR'] = '.cache'
models = {
    'sshleifer/distilbart-cnn-6-6': lambda: MCAForConditionalGeneration.from_pretrained('sshleifer/distilbart-cnn-6-6',
                                                                                        cache_dir=os.environ[
                                                                                            'CACHE_DIR']
                                                                                        ),
    'facebook/bart-large-cnn': lambda: MCAForConditionalGeneration.from_pretrained('facebook/bart-large-cnn',
                                                                                   cache_dir=os.environ['CACHE_DIR']
                                                                                   ),
    'nbtpj/mc-bart-base-mqa-fine-tune': lambda: MCAForConditionalGeneration.from_pretrained(
        'nbtpj/mc-bart-base-mqa-fine-tune',
        cache_dir=os.environ['CACHE_DIR']),
    'yjernite/bart_eli5': lambda: MCAForConditionalGeneration.from_pretrained('yjernite/bart_eli5',
                                                                              cache_dir=os.environ['CACHE_DIR']
                                                                              ),
    'facebook/bart-base': lambda: MCAForConditionalGeneration.from_pretrained('facebook/bart-base',
                                                                              cache_dir=os.environ['CACHE_DIR']),

}

tokenizers = {
    model_name: lambda: MCATokenizer.from_pretrained(model_name,
                                                     cache_dir=os.environ['CACHE_DIR']) for model_name in models
}

# non mca
models.update({
    'facebook/bart-base-non-mca': lambda: BartForConditionalGeneration.from_pretrained('facebook/bart-base',
                                                                                       cache_dir=os.environ[
                                                                                           'CACHE_DIR']),
    'sshleifer/distilbart-cnn-6-6-non-mca': lambda: BartForConditionalGeneration.from_pretrained(
        'sshleifer/distilbart-cnn-6-6',
        cache_dir=os.environ['CACHE_DIR']),
    'sshleifer/distilbart-cnn-6-6-non-mca-with-query': lambda: BartForConditionalGeneration.from_pretrained(
        'sshleifer/distilbart-cnn-6-6',
        cache_dir=os.environ['CACHE_DIR']),
    'facebook/bart-large-cnn-non-mca': lambda: BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn',
                                                                                            cache_dir=os.environ[
                                                                                                'CACHE_DIR']),
    'facebook/bart-large-cnn-non-mca-with-query': lambda: BartForConditionalGeneration.from_pretrained(
        'facebook/bart-large-cnn',
        cache_dir=os.environ['CACHE_DIR']),
    'yjernite/bart_eli5-non-mca': lambda: BartForConditionalGeneration.from_pretrained('yjernite/bart_eli5',
                                                                                       cache_dir=os.environ[
                                                                                           'CACHE_DIR']),
    'nbtpj/bart-base-mqa-fine-tune-non-mca': lambda: BartForConditionalGeneration.from_pretrained(
        'nbtpj/bart-base-mqa-fine-tune',
        cache_dir=os.environ['CACHE_DIR']),
    'nbtpj/bart-base-rmqa-fine-tune-non-mca': lambda: BartForConditionalGeneration.from_pretrained(
        'nbtpj/bart-base-rmqa-fine-tune',
        cache_dir=os.environ['CACHE_DIR']),
    'google/pegasus-cnn_dailymail': lambda: PegasusForConditionalGeneration.from_pretrained(
        'google/pegasus-cnn_dailymail',
        cache_dir=os.environ['CACHE_DIR']),
    'google/bigbird-pegasus-large-bigpatent': lambda: BigBirdPegasusForConditionalGeneration.from_pretrained(
        "google/bigbird-pegasus-large-bigpatent",
        cache_dir=os.environ['CACHE_DIR']),
    'google/pegasus-cnn_dailymail-with-query': lambda: PegasusForConditionalGeneration.from_pretrained(
        'google/pegasus-cnn_dailymail',
        cache_dir=os.environ['CACHE_DIR']),
    'google/bigbird-pegasus-large-bigpatent-with-query': lambda: BigBirdPegasusForConditionalGeneration.from_pretrained(
        "google/bigbird-pegasus-large-bigpatent",
        cache_dir=os.environ['CACHE_DIR']),
})

tokenizers.update({
    'facebook/bart-base-non-mca': lambda: AutoTokenizer.from_pretrained('facebook/bart-base',
                                                                        cache_dir=os.environ['CACHE_DIR']),
    'sshleifer/distilbart-cnn-6-6-non-mca': lambda: AutoTokenizer.from_pretrained('sshleifer/distilbart-cnn-6-6',
                                                                                  cache_dir=os.environ['CACHE_DIR']
                                                                                  ),
    'sshleifer/distilbart-cnn-6-6-non-mca-with-query': lambda: AutoTokenizer.from_pretrained(
        'sshleifer/distilbart-cnn-6-6',
        cache_dir=os.environ['CACHE_DIR']),
    'facebook/bart-large-cnn-non-mca': lambda: AutoTokenizer.from_pretrained('facebook/bart-large-cnn',
                                                                             cache_dir=os.environ['CACHE_DIR']
                                                                             ),
    'facebook/bart-large-cnn-non-mca-with-query': lambda: AutoTokenizer.from_pretrained('facebook/bart-large-cnn',
                                                                                        cache_dir=os.environ[
                                                                                            'CACHE_DIR']
                                                                                        ),

    'yjernite/bart_eli5-non-mca': lambda: AutoTokenizer.from_pretrained('yjernite/bart_eli5',
                                                                        cache_dir=os.environ['CACHE_DIR']),
    'nbtpj/bart-base-mqa-fine-tune-non-mca': lambda: AutoTokenizer.from_pretrained('nbtpj/bart-base-mqa-fine-tune',
                                                                                   cache_dir=os.environ['CACHE_DIR']
                                                                                   ),
    'nbtpj/bart-base-rmqa-fine-tune-non-mca': lambda: AutoTokenizer.from_pretrained('nbtpj/bart-base-rmqa-fine-tune',
                                                                                    cache_dir=os.environ['CACHE_DIR']
                                                                                    ),
    'google/pegasus-cnn_dailymail': lambda: AutoTokenizer.from_pretrained('google/pegasus-cnn_dailymail',
                                                                          cache_dir=os.environ['CACHE_DIR']
                                                                          ),
    'google/bigbird-pegasus-large-bigpatent': lambda: AutoTokenizer.from_pretrained(
        "google/bigbird-pegasus-large-bigpatent",
        cache_dir=os.environ['CACHE_DIR']),
    'google/pegasus-cnn_dailymail-with-query': lambda: AutoTokenizer.from_pretrained('google/pegasus-cnn_dailymail',
                                                                                     cache_dir=os.environ['CACHE_DIR']
                                                                                     ),
    'google/bigbird-pegasus-large-bigpatent-with-query': lambda: AutoTokenizer.from_pretrained(
        "google/bigbird-pegasus-large-bigpatent",
        cache_dir=os.environ['CACHE_DIR']),

})

print('Loading MEDIQA-AnS dataset')
testset = load_dataset("nbtpj/BioNLP2021", split="test",
                       cache_dir=os.environ['CACHE_DIR'])
# testset = testset.select(range(5))
print('_______________________________')


def reformat_qa(example):
    mul_doc = example['text'].replace('<SS>', '')
    question = example['question'].replace('<SS>', '')
    example['summ_abs'] = example['summ_abs'].replace('<SS>', ' ')
    DOCUMENT_SPLITER: str = '<DOC>'
    multi_doc = mul_doc.split(DOCUMENT_SPLITER)
    if question is not None:
        context = [question + '<||||>' + context for context in multi_doc]
    return {'context': context, 'answer': example['summ_abs']}


def reformat_qa_bart_style(example):
    mul_doc = example['text'].replace('<SS>', '')
    question = example['question'].replace('<SS>', '')
    example['summ_abs'] = example['summ_abs'].replace('<SS>', ' ')
    DOCUMENT_SPLITER: str = '<DOC>'
    multi_doc = mul_doc.split(DOCUMENT_SPLITER)
    if question is not None:
        context = [question + '<s>' + context for context in multi_doc]
    return {'context': context, 'answer': example['summ_abs']}


def reformat_qa_single_channel_bart_style(example):
    mul_doc = example['text'].replace('<SS>', '')
    question = example['question'].replace('<SS>', '')
    example['summ_abs'] = example['summ_abs'].replace('<SS>', ' ')
    DOCUMENT_SPLITER: str = '<DOC>'
    multi_doc = mul_doc.split(DOCUMENT_SPLITER)
    context = question + '<s>' + '<s>'.join(multi_doc)
    return {'context': context, 'answer': example['summ_abs']}


def reformat_qa_single_channel(example):
    mul_doc = example['text'].replace('<SS>', '')
    question = example['question'].replace('<SS>', '')
    example['summ_abs'] = example['summ_abs'].replace('<SS>', ' ')
    DOCUMENT_SPLITER: str = '<DOC>'
    multi_doc = mul_doc.split(DOCUMENT_SPLITER)
    context = question + '<||||>' + '<s>'.join(multi_doc)
    return {'context': context, 'answer': example['summ_abs']}


def reformat_non_qa(example):
    mul_doc = example['text'].replace('<SS>', '')
    example['summ_abs'] = example['summ_abs'].replace('<SS>', ' ')
    DOCUMENT_SPLITER: str = '<DOC>'
    multi_doc = mul_doc.split(DOCUMENT_SPLITER)
    return {'context': multi_doc, 'answer': example['summ_abs']}


def reformat_non_qa_concat(example):
    mul_doc = example['text'].replace('<SS>', '')
    example['summ_abs'] = example['summ_abs'].replace('<SS>', ' ')
    DOCUMENT_SPLITER: str = '<DOC>'
    multi_doc = mul_doc.split(DOCUMENT_SPLITER)
    return {'context': '<s>'.join(multi_doc), 'answer': example['summ_abs']}


testset_qa = testset.map(reformat_qa, remove_columns=['text', 'question', 'key', 'summ_abs', 'summ_ext'])
testset_non_qa = testset.map(reformat_non_qa, remove_columns=['text', 'question', 'key', 'summ_abs', 'summ_ext'])
testset_nonqa_concat = testset.map(reformat_non_qa_concat,
                                   remove_columns=['text', 'question', 'key', 'summ_abs', 'summ_ext'])
testset_qa_bart_style = testset.map(reformat_qa_bart_style,
                                    remove_columns=['text', 'question', 'key', 'summ_abs', 'summ_ext'])
testset_qa_single_channel = testset.map(reformat_qa_single_channel,
                                        remove_columns=['text', 'question', 'key', 'summ_abs', 'summ_ext'])
testset_qa_single_channel_bart_style = testset.map(reformat_qa_single_channel_bart_style,
                                                   remove_columns=['text', 'question', 'key', 'summ_abs', 'summ_ext'])

datasets = {
    'sshleifer/distilbart-cnn-6-6': testset_non_qa,
    'facebook/bart-large-cnn': testset_non_qa,
    'nbtpj/mc-bart-base-mqa-fine-tune': testset_qa,
    'yjernite/bart_eli5': testset_qa_bart_style,
    'google/pegasus-cnn_dailymail': testset_nonqa_concat,
    'google/bigbird-pegasus-large-bigpatent': testset_nonqa_concat,
    'google/pegasus-cnn_dailymail-with-query': testset_qa_single_channel,
    'google/bigbird-pegasus-large-bigpatent-with-query': testset_qa_single_channel,
    'sshleifer/distilbart-cnn-6-6-non-mca': testset_nonqa_concat,
    'sshleifer/distilbart-cnn-6-6-non-mca-with-query': testset_qa_single_channel_bart_style,
    'facebook/bart-large-cnn-non-mca': testset_nonqa_concat,
    'facebook/bart-large-cnn-non-mca-with-query': testset_qa_single_channel_bart_style,
    'yjernite/bart_eli5-non-mca': testset_qa_single_channel_bart_style,
    'nbtpj/bart-base-mqa-fine-tune-non-mca': testset_qa_single_channel,
    'nbtpj/bart-base-rmqa-fine-tune-non-mca': testset_qa_single_channel,
    'facebook/bart-base': testset_qa_bart_style,
    'facebook/bart-base-non-mca': testset_qa_single_channel_bart_style,
}

model_name_map = {
    'nbtpj/mc-bart-base-mqa-fine-tune': 'MCA+PMC',
    'nbtpj/bart-base-mqa-fine-tune-non-mca': 'BART',
    'nbtpj/bart-base-rmqa-fine-tune-non-mca': 'BART+PMC',
    'facebook/bart-base': 'MCA(non-finetune)',
    'facebook/bart-base-non-mca': 'BART(non-finetune)',
    'sshleifer/distilbart-cnn-6-6': 'distil-BART+MCA without query',
    'facebook/bart-large-cnn': 'BART-large+MCA without query',
    'yjernite/bart_eli5': 'BART-eli5+MCA with query',
    'sshleifer/distilbart-cnn-6-6-non-mca': 'distil-BART without query',
    'sshleifer/distilbart-cnn-6-6-non-mca-with-query': 'distil-BART with query',
    'facebook/bart-large-cnn-non-mca': 'BART-large without query',
    'facebook/bart-large-cnn-non-mca-with-query': 'BART-large with query',
    'yjernite/bart_eli5-non-mca': 'BART-eli5',
    'google/pegasus-cnn_dailymail': 'PEGASUS without query',
    'google/bigbird-pegasus-large-bigpatent': 'BigBird without query',
    'google/pegasus-cnn_dailymail-with-query': 'PEGASUS with query',
    'google/bigbird-pegasus-large-bigpatent-with-query': 'BigBird with query',
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

    summary = example['answer']
    example['labels'] = tokenizer(summary, truncation=True)['input_ids']
    return example


print('Tokenizing MEDIQA-AnS dataset')
tokenize_testsets = {}

for model_name in models:
    tokenizer = tokenizers[model_name]()
    tokenize_testsets[model_name] = datasets[model_name].map(tokenize, remove_columns=['context', 'answer'],
                                                             fn_kwargs={'tokenizer': tokenizer})
    if 'non-mca' not in model_name and 'google' not in model_name:
        assert isinstance(tokenize_testsets[model_name][0]['input_ids'][0], list), model_name
    else:
        assert isinstance(tokenize_testsets[model_name][0]['input_ids'][0], int), model_name
print('_______________________________')

if os.path.isfile(f'{dirname}/mediqaans-test-predicts-all-iter-retuls.bin'):
    with open(f'{dirname}/mediqaans-test-predicts-all-iter-retuls.bin', 'rb') as f:
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
print('Testing multiple length-ranges')
for (CONTROL_MIN, CONTROL_MAX) in length_ranges:
    predicts = {}
    results = {}
    decoded_texts = {}
    decoded_labels = {}

    if os.path.isfile(f'{dirname}/mediqaans-predicts-length-range-{CONTROL_MIN}-{CONTROL_MAX}.bin'):
        with open(f'{dirname}/mediqaans-predicts-length-range-{CONTROL_MIN}-{CONTROL_MAX}.bin', 'rb') as f:
            predicts = pickle.load(f)

    for model_name in models:
        data_collator = None
        tokenizer = tokenizers[model_name]()

        if 'non-mca' not in model_name and 'google' not in model_name:
            data_collator = DataCollatorForMultipleChannelAttention(tokenizer=tokenizer)
        else:
            data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer)

        if model_name not in predicts:
            model = models[model_name]()
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
                                                               golden=testset['summ_abs'])

        decoded_texts[model_name] = decoded_preds
        results[model_name] = result
        results[model_name]['overflow_ratio'] = round(np.mean(tokenize_testsets[model_name]['overflow_ratio']), 4)
        decoded_labels[model_name] = decoded_label

        print('___________________')
        print(model_name)
        print(result)
        print('___________________')
    with open(f'{dirname}/mediqaans-predicts-length-range-{CONTROL_MIN}-{CONTROL_MAX}.bin', 'wb') as f:
        pickle.dump(predicts, f)
    ttresults[f'{CONTROL_MIN}-{CONTROL_MAX}'] = results
with open(f'{dirname}/mediqaans-test-predicts-all-iter-retuls.bin', 'wb') as f:
    pickle.dump(ttresults, f)

print('Writing meta to files')
fixed_rs = ttresults[f'100-160']
df = pd.DataFrame.from_dict({model_name_map[model_name]: fixed_rs[model_name] for model_name in fixed_rs})
df = df.transpose()
df.to_csv(f'{dirname}/documents/MEDIQA-AnS-full-fixed-setting.csv')
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

tokenizer = list(tokenizers.values())[0]()  # they share a same tokenizer of facebook
encode_summary = lambda example: {'ids': tokenizer(example['summ_abs'], truncation=False)['input_ids']}
summary_ids = testset.map(encode_summary, remove_columns=testset.column_names)

summary_ids = summary_ids['ids']
summary_lens = [len(ids) for ids in summary_ids]
des = pd.Series(summary_lens).describe()
with open(f'{dirname}/documents/MEDIQA-AnS-scores.pk', 'wb+') as f:
    to_write = (des, statistics_by_model, model_name_map)
    pickle.dump(to_write, f)
print('------------------------------')

print('Testing adaptive setting')
predicts = {}
if os.path.isfile(f'{dirname}/mediqaans-perfect-length-predicts.bin'):
    with open(f'{dirname}/mediqaans-perfect-length-predicts.bin', 'rb') as f:
        predicts = pickle.load(f)

results = {}
decoded_texts = {}
decoded_labels = {}

for model_name in models:

    print('tokenizing___________________')
    print(model_name)
    data_collator = None
    tokenizer = tokenizers[model_name]()
    if 'non-mca' not in model_name and 'google' not in model_name:
        data_collator = DataCollatorForMultipleChannelAttention(tokenizer=tokenizer)
    else:
        data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer)

    if model_name not in predicts:
        model = models[model_name]()
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
                                                           golden=testset['summ_abs'])
    decoded_texts[model_name] = decoded_preds
    results[model_name] = result
    results[model_name]['overflow_ratio'] = round(np.mean(tokenize_testsets[model_name]['overflow_ratio']), 4)
    decoded_labels[model_name] = decoded_label

    print('___________________')
    print(model_name)
    print(result)
    print('___________________')

with open(f'{dirname}/mediqaans-perfect-length-predicts.bin', 'wb') as f:
    pickle.dump(predicts, f)
print('Saving results')
df = pd.DataFrame.from_dict({model_name_map[model_name]: results[model_name] for model_name in results})
df = df.transpose()
df.to_csv(
    f'{dirname}/documents/MEDIQA-AnS-full-adaptive-setting.csv')
print('------------------------------')
