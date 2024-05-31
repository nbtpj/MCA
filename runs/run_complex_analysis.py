import pandas as pd
import torch
from mca_bart import MCAForConditionalGeneration as MCAModel
from torch.profiler import profile, record_function, ProfilerActivity
from tqdm.auto import tqdm
from transformers import BigBirdPegasusConfig, BartConfig, BigBirdPegasusForConditionalGeneration, \
    BartForConditionalGeneration
import re

share_config = {
    "d_model": 192,
    "decoder_attention_heads": 3,
    "decoder_ffn_dim": 768,
    "decoder_layers": 1,
    "encoder_attention_heads": 3,
    "encoder_ffn_dim": 768,
    "encoder_layers": 1,
    "max_position_embeddings": 16384,
    "no_repeat_ngram_size": 3,
    "vocab_size": 200
}

bigbird_config = BigBirdPegasusConfig(**share_config)
bigbird_config.block_size = 32
bigbird_model_32 = BigBirdPegasusForConditionalGeneration(bigbird_config)
bigbird_config.block_size = 16
bigbird_model_16 = BigBirdPegasusForConditionalGeneration(bigbird_config)
bigbird_config.block_size = 64
bigbird_model_64 = BigBirdPegasusForConditionalGeneration(bigbird_config)
bart_config = BartConfig(**share_config)
bart_model = BartForConditionalGeneration(bart_config)
mca_model = MCAModel(bart_config)

models = [
    ['bigbird_model_16', bigbird_model_16],
    ['bigbird_model_32', bigbird_model_32],
    ['bigbird_model_64', bigbird_model_64],
    ['bart_model', bart_model],
    ['mca_model', mca_model]
]

for i, (model_name, model) in enumerate(models):
    models[i][-1] = models[i][-1].to('cuda:0')
    print(model_name)
    print(sum(p.numel() for p in model.parameters()))
    print('______________')

torch.cuda.empty_cache()
batch_size = 1
channel_size = 2 ** 8  # token/channel(document)

# profile encoding+decoding process
with torch.no_grad():
    with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU, ],
                 record_shapes=False,
                 profile_memory=True) as prof:
        for i in tqdm(range(2, 32)):
            i *= 2
            num_channel = i
            input_tensor = torch.randint(1, share_config['vocab_size'] - 1, (batch_size, channel_size * i)).to('cuda')
            input_for_mca = torch.reshape(input_tensor, (batch_size, i, -1)).to('cuda')
            decoder_tensor = torch.randint(1, share_config['vocab_size'] - 1, (batch_size, channel_size)).to('cuda')
            for model_name, model in models:
                if 'mca' not in model_name:
                    with record_function(f"{model_name}_test_model_{i}"):
                        model(input_ids=input_tensor, decoder_input_ids=decoder_tensor)
                else:
                    with record_function(f"{model_name}_test_model_{i}"):
                        model(input_ids=input_for_mca, decoder_input_ids=decoder_tensor)

tab = []
lines = prof.key_averages().table(sort_by="cuda_time_total", row_limit=10000).split('\n')
for line in lines:
    if '----' not in line:
        line = [i.strip() for i in line.split('  ') if i]
        if len(line) == 15 and ('_test_model_' in line[0] or 'Name' in line[0]):
            tab.append(line)
columns = tab[0]
columns.append('Input Length in Token')

data = []
for line in tab[1:]:
    seq_length = int(line[0].split('_')[-1]) * 2 ** 8
    model_name = line[0].split('_test_model_')[0]
    line[0] = model_name
    line.append(seq_length)
    data.append(line)

tab = pd.DataFrame(data, columns=columns, index=None)


def convert_mem(thing):
    if 'Gb' in thing:
        return -1000 * float(thing.replace('Gb', ''))
    if 'Mb' in thing:
        return -1 * float(thing.replace('Mb', ''))
    return thing


def convert_time(thing):
    if 'us' in thing:
        return float(thing.replace('us', '')) / 1000
    if 'ms' in thing:
        return float(thing.replace('ms', ''))
    return thing



for column in tab.columns:
    if all([isinstance(r, str) and ('ms' in r or 'us' in r) for r in tab[column].values.tolist()]):
        tab[column + ' in ms'] = [convert_time(r) for r in tab[column].values]
    if all([isinstance(r, str) and ('Gb' in r or 'Mb' in r) for r in tab[column].values.tolist()]):
        tab[column + ' in Mb'] = [convert_mem(r) for r in tab[column].values]
target_tab = tab.loc[:, ['Name', 'CUDA time avg in ms', 'Self CUDA Mem in Mb', 'Input Length in Token']]
target_tab.to_csv('./documents/encode-decode-time-mem.csv', index=False)

torch.cuda.empty_cache()

# profile encoding-only process
with torch.no_grad():
    with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU, ],
                 record_shapes=False,
                 profile_memory=True) as prof:
        for i in tqdm(range(2, 32)):
            i *= 2
            num_channel = i
            input_tensor = torch.randint(1, share_config['vocab_size'] - 1, (batch_size, channel_size * i)).to('cuda')
            input_for_mca = torch.reshape(input_tensor, (batch_size, i, -1)).to('cuda')
            decoder_tensor = torch.randint(1, share_config['vocab_size'] - 1, (batch_size, channel_size)).to('cuda')
            for model_name, model in models:
                if 'mca' not in model_name:
                    with record_function(f"{model_name}_test_model_{i}"):
                        model.get_encoder()(input_ids=input_tensor)
                else:
                    with record_function(f"{model_name}_test_model_{i}"):
                        model.encode(input_for_mca)

tab = []
lines = prof.key_averages().table(sort_by="cuda_time_total", row_limit=10000).split('\n')
for line in lines:
    if '----' not in line:
        line = [i.strip() for i in line.split('  ') if i]
        if len(line) == 15 and ('_test_model_' in line[0] or 'Name' in line[0]):
            tab.append(line)
columns = tab[0]
columns.append('Input Length in Token')

data = []
for line in tab[1:]:
    seq_length = int(line[0].split('_')[-1]) * 2 ** 8
    model_name = line[0].split('_test_model_')[0]
    line[0] = model_name
    line.append(seq_length)
    data.append(line)

tab = pd.DataFrame(data, columns=columns, index=None)

for column in tab.columns:
    if all([isinstance(r, str) and ('ms' in r or 'us' in r) for r in tab[column].values.tolist()]):
        tab[column + ' in ms'] = [convert_time(r) for r in tab[column].values]
    if all([isinstance(r, str) and ('Gb' in r or 'Mb' in r) for r in tab[column].values.tolist()]):
        tab[column + ' in Mb'] = [convert_mem(r) for r in tab[column].values]
target_tab = tab.loc[:, ['Name', 'CUDA time avg in ms', 'Self CUDA Mem in Mb', 'Input Length in Token']]
target_tab.to_csv('./documents/encode-time-mem.csv', index=False)

