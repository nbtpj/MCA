#!/usr/local/bin/python3

import os
os.chdir('../')

import torch
from mca_bart import MCAForConditionalGeneration, MCATokenizer

dirname = os.path.dirname(__file__)

print('_______reading examples__________')
texts = open(f'{dirname}/samples.txt', encoding='utf8').read().split('\n\n')
prompt = lambda context: '{}'.format(context)

texts = [prompt(''.join(text.split('\n'))) for text in texts]
context = [
    texts[:3],
    texts[:5]
]

print('_______loading models__________')
tokenizer = MCATokenizer.from_pretrained('philschmid/bart-large-cnn-samsum', cache_dir=f'{dirname}/../.cache')
gen_model = MCAForConditionalGeneration.from_pretrained('philschmid/bart-large-cnn-samsum',
                                                        cache_dir=f'{dirname}/../.cache')


print('_______generating__________')
ip = tokenizer(context, return_tensors='pt')
if torch.cuda.is_available():
    print('CUDA device is found, using GPU instead of CPU. '
          'Note that this test supports single-GPU only')
    gen_model = gen_model.to('cuda:0')
    ip = {k: v.to('cuda:0') for k, v in ip.items()}
else:
    try:
        if torch.backends.mps.is_available():
            print('MPS device is found, using MPS instead of CPU.')
            mps_device = torch.device("mps")
            gen_model = gen_model.to(mps_device)
            ip = {k: v.to(mps_device) for k, v in ip.items()}
    except:
        pass

with torch.no_grad():
    outputs = gen_model.generate(**ip,
                                 output_attentions=True,
                                 output_hidden_states=True,
                                 force_lazy=True, lazy_batch_size=2,
                                 max_length=60, min_length=0, use_cache=True)

print('______decoding generated tokens______')
generate_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)

print(generate_texts)


