# Zero-cost Transition to Multi-Document Processing in Summarization with Multi-Channel Attention
___

Author Implementation of the paper "Zero-cost Transition to Multi-Document Processing in Summarization with Multi-Channel Attention"
___
## Installation

```commandline
git clone https://github.com/nbtpj/MCA
cd MCA
pip install --upgrade pip
pip install .
```

## Paper Reproduction
We publish all results files in directory [`./runs`](./runs).
For the complete result reproduction, firstly remove all these files then refer to [this notebook file](./runs/Visualization.ipynb).

## Usage

```python
import torch

from mca_bart import MCAForConditionalGeneration, MCATokenizer

multi_doc_1 = [
    '''We uncover the possibility of re-utilizing single-doc estimator in multi-doc problem, in text summarization class (Section 3.1). We propose factorization methods to approximate the target probabilities on certain problems of multi-doc sumarization, namely Vertical Scaling. Vertical Scaling is zero-transition cost approach, while having linear complexity and can adopt single-doc pre-optimized estimators''',
    '''We introduce the encoder-decoder-based Multi-Channel Attention (MCA) architecture as the application of our vertical scaling  solution  (Section 3.2). MCA is developed on the idea of BART, but is proved to be more computationally fficient. Additionally, we also show that MCA can re-utilize the BART parameters directly.''',
    '''We empirically show the improvement of our solutions in handling low-training-resource multi-doc summarization tasks in Section 4.1 and Section 4.2. Experiments show that in both cases of direct utilizing and fine-tuning, MCA can attain the level of full attention accuracy without computational and length limitation, in the multi-doc summarization problem class.'''
]

tokenizer = MCATokenizer.from_pretrained('philschmid/bart-large-cnn-samsum')
gen_model = MCAForConditionalGeneration.from_pretrained('philschmid/bart-large-cnn-samsum')

ip = tokenizer([multi_doc_1, ], return_tensors='pt', padding=True, truncation=True)
if torch.cuda.is_available():
    gen_model = gen_model.to('cuda:0')
    ip = {k: v.to('cuda:0') for k, v in ip.items()}

with torch.no_grad():
    outputs = gen_model.generate(**ip)

generate_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)

print(generate_texts)
# ['In Section 3.1, we show that MCA can be used to solve the problem of multi-doc summarizing in a more efficient way than the previous solution. In section 3.2, we also show that the solution can be re-used to solve a different problem.',]
```

## References
If you use our work, please cite to us:
```

```