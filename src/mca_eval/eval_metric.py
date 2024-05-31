from __future__ import annotations

from itertools import chain
from typing import List, Tuple, Union, Optional

import numpy as np
import pandas as pd
from evaluate import load
from rouge_score import rouge_scorer
from transformers import PreTrainedTokenizer
import torch

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
bertscore = load("bertscore")


def cal_rouge(predictions, references) -> dict:
    def flatten_score(f, p) -> dict:
        pooling_type = 'max'
        if isinstance(f, list):
            scores = scorer.score_multi(f, p)
        else:
            scores = scorer.score(f, p)
            pooling_type = 'avg'
        rs = {}
        for score_type in scores:
            score = scores[score_type]
            d = {f'{pooling_type}-{score_type}-precision': score.precision,
                 f'{pooling_type}-{score_type}-recall': score.recall,
                 f'{pooling_type}-{score_type}-fmeasure': score.fmeasure}
            rs.update(d)
        return rs

    scores = [flatten_score(f, p) for f, p in zip(references, predictions)]

    return pd.DataFrame(scores).describe().iloc[1, :].to_dict()


def cal_bert(predictions, references) -> dict:
    pooling_type = 'avg'
    has_multi_ref = None
    if isinstance(references[0], list):
        new_predictions = []
        new_references = []
        has_multi_ref = []
        for p, rs in zip(predictions, references):
            start_point = len(new_references)
            for r in rs:
                new_predictions.append(p),
                new_references.append(r)
            end_point = len(new_references)
            has_multi_ref.append((start_point, end_point))

        predictions = new_predictions
        references = new_references

    bertscores = bertscore.compute(predictions=predictions, references=references, model_type="roberta-large", device=device)
    bert_type = ['precision', 'recall', 'f1']
    rs = {
        f'{pooling_type}-bert-{bst}': np.mean(bertscores[f'{bst}']) for bst in bert_type
    }
    if has_multi_ref is not None:
        pooling_type = 'max'

        bf = {f'{pooling_type}-bert-{bst}': [] for bst in bert_type}
        for (start, end) in has_multi_ref:
            best_point = start + np.argmax(bertscores['f1'][start:end])
            for bst in bert_type:
                bf[f'{pooling_type}-bert-{bst}'].append(bertscores[f'{bst}'][best_point])
        bf = {f'{pooling_type}-bert-{bst}': np.mean(bf[f'{pooling_type}-bert-{bst}']) for bst in bert_type}
        rs.update(bf)

    return rs


def ids_to_text(batch_ids, tokenizer):
    return tokenizer.batch_decode(batch_ids, skip_special_tokens=True)


def metrics_for_txt(decoded_preds, decoded_labels):
    result = cal_rouge(predictions=decoded_preds, references=decoded_labels)  # max pooling
    bertscores = cal_bert(predictions=decoded_preds, references=decoded_labels)  # max & avg pooling
    if isinstance(decoded_labels[0], list):
        new_predictions = []
        new_references = []
        for p, rs in zip(decoded_preds, decoded_labels):
            for r in rs:
                new_predictions.append(p),
                new_references.append(r)
        decoded_preds = new_predictions
        decoded_labels = new_references
    avg_gouge = cal_rouge(predictions=decoded_preds, references=decoded_labels)  # avg pooling
    result.update(avg_gouge)
    result.update(bertscores)
    return result


def compute_metrics(eval_pred: Tuple,
                    tokenizer: PreTrainedTokenizer,
                    golden: Optional[Union[List[str], List[List[str]]]] = None):
    predictions, labels = eval_pred
    predictions = np.where(predictions >= 0, predictions, tokenizer.pad_token_id)

    decoded_preds = ids_to_text(predictions, tokenizer)
    decoded_preds = [' '.join(decoded_pred.split()) for decoded_pred in decoded_preds]
    decoded_labels = golden
    if decoded_labels is None:
        assert labels is not None, "Both labels and golden are None! No reference is found"
        labels = np.where(labels >= 0, labels, tokenizer.pad_token_id)
        decoded_labels = ids_to_text(labels, tokenizer)
        exp_lens = [np.count_nonzero(lab != tokenizer.pad_token_id) for lab in labels]
        result = {"expected_len": np.mean(exp_lens)}
    else:
        refs = decoded_labels
        if isinstance(decoded_labels[0], List):
            refs = list(chain(*decoded_labels))
        expected_ids = tokenizer.batch_encode_plus(refs, padding=False)['input_ids']
        exp_lens = [len(lab) for lab in expected_ids]
        result = {"expected_len": np.mean(exp_lens)}
    result.update(metrics_for_txt(decoded_preds, decoded_labels))
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
    result["gen_len"] = np.mean(prediction_lens)

    result = {k: round(v, 4) for k, v in result.items()}
    return result, decoded_preds, decoded_labels
