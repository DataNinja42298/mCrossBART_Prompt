from dataclasses import dataclass
import numpy as np
import pandas as pd
import os
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from typing import Optional, Tuple, List

from fsc_reward import PromptedTextStyleTransferReward


class PromptedClassificationDataset(Dataset):
    def __init__(
        self, 
        source_texts: List[str], 
        target_labels: List[str]
    ):
        assert len(source_texts) == len(target_labels)
        self.source_texts = source_texts
        self.target_labels = target_labels

    def __len__(self):
        return len(self.source_texts)

    def __getitem__(self, idx):
        item = {'source_texts': self.source_texts[idx],
                'target_labels': self.target_labels[idx]}
        return item


def make_few_shot_classification_dataset(
        config: "DictConfig") -> Tuple[PromptedClassificationDataset]: 
    data_dict = {}
    for split in ['train', 'dev', 'test']: 
        source_texts, target_labels = \
            load_few_shot_classification_dataset(config.dataset, 
                                                 config.dataset_seed, 
                                                 split, config.base_path, 
                                                 config.num_shots)
        fsc_dataset = PromptedClassificationDataset(source_texts, 
                                                    target_labels)
        data_dict[split] = fsc_dataset

    return (data_dict['train'], data_dict['dev'], data_dict['test'])


def load_few_shot_classification_dataset(
    dataset: str,
    dataset_seed: Optional[int],
    split: str,
    base_path: str,
    num_shots: int
) -> Tuple[List[str]]:
    assert dataset in ['sst-3']
    assert split in ['train', 'dev', 'test']
    assert num_shots in [5000]

    seed_dict = {0:'en-fr'}
    seed_path = seed_dict[dataset_seed]
    filepath = f'{num_shots}-shot/{dataset}/{seed_path}/{split}.tsv'
    full_filepath = os.path.join(base_path, filepath)
    df = pd.read_csv(full_filepath, sep='\t', error_bad_lines=False)
    if split == "train" or split == "dev":
        if 'text' in df:
            source_texts = df.text.tolist()[:32]
        else:
            source_texts = df.src.tolist()[:32]
        target_labels = df.tgt.tolist()[:32]
    else:
        if 'text' in df:
            source_texts = df.text.tolist()
        else:
            source_texts = df.src.tolist()
        target_labels = df.tgt.tolist()

    return (source_texts, target_labels)


# Key: (dataset, dataset_seed, split)
style_classifier_dict = {
    ('yelp', None, 'train'): './style_classifiers/yelp-bert-base-uncased-train/',
    ('yelp', None, 'test'): './style_classifiers/yelp-bert-base-uncased-test/',
    ('shakespeare', 0, 'train'): './style_classifiers/shakespeare-bert-base-uncased-train-100-0/',
    ('shakespeare', 1, 'train'): './style_classifiers/shakespeare-bert-base-uncased-train-100-1/',
    ('shakespeare', 2, 'train'): './style_classifiers/shakespeare-bert-base-uncased-train-100-2/',
    ('shakespeare', None, 'test'): './style_classifiers/shakespeare-bert-base-uncased-test-all/'}
def get_style_classifier(
    split: str,
    config: "DictConfig"
) -> str:
    dataset_seed = config.dataset_seed if split != 'test' else None
    return style_classifier_dict[(config.dataset, dataset_seed, split)]



def make_prompted_text_style_transfer_reward(
        config: "DictConfig") -> PromptedTextStyleTransferReward:
    return PromptedTextStyleTransferReward(
        config.task_lm, config.is_mask_lm, config.task_top_k,
        config.style_tokenizer,
        config.style_batch_size, config.pad_token, config.num_repeats,
        config.num_samples, config.num_bootstraps, config.compute_zscore,
        config.lower_outputs, config.control_output_length,
        config.template, config.end_punct)


@dataclass
class PromptedTextStyleTransferRewardConfig:
    task_lm: str = 'csebuetnlp/mT5_m2m_crossSum_enhanced'  # 'mbart-large-cc25'
    is_mask_lm: Optional[bool] = None
    task_top_k: int = 10
    style_tokenizer: Optional[str] = None
    style_batch_size: int = 32
    pad_token: str = '<|endoftext|>'
    num_repeats: int = 4
    num_samples: int = 32
    num_bootstraps: int = 4
    compute_zscore: bool = True  # Whether to compute z-score of rewards
    lower_outputs: bool = False  # Whether to convert all outputs to lower case
    control_output_length: bool = False
    template: str = '{prompt}"{sentence_1}" "'
    end_punct: str = '"'


@dataclass
class TextStyleTransferDatasetConfig:
    dataset: str = "sst-2"
    dataset_seed: Optional[int] = None
    # direction: str = "???"
    base_path: str = './data'
    max_size: Optional[int] = None
    max_length: Optional[int] = None
    max_length_tokenizer: Optional[str] = None
    num_shots: int = 5000
