import sys
sys.path.append('..')  # A hack
from typing import Optional
from omegaconf import DictConfig, OmegaConf
from rlprompt.utils.utils import (colorful_print, compose_hydra_config_store,
                                  get_hydra_output_dir)
from tst_modules import PromptedGenerator, TextStyleTransferOutputSelector
from fsc_helpers import (TextStyleTransferDatasetConfig,
                         PromptedTextStyleTransferRewardConfig,
                         load_few_shot_classification_dataset,)
import hydra
from dataclasses import dataclass

import re
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from rouge import Rouge

from bert_score import BERTScorer

@dataclass
class TextStyleTransferEvaluationConfig:
    prompt_0_to_1: Optional[str] = None
    prompt_1_to_0: Optional[str] = None


# Compose default config
config_list = [TextStyleTransferDatasetConfig,
               PromptedTextStyleTransferRewardConfig,
               TextStyleTransferEvaluationConfig]
cs = compose_hydra_config_store('base_eval', config_list)


@hydra.main(version_base=None, config_path="./", config_name="eval_config")
def main(config: DictConfig):
    source_texts, target_labels = \
        load_few_shot_classification_dataset(config.dataset,
                                             config.dataset_seed,
                                             "test", config.base_path,
                                             config.num_shots)

    style_rewards = []
    sum_rewards = []
    content_rewards = []
    rouge = Rouge()

    WHITESPACE_HANDLER = lambda k: re.sub('\s+', ' ', re.sub('\n+', ' ', k.strip()))

    model_name = "/root/mjs/TST/examples/few-shot-classification/mT5_m2m_crossSum_enhanced"
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    get_lang_id = lambda lang: tokenizer._convert_token_to_id(
        model.config.task_specific_params["langid_map"][lang][1]
    )

    target_lang = "english"  # for a list of available language names see below

    for source, target in zip(source_texts, target_labels):
        input_ids = tokenizer(
            [WHITESPACE_HANDLER(source)],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=512
        )["input_ids"]

        output_ids = model.generate(
            input_ids=input_ids,
            decoder_start_token_id=get_lang_id(target_lang),
            max_length=84,
            no_repeat_ngram_size=2,
            num_beams=4,
        )[0]

        summary = tokenizer.decode(
            output_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )

        scores = rouge.get_scores(summary, target)
        prob = scores[0]["rouge-1"]["f"]
        prob1 = scores[0]["rouge-2"]["f"]
        prob2 = scores[0]["rouge-l"]["f"]
        # print(prob)
        # prob = ((c['label'] == target_label) * c['score']
        #         + (c['label'] != target_label) * (1 - c['score']))
        # print("prob:", prob*100)
        style_rewards.append(prob * 100)
        sum_rewards.append(prob1 * 100)
        content_rewards.append(prob2 * 100)

    # 计算 rewards 列表的平均值
    average_rewards = sum(content_rewards) / len(content_rewards)

    # 计算 contents 列表的平均值
    average_contents = sum(sum_rewards) / len(sum_rewards)

    # 计算 styles 列表的平均值
    average_styles = sum(style_rewards) / len(style_rewards)

    # 打印平均值
    print("rouge-1：", average_styles)
    print("rouge-2：", average_contents)
    print("rouge-l：", average_rewards)

if __name__ == "__main__":
    main()


