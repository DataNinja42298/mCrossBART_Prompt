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
    colorful_print(OmegaConf.to_yaml(config), fg='red')
    output_dir = get_hydra_output_dir()

    device_id = 0
    generator = PromptedGenerator(config.task_lm, config.template,
                                  config.end_punct, config.pad_token,
                                  device_id, config.lower_outputs,
                                  config.control_output_length)
    # train_style_classifier = \
    #     os.path.join('..', get_style_classifier('train', config))
    selector = TextStyleTransferOutputSelector(config.style_tokenizer,
                                               config.style_batch_size,
                                               device_id)

    prompt = config.prompt
    print('Prompt:', prompt)

    source_texts, target_labels = \
        load_few_shot_classification_dataset(config.dataset,
                                             config.dataset_seed,
                                             "test", config.base_path,
                                             config.num_shots)

    top_p = 1.0
    generated_texts = generator.sample_generate_batch(
        prompt, source_texts, config.num_samples, config.task_top_k, top_p)
    output_texts, rewards, contents, styles = selector.select_outputs_batch(
        source_texts, generated_texts, target_labels)

    # 计算 rewards 列表的平均值
    average_rewards = sum(rewards) / len(rewards)

    # 计算 contents 列表的平均值
    average_contents = sum(contents) / len(contents)

    # 计算 styles 列表的平均值
    average_styles = sum(styles) / len(styles)

    # 打印平均值
    print("rouge-1：", average_styles)
    print("rouge-2：", average_rewards)
    print("rouge-l：", average_contents)

    add = []
    bert_scorer = BERTScorer('roberta-large', device=0, rescale_with_baseline=False, lang='en')
    # bert_scorer = BERTScorer('bert-base-chinese', device=0, rescale_with_baseline=False, lang='zh')


    for line1, line2 in zip(output_texts, target_labels):
        ctc_scores = bert_scorer.score([line1], [line2])[2]
        add.extend(ctc_scores.tolist())

    sum_content_rewards = sum(add)
    average_content_reward = sum_content_rewards / len(add)

    print("BERTScore：", average_content_reward)

    file = open("output_file.txt", "w")
    for value in output_texts:
        file.write(str(value) + "\n")

    del generator
    del selector

if __name__ == "__main__":
    main()
