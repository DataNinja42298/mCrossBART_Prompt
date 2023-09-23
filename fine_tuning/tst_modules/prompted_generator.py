from transformers import AutoTokenizer, pipeline
from torch.utils.data import Dataset
from tqdm import tqdm
import json
from typing import Optional, List

from transformers import MBartForConditionalGeneration, MBartTokenizer
import torch
from transformers import MBartForConditionalGeneration, MBart50Tokenizer, MBartTokenizer, MBart50TokenizerFast

import re
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class PromptedGenerator:
    def __init__(
        self,
        model: str,
        template: str,
        end_punct: str,
        pad_token: str,
        device_id: int,
        lower_outputs: bool,
        control_output_length: bool
    ):

        # self.tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-cc25", src_lang="my_MM",
        #                                                  tgt_lang="zh_CN")
        # self.model = MBartForConditionalGeneration.from_pretrained("/root/mjs/clidm/outputs/mbart50_zh")

        # model_name = "csebuetnlp/mT5_m2m_crossSum_enhanced"
        model_name = "/root/mjs/TST/examples/few-shot-classification/mT5_m2m_crossSum_enhanced"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

        self.template = template
        self.end_punct = end_punct
        self.lower_outputs = lower_outputs
        self.control_output_length = control_output_length

    def _get_max_new_tokens(
        self,
        seq_len: int,
        control_output_length: Optional[bool] = None
    ) -> Optional[int]:
        if control_output_length is None:
            control_output_length = self.control_output_length
        if control_output_length:
            # This hack tends to speed up generation compared to default
            return max(1.5 * seq_len, seq_len + 10)
        else:
            return None

    def sample_generate(
        self,
        prompt: str,
        source_text: str,
        num_samples: int,
        top_k: Optional[int],
        top_p: float,
        lower_outputs: Optional[bool] = None,
        control_output_length: Optional[bool] = None,
        **kwargs
    ) -> List[str]:
        # Used for controlling output length
        # formatted_template = self.template.format(prompt=prompt,
        #                                           sentence_1=source_text)
        # print(formatted_template)

        src_len = len(self.tokenizer(source_text)['input_ids'])
        max_new_tokens = self._get_max_new_tokens(
            src_len, control_output_length=control_output_length)
        pad_token_id = self.tokenizer.pad_token_id

        WHITESPACE_HANDLER = lambda k: re.sub('\s+', ' ', re.sub('\n+', ' ', k.strip()))

        get_lang_id = lambda lang: self.tokenizer._convert_token_to_id(
            self.model.config.task_specific_params["langid_map"][lang][1]
        )

        target_lang = "french"  # for a list of available language names see below

        input_ids = self.tokenizer(
            [WHITESPACE_HANDLER(prompt + "[summarize]" + source_text)],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=512
        )["input_ids"]

        output_ids = self.model.generate(
            input_ids=input_ids,
            decoder_start_token_id=get_lang_id(target_lang),
            max_length=84,
            no_repeat_ngram_size=2,
            num_beams=32,
            num_return_sequences=32  # 将这个值设置为32
        )

        generated_texts = []
        for i, output_seq in enumerate(output_ids):
            generated_str = self.tokenizer.decode(
                output_seq,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )

            generated_texts.append(generated_str.strip())
        return generated_texts

        # input_ids = self.tokenizer.encode(prompt + "[summarize]" + source_text, return_tensors='pt', truncation=True, max_length=128)
        #
        # generated_ids = self.model.generate(
        #     input_ids=input_ids,
        #     # attention_mask=attention_mask,
        #     do_sample=True,
        #     num_return_sequences=32,
        #     use_cache=True,
        #     num_beams=5,
        #     max_length=128,
        #     decoder_start_token_id=self.tokenizer.lang_code_to_id["zh_CN"],
        # )
        #
        # generated_str = self.tokenizer.batch_decode(generated_ids.tolist(), skip_special_tokens=True)
        # print("generated_str:", generated_str)
        #
        # generated_texts = []

    def sample_generate_batch(
        self,
        prompt: str,
        source_texts: List[str],
        num_samples: int,
        top_k: Optional[int],
        top_p: float,
        lower_outputs: Optional[bool] = None,
        control_output_length: Optional[bool] = None,
        **kwargs
    ) -> List[List[str]]:
        all_generated_texts = []
        for i, source_text in tqdm(enumerate(source_texts),
                                   total=len(source_texts)):
            generated_texts = self.sample_generate(
                prompt, source_text, num_samples, top_k, top_p,
                lower_outputs=lower_outputs,
                control_output_length=control_output_length)
            all_generated_texts.append(generated_texts)
        return all_generated_texts

    def postprocess_output(
        self,
        text: str,
        end_punct: Optional[str] = None,
        lower_outputs: Optional[bool] = None
    ) -> str:
        if end_punct is None:
            end_punct = self.end_punct
        if lower_outputs is None:
            lower_outputs = self.lower_outputs

        try:
            end = text.index(end_punct)
        except ValueError:
            end = len(text)
        text = text[:end].strip()

        try:
            end = text.index('.')
        except ValueError:
            end = len(text)
        try:
            end = min(end, text.index('!'))
        except ValueError:
            end = end
        try:
            end = min(end, text.index('?'))
        except ValueError:
            end = end

        text = text[:end+1].strip()
        if lower_outputs:
            text = text.lower()

        return text
