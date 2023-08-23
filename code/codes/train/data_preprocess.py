import logging
import sys
from dataclasses import dataclass, field
from typing import Optional
import torch
import copy

import datasets
from datasets import load_dataset

import transformers
from transformers import (
    HfArgumentParser,
    T5Tokenizer,
    LlamaTokenizer,
    set_seed,
)

q_pre = "<s>\n"
qa_link = "\n"
a_pos = "\n</s>"

logger = logging.getLogger(__name__)

@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )

@dataclass
class DataTrainingArguments:
    data_path: Optional[str] = field(default=None, metadata={"help": "The input training data file (a jsonlines)."})
    model_max_length: int = field(
        default=2048,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    preprocessed_path: str = field(
        default=None, metadata={"help": "Path to the preprocessed training data."}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )

def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments))
    model_args, data_args = parser.parse_args_into_dataclasses()
    
    data_files = {}
    data_files["train"] = data_args.data_path
    raw_datasets = load_dataset(
        "json",
        data_files=data_files
    )
    column_names = raw_datasets["train"].column_names
    print("load dataset finished")

    if "t5" in model_args.model_name_or_path:
        # use truncation_side='left' to preserve linking between end of prompt and target labels
        tokenizer = T5Tokenizer.from_pretrained(model_args.model_name_or_path, truncation_side='left')

        def preprocess_function(examples):
            src_inputs = [q_pre + example[0]["value"] + qa_link for example in examples["conversations"]]
            src_model_inputs = tokenizer(src_inputs, max_length=data_args.model_max_length, padding='longest', truncation=True, add_special_tokens=False)
            trg_inputs = [example[1]["value"] + a_pos for example in examples["conversations"]]
            trg_model_inputs = tokenizer(trg_inputs, max_length=data_args.model_max_length, padding='longest', truncation=True, add_special_tokens=False)
            src_model_inputs["labels"] = [
                [(l if l != tokenizer.pad_token_id else label_ignore_id) for l in label] for label in trg_model_inputs["input_ids"]
            ]
            return src_model_inputs
    else:
        # use truncation_side='left' to preserve linking between end of prompt and target labels
        tokenizer = LlamaTokenizer.from_pretrained(model_args.model_name_or_path, truncation_side='left')

        def preprocess_function(examples):
            inputs = [q_pre + example[0]["value"] + qa_link + example[1]["value"] + a_pos for example in examples["conversations"]]
            model_inputs = tokenizer(inputs, max_length=data_args.model_max_length, padding="longest", truncation=True, add_special_tokens=False)
            model_inputs["labels"] = copy.deepcopy(model_inputs["input_ids"])
            for e_i, example in enumerate(examples["conversations"]):
                source_text = q_pre + example[0]["value"] + qa_link
                target_text = example[1]["value"] + a_pos
                source_ids = tokenizer.encode(source_text, add_special_tokens=False)
                target_ids = tokenizer.encode(target_text, add_special_tokens=False)
                if len(source_ids) >= data_args.model_max_length:
                    model_inputs["labels"][e_i] = [label_ignore_id] * data_args.model_max_length
                    continue
                else:
                    model_inputs["labels"][e_i][:len(source_ids)] = [label_ignore_id] * len(source_ids)
                    if len(target_ids) + len(source_ids) >= len(model_inputs["input_ids"][e_i]):
                        continue
                    else:
                        model_inputs["labels"][e_i][(len(target_ids) + len(source_ids)):] = [label_ignore_id] * (len(model_inputs["input_ids"][e_i]) - len(target_ids) - len(source_ids))
            model_inputs["input_ids"] = torch.tensor(model_inputs["input_ids"])
            model_inputs["labels"] = torch.tensor(model_inputs["labels"])
            model_inputs["attention_mask"] = model_inputs["input_ids"].ne(tokenizer.pad_token_id)
            return model_inputs
    
    label_ignore_id = -100

    print("start data preprocess")
    train_dataset = raw_datasets["train"]
    train_dataset = train_dataset.map(
        preprocess_function,
        batched=True,
        batch_size=len(train_dataset),
        remove_columns=column_names,
        num_proc=data_args.preprocessing_num_workers,
        load_from_cache_file=False,
        desc="Running tokenizer on train dataset"
    )
    train_dataset.save_to_disk(data_args.preprocessed_path)
    print("data preprocess finished")

if __name__ == "__main__":
    main()