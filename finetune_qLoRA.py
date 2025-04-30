import os
import copy
import logging
from dataclasses import dataclass, field
from typing import Dict, Sequence

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    BitsAndBytesConfig,
    HfArgumentParser,
)
from peft import LoraConfig, get_peft_model
from dataclasses import dataclass, field
from typing import Dict, Sequence

from transformers.utils.dummy_pt_objects import Data2VecTextForTokenClassification

IGNORE_INDEX = -100

# data formatting is identical to before
def load_and_tokenize(data_path, tokenizer):
    import json
    with open(data_path) as f:
        examples = json.load(f)
    texts = []
    for ex in examples:
        instr = ex["instruction"]
        inp   = ex.get("input","")
        prompt = (
          f"Below is an instruction that describes a task"
          + (f", paired with an input. Write a response.\n\n### Instruction:\n{instr}"
             + (f"\n\n### Input:\n{inp}" if inp else "")
             + "\n\n### Response:")
        )
        target = ex["output"] + tokenizer.eos_token
        texts.append((prompt, target))
    # tokenize on the fly in collator
    return texts

class QLoRADataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self): return len(self.texts)

    def __getitem__(self, i):
        prompt, target = self.texts[i]
        # concatenate and then mask
        enc = self.tokenizer(
            prompt, return_tensors="pt",
            max_length=self.max_length, truncation=True
        )
        out = self.tokenizer(
            prompt + target,
            return_tensors="pt",
            max_length=self.max_length, truncation=True,
            padding="max_length"
        )
        input_ids = out.input_ids[0]
        labels = input_ids.clone()
        labels[: enc.input_ids.ne(self.tokenizer.pad_token_id).sum()] = IGNORE_INDEX
        return dict(input_ids=input_ids, labels=labels, attention_mask=out.attention_mask[0])

@dataclass
class DataCollator:
    tokenizer: AutoTokenizer

    def __call__(self, features: Sequence[Dict[str, torch.Tensor]]):
        batch = {
            "input_ids": torch.stack([f["input_ids"] for f in features]),
            "labels":    torch.stack([f["labels"]    for f in features]),
            "attention_mask": torch.stack([f["attention_mask"] for f in features]),
        }
        return batch

@dataclass
class ModelArguments:
    "Arguments for pretrained model to load"
    model_name_or_path: str = field(metadata={'help': 'Path to pretrained model'})

@dataclass
class DataArguments:
    "Arguments for training data"
    data_path: str = field(metadata={'help': 'Path to training JSON'})

def main():
    from transformers import HfArgumentParser
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # configure 4-bit quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    # load model & tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        quantization_config=bnb_config,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, use_fast=False
    )
    tokenizer.pad_token_id = tokenizer.eos_token_id

    # inject LoRA adapters
    lora_cfg = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)

    # prepare data
    texts = load_and_tokenize(data_args.data_path, tokenizer)
    train_ds = QLoRADataset(texts, tokenizer)
    collator = DataCollator(tokenizer)

    # trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        data_collator=collator,
    )
    trainer.train()
    trainer.save_model(training_args.output_dir)

if __name__ == "__main__":
    main()
