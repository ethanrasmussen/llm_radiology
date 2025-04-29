import os
# Avoid parallelism deadlocks with tokenizers
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, set_seed
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq
from datasets import Dataset
import pandas as pd
import numpy as np

from CXRMetric.CheXbert.src.label import label

# Label constants
MISSING = 0
POSITIVE = 1
NEGATIVE = 2
UNCERTAIN = 3

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chexbert_path", type=str, required=True,
                        help="Path to the CheXbert model checkpoint")
    parser.add_argument("--dataset_path", type=str, required=True,
                        help="Path to CSV file containing `report` column")
    parser.add_argument("--output_dir", type=str, default="./report_cleaning",
                        help="Directory to save outputs.")
    parser.add_argument("--outfile", type=str, default="clean_output.csv",
                        help="Name of the output CSV file.")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1,
                        help="Batch size for evaluation/inference.")
    parser.add_argument("--model_id", type=str, default="google/flan-t5-XXL",
                        help="Hugging Face model identifier.")
    parser.add_argument("--generation_max_length", type=int, default=200,
                        help="Max tokens to generate.")
    parser.add_argument("--top_k", type=int, default=1,
                        help="Top-k sampling parameter.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed.")
    parser.add_argument("--deepspeed", type=str, default=None,
                        help="Path to deepspeed config file, if using.")
    parser.add_argument("--bf16", type=bool,
                        default=(torch.cuda.get_device_capability()[0] >= 8),
                        help="Whether to use bfloat16.")
    args, _ = parser.parse_known_args()
    return args


def label_heuristic(args, output_list):
    """
    Runs CheXbert on the original and cleaned reports, returns a boolean mask
    of which reports changed their positive/negative labels.
    """
    pos_path = os.path.join(args.output_dir, "gt_pos_labels.pt")
    neg_path = os.path.join(args.output_dir, "gt_neg_labels.pt")
    # Compute ground truth once
    if not (os.path.exists(pos_path) and os.path.exists(neg_path)):
        df = pd.read_csv(args.dataset_path).fillna('_')
        tmp_csv = os.path.join(args.output_dir, "gt_pre_chexbert.csv")
        df[['report']].to_csv(tmp_csv, index=False)
        y_gt = label(args.chexbert_path, tmp_csv, use_gpu=False)
        y_gt = np.array(y_gt).T
        # Positive mask
        y_gt_pos = y_gt.copy()
        y_gt_pos[(y_gt_pos == NEGATIVE) | (y_gt_pos == UNCERTAIN)] = 0
        # Negative mask
        y_gt_neg = y_gt.copy()
        y_gt_neg[(y_gt_neg == POSITIVE) | (y_gt_neg == UNCERTAIN)] = 0
        y_gt_neg[y_gt_neg == NEGATIVE] = 1
        torch.save(y_gt_pos, pos_path)
        torch.save(y_gt_neg, neg_path)
        os.remove(tmp_csv)
    # Load with weights_only=False to allow numpy unpickling
    y_gt_pos = torch.load(pos_path, weights_only=False)
    y_gt_neg = torch.load(neg_path, weights_only=False)

    # Generate labels on cleaned outputs
    df_gen = pd.DataFrame(output_list, columns=['report'])
    df_gen = df_gen.replace('REMOVED', '_')
    tmp_csv = os.path.join(args.output_dir, 'gen_pre_chexbert.csv')
    df_gen.to_csv(tmp_csv, index=False)
    y_gen = label(args.chexbert_path, tmp_csv, use_gpu=False)
    y_gen = np.array(y_gen).T
    os.remove(tmp_csv)
    # Masks on generated
    y_gen_pos = y_gen.copy()
    y_gen_pos[(y_gen_pos == NEGATIVE) | (y_gen_pos == UNCERTAIN)] = 0
    y_gen_neg = y_gen.copy()
    y_gen_neg[(y_gen_neg == POSITIVE) | (y_gen_neg == UNCERTAIN)] = 0
    y_gen_neg[y_gen_neg == NEGATIVE] = 1

    # Compare
    pos_diff = np.logical_xor(y_gt_pos, y_gen_pos).any(axis=1)
    neg_diff = np.logical_xor(y_gt_neg, y_gen_neg).any(axis=1)
    return np.logical_or(pos_diff, neg_diff)


def predict(args, model, tokenizer, data_collator,
            instructions, examples, report_list, outfile):
    """
    Runs Seq2SeqTrainer.predict on the dataset and applies label heuristic.
    """
    # Preprocessing function with truncation
    def preprocess_function(batch):
        inputs = tokenizer(
            batch['input_text'],
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors='pt'
        )
        inputs['labels'] = inputs.input_ids.clone()
        return inputs

    # Build dataset
    input_list = [instructions.format(EXAMPLES=examples, INPUT_QUERY=sent)
                  for sent in report_list]
    ds = Dataset.from_dict({'input_text': input_list})
    ds = ds.map(
        preprocess_function,
        batched=True,
        remove_columns=['input_text'],
        load_from_cache_file=False,
        desc='Tokenizing inputs'
    )

    os.makedirs(args.output_dir, exist_ok=True)

    # Trainer args
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        predict_with_generate=True,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        generation_max_length=args.generation_max_length,
        fp16=False,
        bf16=args.bf16,
        deepspeed=args.deepspeed,
        logging_strategy='steps',
        logging_steps=500,
    )
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=ds,
        eval_dataset=ds,
        data_collator=data_collator,
    )

    # Run prediction
    predict_results = trainer.predict(ds)

    if trainer.is_world_process_zero():
        # Decode
        preds = np.where(
            predict_results.predictions != -100,
            predict_results.predictions,
            tokenizer.pad_token_id
        )
        decoded = tokenizer.batch_decode(
            preds, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        # Apply heuristic: if labels changed, keep original report
        diff_mask = label_heuristic(args, decoded)
        final_reports = [orig if change else gen
                         for gen, orig, change in zip(decoded, report_list, diff_mask)]
        df_out = pd.DataFrame(final_reports, columns=['report'])
        df_out.to_csv(os.path.join(args.output_dir, outfile), index=False)

    # Free GPU memory
    del predict_results
    torch.cuda.empty_cache()

if __name__ == '__main__':
    args = parse_args()
    set_seed(args.seed)

    # Model & tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        args.model_id, torch_dtype=torch.bfloat16
    )
    model.eval()
    # Resize embeddings if tokenizer changed
    if len(tokenizer) != model.get_input_embeddings().weight.size(0):
        model.resize_token_embeddings(len(tokenizer))

    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=tokenizer.pad_token_id,
        pad_to_multiple_of=8
    )

    # Read input reports
    df_main = pd.read_csv(args.dataset_path)
    report_list = df_main['report'].fillna('_').tolist()

    # Iterate over cleaning rules
    RULES = list(range(1, 8))
    for i in RULES:
        rule = f'rewrite{i}'
        print(f'Running {rule}')
        instr_path = f'./prompts/report_clean_rules/{rule}_instructions.txt'
        fewshot_path = f'./prompts/report_clean_rules/{rule}_sen_fewshot.txt'
        instructions = open(instr_path).read()
        examples = open(fewshot_path).read()
        outfile = f'{rule}_intermediate.csv'

        predict(args, model, tokenizer, data_collator,
                instructions, examples, report_list, outfile)
        # Synchronize in distributed context
        torch.distributed.barrier()
        # Load outputs for next iteration
        report_list = pd.read_csv(
            os.path.join(args.output_dir, outfile)
        )['report'].tolist()
