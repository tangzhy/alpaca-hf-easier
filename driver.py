#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import os
import copy
import json
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence, List

import datasets
import torch
import torch.distributed as dist
import transformers

from peft import LoraConfig, TaskType, get_peft_model
from peft import PeftModel, PeftConfig

from torch.utils.data import Dataset
from torch.utils.data import DataLoader, SequentialSampler
from transformers import Trainer

from tqdm import tqdm

import utils

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "</s>"
DEFAULT_UNK_TOKEN = "</s>"
PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    peft_model_id: Optional[str] = field(default="facebook/opt-125m")
    use_lora: bool = field(
        default=False,
        metadata={"help": "Whether to lora."},
    )
    lora_r: int = field(
        default=8,
        metadata={"help": "the rank of the lora parameters. The smaller lora_r is , the fewer parameters lora has."},
    )
    lora_alpha: int = field(
        default=32,
        metadata={"help": "Merging ratio between the fine-tuned model and the original. This is controlled by a parameter called alpha in the paper."},
    )
    lora_dropout: float = field(
        default=0.1,
        metadata={"help": "The dropout rate in lora.linear."},
    )
    lora_target_modules: str = field(
        default="q_proj,v_proj",
    )

    def __post_init__(self):
        self.lora_target_modules = self.lora_target_modules.split(",")


@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    max_new_tokens: int = field(
        default=100,
    )
    temperature: float = field(
        default=0.0,
    )


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def preprocess(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """Preprocess the data by tokenizing."""
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=labels)

def preprocess4predict(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict: 
    sources_tokenized = _tokenize_fn(sources, tokenizer)
    input_ids = sources_tokenized["input_ids"]
    input_list = []
    for inp, inpl in zip(sources, input_ids):
        input_list.append({"input": inp, "input_ids": inpl})
    return input_list

class PredictDataset(Dataset):
    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer):
        super(PredictDataset, self).__init__()
        logging.warning("Loading data...")
        list_data_dict = utils.jload(data_path)

        logging.warning("Formatting inputs...")
        prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]
        sources = [
            prompt_input.format_map(example) if example.get("input", "") != "" else prompt_no_input.format_map(example)
            for example in list_data_dict
        ]

        logging.warning("Tokenizing inputs... This may take some time...")
        input_list = preprocess4predict(sources, tokenizer)

        self.ds = datasets.Dataset.from_list(input_list)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        item = self.ds[i]
        item["input_ids"] = torch.tensor(item["input_ids"], dtype=torch.int32)
        return item

class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer):
        super(SupervisedDataset, self).__init__()
        logging.warning("Loading data...")
        list_data_dict = utils.jload(data_path)

        logging.warning("Formatting inputs...")
        prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]
        sources = [
            prompt_input.format_map(example) if example.get("input", "") != "" else prompt_no_input.format_map(example)
            for example in list_data_dict
        ]
        targets = [f"{example['output']}{tokenizer.eos_token}" for example in list_data_dict]

        logging.warning("Tokenizing inputs... This may take some time...")
        data_dict = preprocess(sources, targets, tokenizer)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

@dataclass
class DataCollatorForPredictDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        inputs = [instance["input"] for instance in instances]
        input_ids = [instance["input_ids"] for instance in instances]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        return dict(
            inputs=inputs, 
            input_ids=input_ids,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = SupervisedDataset(tokenizer=tokenizer, data_path=data_args.data_path)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)


def driver():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if training_args.do_train:
        logging.warning("Doing Training...")

        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
        ) 

        if model_args.use_lora:
            logging.warning("Using LoRA...")
            peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False, r=model_args.lora_r, target_modules=model_args.lora_target_modules, lora_alpha=model_args.lora_alpha, lora_dropout=model_args.lora_dropout)
            model = get_peft_model(model, peft_config)
            model.print_trainable_parameters()

        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            use_fast=False,
        )
        if tokenizer.pad_token is None:
            smart_tokenizer_and_embedding_resize(
                special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
                tokenizer=tokenizer,
                model=model,
            )
        if "llama" in model_args.model_name_or_path:
            tokenizer.add_special_tokens(
                {
                    "eos_token": DEFAULT_EOS_TOKEN,
                    "bos_token": DEFAULT_BOS_TOKEN,
                    "unk_token": DEFAULT_UNK_TOKEN,
                }
            )

        data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
        trainer = Trainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)
        trainer.train()

        if dist.is_initialized():
            dist.barrier()

        if model_args.use_lora:
            # Save Adapter 
            if trainer.is_world_process_zero(): 
                model.save_pretrained(training_args.output_dir)
        else:
            trainer.save_model() 
            if trainer.is_world_process_zero(): 
                tokenizer.save_pretrained(training_args.output_dir)

    elif training_args.do_predict and model_args.use_lora:
        logging.warning("Doing Predicting with LoRA...")

        if training_args.local_rank in [-1, 0]:
            os.makedirs(training_args.output_dir, exist_ok=True)
        if dist.is_initialized():
            dist.barrier()

        peft_model_id = model_args.peft_model_id
        config = PeftConfig.from_pretrained(peft_model_id)
        model = transformers.AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path, cache_dir=training_args.cache_dir)
        model = PeftModel.from_pretrained(model, peft_model_id) 
        if training_args.fp16:
            logging.warning("Move model to fp16..")
            model.half()
        model.to(training_args.device)
        model.eval()

        tokenizer = transformers.AutoTokenizer.from_pretrained(
            config.base_model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            use_fast=False,
        )
        if tokenizer.pad_token is None:
            smart_tokenizer_and_embedding_resize(
                special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
                tokenizer=tokenizer,
                model=model,
            )
        if "llama" in config.base_model_name_or_path:
            tokenizer.add_special_tokens(
                {
                    "eos_token": DEFAULT_EOS_TOKEN,
                    "bos_token": DEFAULT_BOS_TOKEN,
                    "unk_token": DEFAULT_UNK_TOKEN,
                }
            )

        predict_dataset = PredictDataset(tokenizer=tokenizer, data_path=data_args.data_path)
        predict_dataset.ds = predict_dataset.ds.shard(training_args.world_size, training_args.local_rank) 
        predict_dataloader = DataLoader( 
                predict_dataset, 
                batch_size=training_args.per_device_eval_batch_size, 
                collate_fn=DataCollatorForPredictDataset(tokenizer=tokenizer),
                shuffle=False, 
                drop_last=False, 
                num_workers=training_args.dataloader_num_workers,
            ) 
        pbar = tqdm(predict_dataloader, desc="Predicting...") if training_args.local_rank in [-1, 0] else predict_dataloader 

        fw = open(os.path.join(training_args.output_dir, f"part-{training_args.local_rank}.json"), "w")
        with torch.cuda.amp.autocast() if training_args.fp16 else nullcontext():
            for batch in pbar: 
                inputs = batch.pop("inputs")
                with torch.no_grad(): 
                    for k, v in batch.items():
                        batch[k] = v.to(training_args.device)
                    generate_ids = model.generate(**batch, max_new_tokens=training_args.max_new_tokens, temperature=training_args.temperature, synced_gpus=True, pad_token_id=tokenizer.eos_token_id)
                    generate_texts = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                    for inp, gen in zip(inputs, generate_texts):
                        inp_len = len(inp)
                        o = {"input": inp, "generate": gen[inp_len:]}
                        o = json.dumps(o, ensure_ascii=False)
                        fw.write(o + "\n")
        fw.close()

        if dist.is_initialized():
            dist.barrier()

    elif training_args.do_predict and (not model_args.use_lora):
        logging.warning("Doing Predicting without LoRA...")

        if training_args.local_rank in [-1, 0]:
            os.makedirs(training_args.output_dir, exist_ok=True)
        if dist.is_initialized():
            dist.barrier()

        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
        ) 
        if training_args.fp16:
            logging.warning("Move model to fp16..")
            model.half()
        model.to(training_args.device)
        model.eval()

        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            use_fast=False,
        )
        if tokenizer.pad_token is None:
            smart_tokenizer_and_embedding_resize(
                special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
                tokenizer=tokenizer,
                model=model,
            )
        if "llama" in model_args.model_name_or_path:
            tokenizer.add_special_tokens(
                {
                    "eos_token": DEFAULT_EOS_TOKEN,
                    "bos_token": DEFAULT_BOS_TOKEN,
                    "unk_token": DEFAULT_UNK_TOKEN,
                }
            )

        predict_dataset = PredictDataset(tokenizer=tokenizer, data_path=data_args.data_path)
        predict_dataset.ds = predict_dataset.ds.shard(training_args.world_size, training_args.local_rank) 
        predict_dataloader = DataLoader( 
                predict_dataset, 
                batch_size=training_args.per_device_eval_batch_size, 
                collate_fn=DataCollatorForPredictDataset(tokenizer=tokenizer),
                shuffle=False, 
                drop_last=False, 
                num_workers=training_args.dataloader_num_workers,
            ) 
        pbar = tqdm(predict_dataloader, desc="Predicting...") if training_args.local_rank in [-1, 0] else predict_dataloader 

        fw = open(os.path.join(training_args.output_dir, f"part-{training_args.local_rank}.json"), "w")
        with torch.cuda.amp.autocast() if training_args.fp16 else nullcontext():
            for batch in pbar: 
                inputs = batch.pop("inputs")
                with torch.no_grad(): 
                    for k, v in batch.items():
                        batch[k] = v.to(training_args.device)
                    generate_ids = model.generate(**batch, max_new_tokens=training_args.max_new_tokens, temperature=training_args.temperature, synced_gpus=True, pad_token_id=tokenizer.eos_token_id)
                    generate_texts = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                    for inp, gen in zip(inputs, generate_texts):
                        inp_len = len(inp)
                        o = {"input": inp, "generate": gen[inp_len:]}
                        o = json.dumps(o, ensure_ascii=False)
                        fw.write(o + "\n")
        fw.close()

        if dist.is_initialized():
            dist.barrier()

    else:
        raise Exception("You must specify whether to do train or do predict.")


if __name__ == "__main__":
    driver()
