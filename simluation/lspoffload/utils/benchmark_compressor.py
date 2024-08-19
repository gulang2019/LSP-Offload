import copy
import logging
from dataclasses import dataclass
from typing import Dict, Sequence

import torch
import torch.nn as nn
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import transformers
from lspoffload.optim import LspDummy
from transformers.trainer_utils import check_target_module_exists


logger = logging.getLogger(__name__)

# Utils for SFT
IGNORE_INDEX = -100
EOT_TOKEN = "<|EOT|>"


def build_instruction_prompt(instruction: str):
    return """
You are an AI programming assistant, utilizing the DeepSeek Coder model, developed by DeepSeek Company, and you only answer questions related to computer science. For politically sensitive questions, security and privacy issues, and other non-computer science questions, you will refuse to answer.
### Instruction:
{}
### Response:
""".format(instruction.strip()).lstrip()


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

    return {
        "input_ids": input_ids,
        "labels": labels,
        "input_ids_lens": input_ids_lens,
        "labels_lens": labels_lens,
    }


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
    return {"input_ids": input_ids, "labels": labels}


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = [torch.tensor(x) for x in input_ids]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = [torch.tensor(x) for x in labels]
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": input_ids.ne(self.tokenizer.pad_token_id),
        }


def train_tokenize_function(examples, tokenizer):
    sources = [build_instruction_prompt(instruction) for instruction in examples["instruction"]]
    targets = [f"{output}\n{EOT_TOKEN}" for output in examples["output"]]
    data_dict = preprocess(sources, targets, tokenizer)
    return data_dict


def prepare_inputs(inputs, device):
    return {key: value.to(device) for key, value in inputs.items()}


class BenchmarkContext:
    def __init__(
        self,
        model_path: str,
        data_path: str,
        optim_target_modules: str,
        tokenizer_path: str | None = None,
        model_max_length: int = 1024,
        num_update_samples: int = 128,
        num_test_samples: int = 128,
        batch_size: int = 8,
        device: str = "cuda",
    ):
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        self.data_path = data_path
        self.model_max_length = model_max_length
        self.optim_target_modules = optim_target_modules.split(",")

        # Load tokenizer, dataset and model
        self.device = torch.device(device)
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_path if tokenizer_path is None else tokenizer_path,
            model_max_length=model_max_length,
            padding_side="right",
            use_fast=True,
            trust_remote_code=True,
        )
        raw_train_datasets = load_dataset(data_path, split="train")

        train_dataset = raw_train_datasets.map(
            train_tokenize_function,
            batched=True,
            batch_size=3000,
            num_proc=32,
            remove_columns=raw_train_datasets.column_names,
            load_from_cache_file=True,  # not args.overwrite_cache
            desc="Running Encoding",
            fn_kwargs={"tokenizer": tokenizer},
        )
        self.data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)

        model = transformers.AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16).to(
            self.device
        )
        print(model)

        model.gradient_checkpointing_enable()
        self.model = model

        # Sample data
        ds = train_dataset.shuffle(seed=42).select(range(num_update_samples + num_test_samples))
        ds = ds.train_test_split(test_size=num_test_samples, seed=42)

        dev_ds = ds["train"]
        test_ds = ds["test"]

        dev_grads, test_grads = self._sample_gradients(dev_ds=dev_ds, test_ds=test_ds, batch_size=batch_size)

        self.dev_grads = dev_grads
        self.test_grads = test_grads

    def _init_optimizer(self, optim_args, layerwise=False):
        lspoffload_params = []
        lspoffload_params_names = []
        if not layerwise:
            for module_name, module in self.model.named_modules():
                target_module_exist, is_regex = check_target_module_exists(
                    self.optim_target_modules, module_name, return_is_regex=True
                )

                if not target_module_exist:
                    continue

                if not isinstance(module, nn.Linear):
                    if target_module_exist and not is_regex:
                        logger.warning(
                            f"{module_name} is not a linear layer, but it is included in the target modules. "
                        )
                    continue

                lspoffload_params.append(module.weight)
                lspoffload_params_names.append(module_name + ".weight")
        else:
            lspoffload_param_groups = []
            for target_module_name in self.optim_target_modules:
                group = []
                for module_name, module in self.model.named_modules():
                    target_module_exist, is_regex = check_target_module_exists(
                        [target_module_name], module_name, return_is_regex=True
                    )

                    if not target_module_exist:
                        continue

                    if not isinstance(module, nn.Linear):
                        if target_module_exist and not is_regex:
                            logger.warning(
                                f"{module_name} is not a linear layer, but it is included in the target modules. "
                            )
                        continue

                    if module in lspoffload_params:
                        logger.warning(f"{module_name} is already included in the lspoffload_params. ")

                    group.append(module.weight)
                    lspoffload_params.append(module.weight)
                    lspoffload_params_names.append(module_name + ".weight")
                lspoffload_param_groups.append(group)

        if len(lspoffload_params) == 0:
            raise ValueError("No target modules found.")

        non_lspoffload_params = [
            param for name, param in self.model.named_parameters() if name not in lspoffload_params_names
        ]

        if not layerwise:
            param_groups = [
                {"params": non_lspoffload_params},
                {"params": lspoffload_params, "use_lsp": True},
            ]
        else:
            param_groups = [
                {"params": non_lspoffload_params},
                *[{"params": params, "use_lsp": True} for params in lspoffload_param_groups],
            ]

        optimizer = LspDummy(param_groups, optim_args)

        return optimizer

    @staticmethod
    def _store_gradients(model):
        grads = {}
        for name, param in model.named_parameters():
            if param.grad is not None:
                grads[name] = param.grad.clone().cpu()
        return grads

    @staticmethod
    def _restore_gradients(model, grads):
        for name, param in model.named_parameters():
            if name in grads:
                param.grad = grads[name].clone().to(param.device)

    def _sample_gradients(self, dev_ds, test_ds, batch_size):
        dev_dataloder = DataLoader(
            dev_ds,
            collate_fn=self.data_collator,
            batch_size=batch_size,
            num_workers=4,
        )

        test_dataloader = DataLoader(
            test_ds,
            collate_fn=self.data_collator,
            batch_size=batch_size,
            num_workers=4,
        )

        self.model.zero_grad()
        for dev_batch in tqdm(dev_dataloder, desc="Sampling gradients for compressor update"):
            loss = self.model(**prepare_inputs(dev_batch, self.device)).loss
            loss = loss / len(dev_dataloder)
            loss.backward()

        nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        dev_grads = self._store_gradients(self.model)

        self.model.zero_grad()
        for test_batch in tqdm(test_dataloader, desc="Sampling gradients for test"):
            loss = self.model(**prepare_inputs(test_batch, self.device)).loss
            loss = loss / len(test_dataloader)
            loss.backward()

        nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        test_grads = self._store_gradients(self.model)

        torch.cuda.empty_cache()
        return dev_grads, test_grads

    def benchmark(self, optim_args, save_grad_dir=None, layerwise=False):
        optimizer = self._init_optimizer(optim_args, layerwise=layerwise)
        # initialize the compressor

        self.model.zero_grad()
        self._restore_gradients(self.model, self.dev_grads)
        optimizer.update_compressors()
        dev_loss = optimizer.step()

        self.model.zero_grad()
        self._restore_gradients(self.model, self.test_grads)
        nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        test_loss = optimizer.step(save_grad_dir=save_grad_dir)

        del optimizer

        return dev_loss, test_loss
