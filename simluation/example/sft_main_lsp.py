import copy
import logging
import math
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

import torch
from datasets import load_dataset
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import lspoffload
import lspoffload.optim
import transformers
from transformers.trainer_utils import check_target_module_exists


logger = logging.getLogger(__name__)


IGNORE_INDEX = -100
EOT_TOKEN = "<|EOT|>"


def build_instruction_prompt(instruction: str):
    return """
You are an AI programming assistant, utilizing the DeepSeek Coder model, developed by DeepSeek Company, and you only answer questions related to computer science. For politically sensitive questions, security and privacy issues, and other non-computer science questions, you will refuse to answer.
### Instruction:
{}
### Response:
""".format(instruction.strip()).lstrip()


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="deepseek-ai/deepseek-coder-6.7b-instruct")


@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})


@dataclass
class TrainingArguments:
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    train_batch_size: int = field(default=1, metadata={"help": "Batch size for training."})
    num_dataloader_workers: int = field(default=0, metadata={"help": "Number of workers for the data loader."})
    gradient_accumulation_steps: int = field(default=1, metadata={"help": "Number of gradient accumulation steps."})
    gradient_checkpointing: bool = field(default=False, metadata={"help": "Whether to use gradient checkpointing."})
    lr: float = field(default=5e-5, metadata={"help": "The initial learning rate for AdamW."})
    lr_min: float = field(default=0.0, metadata={"help": "The minimum learning rate for the scheduler."})
    weight_decay: float = field(default=0.01, metadata={"help": "Weight decay for AdamW."})
    beta1: float = field(default=0.9, metadata={"help": "Beta1 for AdamW."})
    beta2: float = field(default=0.999, metadata={"help": "Beta2 for AdamW."})
    epsilon: float = field(default=1e-8, metadata={"help": "Epsilon for AdamW."})
    max_epochs: int = field(default=3, metadata={"help": "Maximum number of epochs to train."})
    cuda: bool = field(default=True, metadata={"help": "Whether to use CUDA."})
    lr_scheduler_type: str = field(default="linear", metadata={"help": "Learning rate scheduler."})
    warmup_steps: int = field(default=0, metadata={"help": "Number of warmup steps."})
    cache_dir: str = field(default="./cache", metadata={"help": "Path to the cache directory."})
    log_interval: int = field(default=10, metadata={"help": "Log interval."})
    output_dir: str = field(default="./output", metadata={"help": "Path to the output directory."})
    max_grad_norm: float = field(default=1.0, metadata={"help": "Maximum gradient norm."})

    # Compressor
    optim_args: Optional[str] = field(default=None, metadata={"help": "Optimizer arguments."})
    optim_target_modules: Optional[str] = field(default=None, metadata={"help": "Optimizer target."})
    compressor_name: Optional[str] = field(default=None, metadata={"help": "Compressor name."})
    compressor_update_freq: int = field(default=1000, metadata={"help": "Compressor update frequency."})
    compressor_update_num_samples: int = field(default=128, metadata={"help": "Compressor update number of samples."})
    compressor_layerwise: bool = field(default=False, metadata={"help": "Compressor layerwise."})

    # Lora
    lora: bool = field(default=False, metadata={"help": "Whether to use Lora."})
    lora_args: Optional[str] = field(default=None, metadata={"help": "Lora arguments."})


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


def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    device = torch.device("cuda" if torch.cuda.is_available() and training_args.cuda else "cpu")

    print("=" * 100)
    print(training_args)
    print("=" * 100)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=True,
        trust_remote_code=True,
    )

    print("PAD Token:", tokenizer.pad_token, tokenizer.pad_token_id)
    print("BOS Token", tokenizer.bos_token, tokenizer.bos_token_id)
    print("EOS Token", tokenizer.eos_token, tokenizer.eos_token_id)

    print("Load tokenizer from {} over.".format(model_args.model_name_or_path))

    raw_train_datasets = load_dataset(
        data_args.data_path,
        split="train",
        cache_dir=training_args.cache_dir,
    )

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
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2"
    ).to(device)

    if training_args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    print("Load model from {} over.".format(model_args.model_name_or_path))

    # Lora
    if training_args.lora:
        if training_args.optim not in ["adamw_torch"]:
            raise ValueError("Lora only supports PyTorch AdamW optimizer.")
        from peft import LoraConfig, TaskType, get_peft_model

        # Parse lora_args
        if training_args.lora_args:
            lora_args = {}
            for mapping in training_args.lora_args.replace(" ", "").split(","):
                key, value = mapping.split("=")

                # auto convert to int or float
                try:
                    value = int(value)
                except ValueError:
                    try:
                        value = float(value)
                    except ValueError:
                        pass

                lora_args[key] = value
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=lora_args.get("rank", 8),
            lora_alpha=lora_args.get("alpha", 0.1),
            lora_dropout=lora_args.get("dropout", 0.1),
        )

        model = get_peft_model(model, peft_config)

        model.print_trainable_parameters()

    # Parse optim_args and optim_target_modules
    optim_args = {}
    if training_args.optim_args:
        for mapping in training_args.optim_args.replace(" ", "").split(","):
            key, value = mapping.split("=")

            # auto convert to int or float
            try:
                value = int(value)
            except ValueError:
                try:
                    value = float(value)
                except ValueError:
                    pass

            optim_args[key] = value

    if training_args.optim_target_modules:
        optim_target_modules = training_args.optim_target_modules.replace(" ", "").split(",")
        if len(optim_target_modules) == 1:
            optim_target_modules = optim_target_modules[0]
    else:
        optim_target_modules = None

    is_lspoffload = False

    # Init Optimizer
    if training_args.optim == "adamw_torch":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=training_args.lr,
            weight_decay=training_args.weight_decay,
            betas=(training_args.beta1, training_args.beta2),
            eps=training_args.epsilon,
        )
    elif training_args.optim == "adamw_lsp":
        # Orgnize the parameters

        if optim_target_modules is None:
            raise ValueError("optim_target_modules is required for LSP optimizer.")

        if training_args.compressor_name is None:
            raise ValueError("compressor_name is required for LSP optimizer.")

        optim_args["_compressor_name"] = training_args.compressor_name

        lspoffload_params = []
        lspoffload_params_names = []

        if not training_args.compressor_layerwise:
            for module_name, module in model.named_modules():
                target_module_exist, is_regex = check_target_module_exists(
                    optim_target_modules, module_name, return_is_regex=True
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
            for target_module_name in optim_target_modules:
                group = []
                for module_name, module in model.named_modules():
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
            param for name, param in model.named_parameters() if name not in lspoffload_params_names
        ]

        if not training_args.compressor_layerwise:
            param_groups = [
                {"params": non_lspoffload_params},
                {"params": lspoffload_params, "use_lsp": True},
            ]
        else:
            param_groups = [
                {"params": non_lspoffload_params},
                *[{"params": params, "use_lsp": True} for params in lspoffload_param_groups],
            ]

        optimizer = lspoffload.optim.LspAdamW(
            param_groups,
            lr=training_args.lr,
            weight_decay=training_args.weight_decay,
            betas=(training_args.beta1, training_args.beta2),
            eps=training_args.epsilon,
            lsp_args=optim_args,
        )

        is_lspoffload = True
    else:
        raise ValueError("Invalid optimizer: {}".format(training_args.optim))

    if is_lspoffload:
        # Prepare dev_dataloader
        if training_args.compressor_update_num_samples % training_args.train_batch_size != 0:
            raise ValueError("compress_update_num_samples should be divisible by train_batch_size.")
        ds = train_dataset.train_test_split(test_size=training_args.compressor_update_num_samples)
        train_dataset = ds["train"]
        dev_dataset = ds["test"]
        dev_dataloader = DataLoader(
            dev_dataset,
            batch_size=training_args.train_batch_size,
            shuffle=False,
            collate_fn=data_collator,
            num_workers=training_args.num_dataloader_workers,
        )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=training_args.train_batch_size,
        shuffle=True,
        collate_fn=data_collator,
        num_workers=training_args.num_dataloader_workers,
    )
    num_training_steps = (
        int(math.ceil(len(train_dataloader) / training_args.gradient_accumulation_steps)) * training_args.max_epochs
    )

    if training_args.lr_scheduler_type == "cosine_with_min_lr":
        scheduler_kwargs = {"min_lr": training_args.lr_min}
    else:
        scheduler_kwargs = {}

    lr_scheduler = transformers.get_scheduler(
        training_args.lr_scheduler_type,
        optimizer,
        num_warmup_steps=training_args.warmup_steps,
        num_training_steps=num_training_steps,
        scheduler_specific_kwargs=scheduler_kwargs,
    )

    progress_bar = tqdm(total=num_training_steps, desc="Training")
    writer = SummaryWriter(log_dir=f"{training_args.output_dir}/runs")

    num_steps = 0
    optimizer.zero_grad()

    tr_loss = torch.tensor(0.0).to(device)
    for epoch in range(training_args.max_epochs):
        model.train()
        for idx, batch in enumerate(train_dataloader):
            # The begining of a gradient accumulation step
            if idx % training_args.gradient_accumulation_steps == 0:
                if is_lspoffload and num_steps % training_args.compressor_update_freq == 0:
                    optimizer.zero_grad()
                    for dev_batch in tqdm(dev_dataloader, desc="Sampling gradients"):
                        inputs = prepare_inputs(dev_batch, device)
                        outputs = model(**inputs)
                        loss = outputs["loss"]
                        loss = loss / len(dev_dataloader)
                        loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), training_args.max_grad_norm)
                    optimizer.update_compressors()

            inputs = prepare_inputs(batch, device)
            outputs = model(**inputs)
            loss = outputs["loss"]
            loss = loss / training_args.gradient_accumulation_steps
            loss.backward()

            tr_loss += loss

            if (idx + 1) % training_args.gradient_accumulation_steps == 0 or (idx + 1) == len(train_dataloader):
                num_steps += 1

                # Gradient clipping
                grad_norm = nn.utils.clip_grad_norm_(model.parameters(), training_args.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()

                optimizer.zero_grad()

                progress_bar.update(1)
                if num_steps % training_args.log_interval == 0:
                    # Calculate the gradient norm
                    tr_loss_scalar = round(tr_loss.detach().item() / training_args.log_interval, 4)
                    current_lr = lr_scheduler.get_last_lr()[0]
                    writer.add_scalar("loss/train", tr_loss_scalar, num_steps)
                    writer.add_scalar("lr/train", current_lr, num_steps)
                    writer.add_scalar("grad_norm/train", grad_norm, num_steps)
                    progress_bar.set_postfix({"loss": tr_loss_scalar, "lr": current_lr})

                    tr_loss -= tr_loss
        print(f"Epoch {epoch + 1} finished.")

        # Save model
        if training_args.output_dir:
            current_output_dir = f"{training_args.output_dir}/epoch-{epoch}"
            model.save_pretrained(current_output_dir)
    progress_bar.close()


if __name__ == "__main__":
    train()
