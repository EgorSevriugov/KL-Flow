#!/usr/bin/env python3
"""
Script to download datasets specified in config file, then tokenize and save
a pretokenized copy for faster training.

Usage:
    python download_dataset.py configs/config_tinystories_unconditional.yaml
    python download_dataset.py configs/config_conditional_alpaca.yaml --output_dir ./data/custom_location
"""

import argparse
import os
import sys
from pathlib import Path

from datasets import load_dataset, load_from_disk, DatasetDict, Dataset
from omegaconf import OmegaConf
from transformers import AutoTokenizer


# -----------------------------------------------------------------------------
# Config Loading and Merging
# -----------------------------------------------------------------------------

def load_config(experiment_config_path: str) -> OmegaConf:
    """
    Load and merge default config with experiment config.
    
    Args:
        experiment_config_path: Path to experiment-specific config file
    
    Returns:
        Merged configuration
    """
    # Get the directory containing this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_config_path = os.path.join(script_dir, "configs", "default_config.yaml")
    
    # Load default config
    if os.path.exists(default_config_path):
        default_config = OmegaConf.load(default_config_path)
        print(f"Loaded default config from: {default_config_path}")
    else:
        print(f"Warning: Default config not found at {default_config_path}")
        default_config = OmegaConf.create()
    
    # Load experiment config
    experiment_config = OmegaConf.load(experiment_config_path)
    print(f"Loaded experiment config from: {experiment_config_path}")
    
    # Merge configs (experiment config overrides defaults)
    config = OmegaConf.merge(default_config, experiment_config)
    print("Merged default and experiment configs")
    
    return config


def tokenize_dataset(dataset: DatasetDict, config) -> DatasetDict:
    """Tokenize all splits; add input_ids (and prompt_length for conditional)."""
    tokenizer = AutoTokenizer.from_pretrained(config.data.tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = "<|PAD|>"
    max_length = config.data.sequence_length
    condition = config.data.condition
    text_field = config.data.get("text_field", "text")
    prompt_field = config.data.get("prompt_field", "instruction")
    response_field = config.data.get("response_field", "output")

    def tokenize_unconditional(examples):
        out = tokenizer(
            examples[text_field],
            add_special_tokens=True,
            truncation=True,
            max_length=max_length,
            padding=False,
            return_tensors=None,
        )
        return {"input_ids": out["input_ids"]}

    def tokenize_conditional(examples):
        prompts = examples[prompt_field]
        responses = examples[response_field]
        p_enc = tokenizer(prompts, add_special_tokens=True, truncation=True, max_length=max_length // 2, padding=False, return_tensors=None)
        r_enc = tokenizer(responses, add_special_tokens=False, truncation=True, max_length=max_length // 2, padding=False, return_tensors=None)
        eos = tokenizer.eos_token_id
        input_ids_list = []
        prompt_lengths = []
        for p_ids, r_ids in zip(p_enc["input_ids"], r_enc["input_ids"]):
            combined = p_ids + r_ids + [eos]
            input_ids_list.append(combined[:max_length])
            prompt_lengths.append(len(p_ids))
        return {"input_ids": input_ids_list, "prompt_length": prompt_lengths}

    fn = tokenize_conditional if condition else tokenize_unconditional
    num_proc = max(1, (os.cpu_count() or 2) - 1)
    out = {}
    for split_name, ds in dataset.items():
        out[split_name] = ds.map(
            fn,
            batched=True,
            remove_columns=ds.column_names,
            num_proc=num_proc,
            desc=f"Tokenize {split_name}",
        )
    return DatasetDict(out)


def parse_args():
    parser = argparse.ArgumentParser(description="Download dataset specified in config file")
    
    parser.add_argument(
        "config_path",
        type=str,
        help="Path to config YAML file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory to save dataset (default: ./data/{dataset_name})"
    )
    parser.add_argument(
        "--skip_if_exists",
        action="store_true",
        help="Skip download if dataset already exists locally"
    )

    return parser.parse_args()


def download_dataset(config_path: str, output_dir: str = None, skip_if_exists: bool = False):
    """
    Download dataset, then tokenize and save pretokenized copy.
    
    Args:
        config_path: Path to config YAML file
        output_dir: Optional output directory (default: ./data/{dataset_name})
        skip_if_exists: Skip download if dataset already exists at output path
    """
    config = load_config(config_path)
    dataset_path = config.data.dataset_path
    print(f"\nDataset: {dataset_path}")

    dataset = None
    final_output_path = None
    if os.path.exists(dataset_path):
        print(f"Dataset path already exists locally: {dataset_path}")
        final_output_path = Path(dataset_path)
        print("Loading from disk for tokenization...")
        dataset = load_from_disk(dataset_path)
        if not isinstance(dataset, DatasetDict):
            dataset = DatasetDict({k: dataset[k] for k in dataset.keys()}) if hasattr(dataset, "keys") else DatasetDict({"train": dataset})

    dataset_name = dataset_path.split("/")[-1] if "/" in dataset_path else dataset_path
    if output_dir is None:
        output_dir = f"./data/{dataset_name}"
    output_path = Path(output_dir)

    if final_output_path is None and skip_if_exists and output_path.exists() and (output_path / "dataset_info.json").exists():
        print(f"\nDataset already exists at: {output_path}")
        print(f"Skipping download. Use without --skip_if_exists to re-download.")
        final_output_path = output_path
        print("Loading from disk for tokenization...")
        dataset = load_from_disk(str(output_path))
        if not isinstance(dataset, DatasetDict):
            dataset = DatasetDict({k: dataset[k] for k in dataset.keys()}) if hasattr(dataset, "keys") else DatasetDict({"train": dataset})

    if final_output_path is None:
        # Download dataset (not already local)
        print(f"\nDownloading dataset from HuggingFace Hub...")
        print(f"This may take a while depending on dataset size...\n")
        try:
            dataset_config = config.data.get("dataset_config", None)
            if dataset_config:
                print(f"Loading with config: {dataset_config}")
                dataset = load_dataset(dataset_path, dataset_config)
            else:
                dataset = load_dataset(dataset_path)
            print(f"Dataset loaded successfully!")
            print(f"\nDataset splits: {list(dataset.keys())}")
            for split_name, split_data in dataset.items():
                print(f"  {split_name}: {len(split_data)} examples")
                if len(split_data) > 0:
                    print(f"    Columns: {split_data.column_names}")
            if config.data.get("create_val_split", False):
                if "validation" not in dataset.keys() and "train" in dataset.keys():
                    print(f"\nCreating validation split from train...")
                    val_size = config.data.get("val_split_percentage", 0.05)
                    train_val = dataset["train"].train_test_split(test_size=val_size, seed=42)
                    dataset = DatasetDict({"train": train_val["train"], "validation": train_val["test"]})
                    print(f"Created validation split: {len(dataset['validation'])} examples")
            print(f"\nSaving dataset to: {output_path}")
            output_path.mkdir(parents=True, exist_ok=True)
            dataset.save_to_disk(output_path)
            final_output_path = output_path
            print(f"\n✓ Dataset downloaded and saved successfully!")
            print(f"  Location: {output_path.absolute()}")
        except Exception as e:
            print(f"\nError downloading dataset: {e}")
            print(f"  - Dataset name incorrect, auth (huggingface-cli login), or network")
            sys.exit(1)

    # Always tokenize and save pretokenized
    if dataset is not None and final_output_path is not None:
        tokenized_path = config.data.get("pretokenized_path") or str(final_output_path) + ".tokenized"
        tokenized_path = Path(tokenized_path)
        print(f"\nTokenizing dataset (tokenizer: {config.data.tokenizer_name})...")
        if not isinstance(dataset, DatasetDict):
            dataset = load_from_disk(str(final_output_path))
            dataset = DatasetDict({k: dataset[k] for k in dataset.keys()}) if hasattr(dataset, "keys") else DatasetDict({"train": dataset})
        tokenized = tokenize_dataset(dataset, config)
        tokenized_path.parent.mkdir(parents=True, exist_ok=True)
        tokenized.save_to_disk(tokenized_path)
        print(f"✓ Pretokenized dataset saved to: {tokenized_path}")
    else:
        print("No dataset loaded; cannot tokenize.")

    if final_output_path is not None:
        print(f"\n" + "="*60)
        print("To use this dataset in training, update your config file:")
        print("="*60)
        print(f"data:")
        print(f"    dataset_path: \"{final_output_path.absolute()}\"")
        print(f"    tokenizer_name: \"{config.data.tokenizer_name}\"")
        tok_path = config.data.get("pretokenized_path") or str(final_output_path) + ".tokenized"
        print(f"    pretokenized_path: \"{Path(tok_path).absolute()}\"")
        print("="*60)
        return str(final_output_path.absolute())


def main():
    args = parse_args()
    download_dataset(args.config_path, args.output_dir, args.skip_if_exists)


if __name__ == "__main__":
    main()
