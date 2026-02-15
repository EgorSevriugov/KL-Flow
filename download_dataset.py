#!/usr/bin/env python3
"""
Script to download datasets specified in config file.

Usage:
    python download_dataset.py configs/config_tinystories_unconditional.yaml
    python download_dataset.py configs/config_conditional_alpaca.yaml --output_dir ./data/custom_location
"""

import argparse
import os
import sys
from pathlib import Path

from datasets import load_dataset, DatasetDict
from omegaconf import OmegaConf


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
    Download dataset specified in config file.
    
    Args:
        config_path: Path to config YAML file
        output_dir: Optional output directory (default: ./data/{dataset_name})
        skip_if_exists: Skip if dataset already exists
    """
    # Load and merge config
    config = load_config(config_path)
    
    # Extract dataset information
    dataset_path = config.data.dataset_path
    print(f"\nDataset: {dataset_path}")
    
    # Determine if this is a HuggingFace dataset or local path
    if os.path.exists(dataset_path):
        print(f"Dataset path already exists locally: {dataset_path}")
        return dataset_path
    
    # Parse dataset name for output directory
    dataset_name = dataset_path.split("/")[-1] if "/" in dataset_path else dataset_path
    
    # Set output directory
    if output_dir is None:
        output_dir = f"./data/{dataset_name}"
    
    output_path = Path(output_dir)
    
    # Check if dataset already exists
    if skip_if_exists and output_path.exists() and (output_path / "dataset_info.json").exists():
        print(f"\nDataset already exists at: {output_path}")
        print(f"Skipping download. Use without --skip_if_exists to re-download.")
        print(f"\nUpdate your config to use local dataset:")
        print(f"  dataset_path: \"{output_path.absolute()}\"")
        return str(output_path.absolute())
    
    # Download dataset
    print(f"\nDownloading dataset from HuggingFace Hub...")
    print(f"This may take a while depending on dataset size...\n")
    
    try:
        # Check if dataset has a configuration
        dataset_config = config.data.get("dataset_config", None)
        
        if dataset_config:
            print(f"Loading with config: {dataset_config}")
            dataset = load_dataset(dataset_path, dataset_config)
        else:
            dataset = load_dataset(dataset_path)
            
        print(f"Dataset loaded successfully!")
        
        # Print dataset info
        print(f"\nDataset splits: {list(dataset.keys())}")
        for split_name, split_data in dataset.items():
            print(f"  {split_name}: {len(split_data)} examples")
            if len(split_data) > 0:
                print(f"    Columns: {split_data.column_names}")
        
        # Create validation split if needed
        if config.data.get("create_val_split", False):
            if "validation" not in dataset.keys() and "train" in dataset.keys():
                print(f"\nCreating validation split from train...")
                val_size = config.data.get("val_split_percentage", 0.05)
                train_val = dataset["train"].train_test_split(test_size=val_size, seed=42)
                dataset = DatasetDict({
                    "train": train_val["train"],
                    "validation": train_val["test"],
                })
                print(f"Created validation split: {len(dataset['validation'])} examples")
        
        # Save to disk
        print(f"\nSaving dataset to: {output_path}")
        output_path.mkdir(parents=True, exist_ok=True)
        dataset.save_to_disk(output_path)
        
        print(f"\n✓ Dataset downloaded and saved successfully!")
        print(f"  Location: {output_path.absolute()}")
        
        # Print usage instructions
        print(f"\n" + "="*60)
        print("To use this dataset in training, update your config file:")
        print("="*60)
        print(f"data:")
        print(f"    dataset_path: \"{output_path.absolute()}\"")
        print(f"    tokenizer_name: \"{config.data.tokenizer_name}\"")
        print("="*60)
        
        return str(output_path.absolute())
        
    except Exception as e:
        print(f"\nError downloading dataset: {e}")
        print(f"\nPossible reasons:")
        print(f"  - Dataset name is incorrect")
        print(f"  - Dataset requires authentication (run: huggingface-cli login)")
        print(f"  - Network connection issues")
        print(f"  - Dataset has been moved or deleted")
        sys.exit(1)


def main():
    args = parse_args()
    download_dataset(args.config_path, args.output_dir, args.skip_if_exists)


if __name__ == "__main__":
    main()
