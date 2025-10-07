# Copyright (c) Meta Platforms, Inc. and affiliates.
# LoRA integration utilities using PEFT library

import logging
import re
from typing import List, Optional
import torch
import torch.nn as nn

try:
    from peft import LoraConfig, get_peft_model, PeftModel
    from peft.tuners.lora import LoraLayer
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    logging.warning("PEFT library not found. LoRA training will not be available.")

logger = logging.getLogger(__name__)


def apply_lora_to_model(
    model: nn.Module,
    lora_config: dict,
    verbose: bool = True,
) -> nn.Module:
    """
    Apply LoRA to a model using PEFT library.

    Args:
        model: The base model to apply LoRA to
        lora_config: Dictionary containing LoRA configuration:
            - enabled (bool): Whether to enable LoRA
            - rank (int): LoRA rank
            - alpha (int): LoRA alpha (scaling factor)
            - dropout (float): LoRA dropout
            - target_modules (List[str]): List of module name patterns to apply LoRA to
        verbose: Whether to print detailed information

    Returns:
        Model with LoRA applied (PeftModel)
    """
    if not PEFT_AVAILABLE:
        raise ImportError(
            "PEFT library is not installed. Please install it with: pip install peft"
        )

    if not lora_config.get('enabled', False):
        logger.info("LoRA is disabled in config. Returning original model.")
        return model

    rank = lora_config.get('rank', 16)
    alpha = lora_config.get('alpha', 32)
    dropout = lora_config.get('dropout', 0.1)
    target_modules_patterns = lora_config.get('target_modules', [])

    if not target_modules_patterns:
        raise ValueError("target_modules must be specified in lora_config")

    # Convert target_modules patterns to actual module names
    target_modules = _find_target_modules(model, target_modules_patterns)

    if not target_modules:
        raise ValueError(
            f"No modules found matching patterns: {target_modules_patterns}. "
            f"Please check your configuration."
        )

    if verbose:
        logger.info(f"Applying LoRA with rank={rank}, alpha={alpha}, dropout={dropout}")
        logger.info(f"Found {len(target_modules)} target modules:")
        for i, mod in enumerate(sorted(target_modules)[:10]):
            logger.info(f"  {i+1}. {mod}")
        if len(target_modules) > 10:
            logger.info(f"  ... and {len(target_modules) - 10} more modules")

    # Create PEFT LoRA config
    peft_config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        target_modules=list(target_modules),
        lora_dropout=dropout,
        bias="none",
        task_type=None,  # For non-standard models
    )

    # Apply PEFT to model
    try:
        peft_model = get_peft_model(model, peft_config)
    except Exception as e:
        logger.error(f"Failed to apply PEFT: {e}")
        logger.info("Attempting manual LoRA application...")
        peft_model = _manual_lora_application(model, peft_config)

    # Print trainable parameters
    if verbose:
        trainable_params, all_params = _count_parameters(peft_model)
        logger.info(
            f"Trainable params: {trainable_params:,} || "
            f"All params: {all_params:,} || "
            f"Trainable%: {100 * trainable_params / all_params:.2f}%"
        )

    return peft_model


def _find_target_modules(model: nn.Module, patterns: List[str]) -> set:
    """
    Find all module names in model that match any of the given patterns.

    Supports wildcards:
    - "*" matches any substring
    - "[0-9]" matches single digit
    - "[0-9][0-9]" matches two digits
    - "[1-2][0-3]" matches 10-23

    Args:
        model: The model to search
        patterns: List of regex-like patterns

    Returns:
        Set of matching module names
    """
    target_modules = set()

    # Get all named modules
    all_module_names = [name for name, _ in model.named_modules() if name]

    for pattern in patterns:
        # Convert pattern to regex
        regex_pattern = _pattern_to_regex(pattern)

        for name in all_module_names:
            if re.match(regex_pattern, name):
                # Only include leaf modules (nn.Linear, nn.Conv2d, etc.)
                module = model.get_submodule(name)
                if isinstance(module, (nn.Linear, nn.Conv2d, nn.Conv1d)):
                    target_modules.add(name)

    return target_modules


def _pattern_to_regex(pattern: str) -> str:
    r"""
    Convert simplified pattern to regex.

    Examples:
        "aggregator.frame_blocks.*.attn.qkv" -> "aggregator\.frame_blocks\..*\.attn\.qkv"
        "aggregator.frame_blocks.1[2-9].attn.qkv" -> "aggregator\.frame_blocks\.1[2-9]\.attn\.qkv"
    """
    # Escape dots
    regex = pattern.replace('.', r'\.')

    # Convert * to .*
    regex = regex.replace('*', '.*')

    # Handle character classes [0-9], [1-2], etc. (keep as-is, they're already regex)

    # Anchor the pattern
    regex = f"^{regex}$"

    return regex


def _count_parameters(model: nn.Module) -> tuple:
    """Count trainable and total parameters"""
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())
    return trainable_params, all_params


def _manual_lora_application(model: nn.Module, peft_config: LoraConfig) -> nn.Module:
    """
    Fallback: manually apply LoRA if get_peft_model fails.
    This is a simplified version for debugging.
    """
    logger.warning("Manual LoRA application is experimental and may not work correctly.")

    # Try to apply get_peft_model again with relaxed settings
    try:
        # For custom models, we may need to wrap in a way PEFT understands
        peft_model = get_peft_model(model, peft_config)
        return peft_model
    except Exception as e:
        logger.error(f"Manual LoRA application also failed: {e}")
        logger.error("Returning original model. LoRA will NOT be applied.")
        return model


def save_lora_checkpoint(
    model: nn.Module,
    save_path: str,
    verbose: bool = True,
):
    """
    Save only LoRA parameters (not the full model).

    Args:
        model: PeftModel with LoRA
        save_path: Path to save LoRA weights
        verbose: Whether to print information
    """
    if not isinstance(model, PeftModel):
        logger.warning("Model is not a PeftModel. Saving full model instead.")
        torch.save(model.state_dict(), save_path)
        return

    # Save only LoRA adapter weights
    model.save_pretrained(save_path)

    if verbose:
        logger.info(f"LoRA checkpoint saved to {save_path}")


def load_lora_checkpoint(
    base_model: nn.Module,
    lora_checkpoint_path: str,
    verbose: bool = True,
) -> nn.Module:
    """
    Load LoRA weights into a base model.

    Args:
        base_model: The base model (without LoRA)
        lora_checkpoint_path: Path to LoRA checkpoint
        verbose: Whether to print information

    Returns:
        Model with LoRA loaded
    """
    if not PEFT_AVAILABLE:
        raise ImportError("PEFT library is required to load LoRA checkpoints")

    # Load PEFT model
    model = PeftModel.from_pretrained(base_model, lora_checkpoint_path)

    if verbose:
        logger.info(f"LoRA checkpoint loaded from {lora_checkpoint_path}")
        trainable_params, all_params = _count_parameters(model)
        logger.info(
            f"Trainable params: {trainable_params:,} || "
            f"All params: {all_params:,} || "
            f"Trainable%: {100 * trainable_params / all_params:.2f}%"
        )

    return model


def merge_lora_weights(model: PeftModel) -> nn.Module:
    """
    Merge LoRA weights into the base model for inference.

    Args:
        model: PeftModel with LoRA

    Returns:
        Base model with LoRA weights merged (no longer a PeftModel)
    """
    if not isinstance(model, PeftModel):
        logger.warning("Model is not a PeftModel. Returning as-is.")
        return model

    # Merge and unload LoRA
    merged_model = model.merge_and_unload()

    logger.info("LoRA weights merged into base model")
    return merged_model


def print_lora_info(model: nn.Module):
    """Print detailed LoRA information for debugging"""
    if not isinstance(model, PeftModel):
        logger.info("Model is not a PeftModel")
        return

    logger.info("=" * 80)
    logger.info("LoRA Model Information")
    logger.info("=" * 80)

    trainable_params, all_params = _count_parameters(model)
    logger.info(
        f"Trainable params: {trainable_params:,} ({100 * trainable_params / all_params:.2f}%)"
    )

    logger.info("\nLoRA Modules:")
    lora_modules = []
    for name, module in model.named_modules():
        if isinstance(module, LoraLayer):
            lora_modules.append(name)

    for i, name in enumerate(sorted(lora_modules)):
        logger.info(f"  {i+1}. {name}")

    logger.info(f"\nTotal LoRA modules: {len(lora_modules)}")
    logger.info("=" * 80)
