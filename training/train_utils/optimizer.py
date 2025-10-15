# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import itertools
import re
from typing import Any, Dict, Iterable, List, Mapping, Set, Union

import hydra
import torch
import torch.nn as nn
from torch import Tensor
from omegaconf import OmegaConf

# -----------------------------------------------------------------------------
# Optimizer wrapper
# -----------------------------------------------------------------------------

class OptimizerWrapper:
    """Wraps a torch.optim.Optimizer and its schedulers (if any)."""

    def __init__(self, optimizer: torch.optim.Optimizer, schedulers=None) -> None:
        self.optimizer = optimizer
        self.schedulers = schedulers
        self._validate_optimizer_schedulers()
        self.step_schedulers(0.0)

    # ---------------------------------------------------------------------
    # Public API mirroring torch.optim.Optimizer
    # ---------------------------------------------------------------------

    def step(self, where: float = 1.0, closure=None):
        """Update the optimizer & its schedulers."""
        self.step_schedulers(where)
        return self.optimizer.step(closure)

    def zero_grad(self, *args, **kwargs):
        return self.optimizer.zero_grad(*args, **kwargs)

    def _validate_optimizer_schedulers(self):
        if self.schedulers is None:
            return
        for _, sched_map in enumerate(self.schedulers):
            for option, _ in sched_map.items():
                assert option in self.optimizer.defaults, (
                    f"Optimizer option {option} not found in {self.optimizer}. "
                    f"Valid options are {self.optimizer.defaults.keys()}"
                )

    def step_schedulers(self, where: float) -> None:
        if self.schedulers is None:
            return
        for i, param_group in enumerate(self.optimizer.param_groups):
            for option, scheduler in self.schedulers[i].items():
                param_group[option] = scheduler(where)


# -----------------------------------------------------------------------------
# Validation helpers
# -----------------------------------------------------------------------------


def validate_param_group_params(param_groups: List[Dict], model: nn.Module):
    """Ensure param groups are non-overlapping and include all model params."""

    for pg in param_groups:
        assert len(pg["params"]) == len(set(pg["params"]))

    parameters = [set(pg["params"]) for pg in param_groups]
    model_parameters = {p for _, p in model.named_parameters()}

    for p1, p2 in itertools.permutations(parameters, 2):
        assert p1.isdisjoint(p2), "Parameter groups should be disjoint"

    assert set.union(*parameters) == model_parameters, (
        "Parameter groups must cover ALL model parameters "
        f"(found {len(set.union(*parameters))} / {len(model_parameters)})"
    )


# -----------------------------------------------------------------------------
# Glob helpers for pattern matching
# -----------------------------------------------------------------------------

from wcmatch import fnmatch

GLOB_FLAGS = (
    fnmatch.CASE       # case-sensitive
    | fnmatch.DOTMATCH # '*' also matches '.'
    | fnmatch.EXTMATCH # extended patterns like *(foo|bar)
    | fnmatch.SPLIT    # "pat1|pat2" works out-of-the-box
)


def get_full_parameter_name(module_name: str, param_name: str) -> str:
    return param_name if module_name == "" else f"{module_name}.{param_name}"


def get_module_cls_to_param_names(model: nn.Module) -> Dict[type, Set[str]]:
    """Map each module class to the *immediate* param names it owns."""
    mapping: Dict[type, Set[str]] = {}
    for module_name, module in model.named_modules():
        module_cls = type(module)
        mapping.setdefault(module_cls, set())
        for pname, _ in module.named_parameters(recurse=False):
            mapping[module_cls].add(get_full_parameter_name(module_name, pname))
    return mapping


def unix_param_pattern_to_parameter_names(filter_param_names: Union[List[str], None],
                                           parameter_names: Set[str]) -> Set[str]:
    if filter_param_names is None:
        return set()
    allowed = []
    for pat in filter_param_names:
        matches = set(fnmatch.filter(parameter_names, pat, flags=GLOB_FLAGS))
        if not matches:
            raise AssertionError(f"Pattern {pat} matched no parameters")
        logging.info(f"Matches for param pattern [{pat}]: {matches}")
        allowed.append(matches)
    return set.union(*allowed)


def unix_module_cls_pattern_to_parameter_names(filter_module_cls_names: Union[List[str], None],
                                               module_cls_to_param_names: Dict[type, Set[str]]) -> Set[str]:
    if filter_module_cls_names is None:
        return set()
    allowed = []
    for cls_name in filter_module_cls_names:
        module_cls = hydra.utils.get_class(cls_name)
        if module_cls not in module_cls_to_param_names:
            raise AssertionError(f"Module class {cls_name} not found in model")
        params = module_cls_to_param_names[module_cls]
        if not params:
            raise AssertionError(f"Module class {cls_name} has no parameters")
        logging.info(f"Matches for module [{cls_name}]: {params}")
        allowed.append(params)
    return set.union(*allowed)


def _unix_pattern_to_parameter_names(scheduler_cfg,
                                     parameter_names: Set[str],
                                     module_cls_to_param_names: Dict[type, Set[str]]):
    if "param_names" not in scheduler_cfg and "module_cls_names" not in scheduler_cfg:
        return None
    return unix_param_pattern_to_parameter_names(
        scheduler_cfg.get("param_names"), parameter_names
    ).union(
        unix_module_cls_pattern_to_parameter_names(
            scheduler_cfg.get("module_cls_names"), module_cls_to_param_names
        )
    )


# -----------------------------------------------------------------------------
# Scheduler helpers
# -----------------------------------------------------------------------------


def set_default_parameters(scheduler_cfgs: List[dict], all_parameter_names: Set[str]):
    """Ensure exactly one scheduler per option acts as the default."""
    specified = [cfg["parameter_names"] for cfg in scheduler_cfgs if cfg["parameter_names"]]

    default_params = (
        all_parameter_names if not specified else all_parameter_names - set.union(*specified)
    )

    default_count = 0
    for cfg in scheduler_cfgs:
        if cfg["parameter_names"] is None:
            cfg["parameter_names"] = default_params
            default_count += 1
    assert default_count <= 1, "At most one default scheduler per option"

    if default_count == 0:
        scheduler_cfgs.append({"parameter_names": default_params})


def name_constraints_to_parameters(param_constraints: List[Set[str]],
                                   named_parameters: Dict[str, Tensor]) -> List[Tensor]:
    matching_names = set.intersection(*param_constraints)
    return [v for k, v in named_parameters.items() if k in matching_names]


def map_scheduler_cfgs_to_param_groups(all_scheduler_cfgs: Iterable[List[dict]],
                                       named_parameters: Dict[str, Tensor]):
    """Produce param groups & schedulers that torch.optim can consume."""
    schedulers: List[Dict[str, Any]] = []
    param_groups: List[Dict[str, List[Tensor]]] = []

    for cfgs in itertools.product(*all_scheduler_cfgs):
        param_constraints = [cfg["parameter_names"] for cfg in cfgs]
        matching = name_constraints_to_parameters(param_constraints, named_parameters)
        if not matching:
            continue  # no intersection of params for this combo
        schedulers.append({cfg["option"]: cfg["scheduler"] for cfg in cfgs if "option" in cfg})
        param_groups.append({"params": matching})

    return schedulers, param_groups


# -----------------------------------------------------------------------------
# Public factory functions
# -----------------------------------------------------------------------------


def _resolve_conf(conf):
    if conf is None:
        return None
    if OmegaConf.is_config(conf):
        return OmegaConf.to_container(conf, resolve=True)
    return conf


def _build_zero_weight_decay_param_groups(
    *,
    named_parameters: Dict[str, Tensor],
    optimizer_conf: Any,
    strategy_conf: Mapping[str, Any],
) -> Union[None, List[Dict[str, Any]]]:
    """Create param groups that apply zero weight decay to bias/norm/LoRA parameters."""
    strategy_conf = _resolve_conf(strategy_conf)
    if not strategy_conf:
        return None

    zero_wd_conf = strategy_conf.get("zero_weight_decay")
    if not zero_wd_conf:
        return None
    if isinstance(zero_wd_conf, bool):
        if not zero_wd_conf:
            return None
        zero_wd_conf = {}
    zero_wd_conf = dict(zero_wd_conf)

    apply_bias = bool(zero_wd_conf.get("bias", True))
    apply_norm = bool(zero_wd_conf.get("norm", True))
    apply_lora = bool(zero_wd_conf.get("lora", True))
    extra_patterns: List[str] = list(zero_wd_conf.get("patterns", []))
    norm_keywords: List[str] = [
        keyword.lower() for keyword in zero_wd_conf.get("norm_keywords", ["norm", "bn"])
    ]
    lora_keywords: List[str] = [
        keyword.lower() for keyword in zero_wd_conf.get("lora_keywords", ["lora"])
    ]

    optimizer_defaults = _resolve_conf(optimizer_conf) or {}
    base_lr = optimizer_defaults.get("lr")
    base_weight_decay = optimizer_defaults.get("weight_decay", 0.0)

    decay_params: List[Tensor] = []
    zero_decay_params: List[Tensor] = []

    for name, param in named_parameters.items():
        lower_name = name.lower()

        def _matches(patterns: List[str]) -> bool:
            return any(pattern in lower_name for pattern in patterns)

        def _extra_matches(pattern_list: List[str]) -> bool:
            return any(re.search(pattern, name) for pattern in pattern_list)

        is_bias = name.endswith(".bias")
        is_norm_like = param.ndim <= 1 or _matches(norm_keywords)
        is_lora = _matches(lora_keywords)

        if (
            (apply_lora and is_lora)
            or (apply_bias and is_bias)
            or (apply_norm and is_norm_like)
            or _extra_matches(extra_patterns)
        ):
            zero_decay_params.append(param)
        else:
            decay_params.append(param)

    if not decay_params and not zero_decay_params:
        return None

    param_groups: List[Dict[str, Any]] = []

    if decay_params:
        group = {"params": decay_params}
        if base_lr is not None:
            group["lr"] = base_lr
        group["weight_decay"] = base_weight_decay
        param_groups.append(group)

    if zero_decay_params:
        group = {"params": zero_decay_params, "weight_decay": 0.0}
        if base_lr is not None:
            group["lr"] = base_lr
        param_groups.append(group)

    logging.info(
        "Optimizer param grouping applied: %d decay params, %d zero-wd params.",
        len(decay_params),
        len(zero_decay_params),
    )
    return param_groups


def construct_optimizer(model: nn.Module,
                        optimizer_conf: Any,
                        options_conf: Union[Mapping[str, List], None] = None,
                        param_group_modifiers_conf: Union[List, None] = None,
                        param_groups_conf: Union[Mapping[str, Any], None] = None,
                        validate_param_groups: bool = True) -> OptimizerWrapper:
    """Build an OptimizerWrapper from hydra configs.

    *No* allowlist handling – we always optimize *all* model parameters.
    """

    named_parameters = dict(model.named_parameters())
    all_parameter_names = set(named_parameters.keys())
    module_cls_to_all_param_names = get_module_cls_to_param_names(model)

    # ──────────────────────────────────────────────────────────────────
    # No scheduler case – simple & fast
    # ──────────────────────────────────────────────────────────────────
    if not options_conf:
        param_groups = None
        if param_groups_conf:
            param_groups = _build_zero_weight_decay_param_groups(
                named_parameters=named_parameters,
                optimizer_conf=optimizer_conf,
                strategy_conf=param_groups_conf,
            )
            if param_groups and validate_param_groups:
                validate_param_group_params(param_groups, model)

        if param_groups:
            optimizer = hydra.utils.instantiate(optimizer_conf, param_groups)
        else:
            optimizer = hydra.utils.instantiate(optimizer_conf, named_parameters.values())
        return OptimizerWrapper(optimizer)

    # ──────────────────────────────────────────────────────────────────
    # Build option-specific scheduler configs
    # ──────────────────────────────────────────────────────────────────
    scheduler_cfgs_per_option = hydra.utils.instantiate(options_conf)
    all_scheduler_cfgs: List[List[dict]] = []

    for option, cfg_list in scheduler_cfgs_per_option.items():
        for cfg in cfg_list:
            cfg.option = option  # annotate
            cfg.parameter_names = _unix_pattern_to_parameter_names(
                cfg, all_parameter_names, module_cls_to_all_param_names
            )
        set_default_parameters(cfg_list, all_parameter_names)
        all_scheduler_cfgs.append(cfg_list)

    # User-provided modifiers (rare)
    if param_group_modifiers_conf:
        for modifier in param_group_modifiers_conf:
            modifier = hydra.utils.instantiate(modifier)
            all_scheduler_cfgs = modifier(scheduler_cfgs=all_scheduler_cfgs, model=model)

    # Map scheduler cfg combos to optimizer param groups
    schedulers, param_groups = map_scheduler_cfgs_to_param_groups(
        all_scheduler_cfgs, named_parameters
    )

    if validate_param_groups:
        validate_param_group_params(param_groups, model)

    optimizer = hydra.utils.instantiate(optimizer_conf, param_groups)
    return OptimizerWrapper(optimizer, schedulers)


def construct_optimizers(model: nn.Module, optim_conf) -> Union[List[OptimizerWrapper], None]:
    """Convenience wrapper producing a *single* OptimizerWrapper list."""
    if optim_conf is None:
        return None

    options_conf = getattr(optim_conf, "options", None)
    param_group_modifiers_conf = getattr(optim_conf, "param_group_modifiers", None)
    param_groups_conf = getattr(optim_conf, "param_groups", None)

    optimizer = construct_optimizer(
        model,
        optim_conf.optimizer,
        options_conf,
        param_group_modifiers_conf,
        param_groups_conf,
        validate_param_groups=True,
    )
    return [optimizer]
