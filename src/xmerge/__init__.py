"""xmerge: Merge LLMs across architectures and sizes leveraging representation-level merging.

Includes weight blending for same-architecture models, zero-init bridge networks for
cross-architecture, and standalone merged models via weight translation + distillation.

CLI: xmerge merge --config config.json | eval | list | clean
"""

import logging

from . import merge_prod, merge_stream, utils
from .merge_prod import (
    DEVICE,
    SAVE_DIR,
    CkaComputer,
    MLPBridge,
    OptimalBridge,
    activation_similarity,
    build_bridge,
    clean,
    generate_bridge,
    load_merged,
    load_texts,
    merge_diff_arch,
    merge_diff_arch_standalone,
    merge_diff_arch_streamed,
    merge_same_arch,
    merge_same_arch_bridge,
    ppl,
    stitch_generate,
    train_bridge_cached,
    train_bridge_v2,
    verify_generations,
)
from .utils import (
    build_token_map,
    compute_ppl,
    generate_text,
    hidden_dim,
    proportional_map,
    resolve_device,
    svd_project,
    validate_model_pair,
)

__all__ = [
    # Core merge functions
    "merge_same_arch",
    "merge_same_arch_bridge",
    "train_bridge_v2",
    "train_bridge_cached",
    "merge_diff_arch",
    "merge_diff_arch_streamed",
    "merge_diff_arch_standalone",
    # Generation
    "stitch_generate",
    "generate_bridge",
    "verify_generations",
    "load_merged",
    # Bridge modules
    "OptimalBridge",
    "MLPBridge",
    "CkaComputer",
    "activation_similarity",
    "build_bridge",
    # Utilities
    "svd_project",
    "proportional_map",
    "compute_ppl",
    "generate_text",
    "build_token_map",
    "hidden_dim",
    "validate_model_pair",
    "resolve_device",
    "ppl",
    "clean",
    "load_texts",
    "merge_prod",
    "merge_stream",
    "utils",
    # Constants
    "DEVICE",
    "SAVE_DIR",
]

__version__ = "0.2.0"

# Configure default logger
logging.getLogger(__name__).addHandler(logging.NullHandler())
