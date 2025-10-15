"""
HEP Factory
GPU-accelerated ROOT → ML pipeline utilities
"""

from .pickle_factory import PickleFactory
from .data_utils import load_root_to_cudf, sample_cudf, attach_cross_section
from .train_utils import train_xgboost_gpu
from .inference_utils import run_inference_gpu
from .significance import calculate_significance

__all__ = [
    "PickleFactory",
    "load_root_to_cudf",
    "sample_cudf",
    "attach_cross_section",
    "train_xgboost_gpu",
    "run_inference_gpu",
    "calculate_significance",
]
