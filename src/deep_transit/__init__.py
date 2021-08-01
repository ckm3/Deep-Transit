# -*- coding: utf-8 -*-

__author__ = "Kaiming Cui, Junjie Liu"
__license__ = "MIT"
__version__ = "0.1.0"

from . import config
from .dt_lightcurve import (
    DeepTransit,
    detrend_light_curve,
    smooth_light_curve,
    plot_lc_with_bboxes,
    select_lc_from_bboxes)
from .train import train
from ._utils import save_checkpoint_to_model

__all__ = [
    "config",
    "DeepTransit",
    "detrend_light_curve",
    "smooth_light_curve",
    "plot_lc_with_bboxes",
    "select_lc_from_bboxes",
    "train",
    "save_checkpoint_to_model"
]