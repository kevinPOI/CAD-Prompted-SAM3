# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

# pyre-unsafe

__version__ = "0.1.0"

__all__ = ["build_sam3_image_model", "build_sam3_predictor"]


def build_sam3_image_model(*args, **kwargs):
    from .model_builder import build_sam3_image_model as _build_sam3_image_model

    return _build_sam3_image_model(*args, **kwargs)


def build_sam3_predictor(*args, **kwargs):
    from .model_builder import build_sam3_predictor as _build_sam3_predictor

    return _build_sam3_predictor(*args, **kwargs)
