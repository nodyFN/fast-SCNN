"""Fast-SCNN model package."""

from models.fast_scnn import FastSCNN, FastSCNNMattingAdapter, count_parameters
from models.fast_scnn_salient import FastSCNNSalient

__all__ = ["FastSCNN", "FastSCNNSalient", "FastSCNNMattingAdapter", "count_parameters"]
