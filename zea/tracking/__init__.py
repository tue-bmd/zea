"""Tracking module.

This module provides point tracking algorithms for 2D and 3D data.

Classes:
    - BaseTracker: Abstract base class for trackers.
    - LucasKanadeTracker: Pyramidal Lucas-Kanade tracker (2D/3D).
    - BlockMatchingTracker: Exhaustive block matching tracker (2D/3D).
    - SegmentationTracker: Segmentation-based tracker using contour matching.

"""

from .base import BaseTracker
from .block_matching import BlockMatchingTracker
from .lucas_kanade import LucasKanadeTracker
from .segmentation import SegmentationTracker

__all__ = [
    "BaseTracker",
    "BlockMatchingTracker",
    "LucasKanadeTracker",
    "SegmentationTracker",
]
