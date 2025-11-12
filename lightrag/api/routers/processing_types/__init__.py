"""
Processing types module for handling different file formats.

This module provides a modular architecture for processing various file types.
Each file type has its own processor module that implements a common interface.
"""

from .base_processor import BaseFileProcessor, ProcessingResult
from .processor_registry import ProcessorRegistry, get_processor

__all__ = [
    "BaseFileProcessor",
    "ProcessingResult",
    "ProcessorRegistry",
    "get_processor",
]

