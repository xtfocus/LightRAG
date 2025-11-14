"""
Processing types module for handling different file formats.

This module provides a modular architecture for processing various file types.
Each file type has its own processor module that implements a common interface.
"""

from .base_processor import BaseFileProcessor, ProcessingResult
from .processor_registry import ProcessorRegistry, get_processor, register_processor
from .processing_txt import TxtFileProcessor
from .processing_excel import ExcelFileProcessor
from .processing_pdf import PdfFileProcessor

# Register all processors
_processor_instances = [
    TxtFileProcessor(),
    ExcelFileProcessor()
    PdfFileProcessor(),
]

# Auto-register all processors
for processor in _processor_instances:
    register_processor(processor)

__all__ = [
    "BaseFileProcessor",
    "ProcessingResult",
    "ProcessorRegistry",
    "get_processor",
    "register_processor",
]
