"""
Processor registry for mapping file extensions to processors.

This module provides a centralized registry that maps file extensions
to their corresponding processor classes.
"""

from typing import Dict, Optional
from pathlib import Path
from .base_processor import BaseFileProcessor, ProcessingResult


class ProcessorRegistry:
    """Registry for file processors.
    
    This class maintains a mapping of file extensions to processor instances.
    """
    
    def __init__(self):
        self._processors: Dict[str, BaseFileProcessor] = {}
        self._extension_map: Dict[str, BaseFileProcessor] = {}
    
    def register(self, processor: BaseFileProcessor) -> None:
        """Register a processor for its supported extensions.
        
        Args:
            processor: Processor instance to register
        """
        for ext in processor.supported_extensions:
            ext_lower = ext.lower()
            if ext_lower in self._extension_map:
                # Allow override but log a warning
                import logging
                logger = logging.getLogger(__name__)
                logger.warning(
                    f"Processor for extension {ext_lower} is being overridden. "
                    f"Previous: {type(self._extension_map[ext_lower]).__name__}, "
                    f"New: {type(processor).__name__}"
                )
            self._extension_map[ext_lower] = processor
    
    def get_processor(self, extension: str) -> Optional[BaseFileProcessor]:
        """Get the processor for a given file extension.
        
        Args:
            extension: File extension (e.g., ".docx")
            
        Returns:
            BaseFileProcessor or None if no processor is registered for the extension
        """
        return self._extension_map.get(extension.lower())
    
    def has_processor(self, extension: str) -> bool:
        """Check if a processor exists for the given extension.
        
        Args:
            extension: File extension (e.g., ".docx")
            
        Returns:
            bool: True if a processor is registered for the extension
        """
        return extension.lower() in self._extension_map
    
    def get_all_extensions(self) -> list[str]:
        """Get all registered file extensions.
        
        Returns:
            list[str]: List of all registered extensions
        """
        return list(self._extension_map.keys())


# Global registry instance
_registry = ProcessorRegistry()


def get_registry() -> ProcessorRegistry:
    """Get the global processor registry.
    
    Returns:
        ProcessorRegistry: The global registry instance
    """
    return _registry


def get_processor(extension: str) -> Optional[BaseFileProcessor]:
    """Convenience function to get a processor for an extension.
    
    Args:
        extension: File extension (e.g., ".docx")
        
    Returns:
        BaseFileProcessor or None if no processor is registered
    """
    return _registry.get_processor(extension)


def register_processor(processor: BaseFileProcessor) -> None:
    """Convenience function to register a processor.
    
    Args:
        processor: Processor instance to register
    """
    _registry.register(processor)

