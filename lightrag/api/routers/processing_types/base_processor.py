"""
Base processor interface for file type processing.

All file processors should inherit from BaseFileProcessor and implement
the process_file method.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional
from dataclasses import dataclass


@dataclass
class ProcessingResult:
    """Result of file processing operation."""
    success: bool
    content: str
    error_description: Optional[str] = None
    original_error: Optional[str] = None


class BaseFileProcessor(ABC):
    """Base class for all file processors.
    
    Each file type processor should inherit from this class and implement
    the process_file method.
    """
    
    @property
    @abstractmethod
    def supported_extensions(self) -> tuple[str, ...]:
        """Return a tuple of file extensions this processor supports.
        
        Returns:
            tuple[str, ...]: Supported file extensions (e.g., (".docx",))
        """
        pass
    
    @abstractmethod
    async def process_file(
        self,
        file_path: Path,
        file_bytes: bytes,
        file_size: int,
        track_id: str,
    ) -> ProcessingResult:
        """Process a file and extract its text content.
        
        Args:
            file_path: Path to the file being processed
            file_bytes: Raw file content as bytes
            file_size: Size of the file in bytes
            track_id: Tracking ID for this processing operation
            
        Returns:
            ProcessingResult: Result containing success status, content, and any errors
            
        Raises:
            Exception: If processing fails unexpectedly
        """
        pass
    
    def can_process(self, extension: str) -> bool:
        """Check if this processor can handle the given file extension.
        
        Args:
            extension: File extension (e.g., ".docx")
            
        Returns:
            bool: True if this processor can handle the extension
        """
        return extension.lower() in [ext.lower() for ext in self.supported_extensions]

