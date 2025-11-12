"""
Plain text file processor.

This is a simple example processor for .txt files that can serve as a template
for team members creating their own file type processors.

To create a new processor:
1. Copy this file and rename it (e.g., processing_docx.py)
2. Update the class name and supported_extensions
3. Implement the process_file method with your file type's extraction logic
4. Register the processor in the __init__.py file
"""

from pathlib import Path
from .base_processor import BaseFileProcessor, ProcessingResult
from lightrag.utils import logger


class TxtFileProcessor(BaseFileProcessor):
    """Processor for plain text (.txt) files.
    
    This processor handles UTF-8 encoded text files. It validates the content
    and ensures it's not empty or binary data.
    """
    
    @property
    def supported_extensions(self) -> tuple[str, ...]:
        """Return supported file extensions for this processor.
        
        Returns:
            tuple[str, ...]: Tuple of supported extensions (e.g., (".txt",))
        """
        return (".txt",)
    
    async def process_file(
        self,
        file_path: Path,
        file_bytes: bytes,
        file_size: int,
        track_id: str,
    ) -> ProcessingResult:
        """Process a .txt file and extract its text content.
        
        This method:
        1. Decodes the file bytes as UTF-8
        2. Validates the content is not empty
        3. Checks for binary data representation
        4. Returns the extracted text content
        
        Args:
            file_path: Path to the file being processed
            file_bytes: Raw file content as bytes
            file_size: Size of the file in bytes (for error reporting)
            track_id: Tracking ID for this processing operation
            
        Returns:
            ProcessingResult: Result containing:
                - success: True if processing succeeded
                - content: Extracted text content (empty if failed)
                - error_description: Human-readable error message (if failed)
                - original_error: Original exception message (if failed)
        """
        try:
            # Step 1: Decode the file as UTF-8
            content = file_bytes.decode("utf-8")
            
            # Step 2: Validate content is not empty
            if not content or len(content.strip()) == 0:
                return ProcessingResult(
                    success=False,
                    content="",
                    error_description="[File Extraction]Empty file content",
                    original_error="File contains no content or only whitespace",
                )
            
            # Step 3: Check for binary data representation
            # Sometimes binary data gets incorrectly decoded as text
            if content.startswith("b'") or content.startswith('b"'):
                return ProcessingResult(
                    success=False,
                    content="",
                    error_description="[File Extraction]Binary data in text file",
                    original_error="File appears to contain binary data representation instead of text",
                )
            
            # Step 4: Success - return the content
            return ProcessingResult(
                success=True,
                content=content,
            )
            
        except UnicodeDecodeError as e:
            # Handle encoding errors
            return ProcessingResult(
                success=False,
                content="",
                error_description="[File Extraction]UTF-8 encoding error, please convert it to UTF-8 before processing",
                original_error=f"File is not valid UTF-8 encoded text: {str(e)}",
            )
            
        except Exception as e:
            # Handle any other unexpected errors
            logger.error(
                f"[File Extraction]Unexpected error processing TXT file {file_path.name}: {str(e)}"
            )
            return ProcessingResult(
                success=False,
                content="",
                error_description="[File Extraction]TXT file processing error",
                original_error=f"Unexpected error: {str(e)}",
            )

