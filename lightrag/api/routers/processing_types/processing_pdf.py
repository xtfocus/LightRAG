"""
PDF file processor.

Handles PDF files using either DOCLING or pypdf, with support for encrypted PDFs
that require password decryption.
"""

from pathlib import Path
from io import BytesIO
from .base_processor import BaseFileProcessor, ProcessingResult
from lightrag.utils import logger
import pipmaster as pm
from ..config import global_args


class PdfFileProcessor(BaseFileProcessor):
    """Processor for PDF files.
    
    This processor handles PDF files using either DOCLING or pypdf as a fallback.
    It supports encrypted PDFs with password decryption when PDF_DECRYPT_PASSWORD
    is configured in the environment.
    """
    
    @property
    def supported_extensions(self) -> tuple[str, ...]:
        """Return supported file extensions for this processor."""
        return (".pdf",)
    
    async def process_file(
        self,
        file_path: Path,
        file_bytes: bytes,
        file_size: int,
        track_id: str,
    ) -> ProcessingResult:
        """Process a PDF file and extract its text content.
        
        This method:
        1. Uses DOCLING if document_loading_engine is set to "DOCLING"
        2. Falls back to pypdf for text extraction
        3. Handles encrypted PDFs with password decryption
        4. Extracts text from all pages
        5. Returns the extracted text content
        
        Args:
            file_path: Path to the file being processed
            file_bytes: Raw file content as bytes
            file_size: Size of the file in bytes
            track_id: Tracking ID for this processing operation
            
        Returns:
            ProcessingResult: Result containing success status and content
        """
        try:
            content = ""
            
            if global_args.document_loading_engine == "DOCLING":
                # Use DOCLING for processing
                if not pm.is_installed("docling"):  # type: ignore
                    pm.install("docling")
                from docling.document_converter import DocumentConverter  # type: ignore
                
                converter = DocumentConverter()
                result = converter.convert(file_path)
                content = result.document.export_to_markdown()
            else:
                # Use pypdf as fallback
                if not pm.is_installed("pypdf"):  # type: ignore
                    pm.install("pypdf")
                if not pm.is_installed("pycryptodome"):  # type: ignore
                    pm.install("pycryptodome")
                from pypdf import PdfReader  # type: ignore
                
                pdf_file = BytesIO(file_bytes)
                reader = PdfReader(pdf_file)
                
                # Check if PDF is encrypted
                if reader.is_encrypted:
                    pdf_password = global_args.pdf_decrypt_password
                    if not pdf_password:
                        # PDF is encrypted but no password provided
                        return ProcessingResult(
                            success=False,
                            content="",
                            error_description="[File Extraction]PDF is encrypted but no password provided",
                            original_error="Please set PDF_DECRYPT_PASSWORD environment variable to decrypt this PDF file",
                        )
                    
                    # Try to decrypt with password
                    try:
                        decrypt_result = reader.decrypt(pdf_password)
                        if decrypt_result == 0:
                            # Password is incorrect
                            return ProcessingResult(
                                success=False,
                                content="",
                                error_description="[File Extraction]Failed to decrypt PDF - incorrect password",
                                original_error="The provided PDF_DECRYPT_PASSWORD is incorrect for this file",
                            )
                    except Exception as decrypt_error:
                        # Decryption process error
                        return ProcessingResult(
                            success=False,
                            content="",
                            error_description="[File Extraction]PDF decryption failed",
                            original_error=f"Error during PDF decryption: {str(decrypt_error)}",
                        )
                
                # Extract text from PDF (encrypted PDFs are now decrypted, unencrypted PDFs proceed directly)
                for page in reader.pages:
                    content += page.extract_text() + "\n"
            
            # Validate content
            if not content or len(content.strip()) == 0:
                return ProcessingResult(
                    success=False,
                    content="",
                    error_description="[File Extraction]Empty file content",
                    original_error="File contains no extractable text content",
                )
            
            return ProcessingResult(
                success=True,
                content=content,
            )
            
        except Exception as e:
            logger.error(
                f"[File Extraction]Error processing PDF file {file_path.name}: {str(e)}"
            )
            return ProcessingResult(
                success=False,
                content="",
                error_description="[File Extraction]PDF processing error",
                original_error=f"Failed to extract text from PDF: {str(e)}",
            )

