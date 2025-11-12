# Development Guide: Creating File Type Processors

This guide will help you create a new file processor for a specific file type (e.g., `.docx`, `.xlsx`, `.pdf`, etc.) that integrates with the LightRAG document processing system.

## Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Step-by-Step Instructions](#step-by-step-instructions)
4. [Interface Requirements](#interface-requirements)
5. [Code Template](#code-template)
6. [Common Patterns](#common-patterns)
7. [Error Handling](#error-handling)
8. [Registration](#registration)
9. [Testing](#testing)
10. [Best Practices](#best-practices)

## Overview

The file processing system uses a modular architecture where each file type has its own processor class. All processors:

- Inherit from `BaseFileProcessor`
- Implement a `process_file()` method that extracts text content
- Return a `ProcessingResult` object indicating success/failure
- Are automatically registered in the system

**Key Files:**
- `base_processor.py` - Defines the interface all processors must implement
- `processor_registry.py` - Manages registration and lookup of processors
- `processing_txt.py` - Example implementation (use as a template)
- `__init__.py` - Registers all processors on module import

## Quick Start

1. Copy `processing_txt.py` and rename it to `processing_[yourfiletype].py`
2. Update the class name (e.g., `DocxFileProcessor`)
3. Update `supported_extensions` to return your file extensions
4. Implement `process_file()` with your extraction logic
5. Register your processor in `__init__.py`
6. Test it!

## Step-by-Step Instructions

### Step 1: Create Your Processor File

Create a new file: `processing_[filetype].py` (e.g., `processing_docx.py`)

```python
# processing_docx.py
from pathlib import Path
from .base_processor import BaseFileProcessor, ProcessingResult
from lightrag.utils import logger

class DocxFileProcessor(BaseFileProcessor):
    # Implementation goes here
    pass
```

### Step 2: Define Supported Extensions

Implement the `supported_extensions` property:

```python
@property
def supported_extensions(self) -> tuple[str, ...]:
    """Return supported file extensions for this processor."""
    return (".docx",)  # Can support multiple: (".docx", ".doc")
```

### Step 3: Implement process_file()

This is the main method where you extract text from the file:

```python
async def process_file(
    self,
    file_path: Path,
    file_bytes: bytes,
    file_size: int,
    track_id: str,
) -> ProcessingResult:
    # Your extraction logic here
    pass
```

### Step 4: Register Your Processor

Add your processor to `__init__.py`:

```python
# In __init__.py
from .processing_docx import DocxFileProcessor

_processor_instances = [
    TxtFileProcessor(),
    DocxFileProcessor(),  # Add your processor here
]
```

That's it! The processor will be automatically registered when the module is imported.

## Interface Requirements

### BaseFileProcessor Interface

Your processor **must** implement these methods:

#### 1. `supported_extensions` (property)

```python
@property
def supported_extensions(self) -> tuple[str, ...]:
    """Return a tuple of file extensions this processor supports."""
    return (".docx",)
```

- Must return a tuple of strings
- Extensions should include the dot (e.g., `".docx"` not `"docx"`)
- Case-insensitive matching is handled automatically

#### 2. `process_file()` (async method)

```python
async def process_file(
    self,
    file_path: Path,      # Path object to the file
    file_bytes: bytes,    # Raw file content as bytes
    file_size: int,       # File size in bytes (for error reporting)
    track_id: str,        # Tracking ID for this operation
) -> ProcessingResult:
    """Process the file and extract text content."""
    pass
```

**Parameters:**
- `file_path`: `Path` object - Use this if you need to access the file on disk
- `file_bytes`: `bytes` - The raw file content (usually what you'll use)
- `file_size`: `int` - File size for error reporting
- `track_id`: `str` - Tracking ID (for logging/debugging)

**Returns:**
- `ProcessingResult` - See below

### ProcessingResult

Always return a `ProcessingResult` object:

```python
from .base_processor import ProcessingResult

# Success case
return ProcessingResult(
    success=True,
    content="extracted text content here",
)

# Error case
return ProcessingResult(
    success=False,
    content="",
    error_description="[File Extraction]Human-readable error message",
    original_error="Original exception message or details",
)
```

**Fields:**
- `success`: `bool` - `True` if processing succeeded
- `content`: `str` - Extracted text content (empty string if failed)
- `error_description`: `Optional[str]` - Human-readable error message (prefixed with `[File Extraction]`)
- `original_error`: `Optional[str]` - Original exception message or technical details

## Code Template

Here's a complete template you can copy and modify:

```python
"""
[File Type] file processor.

Description of what this processor does.
"""

from pathlib import Path
from io import BytesIO
from .base_processor import BaseFileProcessor, ProcessingResult
from lightrag.utils import logger
import pipmaster as pm
from ..config import global_args


class [YourFileType]FileProcessor(BaseFileProcessor):
    """Processor for [file type] files.
    
    Brief description of what this processor handles.
    """
    
    @property
    def supported_extensions(self) -> tuple[str, ...]:
        """Return supported file extensions for this processor."""
        return (".your_ext",)
    
    async def process_file(
        self,
        file_path: Path,
        file_bytes: bytes,
        file_size: int,
        track_id: str,
    ) -> ProcessingResult:
        """Process a [file type] file and extract its text content.
        
        Args:
            file_path: Path to the file being processed
            file_bytes: Raw file content as bytes
            file_size: Size of the file in bytes
            track_id: Tracking ID for this processing operation
            
        Returns:
            ProcessingResult: Result containing success status and content
        """
        try:
            # TODO: Implement your extraction logic here
            # Example steps:
            # 1. Install required dependencies if needed
            # 2. Convert file_bytes to appropriate format
            # 3. Extract text content
            # 4. Return ProcessingResult with success=True
            
            content = ""  # Your extracted content
            
            # Validate content is not empty
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
            # Handle any errors
            logger.error(
                f"[File Extraction]Error processing [FILE TYPE] file {file_path.name}: {str(e)}"
            )
            return ProcessingResult(
                success=False,
                content="",
                error_description="[File Extraction][FILE TYPE] processing error",
                original_error=f"Failed to extract text from [FILE TYPE]: {str(e)}",
            )
```

## Common Patterns

### Pattern 1: Installing Dependencies with pipmaster

If your processor needs external libraries, use `pipmaster` (aliased as `pm`):

```python
import pipmaster as pm

# Check if package is installed, install if not
if not pm.is_installed("python-docx"):
    try:
        pm.install("python-docx")
    except Exception:
        # Fallback package name
        pm.install("docx")

from docx import Document
```

### Pattern 2: Using BytesIO for File-like Operations

Many libraries expect a file-like object. Convert `file_bytes` to `BytesIO`:

```python
from io import BytesIO

# Convert bytes to file-like object
file_obj = BytesIO(file_bytes)
doc = Document(file_obj)
```

### Pattern 3: Using file_path for Libraries That Need Disk Access

Some libraries (like DOCLING) need the file on disk:

```python
# Use file_path directly
converter = DocumentConverter()
result = converter.convert(file_path)  # file_path is a Path object
content = result.document.export_to_markdown()
```

### Pattern 4: Supporting Multiple Processing Engines

If there are multiple ways to process a file type (e.g., DOCLING vs python-docx):

```python
from ..config import global_args

if global_args.document_loading_engine == "DOCLING":
    # Use DOCLING
    if not pm.is_installed("docling"):
        pm.install("docling")
    from docling.document_converter import DocumentConverter
    converter = DocumentConverter()
    result = converter.convert(file_path)
    content = result.document.export_to_markdown()
else:
    # Use alternative library
    if not pm.is_installed("python-docx"):
        pm.install("python-docx")
    from docx import Document
    from io import BytesIO
    docx_file = BytesIO(file_bytes)
    doc = Document(docx_file)
    content = "\n".join([paragraph.text for paragraph in doc.paragraphs])
```

### Pattern 5: Handling Encrypted/Protected Files

For files that might be encrypted (like PDFs):

```python
# Check if file is encrypted
if reader.is_encrypted:
    pdf_password = global_args.pdf_decrypt_password
    if not pdf_password:
        return ProcessingResult(
            success=False,
            content="",
            error_description="[File Extraction]File is encrypted but no password provided",
            original_error="Please set PDF_DECRYPT_PASSWORD environment variable",
        )
    # Try to decrypt
    decrypt_result = reader.decrypt(pdf_password)
    if decrypt_result == 0:
        return ProcessingResult(
            success=False,
            content="",
            error_description="[File Extraction]Failed to decrypt - incorrect password",
            original_error="The provided password is incorrect",
        )
```

## Error Handling

### Always Use Try-Except

Wrap your extraction logic in try-except blocks:

```python
try:
    # Your extraction logic
    content = extract_text(file_bytes)
    return ProcessingResult(success=True, content=content)
except SpecificException as e:
    # Handle specific errors
    return ProcessingResult(
        success=False,
        content="",
        error_description="[File Extraction]Specific error message",
        original_error=str(e),
    )
except Exception as e:
    # Catch-all for unexpected errors
    logger.error(f"[File Extraction]Error processing {file_path.name}: {str(e)}")
    return ProcessingResult(
        success=False,
        content="",
        error_description="[File Extraction][FILE TYPE] processing error",
        original_error=f"Unexpected error: {str(e)}",
    )
```

### Error Message Format

- **error_description**: Should start with `[File Extraction]` and be human-readable
- **original_error**: Can contain technical details, exception messages, etc.

Examples:
```python
error_description="[File Extraction]DOCX processing error"
error_description="[File Extraction]File is encrypted but no password provided"
error_description="[File Extraction]Empty file content"
```

### Logging

Use the logger for important events:

```python
from lightrag.utils import logger

logger.error(f"[File Extraction]Error processing {file_path.name}: {str(e)}")
logger.warning(f"[File Extraction]Warning message")
logger.info(f"[File Extraction]Info message")
```

## Registration

### Automatic Registration

Processors are automatically registered when you add them to `__init__.py`:

```python
# In lightrag/api/routers/processing_types/__init__.py

from .processing_docx import DocxFileProcessor

_processor_instances = [
    TxtFileProcessor(),
    DocxFileProcessor(),  # Add your processor here
]

# Auto-register all processors
for processor in _processor_instances:
    register_processor(processor)
```

### Manual Registration (if needed)

You can also register manually:

```python
from .processor_registry import register_processor
from .processing_docx import DocxFileProcessor

register_processor(DocxFileProcessor())
```

## Testing

### Quick Test

1. Create a test file of your type
2. Upload it through the API
3. Check the logs for any errors
4. Verify the content was extracted correctly

### Testing Locally

You can test your processor directly:

```python
from pathlib import Path
from lightrag.api.routers.processing_types.processing_docx import DocxFileProcessor

# Create processor instance
processor = DocxFileProcessor()

# Read a test file
with open("test.docx", "rb") as f:
    file_bytes = f.read()

# Process it
import asyncio
result = asyncio.run(processor.process_file(
    file_path=Path("test.docx"),
    file_bytes=file_bytes,
    file_size=len(file_bytes),
    track_id="test-123"
))

# Check result
print(f"Success: {result.success}")
print(f"Content length: {len(result.content)}")
if not result.success:
    print(f"Error: {result.error_description}")
```

## Best Practices

### 1. **Keep It Simple**
   - Focus on extracting text content
   - Don't try to preserve formatting unless necessary
   - Return plain text, not HTML or markdown (unless that's the file format)

### 2. **Handle Edge Cases**
   - Empty files
   - Corrupted files
   - Very large files
   - Encrypted/protected files
   - Files with no extractable text

### 3. **Validate Content**
   - Always check if extracted content is empty
   - Return appropriate error messages

### 4. **Use Appropriate Libraries**
   - Choose well-maintained libraries
   - Prefer libraries that work with bytes (don't require disk access)
   - Document any special requirements

### 5. **Error Messages**
   - Be descriptive but concise
   - Include actionable information when possible
   - Follow the `[File Extraction]` prefix convention

### 6. **Performance**
   - Process files efficiently
   - Don't load entire large files into memory unnecessarily
   - Consider streaming for very large files

### 7. **Documentation**
   - Add docstrings to your class and methods
   - Document any special requirements or dependencies
   - Include examples if the usage is complex

## Example: Complete DOCX Processor

Here's a complete example based on the existing DOCX handling code:

```python
"""
DOCX file processor.

Handles Microsoft Word .docx files using either DOCLING or python-docx.
"""

from pathlib import Path
from io import BytesIO
from .base_processor import BaseFileProcessor, ProcessingResult
from lightrag.utils import logger
import pipmaster as pm
from ..config import global_args


class DocxFileProcessor(BaseFileProcessor):
    """Processor for Microsoft Word .docx files."""
    
    @property
    def supported_extensions(self) -> tuple[str, ...]:
        """Return supported file extensions for this processor."""
        return (".docx",)
    
    async def process_file(
        self,
        file_path: Path,
        file_bytes: bytes,
        file_size: int,
        track_id: str,
    ) -> ProcessingResult:
        """Process a .docx file and extract its text content.
        
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
                if not pm.is_installed("docling"):
                    pm.install("docling")
                from docling.document_converter import DocumentConverter
                
                converter = DocumentConverter()
                result = converter.convert(file_path)
                content = result.document.export_to_markdown()
            else:
                # Use python-docx as fallback
                if not pm.is_installed("python-docx"):
                    try:
                        pm.install("python-docx")
                    except Exception:
                        pm.install("docx")
                from docx import Document
                
                docx_file = BytesIO(file_bytes)
                doc = Document(docx_file)
                content = "\n".join(
                    [paragraph.text for paragraph in doc.paragraphs]
                )
            
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
                f"[File Extraction]Error processing DOCX file {file_path.name}: {str(e)}"
            )
            return ProcessingResult(
                success=False,
                content="",
                error_description="[File Extraction]DOCX processing error",
                original_error=f"Failed to extract text from DOCX: {str(e)}",
            )
```

## Need Help?

- Check `processing_txt.py` for a simple example
- Look at existing file type handlers in `document_routes.py` for reference
- Review `base_processor.py` for the complete interface definition
- Ask the team if you're stuck!

## Integration: Replacing Old Code in document_routes.py

After creating your processor, you need to integrate it into `document_routes.py` to replace the old inline processing code.

### Step 1: Locate the Old Code

1. Open `lightrag/api/routers/document_routes.py`
2. Find the `pipeline_enqueue_file()` function (around line 905)
3. Look for the `match ext:` statement (around line 981)
4. Find the `case` statement for your file type (e.g., `case ".docx":`)

### Step 2: Understand the Pattern

The old code looks like this:

```python
case ".docx":
    try:
        # Old inline processing code here
        if global_args.document_loading_engine == "DOCLING":
            # ... lots of code ...
        else:
            # ... more code ...
        content = extracted_text
    except Exception as e:
        error_files = [
            {
                "file_path": str(file_path.name),
                "error_description": "[File Extraction]DOCX processing error",
                "original_error": f"Failed to extract text from DOCX: {str(e)}",
                "file_size": file_size,
            }
        ]
        await rag.apipeline_enqueue_error_documents(error_files, track_id)
        logger.error(f"[File Extraction]Error processing DOCX {file_path.name}: {str(e)}")
        return False, track_id
```

### Step 3: Replace with Processor Pattern

Replace the entire `case ".docx":` block with this pattern:

```python
case ".docx":
    # Use the processor for .docx files
    processor = get_processor(ext)
    if processor:
        result = await processor.process_file(
            file_path, file, file_size, track_id
        )
        if result.success:
            content = result.content
        else:
            # Convert ProcessingResult error to error_files format
            error_files = [
                {
                    "file_path": str(file_path.name),
                    "error_description": result.error_description or "[File Extraction]DOCX processing error",
                    "original_error": result.original_error or "Unknown error",
                    "file_size": file_size,
                }
            ]
            await rag.apipeline_enqueue_error_documents(
                error_files, track_id
            )
            logger.error(
                f"[File Extraction]{result.error_description or 'Error'} in file: {file_path.name}"
            )
            return False, track_id
    else:
        # Fallback to original logic if processor not found (optional)
        # You can remove this if you're confident the processor is registered
        logger.warning(f"No processor found for {ext}, using fallback")
        # ... original code as fallback ...
        return False, track_id
```

### Step 4: Complete Example - DOCX Integration

**Before (old code):**

```python
case ".docx":
    try:
        if global_args.document_loading_engine == "DOCLING":
            if not pm.is_installed("docling"):  # type: ignore
                pm.install("docling")
            from docling.document_converter import DocumentConverter  # type: ignore

            converter = DocumentConverter()
            result = converter.convert(file_path)
            content = result.document.export_to_markdown()
        else:
            if not pm.is_installed("python-docx"):  # type: ignore
                try:
                    pm.install("python-docx")
                except Exception:
                    pm.install("docx")
            from docx import Document  # type: ignore
            from io import BytesIO

            docx_file = BytesIO(file)
            doc = Document(docx_file)
            content = "\n".join(
                [paragraph.text for paragraph in doc.paragraphs]
            )
    except Exception as e:
        error_files = [
            {
                "file_path": str(file_path.name),
                "error_description": "[File Extraction]DOCX processing error",
                "original_error": f"Failed to extract text from DOCX: {str(e)}",
                "file_size": file_size,
            }
        ]
        await rag.apipeline_enqueue_error_documents(
            error_files, track_id
        )
        logger.error(
            f"[File Extraction]Error processing DOCX {file_path.name}: {str(e)}"
        )
        return False, track_id
```

**After (new code using processor):**

```python
case ".docx":
    # Use the processor for .docx files
    processor = get_processor(ext)
    if processor:
        result = await processor.process_file(
            file_path, file, file_size, track_id
        )
        if result.success:
            content = result.content
        else:
            # Convert ProcessingResult error to error_files format
            error_files = [
                {
                    "file_path": str(file_path.name),
                    "error_description": result.error_description or "[File Extraction]DOCX processing error",
                    "original_error": result.original_error or "Unknown error",
                    "file_size": file_size,
                }
            ]
            await rag.apipeline_enqueue_error_documents(
                error_files, track_id
            )
            logger.error(
                f"[File Extraction]{result.error_description or 'Error'} in file: {file_path.name}"
            )
            return False, track_id
    else:
        # Fallback: log warning if processor not found
        logger.error(
            f"[File Extraction]No processor registered for {ext}. Please register DocxFileProcessor in __init__.py"
        )
        return False, track_id
```

### Step 5: Verify Import Statement

Make sure `get_processor` is imported at the top of `document_routes.py`:

```python
# At the top of document_routes.py (around line 29)
from .processing_types import get_processor
```

If it's not there, add it!

### Step 6: What to Remove

**Remove:**
- All the inline processing logic (the `try` block with extraction code)
- The old error handling code (it's now in the processor)
- Any imports that were only used for this file type (if not used elsewhere)

**Keep:**
- The `case ".docx":` statement
- The error handling pattern that converts `ProcessingResult` to `error_files` format
- The `await rag.apipeline_enqueue_error_documents()` call
- The `return False, track_id` on errors

### Step 7: Testing the Integration

1. **Test with a real file:**
   - Upload a `.docx` file through the API
   - Check that it processes correctly
   - Verify the content is extracted

2. **Test error handling:**
   - Try uploading a corrupted file
   - Verify error messages are logged correctly
   - Check that errors are enqueued properly

3. **Check logs:**
   - Look for any errors in the console/logs
   - Verify the processor is being called (check for your log messages)

### Common Issues

**Issue: "No processor found"**
- **Solution:** Make sure you registered your processor in `__init__.py`
- **Solution:** Verify the extension matches exactly (case-insensitive, but include the dot)

**Issue: "Import error"**
- **Solution:** Make sure `from .processing_types import get_processor` is at the top of `document_routes.py`

**Issue: "Content is empty"**
- **Solution:** Check your processor's `process_file()` method returns content correctly
- **Solution:** Verify `result.success` is `True` when content exists

**Issue: "Old code still running"**
- **Solution:** Make sure you replaced the entire `case` block, not just part of it
- **Solution:** Check that there are no syntax errors preventing the new code from running

### Pattern Summary

For any file type, the integration pattern is always the same:

```python
case ".your_ext":
    processor = get_processor(ext)
    if processor:
        result = await processor.process_file(file_path, file, file_size, track_id)
        if result.success:
            content = result.content
        else:
            error_files = [{
                "file_path": str(file_path.name),
                "error_description": result.error_description or "[File Extraction]Processing error",
                "original_error": result.original_error or "Unknown error",
                "file_size": file_size,
            }]
            await rag.apipeline_enqueue_error_documents(error_files, track_id)
            logger.error(f"[File Extraction]{result.error_description or 'Error'} in file: {file_path.name}")
            return False, track_id
    else:
        logger.error(f"No processor found for {ext}")
        return False, track_id
```

Just replace `.your_ext` with your file extension and you're done!

## Checklist

Before submitting your processor, make sure:

- [ ] Your class inherits from `BaseFileProcessor`
- [ ] You've implemented `supported_extensions` property
- [ ] You've implemented `process_file()` method
- [ ] All errors return `ProcessingResult` with `success=False`
- [ ] Success cases return `ProcessingResult` with `success=True` and content
- [ ] Error messages start with `[File Extraction]`
- [ ] You've added your processor to `__init__.py`
- [ ] **You've replaced the old code in `document_routes.py`**
- [ ] **You've verified `get_processor` is imported in `document_routes.py`**
- [ ] You've tested with a real file
- [ ] You've tested error handling
- [ ] Your code follows the existing style
- [ ] You've added appropriate docstrings

Good luck! 🚀

