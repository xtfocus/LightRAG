"""
PDF file processor.

Handles PDF files using either DOCLING or pypdf, with support for encrypted PDFs
that require password decryption.
"""

import asyncio
import base64
import os
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING

import pipmaster as pm
from openai import (APIConnectionError, APITimeoutError, AsyncAzureOpenAI,
                    RateLimitError)
from pydantic import BaseModel
from tenacity import (retry, retry_if_exception_type, stop_after_attempt,
                      wait_exponential)

from lightrag.llm.azure_openai import _azure_openai_complete_inner
from lightrag.utils import logger

from ...config import global_args
from .base_processor import BaseFileProcessor, ProcessingResult

global_args.document_loading_engine == "DOCLING"

if TYPE_CHECKING:
    from PIL import Image
    from pypdf._page import ImageFile, PageObject

# Minimum image dimension threshold for significance check
MIN_IMG_DIMENSION = 100
INFOGRAPHIC_IMAGE_THRESHOLD = 10


def get_page_visual_element_count(page: "PageObject") -> int:
    """Count visual elements (images + form XObjects) on a page."""
    count = len(getattr(page, "images", []))
    try:
        resources = page.get("/Resources")
        if resources and "/XObject" in resources:
            xobjects = resources["/XObject"]
            if hasattr(xobjects, "keys"):
                for key in xobjects.keys():
                    obj = xobjects[key]
                    if hasattr(obj, "get"):
                        subtype = obj.get("/Subtype")
                        if subtype == "/Form":
                            count += 1
    except Exception:
        pass
    return count


def is_landscape_page(page: "PageObject") -> bool:
    """Determine if a PDF page is landscape orientation."""
    try:
        width = float(page.mediabox.width)
        height = float(page.mediabox.height)
        return width > height
    except Exception:
        return False


def is_infographic_page(page: "PageObject") -> bool:
    """Determine if a page should be treated as an infographic."""
    image_count = get_page_visual_element_count(page)
    if is_landscape_page(page):
        return True
    if image_count > INFOGRAPHIC_IMAGE_THRESHOLD:
        return True
    return False


def render_pdf_page_to_image(pdf_bytes: bytes, page_index: int) -> "Image.Image":
    """Render a PDF page to a PIL Image using pypdfium2."""
    if not pm.is_installed("pypdfium2"):  # type: ignore
        pm.install("pypdfium2")
    import pypdfium2 as pdfium  # type: ignore

    pdf_doc = pdfium.PdfDocument(BytesIO(pdf_bytes))
    page = pdf_doc[page_index]
    pil_image = page.render_topil(scale=2.0)
    page.close()
    pdf_doc.close()
    return pil_image


def _get_azure_openai_config() -> tuple[str, str, str, str]:
    """Get Azure OpenAI configuration from environment or global_args.

    Returns:
        Tuple of (deployment, base_url, api_key, api_version)
    """
    deployment = (
        os.getenv("AZURE_OPENAI_DEPLOYMENT") or global_args.llm_model or "gpt-4o"
    )
    base_url = os.getenv("AZURE_OPENAI_ENDPOINT") or global_args.llm_binding_host
    api_key = os.getenv("AZURE_OPENAI_API_KEY") or global_args.llm_binding_api_key
    api_version = os.getenv("AZURE_OPENAI_API_VERSION") or os.getenv(
        "OPENAI_API_VERSION"
    )
    return deployment, base_url, api_key, api_version


class ImageJudgment(BaseModel):
    """Pydantic model for structured image judgment response."""

    is_trivial: bool
    needs_retry: bool
    is_substantial: bool
    reason: str


def check_image_significance(
    width: int, height: int, min_dimension: int = MIN_IMG_DIMENSION
) -> bool:
    """Check if an image is significant based on its dimensions.

    Filters out trivial images like thin lines, small icons, or decorative elements
    that are too small in one or both dimensions.

    Args:
        width: Image width in pixels
        height: Image height in pixels
        min_dimension: Minimum dimension threshold (default: MIN_IMG_DIMENSION)
            Images smaller than this in width, height, or both are considered trivial.

    Returns:
        True if the image is significant (should be processed),
        False if the image is trivial (too thin, too tall, or too small).
    """
    # Image is trivial if:
    # - Width is too small (thin vertical line or tiny icon)
    # - Height is too small (thin horizontal line or tiny icon)
    # - Both dimensions are too small (tiny icon/decorative element)
    if width < min_dimension or height < min_dimension:
        return False

    return True


def get_significant_images(page) -> list["ImageFile"]:
    """Extract and filter significant images from a PDF page.

    Iterates through all images in the page, checks their dimensions,
    and returns only those that are significant (not trivial icons, lines, etc.).

    Args:
        page: A pypdf page object containing images in page.images

    Returns:
        List of significant ImageFile objects from page.images that meet
        the minimum dimension requirements.
    """
    if not pm.is_installed("Pillow"):  # type: ignore
        pm.install("Pillow")
    from PIL import Image  # type: ignore

    significant_images = []

    for img in page.images:
        try:
            # Extract image data and open with PIL
            pil_image = Image.open(BytesIO(img.data))
            width, height = pil_image.size

            # Check if image is significant
            if check_image_significance(width, height):
                significant_images.append(img)
        except Exception as e:
            # If we can't read the image dimensions, log and skip it
            logger.debug(
                f"[File Extraction]Could not read dimensions for image {img.name}: {e}"
            )
            continue

    return significant_images


def image_to_base64(pil_image) -> str:
    """Convert PIL Image to base64 string.

    Args:
        pil_image: PIL Image object

    Returns:
        Base64 encoded string (without data URI prefix)
    """
    if not pm.is_installed("Pillow"):  # type: ignore
        pm.install("Pillow")
    from PIL import Image  # type: ignore

    # Convert RGBA to RGB if needed (for JPEG compatibility)
    if pil_image.mode in ("RGBA", "LA", "P"):
        # Create white background
        rgb_image = Image.new("RGB", pil_image.size, (255, 255, 255))
        if pil_image.mode == "P":
            pil_image = pil_image.convert("RGBA")
        rgb_image.paste(
            pil_image,
            mask=pil_image.split()[-1] if pil_image.mode in ("RGBA", "LA") else None,
        )
        pil_image = rgb_image
    elif pil_image.mode != "RGB":
        pil_image = pil_image.convert("RGB")

    # Convert to bytes
    img_buffer = BytesIO()
    pil_image.save(img_buffer, format="JPEG")
    img_bytes = img_buffer.getvalue()

    # Encode to base64
    return base64.b64encode(img_bytes).decode("utf-8")


async def _call_azure_openai_vision(
    image_base64: str,
    text_prompt: str,
    system_prompt: str | None = None,
) -> str:
    """Helper function to call Azure OpenAI with vision messages.

    Args:
        image_base64: Base64 encoded image string
        text_prompt: Text prompt for the vision model
        system_prompt: Optional system prompt

    Returns:
        Response text from Azure OpenAI
    """
    # Get Azure OpenAI configuration
    deployment, base_url, api_key, api_version = _get_azure_openai_config()

    # Build vision messages
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append(
        {
            "role": "user",
            "content": [
                {"type": "text", "text": text_prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"},
                },
            ],
        }
    )

    # Prepare trace metadata
    default_model = global_args.llm_model or "gpt-4o"
    trace_metadata = {
        "model": deployment or default_model,
        "deployment": deployment or default_model,
        "api_version": api_version,
        "base_url": base_url,
        "enable_cot": False,
    }

    # Call the inner function directly with vision messages
    return await _azure_openai_complete_inner(
        model=deployment or default_model,
        prompt=text_prompt,
        messages=messages,
        base_url=base_url,
        deployment=deployment or default_model,
        api_key=api_key,
        api_version=api_version,
        timeout=None,
        kwargs={},
        trace_metadata=trace_metadata,
    )


@retry(
    stop=stop_after_attempt(10),
    wait=wait_exponential(multiplier=1, min=4, max=60),
    retry=retry_if_exception_type((RateLimitError, APIConnectionError, APITimeoutError))
    | retry_if_exception_type(Exception),
)
async def describe_image(
    image_base64: str, image_index: int, page_num: int
) -> str | None:
    """Generate description for an image using Azure OpenAI Vision API.

    Args:
        image_base64: Base64 encoded image string
        image_index: Index of image within the page (1-indexed)
        page_num: Page number (1-indexed)

    Returns:
        Image description string, or None if all retries failed
    """
    try:
        prompt = "Describe this image in detail, including any text, charts, diagrams, or meaningful visual elements."
        description = await _call_azure_openai_vision(image_base64, prompt)
        return description.strip() if description else None
    except Exception as e:
        logger.error(
            f"[File Extraction]Failed to describe image {image_index} on page {page_num}: {e}"
        )
        return None


@retry(
    stop=stop_after_attempt(10),
    wait=wait_exponential(multiplier=1, min=4, max=60),
    retry=retry_if_exception_type((RateLimitError, APIConnectionError, APITimeoutError))
    | retry_if_exception_type(Exception),
)
async def _call_azure_openai_structured(
    prompt: str,
    system_prompt: str | None,
    response_format: type[BaseModel],
) -> BaseModel:
    """Helper function to call Azure OpenAI with structured output (Pydantic model).

    Args:
        prompt: Text prompt
        system_prompt: Optional system prompt
        response_format: Pydantic model class for structured output

    Returns:
        Parsed Pydantic model instance
    """
    # Get Azure OpenAI configuration
    deployment, base_url, api_key, api_version = _get_azure_openai_config()

    # Build messages
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    # Call the parse endpoint directly to get structured output
    default_model = global_args.llm_model or "gpt-4o"
    openai_async_client = AsyncAzureOpenAI(
        azure_endpoint=base_url,
        azure_deployment=deployment or default_model,
        api_key=api_key,
        api_version=api_version,
    )

    try:
        response = await openai_async_client.beta.chat.completions.parse(
            model=deployment or default_model,
            messages=messages,
            response_format=response_format,
        )

        # Get the parsed object
        parsed = response.choices[0].message.parsed
        if parsed is None:
            raise ValueError("Failed to parse structured response from Azure OpenAI")

        return parsed
    finally:
        await openai_async_client.close()


@retry(
    stop=stop_after_attempt(10),
    wait=wait_exponential(multiplier=1, min=4, max=60),
    retry=retry_if_exception_type((RateLimitError, APIConnectionError, APITimeoutError))
    | retry_if_exception_type(Exception),
)
async def judge_image(description: str) -> dict:
    """Judge if an image is trivial or substantial based on its description.

    Args:
        description: Generated image description

    Returns:
        Dictionary with keys: is_trivial, needs_retry, is_substantial, reason
    """
    prompt = f"""Analyze this image description and determine:
1. Is the image trivial (simple icon, basic shape, decorative line) or substantial (contains information, charts, diagrams, infographics)?
2. Was the previous description a denial of service or error message?
3. Is the description of sufficient quality?

Description: {description}"""

    try:
        # Use structured output with Pydantic model
        judgment = await _call_azure_openai_structured(
            prompt=prompt,
            system_prompt="You are a helpful assistant that analyzes image descriptions.",
            response_format=ImageJudgment,
        )

        return {
            "is_trivial": judgment.is_trivial,
            "needs_retry": judgment.needs_retry,
            "is_substantial": judgment.is_substantial,
            "reason": judgment.reason,
        }
    except Exception as e:
        logger.error(f"[File Extraction]Failed to judge image: {e}")
        # Default to substantial on error
        return {
            "is_trivial": False,
            "needs_retry": False,
            "is_substantial": True,
            "reason": f"Error during judgment: {str(e)}",
        }


async def process_single_image(
    img: "ImageFile", image_index: int, page_num: int
) -> str | None:
    """Process a single image: extract, describe, judge, and format.

    This function performs the loop: create description > review > retry if needed.

    Args:
        img: ImageFile object from pypdf
        image_index: Index of image within the page (1-indexed)
        page_num: Page number (1-indexed)

    Returns:
        Formatted description string like "Image x page y: {description}",
        or None if image is trivial or processing failed
    """
    if not pm.is_installed("Pillow"):  # type: ignore
        pm.install("Pillow")
    from PIL import Image  # type: ignore

    try:
        # Extract image data and convert to PIL Image
        pil_image = Image.open(BytesIO(img.data))

        # Convert to base64
        image_base64 = image_to_base64(pil_image)

        # Generate description
        description = await describe_image(image_base64, image_index, page_num)
        if not description:
            logger.warning(
                f"[File Extraction]Failed to generate description for image {image_index} on page {page_num}"
            )
            return None

        # Judge the image
        judgment = await judge_image(description)

        # If judge says we need retry, retry description generation
        if judgment.get("needs_retry", False):
            logger.info(
                f"[File Extraction]Retrying description for image {image_index} on page {page_num}"
            )
            description = await describe_image(image_base64, image_index, page_num)
            if not description:
                return None
            # Re-judge after retry
            judgment = await judge_image(description)

        # If image is trivial, skip it
        if judgment.get("is_trivial", False):
            logger.debug(
                f"[File Extraction]Skipping trivial image {image_index} on page {page_num}: {judgment.get('reason', '')}"
            )
            return None

        # If image is substantial, return formatted description
        if judgment.get("is_substantial", True):
            return f"Image {image_index} page {page_num}: {description}"

        # Default: skip if not substantial
        return None

    except Exception as e:
        logger.error(
            f"[File Extraction]Error processing image {image_index} on page {page_num}: {e}"
        )
        return None


async def process_page_images(page, page_num: int) -> str:
    """Process all images from a PDF page concurrently.

    Extracts significant images, processes them concurrently, and combines
    their descriptions into a single string.

    Args:
        page: A pypdf page object
        page_num: Page number (1-indexed)

    Returns:
        Combined image descriptions string, or empty string if no images
        or all images were filtered out
    """
    # Get significant images (already filtered by size)
    significant_images = get_significant_images(page)

    if not significant_images:
        return ""

    # Create tasks for concurrent processing
    tasks = []
    for image_index, img in enumerate(significant_images, 1):
        task = process_single_image(img, image_index, page_num)
        tasks.append(task)

    # Process all images concurrently
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Filter out None results, exceptions, and combine descriptions
    image_descriptions = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            logger.error(
                f"[File Extraction]Exception processing image {i+1} on page {page_num}: {result}"
            )
            continue
        if result is not None:
            image_descriptions.append(result)

    # Combine all descriptions with newlines
    if image_descriptions:
        return "\n".join(image_descriptions) + "\n"

    return ""


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
        """Process a PDF file and extract its text and image content.

        This method:
        1. Uses DOCLING if document_loading_engine is set to "DOCLING"
        2. Falls back to pypdf for text extraction
        3. Handles encrypted PDFs with password decryption
        4. Extracts text from all pages
        5. Extracts and describes images from all pages (pypdf path only)
        6. Returns the extracted text and image descriptions

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
                logger.info("Will use DOCLING for processing this pdf")
                if not pm.is_installed("docling"):  # type: ignore
                    pm.install("docling")
                from docling.document_converter import \
                    DocumentConverter  # type: ignore

                converter = DocumentConverter()
                result = converter.convert(file_path)
                content = result.document.export_to_markdown()
            else:
                logger.info("Will use pypdf for processing this pdf")
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

                # Extract text and images from PDF (encrypted PDFs are now decrypted, unencrypted PDFs proceed directly)
                infographic_pages: list[int] = []

                for page_num, page in enumerate(reader.pages, 1):
                    infographic_page = is_infographic_page(page)

                    if infographic_page:
                        infographic_pages.append(page_num)
                        try:
                            full_page_image = render_pdf_page_to_image(
                                file_bytes, page_num - 1
                            )
                            image_base64 = image_to_base64(full_page_image)
                            description = await describe_image(
                                image_base64, image_index=1, page_num=page_num
                            )
                            if description:
                                content += (
                                    f"[Infographic Page {page_num}]\n"
                                    f"{description.strip()}\n"
                                )
                                continue
                            else:
                                logger.warning(
                                    f"[File Extraction]Azure description empty for infographic page {page_num}, falling back to text extraction."
                                )
                        except Exception as e:
                            logger.warning(
                                f"[File Extraction]Failed to process infographic page {page_num}: {e}. Falling back to text extraction."
                            )

                    # Extract text from page
                    page_text = page.extract_text() or ""

                    # Process images from page (concurrent processing)
                    try:
                        image_descriptions = await process_page_images(page, page_num)
                    except Exception as e:
                        # If image processing fails, log and continue with text only
                        logger.warning(
                            f"[File Extraction]Failed to process images on page {page_num}: {e}. Continuing with text extraction only."
                        )
                        image_descriptions = ""

                    # Combine page text and image descriptions
                    content += page_text + "\n" + image_descriptions

                if infographic_pages:
                    logger.info(
                        f"[File Extraction]Infographic pages detected for {file_path.name}: {infographic_pages}"
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
                f"[File Extraction]Error processing PDF file {file_path.name}: {str(e)}"
            )
            return ProcessingResult(
                success=False,
                content="",
                error_description="[File Extraction]PDF processing error",
                original_error=f"Failed to extract text from PDF: {str(e)}",
            )
