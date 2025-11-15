"""
PDF file processor.

Handles PDF files using either DOCLING or pypdf, with support for encrypted PDFs
that require password decryption.
"""

import asyncio
import base64
import json
import os
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Tuple

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

if TYPE_CHECKING:
    from PIL import Image
    from pypdf._page import ImageFile, PageObject

# Minimum image dimension threshold for significance check
MIN_IMG_DIMENSION = 100
INFOGRAPHIC_IMAGE_THRESHOLD = 10
IMAGE_DESCRIPTION_RUNS = 3
ENABLE_ORG_CHART_EXTRACTION = (
    os.getenv("ENABLE_ORG_CHART_EXTRACTION", "true").lower() not in {"0", "false", "no"}
)
ORG_CHART_RECT_THRESHOLD = int(os.getenv("ORG_CHART_RECT_THRESHOLD", "8"))
ORG_CHART_ROOT_AREA_RATIO = float(os.getenv("ORG_CHART_ROOT_AREA_RATIO", "0.6"))
ORG_CHART_MIN_DEPTH = int(os.getenv("ORG_CHART_MIN_DEPTH", "2"))
ORG_CHART_MIN_AREA = float(os.getenv("ORG_CHART_MIN_AREA", "500"))
ORG_CHART_INCLUDE_GEOMETRY = (
    os.getenv("ORG_CHART_INCLUDE_GEOMETRY", "false").lower() in {"1", "true", "yes"}
)


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
    try:
        page = pdf_doc[page_index]
        bitmap = page.render(scale=2.0, rotation=0)
        pil_image = bitmap.to_pil()
        bitmap.close()
        page.close()
        return pil_image
    finally:
        pdf_doc.close()


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


# --- Org-chart extraction helpers ------------------------------------------------


@dataclass
class WordBox:
    text: str
    x0: float
    x1: float
    top: float
    bottom: float


@dataclass
class RectEntity:
    rect_id: str
    page_number: int
    x0: float
    x1: float
    top: float
    bottom: float
    words: List[WordBox] = field(default_factory=list)
    parent_id: Optional[str] = None

    @property
    def width(self) -> float:
        return self.x1 - self.x0

    @property
    def height(self) -> float:
        return self.bottom - self.top

    @property
    def area(self) -> float:
        return max(self.width, 0) * max(self.height, 0)


def group_words_into_lines(words: Iterable[WordBox], line_tol: float = 4.0) -> List[str]:
    sorted_words = sorted(words, key=lambda w: (w.top, w.x0))
    lines: List[List[WordBox]] = []
    for word in sorted_words:
        if not lines:
            lines.append([word])
            continue
        last_line = lines[-1]
        if abs(word.top - last_line[0].top) <= line_tol:
            last_line.append(word)
        else:
            lines.append([word])
    return [" ".join(w.text for w in line).strip() for line in lines]


def assign_words(rectangles: List[RectEntity], words: List[WordBox], padding: float = 1.5) -> None:
    for rect in rectangles:
        for word in words:
            if (
                word.x0 >= rect.x0 - padding
                and word.x1 <= rect.x1 + padding
                and word.top >= rect.top - padding
                and word.bottom <= rect.bottom + padding
            ):
                rect.words.append(word)


def detect_rect_hierarchy(rectangles: List[RectEntity], padding: float = 2.0) -> None:
    ordered = sorted(rectangles, key=lambda r: r.area, reverse=True)
    for child in ordered:
        for parent in ordered:
            if parent is child or parent.area <= child.area:
                continue
            if (
                child.x0 >= parent.x0 - padding
                and child.x1 <= parent.x1 + padding
                and child.top >= parent.top - padding
                and child.bottom <= parent.bottom + padding
            ):
                child.parent_id = parent.rect_id
                break


def serialize_rectangles(rectangles: List[RectEntity]) -> List[Dict[str, Any]]:
    data: List[Dict[str, Any]] = []
    for rect in rectangles:
        lines = group_words_into_lines(rect.words)
        data.append(
            {
                "rect_id": rect.rect_id,
                "bounds": {"x0": rect.x0, "x1": rect.x1, "top": rect.top, "bottom": rect.bottom},
                "width": rect.width,
                "height": rect.height,
                "area": rect.area,
                "label": " ".join(line for line in lines if line).strip(),
                "word_count": len(rect.words),
                "parent_id": rect.parent_id,
            }
        )
    return data


def build_page_hierarchy(rectangles: List[Dict[str, Any]], keep_geometry: bool = False) -> List[Dict[str, Any]]:
    nodes: Dict[str, Dict[str, Any]] = {}
    for rect in rectangles:
        node: Dict[str, Any] = {
            "rect_id": rect["rect_id"],
            "text": rect["label"],
            "word_count": rect["word_count"],
            "children": [],
        }
        if keep_geometry:
            node["bounds"] = rect["bounds"]
            node["area"] = rect["area"]
        nodes[rect["rect_id"]] = node

    roots: List[Dict[str, Any]] = []
    for rect in rectangles:
        parent_id = rect.get("parent_id")
        node = nodes[rect["rect_id"]]
        if parent_id and parent_id in nodes:
            nodes[parent_id]["children"].append(node)
        else:
            roots.append(node)
    return roots


def _compute_max_depth(rectangles: List[Dict[str, Any]]) -> int:
    children_map: Dict[str, List[str]] = {}
    for rect in rectangles:
        parent = rect.get("parent_id")
        if parent:
            children_map.setdefault(parent, []).append(rect["rect_id"])

    def depth(rect_id: str) -> int:
        if rect_id not in children_map:
            return 1
        return 1 + max(depth(child_id) for child_id in children_map[rect_id])

    depths = []
    for rect in rectangles:
        if not rect.get("parent_id"):
            depths.append(depth(rect["rect_id"]))
    return max(depths) if depths else 0


class OrgChartLayoutExtractor:
    """Extract simplified org-chart layouts using PyMuPDF vector data."""

    def __init__(self, pdf_bytes: bytes) -> None:
        """Initialize extractor (lazy-loads pymupdf)."""
        self.keep_geometry = ORG_CHART_INCLUDE_GEOMETRY
        self.doc = None
        self._fitz = None
        if not ENABLE_ORG_CHART_EXTRACTION:
            return
        try:
            if not pm.is_installed("pymupdf"):  # type: ignore
                pm.install("pymupdf")
            import fitz  # type: ignore

            self._fitz = fitz
            self.doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        except Exception as exc:
            logger.warning(f"[File Extraction]Failed to initialize PyMuPDF for org-chart extraction: {exc}")
            self.doc = None

    def close(self) -> None:
        if self.doc:
            try:
                self.doc.close()
            except Exception:
                pass
            self.doc = None

    def _extract_words(self, page: "fitz.Page") -> List[WordBox]:
        words: List[WordBox] = []
        try:
            for entry in page.get_text("words"):
                if len(entry) < 5:
                    continue
                x0, y0, x1, y1, text, *_ = entry
                if not text.strip():
                    continue
                words.append(WordBox(text=text, x0=float(x0), x1=float(x1), top=float(y0), bottom=float(y1)))
        except Exception as exc:
            logger.debug(f"[File Extraction]Failed to extract words for org-chart detection: {exc}")
        return words

    def _load_rectangles(self, page: "fitz.Page", page_number: int) -> List[RectEntity]:
        rectangles: List[RectEntity] = []
        seen: set[Tuple[int, int, int, int]] = set()

        def register(rect_obj: Any) -> None:
            if not rect_obj:
                return
            bbox = (float(rect_obj.x0), float(rect_obj.y0), float(rect_obj.x1), float(rect_obj.y1))
            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]
            area = width * height
            if area < ORG_CHART_MIN_AREA or width <= 0 or height <= 0:
                return
            rounded = tuple(int(round(coord)) for coord in bbox)
            if rounded in seen:
                return
            seen.add(rounded)
            rect_id = f"p{page_number}_rect{len(rectangles) + 1}"
            rectangles.append(
                RectEntity(
                    rect_id=rect_id,
                    page_number=page_number,
                    x0=bbox[0],
                    x1=bbox[2],
                    top=bbox[1],
                    bottom=bbox[3],
                )
            )

        try:
            drawings = page.get_drawings()
        except Exception as exc:
            logger.debug(f"[File Extraction]Failed to read drawings for org-chart detection: {exc}")
            return rectangles

        for drawing in drawings:
            direct_rect = False
            for item in drawing.get("items", []):
                op = item[0]
                geom = item[1] if len(item) > 1 else None
                if op == "re" and self._fitz and isinstance(geom, self._fitz.Rect):
                    register(geom)
                    direct_rect = True
            if direct_rect:
                continue

            points: List[Tuple[float, float]] = []
            for item in drawing.get("items", []):
                for maybe_point in item[1:]:
                    if self._fitz and isinstance(maybe_point, self._fitz.Point):
                        points.append((maybe_point.x, maybe_point.y))
            if len(points) < 4:
                continue
            xs = [pt[0] for pt in points]
            ys = [pt[1] for pt in points]
            register(self._fitz.Rect(min(xs), min(ys), max(xs), max(ys)) if self._fitz else None)

        return rectangles

    def extract_page_layout(self, page_number: int) -> Optional[str]:
        """Return layout JSON for a page if it looks like an org chart."""
        if not self.doc:
            return None
        try:
            page = self.doc[page_number - 1]
        except Exception:
            return None

        rectangles = self._load_rectangles(page, page_number)
        if len(rectangles) < ORG_CHART_RECT_THRESHOLD:
            logger.debug(
                f"[File Extraction]Org-chart detection skipped on page {page_number}: "
                f"{len(rectangles)} rectangles < threshold {ORG_CHART_RECT_THRESHOLD}"
            )
            return None

        words = self._extract_words(page)
        assign_words(rectangles, words)
        detect_rect_hierarchy(rectangles)

        serialized = serialize_rectangles(rectangles)
        page_rect = page.rect if hasattr(page, "rect") else None
        page_area = (page_rect.width * page_rect.height) if page_rect else None

        has_large_root = False
        if page_area:
            for rect in serialized:
                if not rect.get("parent_id") and rect["area"] / page_area >= ORG_CHART_ROOT_AREA_RATIO:
                    has_large_root = True
                    break

        max_depth = _compute_max_depth(serialized)
        if not has_large_root and max_depth < ORG_CHART_MIN_DEPTH:
            logger.debug(
                f"[File Extraction]Org-chart detection skipped on page {page_number}: "
                f"max_depth={max_depth}, has_large_root={has_large_root}"
            )
            return None

        simplified = {
            "page_number": page_number,
            "trees": build_page_hierarchy(serialized, keep_geometry=self.keep_geometry),
        }

        if not simplified["trees"]:
            logger.debug(
                f"[File Extraction]Org-chart detection skipped on page {page_number}: no hierarchy built."
            )
            return None

        logger.info(
            f"[File Extraction]Org-chart layout extracted for page {page_number} "
            f"(rectangles={len(serialized)}, depth={max_depth}, large_root={has_large_root})."
        )
        return json.dumps(simplified, ensure_ascii=False)


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


async def _call_azure_openai_chat(
    prompt: str, system_prompt: str | None = None
) -> str:
    """Helper function to call Azure OpenAI for text-only chat completions."""
    deployment, base_url, api_key, api_version = _get_azure_openai_config()

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    messages.append({"role": "user", "content": prompt})

    default_model = global_args.llm_model or "gpt-4o"
    return await _azure_openai_complete_inner(
        model=deployment or default_model,
        prompt=prompt,
        messages=messages,
        base_url=base_url,
        deployment=deployment or default_model,
        api_key=api_key,
        api_version=api_version,
        timeout=None,
        kwargs={},
        trace_metadata={
            "model": deployment or default_model,
            "deployment": deployment or default_model,
            "api_version": api_version,
            "base_url": base_url,
            "enable_cot": False,
        },
    )


async def _generate_single_image_description(
    image_base64: str, image_index: int, page_num: int, run_index: int
) -> str | None:
    prompt = (
        "You are transcribing an infographic or PDF page snapshot. "
        "Convert every visible element into text, preserving structure. "
        "Transcribe ALL readable text verbatim (word-for-word, including headings, labels, legends, footnotes, and data). "
        "Describe charts, tables, or icons with enough detail that a reader could recreate them. "
        "If layout matters (columns, bullet lists, timelines), explain the ordering. "
        "Do not summarize—provide a faithful textual rendering of the entire image."
    )
    description = await _call_azure_openai_vision(image_base64, prompt)
    return description.strip() if description else None


async def synthesize_image_descriptions(
    descriptions: list[str], image_index: int, page_num: int
) -> str | None:
    prompt_sections = "\n\n".join(
        f"Version {idx + 1}:\n{desc}" for idx, desc in enumerate(descriptions)
    )
    prompt = (
        "You received multiple detailed transcriptions of the same infographic/page. "
        "Combine them into ONE exhaustive transcription without losing any detail. "
        "If different versions contain conflicting information, include all variants "
        "and note the discrepancy. Preserve verbatim text, numeric values, and structure. "
        "The final result must allow a reader to recreate the entire visual precisely.\n\n"
        f"Transcriptions:\n{prompt_sections}"
    )
    combined = await _call_azure_openai_chat(
        prompt,
        system_prompt="You synthesize multiple OCR-like outputs into a single, lossless transcription.",
    )
    return combined.strip() if combined else None


@retry(
    stop=stop_after_attempt(10),
    wait=wait_exponential(multiplier=1, min=4, max=60),
    retry=retry_if_exception_type((RateLimitError, APIConnectionError, APITimeoutError))
    | retry_if_exception_type(Exception),
)
async def describe_image(
    image_base64: str, image_index: int, page_num: int
) -> str | None:
    """Generate a detailed description for an image using multiple LLM passes."""
    descriptions: list[str] = []
    for run_index in range(IMAGE_DESCRIPTION_RUNS):
        try:
            description = await _generate_single_image_description(
                image_base64, image_index, page_num, run_index
            )
            if description:
                descriptions.append(description)
        except Exception as e:
            logger.error(
                f"[File Extraction]Failed run {run_index + 1} describing image {image_index} on page {page_num}: {e}"
            )

    if not descriptions:
        return None

    if len(descriptions) == 1:
        return descriptions[0]

    combined = await synthesize_image_descriptions(descriptions, image_index, page_num)
    return combined or descriptions[0]


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
                org_chart_extractor: Optional[OrgChartLayoutExtractor] = (
                    OrgChartLayoutExtractor(file_bytes) if ENABLE_ORG_CHART_EXTRACTION else None
                )

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

                    extra_layout = ""
                    if org_chart_extractor and not infographic_page:
                        try:
                            layout_json = org_chart_extractor.extract_page_layout(page_num)
                            if layout_json:
                                extra_layout = f"[OrgChartLayoutJSON]\n{layout_json}\n"
                        except Exception as exc:
                            logger.debug(
                                f"[File Extraction]Org-chart extraction failed on page {page_num}: {exc}"
                            )

                    # Combine page text, image descriptions, and layout info
                    content += page_text + "\n" + image_descriptions + extra_layout

                if infographic_pages:
                    logger.info(
                        f"[File Extraction]Infographic pages detected for {file_path.name}: {infographic_pages}"
                    )

                if org_chart_extractor:
                    org_chart_extractor.close()

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
