"""
PDF Analysis Script for Jupyter Notebooks

This script provides functions to analyze PDF files using pypdf, extracting
text and images from each page. Designed for use in Jupyter notebooks with
visual rendering capabilities.
"""

import base64
from io import BytesIO
from pathlib import Path
from typing import Optional, Union, List, Dict, Any
import pipmaster as pm

# Minimum image dimension threshold for significance check
MIN_IMG_DIMENSION = 100


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
    if width < min_dimension or height < min_dimension:
        return False
    return True


def get_significant_images(page) -> list:
    """Extract and filter significant images from a PDF page.

    Args:
        page: A pypdf page object containing images in page.images

    Returns:
        List of significant ImageFile objects from page.images that meet
        the minimum dimension requirements.
    """
    if not pm.is_installed("Pillow"):
        pm.install("Pillow")
    from PIL import Image

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
            # If we can't read the image dimensions, skip it
            print(f"Warning: Could not read dimensions for image {img.name}: {e}")
            continue

    return significant_images


def image_to_base64(pil_image) -> str:
    """Convert PIL Image to base64 string.

    Args:
        pil_image: PIL Image object

    Returns:
        Base64 encoded string (without data URI prefix)
    """
    if not pm.is_installed("Pillow"):
        pm.install("Pillow")
    from PIL import Image

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


def extract_page_images(page, page_num: int, return_pil: bool = False) -> List[Dict[str, Any]]:
    """Extract images from a PDF page.

    Args:
        page: A pypdf page object
        page_num: Page number (1-indexed)
        return_pil: If True, return PIL Image objects; if False, return base64 strings

    Returns:
        List of dictionaries containing image data:
        - 'index': Image index within the page (1-indexed)
        - 'name': Image name from PDF
        - 'width': Image width in pixels
        - 'height': Image height in pixels
        - 'image': PIL Image object (if return_pil=True) or base64 string (if return_pil=False)
        - 'format': Image format (e.g., 'JPEG', 'PNG')
    """
    if not pm.is_installed("Pillow"):
        pm.install("Pillow")
    from PIL import Image

    significant_images = get_significant_images(page)
    extracted_images = []

    for image_index, img in enumerate(significant_images, 1):
        try:
            # Extract image data and open with PIL
            pil_image = Image.open(BytesIO(img.data))
            width, height = pil_image.size

            image_data = {
                "index": image_index,
                "name": img.name,
                "width": width,
                "height": height,
                "format": pil_image.format or "Unknown",
            }

            if return_pil:
                # Return PIL Image object for direct display in notebooks
                image_data["image"] = pil_image
            else:
                # Return base64 string for embedding in HTML
                image_data["image"] = image_to_base64(pil_image)
                image_data["image_format"] = "base64"

            extracted_images.append(image_data)
        except Exception as e:
            print(f"Warning: Error processing image {image_index} on page {page_num}: {e}")
            continue

    return extracted_images


def analyze_pdf(
    pdf_path: Union[str, Path],
    password: Optional[str] = None,
    return_pil_images: bool = True,
    min_image_dimension: int = MIN_IMG_DIMENSION,
) -> Dict[str, Any]:
    """Analyze a PDF file and extract text and images from each page.

    This function is designed for use in Jupyter notebooks. It extracts text
    and images from each page of a PDF and returns them in a structured format
    suitable for rendering in notebooks.

    Args:
        pdf_path: Path to the PDF file
        password: Optional password for encrypted PDFs
        return_pil_images: If True, return PIL Image objects for direct display
                          in notebooks. If False, return base64-encoded strings.
        min_image_dimension: Minimum dimension threshold for filtering significant images

    Returns:
        Dictionary containing:
        - 'file_path': Path to the PDF file
        - 'total_pages': Total number of pages
        - 'is_encrypted': Whether the PDF is encrypted
        - 'pages': List of page data, each containing:
            - 'page_number': Page number (1-indexed)
            - 'text': Extracted text from the page
            - 'text_length': Length of extracted text
            - 'images': List of image data dictionaries (see extract_page_images)
            - 'image_count': Number of images found on the page

    Raises:
        FileNotFoundError: If the PDF file does not exist
        ValueError: If PDF is encrypted and no password is provided, or password is incorrect
        Exception: For other PDF processing errors

    Example:
        >>> result = analyze_pdf("document.pdf")
        >>> print(f"Total pages: {result['total_pages']}")
        >>> for page in result['pages']:
        ...     print(f"Page {page['page_number']}: {len(page['text'])} chars, {page['image_count']} images")
    """
    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    # Install pypdf if needed
    if not pm.is_installed("pypdf"):
        pm.install("pypdf")
    if not pm.is_installed("pycryptodome"):
        pm.install("pycryptodome")
    from pypdf import PdfReader

    # Read PDF file
    with open(pdf_path, "rb") as f:
        pdf_file = BytesIO(f.read())

    reader = PdfReader(pdf_file)
    is_encrypted = reader.is_encrypted

    # Handle encrypted PDFs
    if is_encrypted:
        if not password:
            raise ValueError(
                "PDF is encrypted. Please provide a password using the 'password' parameter."
            )
        try:
            decrypt_result = reader.decrypt(password)
            if decrypt_result == 0:
                raise ValueError("Incorrect password provided for encrypted PDF")
        except Exception as e:
            raise ValueError(f"Failed to decrypt PDF: {str(e)}")

    # Extract data from each page
    pages_data = []
    total_pages = len(reader.pages)

    for page_num, page in enumerate(reader.pages, 1):
        # Extract text
        page_text = page.extract_text() or ""

        # Extract images
        try:
            page_images = extract_page_images(page, page_num, return_pil=return_pil_images)
        except Exception as e:
            print(f"Warning: Error extracting images from page {page_num}: {e}")
            page_images = []

        page_data = {
            "page_number": page_num,
            "text": page_text,
            "text_length": len(page_text),
            "images": page_images,
            "image_count": len(page_images),
        }

        pages_data.append(page_data)

    return {
        "file_path": str(pdf_path),
        "total_pages": total_pages,
        "is_encrypted": is_encrypted,
        "pages": pages_data,
    }


def display_page_in_notebook(page_data: Dict[str, Any], show_images: bool = True) -> None:
    """Display a page's extracted data in a Jupyter notebook.

    This function uses IPython.display to render text and images in a notebook cell.

    Args:
        page_data: Page data dictionary from analyze_pdf
        show_images: Whether to display images (requires PIL Image objects)

    Example:
        >>> result = analyze_pdf("document.pdf")
        >>> for page in result['pages']:
        ...     display_page_in_notebook(page)
    """
    try:
        from IPython.display import display, Markdown, Image, HTML
    except ImportError:
        print("Warning: IPython not available. Cannot display in notebook format.")
        print(f"Page {page_data['page_number']}:")
        print(f"Text ({page_data['text_length']} chars):")
        print(page_data['text'])
        print(f"Images: {page_data['image_count']}")
        return

    # Display page header
    display(Markdown(f"## Page {page_data['page_number']}"))

    # Display text
    if page_data['text']:
        display(Markdown("### Extracted Text"))
        display(Markdown(f"```\n{page_data['text']}\n```"))
    else:
        display(Markdown("*No text found on this page*"))

    # Display images
    if show_images and page_data['images']:
        display(Markdown(f"### Images ({page_data['image_count']})"))
        for img_data in page_data['images']:
            display(Markdown(f"**Image {img_data['index']}**: {img_data['name']} "
                           f"({img_data['width']}x{img_data['height']}px, {img_data['format']})"))
            
            if 'image' in img_data:
                if isinstance(img_data['image'], str):
                    # Base64 encoded image
                    display(Image(data=base64.b64decode(img_data['image'])))
                else:
                    # PIL Image object
                    display(img_data['image'])


def analyze_and_display_pdf(
    pdf_path: Union[str, Path],
    password: Optional[str] = None,
    return_pil_images: bool = True,
    min_image_dimension: int = MIN_IMG_DIMENSION,
) -> Dict[str, Any]:
    """Analyze a PDF and automatically display results in a Jupyter notebook.

    This is a convenience function that combines analyze_pdf and display_page_in_notebook.

    Args:
        pdf_path: Path to the PDF file
        password: Optional password for encrypted PDFs
        return_pil_images: If True, return PIL Image objects for direct display
        min_image_dimension: Minimum dimension threshold for filtering significant images

    Returns:
        Dictionary containing analysis results (same as analyze_pdf)

    Example:
        >>> result = analyze_and_display_pdf("document.pdf")
        >>> # Results are automatically displayed in the notebook
    """
    result = analyze_pdf(
        pdf_path=pdf_path,
        password=password,
        return_pil_images=return_pil_images,
        min_image_dimension=min_image_dimension,
    )

    # Display summary
    try:
        from IPython.display import display, Markdown
        display(Markdown(f"# PDF Analysis: {Path(result['file_path']).name}"))
        display(Markdown(f"**Total Pages:** {result['total_pages']}"))
        display(Markdown(f"**Encrypted:** {result['is_encrypted']}"))
        display(Markdown("---"))
    except ImportError:
        print(f"PDF Analysis: {Path(result['file_path']).name}")
        print(f"Total Pages: {result['total_pages']}")
        print(f"Encrypted: {result['is_encrypted']}")
        print("-" * 50)

    # Display each page
    for page_data in result['pages']:
        display_page_in_notebook(page_data, show_images=return_pil_images)

    return result

