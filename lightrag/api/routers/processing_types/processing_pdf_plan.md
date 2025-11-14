# PDF Image Extraction and Processing Plan

## Overview
Enhance the PDF processor to extract and describe images from PDF pages using Azure OpenAI Vision API. Images are processed concurrently, described, and judged for relevance before being included in the extracted content.

## Architecture

### Processing Flow
1. **Extract text** from each page using `page.extract_text()`
2. **Extract images** from each page using `page.images`
3. **Process all images concurrently** (async/await)
4. **For each image:**
   - Convert to base64 format
   - Generate description via Azure OpenAI Vision API
   - Judge description quality and image relevance
   - Filter out trivial images (icons, simple shapes, lines)
5. **Combine per page:** `page_text + "\n" + image_descriptions + "\n"`
6. **Detect org-chart pages:** Inspect vector rectangles (PyMuPDF) for hierarchical layouts
7. **If detected:** Produce simplified layout JSON for the page
8. **Combine per page:** Append org-chart JSON (if any) to the textual content block
9. **Combine all pages:** Join all page contents into final `content`

### End-to-End Flow (Mermaid)

```mermaid
flowchart TD
    A["Uploaded PDF bytes/path<br/>PdfFileProcessor.process_file"] --> B{"document_loading_engine == 'DOCLING'?"}
    B -->|Yes| C["docling.DocumentConverter<br/>convert -> export_to_markdown"]
    C --> Z{"content.strip()?"}
    B -->|No| D["Ensure pypdf + pycryptodome<br/>PdfReader(BytesIO(bytes))"]
    D --> E{"reader.is_encrypted?"}
    E -->|Yes| F["decrypt with PDF_DECRYPT_PASSWORD"]
    F --> G["for page in enumerate(reader.pages, 1)"]
    E -->|No| G
    G --> H{"is_infographic_page(page)?"}
    H -->|Yes| I["render_pdf_page_to_image<br/>image_to_base64<br/>describe_image()"]
    I -->|description| J["append 'Infographic Page N' block<br/>continue next page"]
    J --> G
    I -->|failure| K["fallback to regular flow"]
    H -->|No| K
    K --> L["page_text = page.extract_text() or ''"]
    L --> M["image_descriptions = process_page_images(page, n)"]
    M --> O{"is_org_chart_page(page)?"}
    O -->|Yes| P["extract_org_chart_layout()<br/>attach layout JSON"]
    P --> N["append page_text + '\\n' + image_descriptions + '\\n' + layout_json"]
    O -->|No| N["append page_text + '\\n' + image_descriptions"]
    N --> G
    Z -->|False/Empty| ERR["ProcessingResult failure:<br/>'[File Extraction]Empty file content'"]
    Z -->|True| OK["ProcessingResult success<br/>content"]```

### Image Processing Pipeline (Per Image)

```
Image Extraction
    ↓
Size Check (optional pre-filter)
    ↓
Convert to Base64
    ↓
LLM Call 1: Generate Description (with retry)
    ↓
LLM Call 2: Judge Description & Image Relevance (with retry)
    ↓
Decision: Include or Skip
    ↓
Format: "Image x page y: {description}"
```

## Implementation Details

### 1. Image Extraction
- **Location:** Only in pypdf path (not DOCLING)
- **Method:** `page.images` from pypdf
- **Format:** PIL Image objects via `Image.open(io.BytesIO(img.data))`
- **Size Check:** 
  - Get dimensions: `width, height = pil_image.size`
  - Filter criteria: Very small images (e.g., < 50x50 pixels) likely trivial
  - This is a pre-filter, but final decision made by judge

### 2. Image Description Generation

**Function:** `async def describe_image(image_base64: str, image_index: int, page_num: int) -> str`

- **Model:** Use environment-configured Azure OpenAI model (from `global_args` or env vars)
- **API Call:** Use `azure_openai_complete_if_cache` with vision format:
  ```python
  messages=[
      {
          "role": "user",
          "content": [
              {"type": "text", "text": "Describe this image in detail, including any text, charts, diagrams, or meaningful visual elements."},
              {
                  "type": "image_url",
                  "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}
              }
          ]
      }
  ]
  ```
- **Retry Logic:** 
  - Use `tenacity` decorator
  - At least 10 retry attempts
  - Exponential backoff
  - Retry on: `RateLimitError`, `APIConnectionError`, `APITimeoutError`, and general exceptions
- **Error Handling:** If all retries fail, return `None` (will be handled by caller)

### 3. Image Judgment

**Function:** `async def judge_image(description: str, image_size: tuple[int, int]) -> dict`

- **Purpose:** 
  - Determine if image is trivial (icon, simple shape, line) or substantial (infographic, chart, meaningful content)
  - Detect if previous call returned denial of service (need retry)
  - Validate description quality
- **Model:** Same environment-configured Azure OpenAI model
- **Input:** 
  - Generated description
  - Image dimensions (width, height)
- **Prompt:** 
  ```
  Analyze this image description and determine:
  1. Is the image trivial (simple icon, basic shape, decorative line) or substantial (contains information, charts, diagrams, infographics)?
  2. Was the previous description a denial of service or error message?
  3. Is the description of sufficient quality?
  
  Image dimensions: {width}x{height} pixels
  Description: {description}
  
  Respond in JSON format:
  {
    "is_trivial": true/false,
    "needs_retry": true/false,
    "is_substantial": true/false,
    "reason": "brief explanation"
  }
  ```
- **Retry Logic:** Same as description generation (10+ attempts, exponential backoff)
- **Size Consideration:** Small dimensions (< 50x50) suggest trivial, but judge makes final decision
- **Return:** Dictionary with judgment results

### 4. Image Format Conversion

**Function:** `def image_to_base64(pil_image: Image) -> str`

- Convert PIL Image to base64 string
- Handle different formats (JPEG, PNG, etc.)
- Convert RGBA to RGB if needed (for JPEG compatibility)
- Return format: base64 string (without data URI prefix)

### 5. Concurrent Processing

**Function:** `async def process_page_images(page: Page, page_num: int) -> str`

- Extract all images from page
- Create tasks for concurrent processing:
  ```python
  tasks = []
  for i, img in enumerate(page.images, 1):
      task = process_single_image(img, i, page_num)
      tasks.append(task)
  
  results = await asyncio.gather(*tasks, return_exceptions=True)
  ```
- Handle exceptions per image (continue processing others)
- Filter out None results and trivial images
- Format and combine descriptions
- Return combined image descriptions string

**Function:** `async def process_single_image(img, image_index: int, page_num: int) -> str | None`

- Extract image data: `Image.open(io.BytesIO(img.data))`
- Get dimensions for size check
- Convert to base64
- Generate description (with retry)
- Judge image (with retry)
- If needs_retry: retry description generation
- If trivial: return None (skip)
- If substantial: return formatted string `f"Image {image_index} page {page_num}: {description}"`

### 6. Integration with Existing Code

**Modify:** `process_file()` method in `PdfFileProcessor`

**Current flow:**
```python
for page in reader.pages:
    content += page.extract_text() + "\n"
```

**New flow:**
```python
for page_num, page in enumerate(reader.pages, 1):
    page_text = page.extract_text()
    image_descriptions = await process_page_images(page, page_num)
    org_chart_layout_json = detect_and_extract_org_chart(page, page_num)
    extra = f"\n[OrgChartLayoutJSON]\n{org_chart_layout_json}" if org_chart_layout_json else ""
    content += page_text + "\n" + image_descriptions + extra + "\n"
```

### 7. Org-Chart Page Detection & Layout JSON

- **Detection heuristic**
  - Use PyMuPDF (`pymupdf`) to collect rectangles and vector paths per page.
  - Candidate criteria (configurable):
    - At least `ORG_CHART_RECT_THRESHOLD` rectangles (default 8) with area above `ORG_CHART_MIN_AREA`.
    - Presence of a container rectangle covering > `ORG_CHART_ROOT_AREA_RATIO` (default 0.6) of the page OR nested rectangles reaching `ORG_CHART_MIN_DEPTH`.
  - Skip if page already marked as general infographic to avoid double work.
- **Extraction**
  - Invoke shared helper (derived from `notebooks/org_chart.py`) with PyMuPDF backend to:
    1. Detect rectangles & assign text.
    2. Build parent/child tree.
    3. Produce simplified JSON: `{ "page_number": n, "trees": [...] }`.
  - Optionally include geometry metadata when `ORG_CHART_INCLUDE_GEOMETRY` is enabled.
- **Attachment**
  - Store JSON string in page metadata (e.g., `page_analysis.org_chart_layout_json`) and append a tagged text block:
    ```
    [OrgChartLayoutJSON]
    {"page":3,"trees":[...]}
    ```
  - Downstream LLM prompt builders include this snippet when present.
- **Configuration**
  - Feature flag `ENABLE_ORG_CHART_EXTRACTION` (default True).
  - Thresholds: `ORG_CHART_RECT_THRESHOLD`, `ORG_CHART_ROOT_AREA_RATIO`, `ORG_CHART_MIN_DEPTH`, `ORG_CHART_MIN_AREA`.
- **Error handling**
  - On extraction failure, log warning and continue; never block page processing.
  - If JSON exceeds size limit (e.g., 4 KB), truncate or summarize before attachment.

## Error Handling Strategy

### Per-Image Errors
- **Description generation fails:** Log error, continue with next image, skip this image
- **Judgment fails:** Log error, treat as substantial (include description to be safe)
- **Image conversion fails:** Log error, skip image
- **All retries exhausted:** Log warning, skip image, continue processing

### Overall Error Handling
- If image processing completely fails: Continue with text extraction only
- Never fail entire PDF processing due to image issues
- Log all errors with context (page number, image index, error type)

## Configuration

### Environment Variables
- Use existing Azure OpenAI configuration:
  - `AZURE_OPENAI_ENDPOINT`
  - `AZURE_OPENAI_API_KEY`
  - `AZURE_OPENAI_DEPLOYMENT` (or `LLM_MODEL`)
  - `AZURE_OPENAI_API_VERSION`

### Model Selection
- Use same model as configured for LLM operations
- Must support vision API (e.g., `gpt-4o`, `gpt-4-vision-preview`)

## Retry Configuration

### Tenacity Decorator
```python
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

@retry(
    stop=stop_after_attempt(10),
    wait=wait_exponential(multiplier=1, min=4, max=60),
    retry=retry_if_exception_type(
        (RateLimitError, APIConnectionError, APITimeoutError)
    ) | retry_if_exception_type(Exception),  # Retry on any exception
)
async def describe_image(...):
    ...
```

## Dependencies

### Required Packages
- `pypdf` (already used)
- `Pillow` (PIL) - for image handling
- `pymupdf` - for rectangle detection & layout parsing
- `tenacity` - for retry logic
- `asyncio` - for concurrent processing
- `base64` - for encoding
- `io` - for BytesIO

### Azure OpenAI
- Already integrated via `lightrag.llm.azure_openai`

## Testing Considerations

### Test Cases
1. PDF with no images (should work as before)
2. PDF with trivial images (icons, lines) - should be filtered
3. PDF with substantial images (charts, diagrams) - should be included
4. PDF with mixed image types
5. PDF with many images (test concurrent processing)
6. API failures (test retry logic)
7. Rate limiting (test exponential backoff)
8. Large images (test memory handling)
9. PDF with hierarchical org-chart diagrams (validate detection + JSON output)

## Performance Considerations

- **Concurrent processing:** Process all images in a page concurrently
- **Memory:** Don't keep all images in memory simultaneously if possible
- **Rate limits:** Exponential backoff helps with rate limits
- **Timeout:** Consider per-image timeout to avoid hanging

## Output Format

### Final Content Structure
```
[Page 1 Text Content]

Image 1 page 1: [Description of first image on page 1]
Image 2 page 1: [Description of second image on page 1]

[Page 2 Text Content]

Image 1 page 2: [Description of first image on page 2]

[OrgChartLayoutJSON]
{"page":2,"trees":[...]}

...
```

### Image Description Format
- **Included images:** `"Image {x} page {y}: {description}"`
- **Trivial images:** Skipped (not included in output)
- **Failed images:** Skipped (not included in output)

## Implementation Order

1. Add image extraction and base64 conversion
2. Implement image description generation with retry
3. Implement image judgment with retry
4. Implement concurrent processing per page
5. Integrate with existing PDF processing flow
6. Add org-chart detection + layout extraction
7. Attach org-chart JSON to page content and metadata
8. Add error handling and logging
9. Test with various PDF types (including org charts)

## Notes

- Only implement for pypdf path (not DOCLING)
- Image extraction is always enabled (not optional)
- Use existing Azure OpenAI integration
- Maintain backward compatibility (PDFs without images should work as before)

