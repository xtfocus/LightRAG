"""
Org Chart / Treemap Extraction Utilities

This notebook-friendly helper scans a PDF that contains a treemap- or org-chart
style visualization, detects the rectangles that form the chart, and associates
each rectangle with the text that falls inside it.  The resulting structure can
then be used to understand which organizations belong to which divisions or to
reconstruct the hierarchy visually.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    import pdfplumber  # type: ignore
except ImportError:
    pdfplumber = None  # type: ignore[assignment]

try:
    import fitz  # type: ignore
except ImportError:
    fitz = None  # type: ignore[assignment]


@dataclass
class Word:
    text: str
    x0: float
    x1: float
    top: float
    bottom: float

    @property
    def center(self) -> Tuple[float, float]:
        return ((self.x0 + self.x1) / 2, (self.top + self.bottom) / 2)


@dataclass
class RectEntity:
    rect_id: str
    page_number: int
    x0: float
    x1: float
    top: float
    bottom: float
    fill: Optional[str] = None
    stroke: Optional[str] = None
    words: List[Word] = field(default_factory=list)
    parent_id: Optional[str] = None

    @property
    def width(self) -> float:
        return self.x1 - self.x0

    @property
    def height(self) -> float:
        return self.bottom - self.top

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def label(self) -> str:
        """Return the concatenated text for the rectangle."""
        lines = group_words_into_lines(self.words)
        return " ".join(lines).strip()

    def contains_word(self, word: Word, padding: float = 1.5) -> bool:
        """Return True if the word's bounding box resides inside the rectangle."""
        return (
            word.x0 >= self.x0 - padding
            and word.x1 <= self.x1 + padding
            and word.top >= self.top - padding
            and word.bottom <= self.bottom + padding
        )


def group_words_into_lines(words: Iterable[Word], line_tolerance: float = 4.0) -> List[str]:
    """Simple heuristic to stitch words into text lines from top to bottom."""
    sorted_words = sorted(words, key=lambda w: (w.top, w.x0))
    lines: List[List[Word]] = []

    for word in sorted_words:
        if not lines:
            lines.append([word])
            continue

        last_line = lines[-1]
        if abs(word.top - last_line[0].top) <= line_tolerance:
            last_line.append(word)
        else:
            lines.append([word])

    joined_lines = [" ".join(w.text for w in line) for line in lines]
    return joined_lines


def extract_words_pdfplumber(page: "pdfplumber.page.Page") -> List[Word]:
    words_data = page.extract_words(
        use_text_flow=True,
        keep_blank_chars=False,
        extra_attrs=["fontname", "size"],
    )
    return [
        Word(
            text=w["text"],
            x0=float(w["x0"]),
            x1=float(w["x1"]),
            top=float(w["top"]),
            bottom=float(w["bottom"]),
        )
        for w in words_data
        if w.get("text")
    ]


def load_rectangles_pdfplumber(
    page: "pdfplumber.page.Page",
    page_number: int,
    min_area: float = 500.0,
) -> List[RectEntity]:
    rectangles: List[RectEntity] = []
    for idx, rect in enumerate(page.rects, 1):
        entity = RectEntity(
            rect_id=f"p{page_number}_rect{idx}",
            page_number=page_number,
            x0=float(rect["x0"]),
            x1=float(rect["x1"]),
            top=float(rect["top"]),
            bottom=float(rect["bottom"]),
            fill=rect.get("non_stroking_color"),
            stroke=rect.get("stroking_color"),
        )
        if entity.area >= min_area:
            rectangles.append(entity)
    return rectangles


def extract_words_pymupdf(page: "fitz.Page") -> List[Word]:
    words_data = page.get_text("words")
    words: List[Word] = []
    for entry in words_data:
        if len(entry) < 5:
            continue
        x0, y0, x1, y1, text, *_ = entry
        if not text.strip():
            continue
        words.append(
            Word(
                text=text,
                x0=float(x0),
                x1=float(x1),
                top=float(y0),
                bottom=float(y1),
            )
        )
    return words


def load_rectangles_pymupdf(
    page: "fitz.Page",
    page_number: int,
    min_area: float = 500.0,
    dedup_precision: float = 1.0,
) -> List[RectEntity]:
    rectangles: List[RectEntity] = []
    seen: set[Tuple[float, float, float, float]] = set()

    def register_bbox(bbox: Tuple[float, float, float, float]) -> None:
        nonlocal rectangles
        rounded = tuple(round(coord / dedup_precision) * dedup_precision for coord in bbox)
        if rounded in seen:
            return
        seen.add(rounded)

        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        area = width * height
        if area < min_area or width <= 0 or height <= 0:
            return

        rect_id = f"p{page_number}_rect{len(rectangles) + 1}"
        rectangles.append(
            RectEntity(
                rect_id=rect_id,
                page_number=page_number,
                x0=float(bbox[0]),
                x1=float(bbox[2]),
                top=float(bbox[1]),
                bottom=float(bbox[3]),
                fill=None,
                stroke=None,
            )
        )

    drawings = page.get_drawings()
    for drawing in drawings:
        direct_rect_found = False
        for item in drawing.get("items", []):
            op = item[0]
            geometry = item[1] if len(item) > 1 else None
            if op == "re" and isinstance(geometry, fitz.Rect):
                register_bbox((geometry.x0, geometry.y0, geometry.x1, geometry.y1))
                direct_rect_found = True
        if direct_rect_found:
            continue

        points: List[Tuple[float, float]] = []
        for item in drawing.get("items", []):
            for maybe_point in item[1:]:
                if isinstance(maybe_point, fitz.Point):
                    points.append((maybe_point.x, maybe_point.y))

        if len(points) < 4:
            continue

        xs = [pt[0] for pt in points]
        ys = [pt[1] for pt in points]
        width = max(xs) - min(xs)
        height = max(ys) - min(ys)
        if width <= 0 or height <= 0:
            continue
        register_bbox((min(xs), min(ys), max(xs), max(ys)))

    return rectangles


def assign_words_to_rectangles(
    rectangles: List[RectEntity],
    words: List[Word],
    padding: float = 1.5,
) -> None:
    for rect in rectangles:
        for word in words:
            if rect.contains_word(word, padding=padding):
                rect.words.append(word)


def serialize_rectangles(rectangles: List[RectEntity]) -> List[Dict[str, Any]]:
    return [
        {
            "rect_id": rect.rect_id,
            "bounds": {
                "x0": rect.x0,
                "x1": rect.x1,
                "top": rect.top,
                "bottom": rect.bottom,
            },
            "width": rect.width,
            "height": rect.height,
            "area": rect.area,
            "label": rect.label,
            "parent_id": rect.parent_id,
        }
        for rect in rectangles
    ]


def detect_hierarchy(rectangles: List[RectEntity], padding: float = 2.0) -> None:
    """Annotate rectangles with parent IDs based on containment."""
    sorted_rects = sorted(rectangles, key=lambda r: r.area, reverse=True)
    for child in sorted_rects:
        for parent in sorted_rects:
            if parent is child:
                continue
            if parent.area <= child.area:
                continue
            if is_rect_inside(child, parent, padding=padding):
                child.parent_id = parent.rect_id
                break


def is_rect_inside(inner: RectEntity, outer: RectEntity, padding: float = 0.0) -> bool:
    return (
        inner.x0 >= outer.x0 - padding
        and inner.x1 <= outer.x1 + padding
        and inner.top >= outer.top - padding
        and inner.bottom <= outer.bottom + padding
    )


def _process_with_pdfplumber(
    pdf_path: Path,
    min_rect_area: float,
    word_padding: float,
    hierarchy_padding: float,
) -> List[Dict[str, Any]]:
    if pdfplumber is None:
        raise ImportError(
            "pdfplumber is not installed. Install it via `pip install pdfplumber` "
            "or use backend='pymupdf'."
        )

    pages_output: List[Dict[str, Any]] = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        for page_number, page in enumerate(pdf.pages, 1):
            words = extract_words_pdfplumber(page)
            rectangles = load_rectangles_pdfplumber(page, page_number, min_area=min_rect_area)
            assign_words_to_rectangles(rectangles, words, padding=word_padding)
            detect_hierarchy(rectangles, padding=hierarchy_padding)
            pages_output.append(
                {"page_number": page_number, "rectangles": serialize_rectangles(rectangles)}
            )
    return pages_output


def _process_with_pymupdf(
    pdf_path: Path,
    min_rect_area: float,
    word_padding: float,
    hierarchy_padding: float,
    dedup_precision: float = 1.0,
) -> List[Dict[str, Any]]:
    if fitz is None:
        raise ImportError(
            "PyMuPDF (fitz) is not installed. Install it via `pip install pymupdf` "
            "or use backend='pdfplumber'."
        )

    pages_output: List[Dict[str, Any]] = []
    doc = fitz.open(str(pdf_path))
    try:
        for page_index in range(len(doc)):
            page = doc[page_index]
            page_number = page_index + 1
            words = extract_words_pymupdf(page)
            rectangles = load_rectangles_pymupdf(
                page, page_number, min_area=min_rect_area, dedup_precision=dedup_precision
            )
            assign_words_to_rectangles(rectangles, words, padding=word_padding)
            detect_hierarchy(rectangles, padding=hierarchy_padding)
            pages_output.append(
                {"page_number": page_number, "rectangles": serialize_rectangles(rectangles)}
            )
    finally:
        doc.close()
    return pages_output


def extract_org_chart(
    pdf_path: Path | str,
    min_rect_area: float = 500.0,
    word_padding: float = 1.5,
    hierarchy_padding: float = 2.0,
    backend: str = "pdfplumber",
    backend_options: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Extract per-rectangle text blocks and inferred hierarchy.

    Args:
        pdf_path: Path to the org-chart PDF.
        min_rect_area: Minimum rectangle area (in PDF units) to keep.
        word_padding: Padding (in PDF units) when assigning words to rectangles.
        hierarchy_padding: Padding tolerance for parent-child containment detection.
        backend: Either 'pdfplumber' (default) or 'pymupdf'.
        backend_options: Optional dict with backend-specific overrides (currently
            supports 'dedup_precision' for pymupdf).
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    backend = backend.lower()
    backend_options = backend_options or {}

    if backend == "pdfplumber":
        pages_output = _process_with_pdfplumber(
            pdf_path=pdf_path,
            min_rect_area=min_rect_area,
            word_padding=word_padding,
            hierarchy_padding=hierarchy_padding,
        )
    elif backend == "pymupdf":
        pages_output = _process_with_pymupdf(
            pdf_path=pdf_path,
            min_rect_area=min_rect_area,
            word_padding=word_padding,
            hierarchy_padding=hierarchy_padding,
            dedup_precision=float(backend_options.get("dedup_precision", 1.0)),
        )
    else:
        raise ValueError("backend must be 'pdfplumber' or 'pymupdf'")

    return {
        "file_path": str(pdf_path),
        "pages_processed": len(pages_output),
        "pages": pages_output,
    }


def summarize_hierarchy(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Flatten the hierarchy info into a quick summary table."""
    rows: List[Dict[str, Any]] = []
    for page in data["pages"]:
        for rect in page["rectangles"]:
            rows.append(
                {
                    "page": page["page_number"],
                    "rect_id": rect["rect_id"],
                    "label": rect["label"],
                    "parent_id": rect["parent_id"],
                    "area": rect["area"],
                }
            )
    return rows


def save_results(data: Dict[str, Any], output_path: Path | str) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return output_path


def preview_results(data: Dict[str, Any], max_rows: int = 10) -> None:
    rows = summarize_hierarchy(data)
    for row in rows[:max_rows]:
        print(
            f"[page {row['page']}] {row['rect_id']} -> {row['label'] or '<no label>'} "
            f"(parent={row['parent_id']}, area={row['area']:.0f})"
        )
    if len(rows) > max_rows:
        print(f"... {len(rows) - max_rows} more rows")


def build_page_hierarchy(
    rectangles: List[Dict[str, Any]], keep_geometry: bool = False
) -> List[Dict[str, Any]]:
    """Build parent-child trees for a page."""
    nodes: Dict[str, Dict[str, Any]] = {}
    for rect in rectangles:
        node = {
            "rect_id": rect["rect_id"],
            "text": rect["label"],
            "children": [],
        }
        if keep_geometry:
            node["bounds"] = rect["bounds"]
            node["area"] = rect["area"]
        nodes[rect["rect_id"]] = node

    roots: List[Dict[str, Any]] = []
    for rect in rectangles:
        node = nodes[rect["rect_id"]]
        parent_id = rect.get("parent_id")
        if parent_id and parent_id in nodes:
            nodes[parent_id]["children"].append(node)
        else:
            roots.append(node)
    for node in nodes.values():
        if not node["children"]:
            node.pop("children", None)
    return roots


def simplify_org_chart_result(
    data: Dict[str, Any], keep_geometry: bool = False
) -> Dict[str, Any]:
    """Return a simplified hierarchy (boxes with contained boxes/text)."""
    return {
        "file_path": data.get("file_path"),
        "pages": [
            {
                "page_number": page["page_number"],
                "trees": build_page_hierarchy(page["rectangles"], keep_geometry=keep_geometry),
            }
            for page in data["pages"]
        ],
    }


def _node_title(node: Dict[str, Any]) -> str:
    text = (node.get("text") or "").strip()
    return text if text else "Unnamed box"


def _describe_node(node: Dict[str, Any], depth: int = 0) -> List[str]:
    indent = "  " * depth
    title = _node_title(node)
    children = node.get("children") or []
    if children:
        child_titles = ", ".join(_node_title(child) for child in children)
        line = f"{indent}- Box '{title}' contains {len(children)} child box(es): {child_titles}."
    else:
        line = f"{indent}- Box '{title}' has no nested boxes."
    lines = [line]
    for child in children:
        lines.extend(_describe_node(child, depth + 1))
    return lines


def describe_simplified_page(page_layout: Dict[str, Any]) -> str:
    page_number = page_layout.get("page_number")
    lines = [f"Org chart layout for page {page_number}:"]
    trees = page_layout.get("trees") or []
    if not trees:
        lines.append("No box hierarchy detected.")
        return "\n".join(lines)
    for idx, tree in enumerate(trees, 1):
        root_title = _node_title(tree)
        lines.append(f"Root box {idx}: '{root_title}'.")
        lines.extend(_describe_node(tree, depth=1))
    return "\n".join(lines)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract org-chart rectangles from a PDF.")
    parser.add_argument("pdf_path", type=Path, help="Path to the treemap/org-chart PDF.")
    parser.add_argument(
        "--min-rect-area",
        type=float,
        default=500.0,
        help="Minimum rectangle area to consider (in PDF units).",
    )
    parser.add_argument(
        "--backend",
        choices=("pdfplumber", "pymupdf"),
        default="pdfplumber",
        help="Backend to extract boxes from vector instructions.",
    )
    parser.add_argument(
        "--dedup-precision",
        type=float,
        default=1.0,
        help="(PyMuPDF only) rounding precision for rectangle deduplication.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path to save the extracted structure as JSON.",
    )
    parser.add_argument(
        "--simplify-json",
        type=Path,
        default=None,
        help="Optional path to save simplified hierarchy (boxes + contained nodes).",
    )
    parser.add_argument(
        "--keep-geometry",
        action="store_true",
        help="Include bounds/area in simplified output.",
    )
    args = parser.parse_args()

    result = extract_org_chart(
        pdf_path=args.pdf_path,
        min_rect_area=args.min_rect_area,
        backend=args.backend,
        backend_options={"dedup_precision": args.dedup_precision},
    )
    preview_results(result)

    if args.output_json:
        saved_path = save_results(result, args.output_json)
        print(f"Saved detailed output to {saved_path}")

    if args.simplify_json:
        simplified = simplify_org_chart_result(
            result,
            keep_geometry=args.keep_geometry,
        )
        saved_path = save_results(simplified, args.simplify_json)
        print(f"Saved simplified hierarchy to {saved_path}")
        for page in simplified["pages"]:
            print(describe_simplified_page(page))

