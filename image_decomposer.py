"""Decompose a grid/composite image into individual sub-images.

Detects seams, gaps, or regular grid patterns, extracts each cell,
and saves them as numbered files.
"""

from pathlib import Path
from typing import List, Tuple, Optional

import cv2
import numpy as np
from PIL import Image


def decompose_image(image_rgb: np.ndarray, output_dir: str) -> Tuple[int, List[str]]:
    """Detect and extract sub-images from a grid/composite image.

    Args:
        image_rgb: Input image as HxWx3 uint8 RGB array.
        output_dir: Directory to save extracted images.

    Returns:
        (count, list of saved file paths)
    """
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape[:2]

    # Phase 1: detect visible seams/gaps
    h_gaps, v_gaps = _detect_gap_seams(gray)

    cells = None
    if h_gaps and v_gaps:
        cells = _extract_cells_from_gaps(image_rgb, h_gaps, v_gaps)
    elif h_gaps or v_gaps:
        # Only one direction has gaps — try gap + line combination
        cells = _extract_cells_partial_gaps(image_rgb, gray, h_gaps, v_gaps)

    # Phase 2: no-gap regular grid detection
    if cells is None:
        cells = _detect_regular_grid(image_rgb, gray)

    # Phase 3: content-based (connected components)
    if cells is None:
        cells = _detect_content_regions(image_rgb, gray)

    if not cells:
        return 0, []

    # Filter blank cells and sort top-to-bottom, left-to-right
    cells = [(y, x, img) for y, x, img in cells if not _is_blank_cell(img)]
    cells.sort(key=lambda c: (c[0], c[1]))

    # Save
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    saved = []
    for i, (_, _, cell_img) in enumerate(cells, 1):
        name = f"{i:03d}.png"
        path = Path(output_dir) / name
        Image.fromarray(cell_img).save(str(path))
        saved.append(str(path))

    return len(saved), saved


# ---------------------------------------------------------------------------
# Phase 1: gap/seam detection
# ---------------------------------------------------------------------------

def _detect_gap_seams(gray: np.ndarray, threshold: float = 5.0, min_gap: int = 2) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    """Find horizontal and vertical gap bands (uniform-colored seams).

    Returns lists of (start, end) pixel positions for each gap band.
    """
    h, w = gray.shape

    # Horizontal gaps: rows with low pixel variance
    h_gaps = []
    row_stds = np.std(gray, axis=1)
    in_gap = False
    gap_start = 0
    for y in range(h):
        if row_stds[y] < threshold:
            if not in_gap:
                gap_start = y
                in_gap = True
        else:
            if in_gap:
                gap_len = y - gap_start
                if gap_len >= min_gap:
                    h_gaps.append((gap_start, y - 1))
                in_gap = False
    if in_gap:
        gap_len = h - gap_start
        if gap_len >= min_gap:
            h_gaps.append((gap_start, h - 1))

    # Vertical gaps: columns with low pixel variance
    v_gaps = []
    col_stds = np.std(gray, axis=0)
    in_gap = False
    gap_start = 0
    for x in range(w):
        if col_stds[x] < threshold:
            if not in_gap:
                gap_start = x
                in_gap = True
        else:
            if in_gap:
                gap_len = x - gap_start
                if gap_len >= min_gap:
                    v_gaps.append((gap_start, x - 1))
                in_gap = False
    if in_gap:
        gap_len = w - gap_start
        if gap_len >= min_gap:
            v_gaps.append((gap_start, w - 1))

    # Filter out edge gaps (borders of the image)
    h_gaps = [(s, e) for s, e in h_gaps if s > 0 and e < h - 1]
    v_gaps = [(s, e) for s, e in v_gaps if s > 0 and e < w - 1]

    # Merge gaps that are very close together (< 3px apart)
    h_gaps = _merge_gaps(h_gaps, 3)
    v_gaps = _merge_gaps(v_gaps, 3)

    return h_gaps, v_gaps


def _merge_gaps(gaps: List[Tuple[int, int]], max_dist: int) -> List[Tuple[int, int]]:
    if not gaps:
        return gaps
    merged = [gaps[0]]
    for s, e in gaps[1:]:
        if s - merged[-1][1] <= max_dist:
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append((s, e))
    return merged


def _extract_cells_from_gaps(
    image: np.ndarray,
    h_gaps: List[Tuple[int, int]],
    v_gaps: List[Tuple[int, int]],
) -> List[Tuple[int, int, np.ndarray]]:
    """Extract cells using both horizontal and vertical gap positions."""
    h_bounds = _gaps_to_bounds(h_gaps, image.shape[0])
    v_bounds = _gaps_to_bounds(v_gaps, image.shape[1])

    if not h_bounds or not v_bounds:
        return []

    cells = []
    for y0, y1 in h_bounds:
        for x0, x1 in v_bounds:
            cell = image[y0:y1 + 1, x0:x1 + 1]
            if cell.shape[0] > 10 and cell.shape[1] > 10:
                cells.append((y0, x0, cell))
    return cells


def _extract_cells_partial_gaps(
    image: np.ndarray,
    gray: np.ndarray,
    h_gaps: List[Tuple[int, int]],
    v_gaps: List[Tuple[int, int]],
) -> Optional[List[Tuple[int, int, np.ndarray]]]:
    """When only one axis has gaps, try to find divisions on the other axis."""
    h, w = image.shape[:2]

    if h_gaps and not v_gaps:
        # Use h_gaps for rows, detect vertical lines
        h_bounds = _gaps_to_bounds(h_gaps, h)
        v_lines = _detect_lines(gray, horizontal=False)
        if not v_lines:
            return None
        v_bounds = _lines_to_bounds(v_lines, w)
    elif v_gaps and not h_gaps:
        v_bounds = _gaps_to_bounds(v_gaps, w)
        h_lines = _detect_lines(gray, horizontal=True)
        if not h_lines:
            return None
        h_bounds = _lines_to_bounds(h_lines, h)
    else:
        return None

    if not h_bounds or not v_bounds:
        return None

    cells = []
    for y0, y1 in h_bounds:
        for x0, x1 in v_bounds:
            cell = image[y0:y1 + 1, x0:x1 + 1]
            if cell.shape[0] > 10 and cell.shape[1] > 10:
                cells.append((y0, x0, cell))
    return cells


def _gaps_to_bounds(gaps: List[Tuple[int, int]], total: int) -> List[Tuple[int, int]]:
    """Convert gap positions to cell boundary positions."""
    if not gaps:
        return []
    bounds = []
    prev_end = 0
    for gap_start, gap_end in gaps:
        cell_end = gap_start - 1
        if cell_end >= prev_end:
            bounds.append((prev_end, cell_end))
        prev_end = gap_end + 1
    # Last cell
    if prev_end < total:
        bounds.append((prev_end, total - 1))
    return bounds


# ---------------------------------------------------------------------------
# Phase 2: regular grid detection (no visible seams)
# ---------------------------------------------------------------------------

def _detect_lines(gray: np.ndarray, horizontal: bool = True) -> List[int]:
    """Detect strong straight lines using Hough transform."""
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(
        edges, rho=1, theta=np.pi / 180,
        threshold=80, minLineLength=gray.shape[1] if horizontal else gray.shape[0],
        maxLineGap=10,
    )
    if lines is None:
        return []

    positions = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if horizontal:
            if abs(y2 - y1) < 5:  # Nearly horizontal
                positions.append((y1 + y2) // 2)
        else:
            if abs(x2 - x1) < 5:  # Nearly vertical
                positions.append((x1 + x2) // 2)

    if not positions:
        return []

    # Cluster nearby positions
    positions.sort()
    clusters = []
    cluster = [positions[0]]
    for p in positions[1:]:
        if p - cluster[-1] < 10:
            cluster.append(p)
        else:
            clusters.append(int(np.median(cluster)))
            cluster = [p]
    clusters.append(int(np.median(cluster)))

    return clusters


def _lines_to_bounds(line_positions: List[int], total: int) -> List[Tuple[int, int]]:
    """Convert line positions to cell boundary pairs."""
    if not line_positions:
        return []
    bounds = []
    prev = 0
    for pos in sorted(line_positions):
        if pos > prev + 10:
            bounds.append((prev, pos - 1))
        prev = pos
    if prev < total - 10:
        bounds.append((prev, total - 1))
    return bounds


def _score_grid_divisions(gray: np.ndarray, rows: int, cols: int) -> float:
    """Score a grid layout by measuring color differences at division lines.

    Higher score = stronger boundaries at the proposed division positions.
    """
    h, w = gray.shape
    cell_h = h // rows
    cell_w = w // cols
    band = max(2, min(cell_h, cell_w) // 20)  # sample width around division

    total_diff = 0.0
    count = 0

    # Horizontal divisions (between rows)
    for r in range(1, rows):
        y = r * cell_h
        if y + band >= h:
            continue
        # Compare band just above vs just below the division
        above = gray[max(0, y - band):y, :].astype(float)
        below = gray[y:y + band, :].astype(float)
        diff = np.mean(np.abs(above.mean(axis=0) - below.mean(axis=0)))
        total_diff += diff
        count += 1

    # Vertical divisions (between columns)
    for c in range(1, cols):
        x = c * cell_w
        if x + band >= w:
            continue
        left = gray[:, max(0, x - band):x].astype(float)
        right = gray[:, x:x + band].astype(float)
        diff = np.mean(np.abs(left.mean(axis=1) - right.mean(axis=1)))
        total_diff += diff
        count += 1

    return total_diff / max(count, 1)


def _detect_regular_grid(
    image: np.ndarray, gray: np.ndarray,
) -> Optional[List[Tuple[int, int, np.ndarray]]]:
    """Try to detect a regular grid when no seams are visible."""
    h, w = gray.shape[:2]

    # Try Hough line detection
    h_lines = _detect_lines(gray, horizontal=True)
    v_lines = _detect_lines(gray, horizontal=False)

    if h_lines and v_lines:
        h_bounds = _lines_to_bounds(h_lines, h)
        v_bounds = _lines_to_bounds(v_lines, w)
        if h_bounds and v_bounds:
            cells = []
            for y0, y1 in h_bounds:
                for x0, x1 in v_bounds:
                    cell = image[y0:y1 + 1, x0:x1 + 1]
                    if cell.shape[0] > 20 and cell.shape[1] > 20:
                        cells.append((y0, x0, cell))
            if len(cells) >= 2:
                return cells

    # Fallback: try common grid sizes, scored by edge strength at divisions
    best_grid = None
    best_score = -1

    for cols in range(2, 8):
        for rows in range(2, 8):
            cell_w = w // cols
            cell_h = h // rows
            if cell_w < 50 or cell_h < 50:
                continue
            remainder_w = w % cols
            remainder_h = h % rows
            if remainder_w > w * 0.05 or remainder_h > h * 0.05:
                continue

            # Score based on color difference at division lines
            edge_score = _score_grid_divisions(gray, rows, cols)
            # Penalize very large grids (prefer fewer, larger cells)
            size_penalty = (rows * cols) * 0.5
            score = edge_score - size_penalty
            if score > best_score:
                best_score = score
                best_grid = (rows, cols)

    if best_grid:
        rows, cols = best_grid
        cell_h = h // rows
        cell_w = w // cols
        cells = []
        for r in range(rows):
            for c in range(cols):
                y0 = r * cell_h
                x0 = c * cell_w
                y1 = y0 + cell_h
                x1 = x0 + cell_w
                cell = image[y0:y1, x0:x1]
                cells.append((y0, x0, cell))
        return cells

    return None


# ---------------------------------------------------------------------------
# Phase 3: content-based detection (irregular layouts)
# ---------------------------------------------------------------------------

def _detect_content_regions(
    image: np.ndarray, gray: np.ndarray,
) -> Optional[List[Tuple[int, int, np.ndarray]]]:
    """Use contour detection to find individual content regions."""
    # Blur and threshold to find content vs background
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # Otsu's thresholding
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Morphological operations to clean up
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) < 2:
        return None

    h, w = gray.shape
    cells = []
    for contour in contours:
        x, y, cw, ch = cv2.boundingRect(contour)
        # Skip tiny regions (< 2% of image area)
        if cw * ch < h * w * 0.02:
            continue
        # Skip nearly-full-image regions
        if cw > w * 0.95 and ch > h * 0.95:
            continue
        cell = image[y:y + ch, x:x + cw]
        cells.append((y, x, cell))

    return cells if len(cells) >= 2 else None


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _is_blank_cell(img: np.ndarray) -> bool:
    """Check if a cell is too small to be meaningful."""
    return img.shape[0] < 5 or img.shape[1] < 5
