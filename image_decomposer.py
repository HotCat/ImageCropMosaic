"""Decompose a grid/composite image into individual sub-images.

Detects seams, gaps, content discontinuities, or regular grid patterns,
extracts each cell, and saves them as numbered files.
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

    # Phase 1: detect visible uniform-colored seams/gaps
    h_gaps, v_gaps = _detect_gap_seams(gray)
    cells = None
    if h_gaps and v_gaps:
        cells = _extract_cells_from_gaps(image_rgb, h_gaps, v_gaps)
    elif h_gaps or v_gaps:
        cells = _extract_cells_partial_gaps(image_rgb, gray, h_gaps, v_gaps)

    # Phase 2: detect content discontinuities (images huddled together)
    if cells is None:
        cells = _detect_discontinuities(image_rgb, gray)

    # Phase 3: regular grid detection
    if cells is None:
        cells = _detect_regular_grid(image_rgb, gray)

    if not cells:
        return 0, []

    # Filter tiny cells and sort top-to-bottom, left-to-right
    cells = [(y, x, img) for y, x, img in cells
             if img.shape[0] >= 10 and img.shape[1] >= 10]
    cells.sort(key=lambda c: (c[0], c[1]))

    # Merge overlapping cells
    cells = _merge_overlapping_cells(cells, h, w)

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
# Phase 1: uniform-colored gap/seam detection
# ---------------------------------------------------------------------------

def _detect_gap_seams(gray: np.ndarray, threshold: float = 5.0,
                      min_gap: int = 2) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    """Find horizontal and vertical gap bands (uniform-colored seams)."""
    h, w = gray.shape

    row_stds = np.std(gray, axis=1)
    col_stds = np.std(gray, axis=0)

    h_gaps = _find_low_variance_bands(row_stds, threshold, min_gap)
    v_gaps = _find_low_variance_bands(col_stds, threshold, min_gap)

    # Filter out edge gaps
    h_gaps = [(s, e) for s, e in h_gaps if s > 0 and e < h - 1]
    v_gaps = [(s, e) for s, e in v_gaps if s > 0 and e < w - 1]

    h_gaps = _merge_nearby(h_gaps, 3)
    v_gaps = _merge_nearby(v_gaps, 3)

    return h_gaps, v_gaps


def _find_low_variance_bands(stds: np.ndarray, threshold: float,
                              min_len: int) -> List[Tuple[int, int]]:
    """Group contiguous low-variance positions into bands."""
    bands = []
    in_band = False
    start = 0
    for i in range(len(stds)):
        if stds[i] < threshold:
            if not in_band:
                start = i
                in_band = True
        else:
            if in_band:
                if i - start >= min_len:
                    bands.append((start, i - 1))
                in_band = False
    if in_band and len(stds) - start >= min_len:
        bands.append((start, len(stds) - 1))
    return bands


def _merge_nearby(gaps: List[Tuple[int, int]], max_dist: int) -> List[Tuple[int, int]]:
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
    h_bounds = _gaps_to_bounds(h_gaps, image.shape[0])
    v_bounds = _gaps_to_bounds(v_gaps, image.shape[1])
    if not h_bounds or not v_bounds:
        return []
    cells = []
    for y0, y1 in h_bounds:
        for x0, x1 in v_bounds:
            cell = image[y0:y1 + 1, x0:x1 + 1]
            cells.append((y0, x0, cell))
    return cells


def _extract_cells_partial_gaps(
    image: np.ndarray, gray: np.ndarray,
    h_gaps: List[Tuple[int, int]], v_gaps: List[Tuple[int, int]],
) -> Optional[List[Tuple[int, int, np.ndarray]]]:
    h, w = image.shape[:2]
    min_cell = max(50, min(h, w) // 8)
    if h_gaps and not v_gaps:
        h_bounds = _gaps_to_bounds(h_gaps, h)
        v_bounds = _discontinuity_bounds(gray, horizontal=False, min_cell=min_cell)
    elif v_gaps and not h_gaps:
        v_bounds = _gaps_to_bounds(v_gaps, w)
        h_bounds = _discontinuity_bounds(gray, horizontal=True, min_cell=min_cell)
    else:
        return None
    if not h_bounds or not v_bounds:
        return None
    cells = []
    for y0, y1 in h_bounds:
        for x0, x1 in v_bounds:
            cell = image[y0:y1 + 1, x0:x1 + 1]
            cells.append((y0, x0, cell))
    return cells if cells else None


def _gaps_to_bounds(gaps: List[Tuple[int, int]], total: int) -> List[Tuple[int, int]]:
    if not gaps:
        return []
    bounds = []
    prev_end = 0
    for gs, ge in gaps:
        if gs - 1 >= prev_end:
            bounds.append((prev_end, gs - 1))
        prev_end = ge + 1
    if prev_end < total:
        bounds.append((prev_end, total - 1))
    return bounds


# ---------------------------------------------------------------------------
# Phase 2: content discontinuity detection (huddled images)
# ---------------------------------------------------------------------------

def _row_diff(gray: np.ndarray) -> np.ndarray:
    """Average absolute difference between each row and the next."""
    return np.mean(np.abs(gray[1:].astype(float) - gray[:-1].astype(float)), axis=1)


def _col_diff(gray: np.ndarray) -> np.ndarray:
    """Average absolute difference between each column and the next."""
    return np.mean(np.abs(gray[:, 1:].astype(float) - gray[:, :-1].astype(float)), axis=0)


def _find_peaks(signal: np.ndarray, min_distance: int = 80,
                prominence_factor: float = 2.5, min_absolute: float = 10.0) -> List[int]:
    """Find peaks in a 1D signal that stand out above the local baseline.

    A peak is a local maximum that is at least prominence_factor * local_median
    and above min_absolute.
    """
    n = len(signal)
    if n < 3:
        return []

    # Smooth the signal to reduce noise
    kernel_size = max(5, n // 50)
    if kernel_size % 2 == 0:
        kernel_size += 1
    smoothed = cv2.GaussianBlur(
        signal.reshape(1, -1).astype(np.float32),
        (1, kernel_size), 0
    ).flatten()

    # Rolling median as baseline
    half_w = max(min_distance * 2, 60)
    baseline = np.zeros(n)
    for i in range(n):
        lo = max(0, i - half_w)
        hi = min(n, i + half_w + 1)
        baseline[i] = np.median(smoothed[lo:hi])

    # Find local maxima well above baseline
    peaks = []
    for i in range(1, n - 1):
        if smoothed[i] > smoothed[i - 1] and smoothed[i] > smoothed[i + 1]:
            if smoothed[i] > baseline[i] * prominence_factor and smoothed[i] > min_absolute:
                if not peaks or i - peaks[-1] >= min_distance:
                    peaks.append(i)

    return peaks


def _discontinuity_bounds(gray: np.ndarray,
                          horizontal: bool = True,
                          min_cell: int = 30) -> List[Tuple[int, int]]:
    """Find cell boundaries by detecting content discontinuities."""
    if horizontal:
        diff = _row_diff(gray)
    else:
        diff = _col_diff(gray)

    total = gray.shape[0] if horizontal else gray.shape[1]
    peaks = _find_peaks(diff, min_distance=min_cell, prominence_factor=1.5)

    if not peaks:
        return []

    # Convert peak positions to cell bounds
    bounds = []
    prev = 0
    for p in peaks:
        if p - prev >= min_cell:
            bounds.append((prev, p))
        prev = p
    if total - prev >= min_cell:
        bounds.append((prev, total - 1))

    return bounds


def _detect_discontinuities(
    image: np.ndarray, gray: np.ndarray,
) -> Optional[List[Tuple[int, int, np.ndarray]]]:
    """Detect images huddled together by finding content discontinuities.

    Works by analyzing row-to-row and column-to-column differences.
    Sudden spikes in these differences indicate boundaries between
    different images, even without visible seams.
    """
    h, w = gray.shape
    min_cell = max(50, min(h, w) // 8)

    # Get horizontal and vertical boundaries
    h_bounds = _discontinuity_bounds(gray, horizontal=True, min_cell=min_cell)
    v_bounds = _discontinuity_bounds(gray, horizontal=False, min_cell=min_cell)

    if not h_bounds and not v_bounds:
        return None

    # If we only found boundaries in one direction, use full span for the other
    if not h_bounds:
        h_bounds = [(0, h - 1)]
    if not v_bounds:
        v_bounds = [(0, w - 1)]

    cells = []
    for y0, y1 in h_bounds:
        for x0, x1 in v_bounds:
            cell = image[y0:y1 + 1, x0:x1 + 1]
            if cell.shape[0] >= min_cell and cell.shape[1] >= min_cell:
                cells.append((y0, x0, cell))

    return cells if len(cells) >= 2 else None


# ---------------------------------------------------------------------------
# Phase 3: regular grid detection
# ---------------------------------------------------------------------------

def _detect_lines(gray: np.ndarray, horizontal: bool = True) -> List[int]:
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    min_len = gray.shape[1] if horizontal else gray.shape[0]
    lines = cv2.HoughLinesP(
        edges, rho=1, theta=np.pi / 180,
        threshold=80, minLineLength=min_len, maxLineGap=10,
    )
    if lines is None:
        return []
    positions = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if horizontal and abs(y2 - y1) < 5:
            positions.append((y1 + y2) // 2)
        elif not horizontal and abs(x2 - x1) < 5:
            positions.append((x1 + x2) // 2)
    if not positions:
        return []
    positions.sort()
    clusters = [[positions[0]]]
    for p in positions[1:]:
        if p - clusters[-1][-1] < 10:
            clusters[-1].append(p)
        else:
            clusters.append([p])
    return [int(np.median(c)) for c in clusters]


def _score_grid_divisions(gray: np.ndarray, rows: int, cols: int) -> float:
    h, w = gray.shape
    cell_h = h // rows
    cell_w = w // cols
    band = max(2, min(cell_h, cell_w) // 20)
    total_diff = 0.0
    count = 0
    for r in range(1, rows):
        y = r * cell_h
        if y + band >= h:
            continue
        above = gray[max(0, y - band):y, :].astype(float)
        below = gray[y:y + band, :].astype(float)
        total_diff += np.mean(np.abs(above.mean(axis=0) - below.mean(axis=0)))
        count += 1
    for c in range(1, cols):
        x = c * cell_w
        if x + band >= w:
            continue
        left = gray[:, max(0, x - band):x].astype(float)
        right = gray[:, x:x + band].astype(float)
        total_diff += np.mean(np.abs(left.mean(axis=1) - right.mean(axis=1)))
        count += 1
    return total_diff / max(count, 1)


def _detect_regular_grid(
    image: np.ndarray, gray: np.ndarray,
) -> Optional[List[Tuple[int, int, np.ndarray]]]:
    h, w = gray.shape[:2]
    h_lines = _detect_lines(gray, horizontal=True)
    v_lines = _detect_lines(gray, horizontal=False)
    if h_lines and v_lines:
        h_bounds = [(0, p - 1) for p in sorted(h_lines) if p > 10]
        h_bounds += [(p, h - 1) for p in sorted(h_lines)[-1:]]
        v_bounds = [(0, p - 1) for p in sorted(v_lines) if p > 10]
        v_bounds += [(p, w - 1) for p in sorted(v_lines)[-1:]]
        # Deduplicate and build proper bounds
        h_bounds = _lines_to_bounds(h_lines, h)
        v_bounds = _lines_to_bounds(v_lines, w)
        if h_bounds and v_bounds:
            cells = []
            for y0, y1 in h_bounds:
                for x0, x1 in v_bounds:
                    cell = image[y0:y1 + 1, x0:x1 + 1]
                    cells.append((y0, x0, cell))
            if len(cells) >= 2:
                return cells

    best_grid = None
    best_score = -1
    for cols in range(2, 8):
        for rows in range(2, 8):
            cell_w = w // cols
            cell_h = h // rows
            if cell_w < 50 or cell_h < 50:
                continue
            if (w % cols) > w * 0.05 or (h % rows) > h * 0.05:
                continue
            edge = _score_grid_divisions(gray, rows, cols)
            score = edge - (rows * cols) * 0.5
            if score > best_score:
                best_score = score
                best_grid = (rows, cols)
    if best_grid:
        rows, cols = best_grid
        cell_h, cell_w = h // rows, w // cols
        return [(r * cell_h, c * cell_w,
                 image[r * cell_h:(r + 1) * cell_h, c * cell_w:(c + 1) * cell_w])
                for r in range(rows) for c in range(cols)]
    return None


def _lines_to_bounds(positions: List[int], total: int) -> List[Tuple[int, int]]:
    if not positions:
        return []
    bounds = []
    prev = 0
    for pos in sorted(positions):
        if pos > prev + 10:
            bounds.append((prev, pos - 1))
        prev = pos
    if prev < total - 10:
        bounds.append((prev, total - 1))
    return bounds


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _merge_overlapping_cells(
    cells: List[Tuple[int, int, np.ndarray]], h: int, w: int,
) -> List[Tuple[int, int, np.ndarray]]:
    """Merge cells that significantly overlap (keep larger)."""
    if len(cells) <= 1:
        return cells

    # Sort by area descending
    cells_sorted = sorted(cells, key=lambda c: c[2].shape[0] * c[2].shape[1], reverse=True)
    kept = []
    for y, x, img in cells_sorted:
        ch, cw = img.shape[:2]
        dominated = False
        for ky, kx, kimg in kept:
            kh, kw = kimg.shape[:2]
            # Check if this cell is mostly inside a kept cell
            ox0 = max(x, kx)
            oy0 = max(y, ky)
            ox1 = min(x + cw, kx + kw)
            oy1 = min(y + ch, ky + kh)
            if ox1 > ox0 and oy1 > oy0:
                overlap = (ox1 - ox0) * (oy1 - oy0)
                area = cw * ch
                if overlap > area * 0.7:
                    dominated = True
                    break
        if not dominated:
            kept.append((y, x, img))

    kept.sort(key=lambda c: (c[0], c[1]))
    return kept
