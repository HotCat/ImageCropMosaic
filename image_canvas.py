from typing import Optional, Tuple, List

import cv2
import numpy as np
from PySide6.QtWidgets import (
    QGraphicsView, QGraphicsScene, QGraphicsPixmapItem,
    QGraphicsItem, QGraphicsEllipseItem, QGraphicsRectItem,
    QGraphicsPathItem
)
from PySide6.QtCore import Qt, Signal, QPointF, QRectF
from PySide6.QtGui import (
    QPixmap, QImage, QPen, QBrush, QColor, QPainter,
    QPainterPath
)

from selection_model import SelectionModel, SelectionType


class PointMarker(QGraphicsEllipseItem):
    POSITIVE_COLOR = QColor(0, 220, 0, 220)
    NEGATIVE_COLOR = QColor(220, 0, 0, 220)
    RADIUS = 7

    def __init__(self, x: int, y: int, is_positive: bool, parent=None):
        super().__init__(
            x - self.RADIUS, y - self.RADIUS,
            self.RADIUS * 2, self.RADIUS * 2, parent
        )
        color = self.POSITIVE_COLOR if is_positive else self.NEGATIVE_COLOR
        self.setBrush(QBrush(color))
        self.setPen(QPen(Qt.white, 2))
        self.setZValue(100)


class MaskOverlay(QGraphicsItem):
    MASK_COLOR = QColor(255, 180, 0, 100)
    CONTOUR_COLOR = QColor(255, 255, 0)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._mask: Optional[np.ndarray] = None
        self._mask_bool: Optional[np.ndarray] = None
        self._overlay_pixmap: Optional[QPixmap] = None
        self._contour_paths: List[QPainterPath] = []
        self._last_size: Tuple[int, int] = (0, 0)
        self.setZValue(50)

    def set_mask(self, mask: Optional[np.ndarray]):
        # Remember size before clearing so boundingRect stays valid for repaint
        if self._mask is not None:
            self._last_size = (self._mask.shape[1], self._mask.shape[0])

        self._mask = None
        self._mask_bool = None
        self._overlay_pixmap = None
        self._contour_paths = []

        if mask is not None:
            self._mask = mask
            self._mask_bool = mask > 0.5 if mask.dtype != bool else mask
            self._last_size = (mask.shape[1], mask.shape[0])
            self._build_overlay()
            self._build_contours()

        self.update()

    def _build_overlay(self):
        if self._mask_bool is None:
            return
        h, w = self._mask_bool.shape[:2]
        overlay = np.zeros((h, w, 4), dtype=np.uint8)
        overlay[self._mask_bool] = [
            self.MASK_COLOR.red(),
            self.MASK_COLOR.green(),
            self.MASK_COLOR.blue(),
            self.MASK_COLOR.alpha()
        ]
        qimg = QImage(overlay.data, w, h, w * 4, QImage.Format_RGBA8888).copy()
        self._overlay_pixmap = QPixmap.fromImage(qimg)

    def _build_contours(self):
        self._contour_paths = []
        if self._mask_bool is None:
            return

        mask_u8 = self._mask_bool.astype(np.uint8) * 255
        contours, _ = cv2.findContours(
            mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        for contour in contours:
            if len(contour) < 2:
                continue
            path = QPainterPath()
            pts = contour.reshape(-1, 2)
            path.moveTo(float(pts[0][0]), float(pts[0][1]))
            for pt in pts[1:]:
                path.lineTo(float(pt[0]), float(pt[1]))
            path.closeSubpath()
            self._contour_paths.append(path)

    def boundingRect(self) -> QRectF:
        w, h = self._last_size
        if w > 0 and h > 0:
            return QRectF(0, 0, w, h)
        return QRectF()

    def paint(self, painter: QPainter, option, widget):
        if self._overlay_pixmap is None:
            return
        painter.drawPixmap(0, 0, self._overlay_pixmap)

        if self._contour_paths:
            painter.setPen(QPen(self.CONTOUR_COLOR, 2))
            painter.setBrush(Qt.NoBrush)
            for path in self._contour_paths:
                painter.drawPath(path)


class ImageCanvas(QGraphicsView):
    clicked = Signal(int, int)
    bbox_selected = Signal(int, int, int, int)

    MODE_VIEW = 0
    MODE_SAM2_POS = 1
    MODE_SAM2_NEG = 2
    MODE_SAM2_BBOX = 3
    MODE_CROP_BBOX = 4

    def __init__(self, parent=None):
        super().__init__(parent)

        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)

        self.setRenderHint(QPainter.Antialiasing)
        self.setRenderHint(QPainter.SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.NoDrag)
        self.setMouseTracking(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setBackgroundBrush(QBrush(QColor(40, 40, 40)))

        self._image_item: Optional[QGraphicsPixmapItem] = None
        self._mask_overlay = MaskOverlay()
        self._selection_rect: Optional[QGraphicsRectItem] = None
        self._rubber_band: Optional[QGraphicsRectItem] = None
        self._drag_rubber: Optional[QGraphicsRectItem] = None
        self._point_markers: List[PointMarker] = []

        self._scene.addItem(self._mask_overlay)

        self._is_dragging = False
        self._is_panning = False
        self._pan_start: Optional[QPointF] = None
        self._drag_start: Optional[Tuple[int, int]] = None
        self._current_mode = self.MODE_VIEW
        self._image_size: Optional[Tuple[int, int]] = None

    def set_image(self, image_rgb: np.ndarray):
        h, w = image_rgb.shape[:2]
        self._image_size = (w, h)
        qimg = QImage(image_rgb.data, w, h, w * 3, QImage.Format_RGB888).copy()
        pixmap = QPixmap.fromImage(qimg)

        if self._image_item is None:
            self._image_item = self._scene.addPixmap(pixmap)
        else:
            self._image_item.setPixmap(pixmap)

        self._image_item.setZValue(0)
        self.setSceneRect(0, 0, w, h)
        self.fitInView(self._image_item, Qt.KeepAspectRatio)

    def set_mode(self, mode: int):
        self._current_mode = mode
        if mode in (self.MODE_SAM2_POS, self.MODE_SAM2_NEG,
                    self.MODE_SAM2_BBOX, self.MODE_CROP_BBOX):
            self.setCursor(Qt.CrossCursor)
        else:
            self.setCursor(Qt.ArrowCursor)

    def update_selection(self, selection_model: SelectionModel):
        # Clear all visual markers first
        for marker in self._point_markers:
            self._scene.removeItem(marker)
        self._point_markers.clear()

        if self._rubber_band is not None:
            self._rubber_band.hide()
        self._hide_selection_rect()

        # Draw prompt markers (points and bbox prompt)
        prompts = selection_model.prompts
        for x, y in prompts.positive_points:
            marker = PointMarker(x, y, is_positive=True)
            self._scene.addItem(marker)
            self._point_markers.append(marker)
        for x, y in prompts.negative_points:
            marker = PointMarker(x, y, is_positive=False)
            self._scene.addItem(marker)
            self._point_markers.append(marker)

        if prompts.bbox is not None:
            x0, y0, x1, y1 = prompts.bbox
            self._ensure_rubber_band()
            self._rubber_band.setRect(QRectF(
                min(x0, x1), min(y0, y1),
                abs(x1 - x0), abs(y1 - y0)
            ))
            self._rubber_band.show()

        # Draw selection result (SAM2 mask or manual bbox)
        if selection_model.selection_type == SelectionType.SAM2_MASK:
            mask = selection_model.selected_mask
            self._mask_overlay.set_mask(mask)
        elif selection_model.selection_type == SelectionType.MANUAL_BBOX:
            bbox = selection_model.manual_bbox
            if bbox:
                self._ensure_selection_rect()
                x0, y0, x1, y1 = bbox
                self._selection_rect.setRect(QRectF(
                    min(x0, x1), min(y0, y1),
                    abs(x1 - x0), abs(y1 - y0)
                ))
                self._selection_rect.show()
            self._mask_overlay.set_mask(None)
        else:
            # No selection — clear everything
            self._mask_overlay.set_mask(None)

    def _ensure_rubber_band(self):
        if self._rubber_band is None:
            self._rubber_band = QGraphicsRectItem()
            self._rubber_band.setPen(QPen(QColor(0, 180, 0), 2, Qt.DashLine))
            self._rubber_band.setBrush(Qt.NoBrush)
            self._rubber_band.setZValue(90)
            self._scene.addItem(self._rubber_band)

    def _ensure_selection_rect(self):
        if self._selection_rect is None:
            self._selection_rect = QGraphicsRectItem()
            self._selection_rect.setBrush(QBrush(QColor(0, 255, 0, 60)))
            self._selection_rect.setPen(QPen(QColor(0, 255, 0), 2))
            self._selection_rect.setZValue(50)
            self._scene.addItem(self._selection_rect)

    def _ensure_drag_rubber(self):
        if self._drag_rubber is None:
            self._drag_rubber = QGraphicsRectItem()
            self._drag_rubber.setPen(QPen(QColor(0, 180, 0), 1, Qt.DashLine))
            self._drag_rubber.setBrush(QBrush(QColor(0, 180, 0, 30)))
            self._drag_rubber.setZValue(95)
            self._scene.addItem(self._drag_rubber)

    def _hide_selection_rect(self):
        if self._selection_rect is not None:
            self._selection_rect.hide()

    def _map_to_image(self, scene_pos: QPointF) -> Tuple[int, int]:
        x, y = int(scene_pos.x()), int(scene_pos.y())
        if self._image_size:
            w, h = self._image_size
            x = max(0, min(x, w - 1))
            y = max(0, min(y, h - 1))
        return x, y

    def mousePressEvent(self, event):
        if event.button() != Qt.LeftButton:
            super().mousePressEvent(event)
            return

        # Ctrl/Cmd+LeftClick = pan view (works even when image fits viewport)
        if event.modifiers() & (Qt.ControlModifier | Qt.MetaModifier):
            self._is_panning = True
            self._pan_start = self.mapToScene(event.pos())
            self.setCursor(Qt.ClosedHandCursor)
            return

        scene_pos = self.mapToScene(event.pos())
        x, y = self._map_to_image(scene_pos)

        if self._current_mode in (self.MODE_SAM2_BBOX, self.MODE_CROP_BBOX):
            self._is_dragging = True
            self._drag_start = (x, y)
            self._ensure_drag_rubber()
            self._drag_rubber.setRect(QRectF(x, y, 0, 0))
            self._drag_rubber.show()
        elif self._current_mode in (self.MODE_SAM2_POS, self.MODE_SAM2_NEG):
            self.clicked.emit(x, y)

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._is_panning and self._pan_start is not None:
            current = self.mapToScene(event.pos())
            dx = current.x() - self._pan_start.x()
            dy = current.y() - self._pan_start.y()
            self.translate(dx, dy)
            return

        if self._is_dragging and self._drag_start is not None:
            scene_pos = self.mapToScene(event.pos())
            x, y = self._map_to_image(scene_pos)
            sx, sy = self._drag_start
            self._drag_rubber.setRect(QRectF(
                min(sx, x), min(sy, y),
                abs(x - sx), abs(y - sy)
            ))
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() != Qt.LeftButton:
            super().mouseReleaseEvent(event)
            return

        if self._is_panning:
            self._is_panning = False
            self._pan_start = None
            # Restore cursor for current mode
            if self._current_mode in (self.MODE_SAM2_POS, self.MODE_SAM2_NEG,
                                       self.MODE_SAM2_BBOX, self.MODE_CROP_BBOX):
                self.setCursor(Qt.CrossCursor)
            else:
                self.setCursor(Qt.ArrowCursor)
            return

        if self._is_dragging and self._drag_start is not None:
            scene_pos = self.mapToScene(event.pos())
            x, y = self._map_to_image(scene_pos)
            sx, sy = self._drag_start

            if abs(x - sx) > 5 and abs(y - sy) > 5:
                self.bbox_selected.emit(sx, sy, x, y)

            self._is_dragging = False
            self._drag_start = None
            if self._drag_rubber is not None:
                self._drag_rubber.hide()

        super().mouseReleaseEvent(event)

    def wheelEvent(self, event):
        factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
        self.scale(factor, factor)

    def fit_to_view(self):
        if self._image_item:
            self.fitInView(self._image_item, Qt.KeepAspectRatio)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._image_item:
            self.fitInView(self._image_item, Qt.KeepAspectRatio)