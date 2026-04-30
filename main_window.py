from pathlib import Path
from typing import Optional

from PySide6.QtWidgets import (
    QMainWindow, QToolBar, QStatusBar, QFileDialog,
    QMessageBox, QWidget, QVBoxLayout, QLabel, QSpinBox,
    QComboBox, QDoubleSpinBox
)
from PySide6.QtCore import Qt, Slot
from PySide6.QtGui import QAction, QActionGroup, QKeySequence

from image_canvas import ImageCanvas
from app_controller import AppController, AppMode
from selection_model import SelectionType
from config import (
    SAM2_CONFIG, CHECKPOINT_PATH, SAVE_DIR,
    SR_RESOLUTION_CHOICES, SR_DEFAULT_RESOLUTION,
    SR_DEFAULT_STRENGTH, SR_DEFAULT_GUIDANCE_SCALE,
    SR_DEFAULT_NUM_STEPS, SR_DEFAULT_PROMPT, SR_DEFAULT_NEGATIVE_PROMPT,
)


class MainWindow(QMainWindow):
    def __init__(self, initial_image: Optional[str] = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Image Crop Tool")
        self.setMinimumSize(900, 700)
        self.resize(1200, 900)

        self._controller = AppController(self)

        self._setup_ui()
        self._setup_toolbar()
        self._setup_statusbar()
        self._connect_signals()

        self._load_sam2_model()
        self._controller.load_sr_model()

        if initial_image:
            self._controller.load_image(initial_image)

    def _setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._canvas = ImageCanvas()
        layout.addWidget(self._canvas)

    def _setup_toolbar(self):
        toolbar = QToolBar("Main Toolbar")
        toolbar.setMovable(False)
        toolbar.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.addToolBar(toolbar)

        # -- File --
        self._open_action = QAction("Open", self)
        self._open_action.setShortcut(QKeySequence.Open)
        self._open_action.setToolTip("Open image file (Cmd+O)")
        toolbar.addAction(self._open_action)

        toolbar.addSeparator()

        # -- SAM2 prompt modes (exclusive radio group) --
        self._mode_group = QActionGroup(self)
        self._mode_group.setExclusive(True)

        self._pos_action = QAction("+Point", self, checkable=True)
        self._pos_action.setShortcut("P")
        self._pos_action.setToolTip("SAM2 positive point (P)")
        self._pos_action.setChecked(True)
        self._mode_group.addAction(self._pos_action)
        toolbar.addAction(self._pos_action)

        self._neg_action = QAction("-Point", self, checkable=True)
        self._neg_action.setShortcut("N")
        self._neg_action.setToolTip("SAM2 negative point (N)")
        self._mode_group.addAction(self._neg_action)
        toolbar.addAction(self._neg_action)

        self._sam2_bbox_action = QAction("SAM2 Box", self, checkable=True)
        self._sam2_bbox_action.setShortcut("B")
        self._sam2_bbox_action.setToolTip("SAM2 bounding box (B)")
        self._mode_group.addAction(self._sam2_bbox_action)
        toolbar.addAction(self._sam2_bbox_action)

        self._confirm_action = QAction("Confirm", self)
        self._confirm_action.setShortcut("Return")
        self._confirm_action.setToolTip("Run SAM2 prediction (Enter)")
        toolbar.addAction(self._confirm_action)

        self._undo_action = QAction("Undo", self)
        self._undo_action.setShortcut(QKeySequence.Undo)
        self._undo_action.setToolTip("Undo last prompt (Cmd+Z)")
        toolbar.addAction(self._undo_action)

        self._clear_action = QAction("Clear", self)
        self._clear_action.setShortcut("Escape")
        self._clear_action.setToolTip("Clear all prompts and mask (Esc)")
        toolbar.addAction(self._clear_action)

        self._multimask_action = QAction("MultiMask", self, checkable=True)
        self._multimask_action.setChecked(True)
        self._multimask_action.setToolTip("Return 3 candidate masks (1/2/3 to cycle)")
        toolbar.addAction(self._multimask_action)

        toolbar.addSeparator()

        # -- Crop mode --
        self._crop_action = QAction("Crop", self, checkable=True)
        self._crop_action.setShortcut("C")
        self._crop_action.setToolTip("Crop bounding box (C)")
        self._mode_group.addAction(self._crop_action)
        toolbar.addAction(self._crop_action)

        toolbar.addSeparator()

        # -- Mosaic --
        self._mosaic_action = QAction("Mosaic", self)
        self._mosaic_action.setShortcut("M")
        self._mosaic_action.setToolTip("Apply mosaic to selection (M)")
        toolbar.addAction(self._mosaic_action)

        toolbar.addWidget(QLabel(" Block:"))
        self._block_size_spin = QSpinBox()
        self._block_size_spin.setRange(2, 100)
        self._block_size_spin.setValue(10)
        self._block_size_spin.setToolTip(
            "Mosaic block size in pixels\n"
            "Small (2-5): fine grain\n"
            "Medium (10-20): standard\n"
            "Large (30+): heavy pixelation"
        )
        toolbar.addWidget(self._block_size_spin)

        toolbar.addSeparator()

        # -- Super Resolution --
        self._sr_action = QAction("SR", self)
        self._sr_action.setShortcut("S")
        self._sr_action.setToolTip("Run Super Resolution on current image (S)")
        toolbar.addAction(self._sr_action)

        toolbar.addWidget(QLabel(" Res:"))
        self._sr_resolution_combo = QComboBox()
        for r in SR_RESOLUTION_CHOICES:
            self._sr_resolution_combo.addItem(f"{r}", r)
        default_idx = SR_RESOLUTION_CHOICES.index(SR_DEFAULT_RESOLUTION)
        self._sr_resolution_combo.setCurrentIndex(default_idx)
        self._sr_resolution_combo.setToolTip("Target resolution (longest side)")
        toolbar.addWidget(self._sr_resolution_combo)

        toolbar.addWidget(QLabel(" Str:"))
        self._sr_strength_spin = QDoubleSpinBox()
        self._sr_strength_spin.setRange(0.1, 1.0)
        self._sr_strength_spin.setSingleStep(0.05)
        self._sr_strength_spin.setValue(SR_DEFAULT_STRENGTH)
        self._sr_strength_spin.setDecimals(2)
        self._sr_strength_spin.setToolTip(
            "Denoising strength\n"
            "0.3-0.5: subtle enhancement\n"
            "0.5-0.7: stronger detail\n"
            ">0.7: may hallucinate"
        )
        toolbar.addWidget(self._sr_strength_spin)

        toolbar.addWidget(QLabel(" Steps:"))
        self._sr_steps_spin = QSpinBox()
        self._sr_steps_spin.setRange(10, 75)
        self._sr_steps_spin.setValue(SR_DEFAULT_NUM_STEPS)
        self._sr_steps_spin.setToolTip("Inference steps (more = better but slower)")
        toolbar.addWidget(self._sr_steps_spin)

        toolbar.addSeparator()

        # -- Save --
        self._save_action = QAction("Save", self)
        self._save_action.setShortcut(QKeySequence.Save)
        self._save_action.setToolTip("Save selection region to file (Cmd+S)")
        toolbar.addAction(self._save_action)

    def _setup_statusbar(self):
        self._statusbar = QStatusBar()
        self.setStatusBar(self._statusbar)
        self._sam2_status = QLabel("SAM2: Not loaded")
        self._sr_status = QLabel("SR: Not loaded")
        self._statusbar.addPermanentWidget(self._sam2_status)
        self._statusbar.addPermanentWidget(self._sr_status)

    def _connect_signals(self):
        # Canvas -> Controller
        self._canvas.clicked.connect(self._controller.handle_canvas_click)
        self._canvas.bbox_selected.connect(self._controller.handle_bbox_drag)

        # Controller -> UI
        self._controller.mode_changed.connect(self._on_mode_changed)
        self._controller.image_loaded.connect(self._on_image_loaded)
        self._controller.image_changed.connect(self._on_image_changed)
        self._controller.selection_changed.connect(self._on_selection_changed)
        self._controller.prompts_changed.connect(self._on_prompts_changed)
        self._controller.sam2_loading.connect(self._on_sam2_loading)
        self._controller.sam2_loaded.connect(self._on_sam2_loaded)
        self._controller.sam2_error.connect(self._on_sam2_error)
        self._controller.sr_loading.connect(self._on_sr_loading)
        self._controller.sr_loaded.connect(self._on_sr_loaded)
        self._controller.sr_error.connect(self._on_sr_error)
        self._controller.status_message.connect(self._statusbar.showMessage)

        # Toolbar actions
        self._open_action.triggered.connect(self._on_open)
        self._pos_action.triggered.connect(lambda: self._controller.set_mode(AppMode.SAM2_POSITIVE))
        self._neg_action.triggered.connect(lambda: self._controller.set_mode(AppMode.SAM2_NEGATIVE))
        self._sam2_bbox_action.triggered.connect(lambda: self._controller.set_mode(AppMode.SAM2_BBOX))
        self._crop_action.triggered.connect(lambda: self._controller.set_mode(AppMode.CROP_BBOX))
        self._confirm_action.triggered.connect(self._controller.confirm_prompts)
        self._undo_action.triggered.connect(self._controller.undo_last_prompt)
        self._clear_action.triggered.connect(self._controller.clear_all)
        self._multimask_action.toggled.connect(self._on_multimask_toggled)
        self._mosaic_action.triggered.connect(self._on_mosaic)
        self._sr_action.triggered.connect(self._on_sr)
        self._save_action.triggered.connect(self._on_save)

    def _load_sam2_model(self):
        if CHECKPOINT_PATH.exists():
            self._controller.load_sam2_model(SAM2_CONFIG, str(CHECKPOINT_PATH))
        else:
            self._sam2_status.setText(f"SAM2: Checkpoint not found")

    @Slot()
    def _on_open(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Image", "",
            "Images (*.png *.jpg *.jpeg *.bmp *.tiff *.webp);;All Files (*)"
        )
        if path:
            self._controller.load_image(path)

    @Slot()
    def _on_mosaic(self):
        block_size = self._block_size_spin.value()
        self._controller.apply_mosaic(block_size)

    @Slot()
    def _on_sr(self):
        if not self._controller.sr_service.is_loaded:
            self._statusbar.showMessage("SR model not loaded yet")
            return
        if not self._controller.image_model.is_loaded:
            self._statusbar.showMessage("No image loaded")
            return
        if self._controller.sr_service.is_upscaling:
            self._statusbar.showMessage("SR already running")
            return

        params = {
            "resolution": self._sr_resolution_combo.currentData(),
            "strength": self._sr_strength_spin.value(),
            "guidance_scale": SR_DEFAULT_GUIDANCE_SCALE,
            "num_inference_steps": self._sr_steps_spin.value(),
            "max_tile_size": 1024,
            "prompt": SR_DEFAULT_PROMPT,
            "negative_prompt": SR_DEFAULT_NEGATIVE_PROMPT,
        }
        self._controller.apply_super_resolution(params)

    @Slot()
    def _on_save(self):
        if not self._controller.selection_model.has_selection:
            QMessageBox.warning(self, "No Selection", "Please make a selection first.")
            return

        SAVE_DIR.mkdir(parents=True, exist_ok=True)

        path, _ = QFileDialog.getSaveFileName(
            self, "Save Selection",
            str(SAVE_DIR / "selection.png"),
            "PNG (*.png);;JPEG (*.jpg);;All Files (*)"
        )
        if path:
            self._controller.save_selection(path)

    @Slot(bool)
    def _on_multimask_toggled(self, checked: bool):
        self._controller.multimask = checked

    @Slot(object)
    def _on_mode_changed(self, mode: AppMode):
        mode_to_canvas = {
            AppMode.VIEW: ImageCanvas.MODE_VIEW,
            AppMode.SAM2_POSITIVE: ImageCanvas.MODE_SAM2_POS,
            AppMode.SAM2_NEGATIVE: ImageCanvas.MODE_SAM2_NEG,
            AppMode.SAM2_BBOX: ImageCanvas.MODE_SAM2_BBOX,
            AppMode.CROP_BBOX: ImageCanvas.MODE_CROP_BBOX,
        }
        self._canvas.set_mode(mode_to_canvas.get(mode, ImageCanvas.MODE_VIEW))

    @Slot()
    def _on_image_loaded(self):
        image = self._controller.image_model.current_image_rgb
        self._canvas.set_image(image)
        self._canvas.fit_to_view()
        name = Path(self._controller.image_model.image_path).name
        self.setWindowTitle(f"Image Crop Tool - {name}")

    @Slot()
    def _on_image_changed(self):
        image = self._controller.image_model.current_image_rgb
        self._canvas.set_image(image)
        self._canvas.fit_to_view()

    @Slot()
    def _on_selection_changed(self):
        self._canvas.update_selection(self._controller.selection_model)

    @Slot()
    def _on_prompts_changed(self):
        self._canvas.update_selection(self._controller.selection_model)

    @Slot(str)
    def _on_sam2_loading(self, message: str):
        self._sam2_status.setText(f"SAM2: {message}")

    @Slot()
    def _on_sam2_loaded(self):
        self._sam2_status.setText("SAM2: Ready")

    @Slot(str)
    def _on_sam2_error(self, error: str):
        self._sam2_status.setText("SAM2: Error")
        QMessageBox.critical(self, "SAM2 Error", error)

    @Slot(str)
    def _on_sr_loading(self, message: str):
        self._sr_status.setText(f"SR: {message}")

    @Slot()
    def _on_sr_loaded(self):
        self._sr_status.setText("SR: Ready")

    @Slot(str)
    def _on_sr_error(self, error: str):
        self._sr_status.setText("SR: Error")
        self._statusbar.showMessage(f"SR Error: {error}")

    def keyPressEvent(self, event):
        key = event.key()
        if key in (Qt.Key_1, Qt.Key_2, Qt.Key_3):
            idx = key - Qt.Key_1
            self._controller.cycle_sam2_mask(idx)
        else:
            super().keyPressEvent(event)