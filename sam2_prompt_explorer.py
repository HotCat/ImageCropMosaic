"""
SAM2 Prompt Explorer — interactive tool for exploring SAM2 prompt types.

Tkinter UI with radio buttons for mode selection and push buttons for actions.
Left-click to add points, drag for bounding box on the image canvas.

Keys:
  1/2/3  — cycle through masks (after Confirm)
  R      — return to prompt editing
  M      — toggle multimask_output
  Z      — undo last prompt in current mode
  C      — clear all prompts
  S      — save mask (after Confirm)
  ESC    — quit
"""

import time
import tkinter as tk
from tkinter import ttk
import cv2
import torch
import numpy as np
from PIL import Image, ImageTk
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# ── Configuration ───────────────────────────────────────────────────────
IMAGE_PATH  = "/home/hotcat/Downloads/defectDetection/data/template.png"
MODEL_CFG   = "configs/sam2.1/sam2.1_hiera_l.yaml"
CHECKPOINT  = "/home/hotcat/Downloads/sam2/checkpoints/sam2.1_hiera_large.pt"

# ── Colors (BGR for OpenCV drawing) ────────────────────────────────────
COL_POS_PT      = (0, 220, 0)
COL_NEG_PT      = (0, 0, 220)
COL_BBOX        = (0, 220, 0)
COL_RUBBER      = (0, 180, 0)
COL_MASK_FILL   = (0, 180, 255)
COL_MASK_CONTOUR= (0, 255, 255)
COL_HUD_TXT     = (240, 240, 240)
COL_HUD_WARN    = (0, 140, 255)


class SAM2Explorer:
    def __init__(self):
        # ── Image data ────────────────────────────────────────────────
        self.img_rgb = np.array(Image.open(IMAGE_PATH).convert("RGB"))
        self.img_bgr = cv2.cvtColor(self.img_rgb, cv2.COLOR_RGB2BGR)
        self.img_h, self.img_w = self.img_rgb.shape[:2]
        print(f"[INFO] Loaded {IMAGE_PATH}: {self.img_w}x{self.img_h}")

        # ── SAM2 model ────────────────────────────────────────────────
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[INFO] Loading SAM2 on {device} ...")
        sam2_model = build_sam2(MODEL_CFG, CHECKPOINT, device=device)
        self.predictor = SAM2ImagePredictor(sam2_model)

        print("[INFO] Computing image embeddings ...")
        t0 = time.perf_counter()
        self.predictor.set_image(self.img_rgb)
        print(f"[INFO] Embeddings ready ({(time.perf_counter()-t0)*1000:.0f} ms).")

        # ── Prompt state ──────────────────────────────────────────────
        self.pos_points: list = []
        self.neg_points: list = []
        self.bbox: tuple | None = None

        self.current_mode = "positive"
        self.is_dragging = False
        self.drag_start: tuple | None = None
        self.drag_current: tuple | None = None

        self.result_mode = False
        self.all_masks: np.ndarray | None = None
        self.all_scores: np.ndarray | None = None
        self.display_idx = 0
        self.multimask = True

        self.hud_message = ""
        self.hud_timer = 0.0

        # ── Display scaling ───────────────────────────────────────────
        self.scale = 1.0
        self.disp_w = self.img_w
        self.disp_h = self.img_h

        # ── Build UI ──────────────────────────────────────────────────
        self._build_ui()
        self._refresh()

    # ══════════════════════════════════════════════════════════════════
    #  UI construction
    # ══════════════════════════════════════════════════════════════════
    def _build_ui(self):
        self.root = tk.Tk()
        self.root.title("SAM2 Prompt Explorer")
        self.root.resizable(True, True)

        # Compute display scale to fit screen
        scr_w = self.root.winfo_screenwidth()
        scr_h = self.root.winfo_screenheight()
        max_w = min(self.img_w, scr_w - 60)
        max_h = min(self.img_h, scr_h - 160)
        self.scale = min(max_w / self.img_w, max_h / self.img_h, 1.0)
        self.disp_w = int(self.img_w * self.scale)
        self.disp_h = int(self.img_h * self.scale)

        # ── Canvas ────────────────────────────────────────────────────
        canvas_frame = tk.Frame(self.root)
        canvas_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(canvas_frame, width=self.disp_w, height=self.disp_h,
                                bg="black", cursor="crosshair")
        self.canvas.pack(padx=5, pady=5)

        # Mouse bindings
        self.canvas.bind("<Button-1>", self._on_click)
        self.canvas.bind("<B1-Motion>", self._on_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)
        self.canvas.bind("<Button-3>", self._on_right_click)

        # ── Control frame ─────────────────────────────────────────────
        ctrl = tk.Frame(self.root)
        ctrl.pack(side=tk.TOP, fill=tk.X, padx=8, pady=4)

        # Mode radio buttons
        tk.Label(ctrl, text="Mode:", font=("Helvetica", 10, "bold")).pack(side=tk.LEFT, padx=(0, 4))

        self.mode_var = tk.StringVar(value="positive")
        modes = [("Pos Pt", "positive"), ("Neg Pt", "negative"), ("BBox", "bbox")]
        for text, val in modes:
            tk.Radiobutton(ctrl, text=text, variable=self.mode_var, value=val,
                           command=self._on_mode_change,
                           font=("Helvetica", 10)).pack(side=tk.LEFT, padx=4)

        ttk.Separator(ctrl, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=10)

        # Action buttons
        self.btn_confirm = tk.Button(ctrl, text="Confirm", command=self._on_confirm,
                                     font=("Helvetica", 10, "bold"), width=8,
                                     bg="#4CAF50", fg="white")
        self.btn_confirm.pack(side=tk.LEFT, padx=4)

        tk.Button(ctrl, text="Clear", command=self._on_clear,
                  font=("Helvetica", 10), width=6).pack(side=tk.LEFT, padx=4)
        tk.Button(ctrl, text="Undo", command=self._on_undo,
                  font=("Helvetica", 10), width=6).pack(side=tk.LEFT, padx=4)

        ttk.Separator(ctrl, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=10)

        # MultiMask checkbox
        self.mm_var = tk.BooleanVar(value=True)
        tk.Checkbutton(ctrl, text="MultiMask", variable=self.mm_var,
                       command=self._on_mm_toggle,
                       font=("Helvetica", 10)).pack(side=tk.LEFT, padx=4)

        # ── Status bar ────────────────────────────────────────────────
        self.status_var = tk.StringVar(value="Ready — click image to add prompts")
        status = tk.Label(self.root, textvariable=self.status_var, anchor=tk.W,
                          font=("Helvetica", 9), relief=tk.SUNKEN, padx=6)
        status.pack(side=tk.BOTTOM, fill=tk.X)

        # Keyboard bindings
        self.root.bind("<Escape>", lambda e: self.root.destroy())
        self.root.bind("<Key-1>", lambda e: self._cycle_mask(0))
        self.root.bind("<Key-2>", lambda e: self._cycle_mask(1))
        self.root.bind("<Key-3>", lambda e: self._cycle_mask(2))
        self.root.bind("<Key-r>", lambda e: self._return_to_edit())
        self.root.bind("<Key-R>", lambda e: self._return_to_edit())
        self.root.bind("<Key-z>", lambda e: self._on_undo())
        self.root.bind("<Key-Z>", lambda e: self._on_undo())
        self.root.bind("<Key-c>", lambda e: self._on_clear())
        self.root.bind("<Key-C>", lambda e: self._on_clear())
        self.root.bind("<Key-s>", lambda e: self._on_save())
        self.root.bind("<Key-S>", lambda e: self._on_save())
        self.root.bind("<Key-m>", lambda e: self._toggle_mm())
        self.root.bind("<Key-M>", lambda e: self._toggle_mm())

        # Keep a reference to the displayed PhotoImage
        self._photo = None

    # ══════════════════════════════════════════════════════════════════
    #  Coordinate mapping
    # ══════════════════════════════════════════════════════════════════
    def _canvas_to_image(self, cx, cy):
        """Map canvas pixel coordinates to original image pixel coordinates."""
        ix = int(cx / self.scale)
        iy = int(cy / self.scale)
        ix = max(0, min(ix, self.img_w - 1))
        iy = max(0, min(iy, self.img_h - 1))
        return ix, iy

    # ══════════════════════════════════════════════════════════════════
    #  Image rendering
    # ══════════════════════════════════════════════════════════════════
    def _render_canvas(self):
        """Build the full display image (with overlays) and push it to the canvas."""
        canvas = self.img_bgr.copy()
        self._draw_prompts(canvas)
        self._draw_hud(canvas)

        # Scale to display size
        if self.scale != 1.0:
            canvas = cv2.resize(canvas, (self.disp_w, self.disp_h),
                                interpolation=cv2.INTER_LINEAR)

        # Convert BGR → RGB → PIL → PhotoImage
        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        self._photo = ImageTk.PhotoImage(pil_img)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self._photo)

    def _draw_prompts(self, canvas):
        if self.bbox is not None:
            x0, y0, x1, y1 = self.bbox
            cv2.rectangle(canvas, (x0, y0), (x1, y1), COL_BBOX, 2)
            cv2.putText(canvas, "BBox", (x0 + 4, y0 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, COL_BBOX, 1, cv2.LINE_AA)

        if self.is_dragging and self.drag_start and self.drag_current:
            cv2.rectangle(canvas, self.drag_start, self.drag_current,
                          COL_RUBBER, 1, cv2.LINE_AA)

        for (px, py) in self.pos_points:
            cv2.circle(canvas, (px, py), 7, COL_POS_PT, -1)
            cv2.circle(canvas, (px, py), 7, (0, 0, 0), 1)
            cv2.drawMarker(canvas, (px, py), (0, 0, 0), cv2.MARKER_CROSS, 12, 1)

        for (px, py) in self.neg_points:
            cv2.circle(canvas, (px, py), 7, COL_NEG_PT, -1)
            cv2.circle(canvas, (px, py), 7, (0, 0, 0), 1)
            cv2.drawMarker(canvas, (px, py), (255, 255, 255),
                           cv2.MARKER_TILTED_CROSS, 12, 2)

    def _draw_hud(self, canvas):
        bar_h = 28
        overlay = canvas[self.img_h - bar_h:self.img_h, :self.img_w].copy()
        cv2.rectangle(overlay, (0, 0), (self.img_w, bar_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.65,
                        canvas[self.img_h - bar_h:self.img_h, :self.img_w],
                        0.35, 0,
                        dst=canvas[self.img_h - bar_h:self.img_h, :self.img_w])

        mode_color = {"positive": COL_POS_PT, "negative": COL_NEG_PT,
                      "bbox": COL_BBOX}[self.current_mode]
        mode_label = {"positive": "POS PT", "negative": "NEG PT",
                      "bbox": "BBOX"}[self.current_mode]
        cv2.putText(canvas, f"  Mode: {mode_label}", (6, self.img_h - 9),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, mode_color, 2, cv2.LINE_AA)

        right = (f"Pos:{len(self.pos_points)}  Neg:{len(self.neg_points)}  "
                 f"BBox:{'Y' if self.bbox else 'N'}")
        tw = cv2.getTextSize(right, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0][0]
        cv2.putText(canvas, right, (self.img_w - tw - 10, self.img_h - 9),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COL_HUD_TXT, 1, cv2.LINE_AA)

        if self.hud_message and time.perf_counter() < self.hud_timer:
            cv2.putText(canvas, self.hud_message, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, COL_HUD_WARN, 2, cv2.LINE_AA)

    def _refresh(self):
        self._render_canvas()
        self._update_status()

    def _update_status(self):
        mode = self.current_mode.upper()
        n_pos = len(self.pos_points)
        n_neg = len(self.neg_points)
        has_box = "Y" if self.bbox else "N"
        mm = "ON" if self.multimask else "OFF"
        self.status_var.set(
            f"Mode: {mode}  |  Pos: {n_pos}  Neg: {n_neg}  BBox: {has_box}  |  "
            f"MultiMask: {mm}  |  Right-click to undo last point"
        )

    def _set_hud(self, msg, duration=2.0):
        self.hud_message = msg
        self.hud_timer = time.perf_counter() + duration

    # ══════════════════════════════════════════════════════════════════
    #  Result display
    # ══════════════════════════════════════════════════════════════════
    def _show_result(self, idx=0):
        if self.all_masks is None:
            return
        mask = self.all_masks[idx]
        score = self.all_scores[idx]

        canvas = self.img_bgr.copy()
        mask_bool = mask > 0.5
        overlay = canvas.copy()
        overlay[mask_bool] = COL_MASK_FILL
        cv2.addWeighted(overlay, 0.4, canvas, 0.6, 0, dst=canvas)

        mask_u8 = (mask_bool.astype(np.uint8)) * 255
        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(canvas, contours, -1, COL_MASK_CONTOUR, 2)

        self._draw_prompts(canvas)

        lines = [
            f"Mask {idx+1}/{len(self.all_masks)}  score: {score:.3f}",
            "1/2/3: cycle  R: back to edit  S: save",
        ]
        for i, line in enumerate(lines):
            cv2.putText(canvas, line, (10, 25 + i * 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, COL_HUD_TXT, 2, cv2.LINE_AA)

        # Scale and display
        if self.scale != 1.0:
            canvas = cv2.resize(canvas, (self.disp_w, self.disp_h),
                                interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        self._photo = ImageTk.PhotoImage(pil_img)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self._photo)

        self.status_var.set(
            f"Result: Mask {idx+1}/{len(self.all_masks)}  score={score:.3f}  |  "
            f"1/2/3: cycle  R: edit  S: save  ESC: quit"
        )

    # ══════════════════════════════════════════════════════════════════
    #  SAM2 inference
    # ══════════════════════════════════════════════════════════════════
    def _on_confirm(self):
        if self.result_mode:
            return

        point_coords_list = []
        point_labels_list = []
        for (px, py) in self.pos_points:
            point_coords_list.append([px, py])
            point_labels_list.append(1)
        for (px, py) in self.neg_points:
            point_coords_list.append([px, py])
            point_labels_list.append(0)

        point_coords = (np.array(point_coords_list, dtype=np.float32)
                        if point_coords_list else None)
        point_labels = (np.array(point_labels_list, dtype=np.int32)
                        if point_labels_list else None)
        box_input = (np.array(self.bbox, dtype=np.float32)[None, :]
                     if self.bbox is not None else None)

        if point_coords is None and box_input is None:
            self._set_hud("No prompts! Add points or bbox first.", 3.0)
            self._refresh()
            return

        print(f"[INFO] Running SAM2: {len(self.pos_points)} pos, "
              f"{len(self.neg_points)} neg, bbox={self.bbox}, multimask={self.multimask}")

        self.status_var.set("Running SAM2 inference...")
        self.root.update_idletasks()

        t0 = time.perf_counter()
        masks, scores, _ = self.predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            box=box_input,
            multimask_output=self.multimask,
        )
        elapsed = (time.perf_counter() - t0) * 1000
        print(f"[INFO] SAM2 inference: {elapsed:.1f} ms  scores: {scores}")

        self.all_masks = masks
        self.all_scores = scores
        self.display_idx = int(np.argmax(scores))
        self.result_mode = True
        self._show_result(self.display_idx)

    def _on_clear(self):
        self.pos_points.clear()
        self.neg_points.clear()
        self.bbox = None
        self.result_mode = False
        self.all_masks = None
        self.all_scores = None
        self._refresh()
        print("[INFO] Cleared all prompts.")

    def _on_undo(self):
        if self.result_mode:
            return
        mode = self.current_mode
        if mode == "positive" and self.pos_points:
            self.pos_points.pop()
        elif mode == "negative" and self.neg_points:
            self.neg_points.pop()
        elif mode == "bbox" and self.bbox is not None:
            self.bbox = None
        else:
            self._set_hud("Nothing to undo", 1.5)
            self._refresh()
            return
        self._refresh()

    def _on_save(self):
        if self.all_masks is None:
            return
        mask = self.all_masks[self.display_idx]
        mask_u8 = (mask > 0.5).astype(np.uint8) * 255
        path = "explorer_mask.png"
        cv2.imwrite(path, mask_u8)
        print(f"[INFO] Saved mask to {path}")
        self._set_hud(f"Saved to {path}", 2.0)
        if self.result_mode:
            self._show_result(self.display_idx)

    def _on_mode_change(self):
        self.current_mode = self.mode_var.get()
        self._update_status()

    def _on_mm_toggle(self):
        self.multimask = self.mm_var.get()
        print(f"[INFO] multimask_output = {'ON' if self.multimask else 'OFF'}")

    def _toggle_mm(self):
        self.multimask = not self.multimask
        self.mm_var.set(self.multimask)

    def _cycle_mask(self, idx):
        if not self.result_mode or self.all_masks is None:
            return
        if idx < len(self.all_masks):
            self.display_idx = idx
            self._show_result(idx)

    def _return_to_edit(self):
        self.result_mode = False
        self._refresh()

    # ══════════════════════════════════════════════════════════════════
    #  Mouse events
    # ══════════════════════════════════════════════════════════════════
    def _on_click(self, event):
        if self.result_mode:
            return
        ix, iy = self._canvas_to_image(event.x, event.y)
        mode = self.current_mode

        if mode in ("positive", "negative"):
            if mode == "positive":
                self.pos_points.append((ix, iy))
            else:
                self.neg_points.append((ix, iy))
            self._refresh()

        elif mode == "bbox":
            self.is_dragging = True
            self.drag_start = (ix, iy)
            self.drag_current = (ix, iy)

    def _on_drag(self, event):
        if not self.is_dragging or self.result_mode:
            return
        if self.current_mode != "bbox":
            return
        ix, iy = self._canvas_to_image(event.x, event.y)
        self.drag_current = (ix, iy)
        self._refresh()

    def _on_release(self, event):
        if not self.is_dragging or self.result_mode:
            return
        if self.current_mode != "bbox":
            return
        self.is_dragging = False
        x0, y0 = self.drag_start
        x1, y1 = self.drag_current
        self.drag_start = None
        self.drag_current = None

        if abs(x1 - x0) < 5 or abs(y1 - y0) < 5:
            self._set_hud("BBox too small, discarded", 2.0)
            self._refresh()
            return

        self.bbox = (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))
        print(f"[INFO] BBox set: {self.bbox}")
        self._refresh()

    def _on_right_click(self, event):
        """Right-click undoes the last prompt in the current mode."""
        self._on_undo()

    # ══════════════════════════════════════════════════════════════════
    #  Run
    # ══════════════════════════════════════════════════════════════════
    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    app = SAM2Explorer()
    app.run()
