"""
Annotation Reviewer GUI Tool

A Tkinter-based tool for reviewing, modifying, and filtering polygon annotations.
"""

import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
from PIL import Image, ImageTk, ImageDraw
import os
import glob
import cv2
import numpy as np
import yaml
from pathlib import Path
from typing import Callable, Optional, List, Dict, Any, Tuple


CONFIG_DIR = Path(__file__).parent.parent.parent.parent / "configs"


class FilterDialog(tk.Toplevel):
    """Dialog for filtering unwanted annotations based on criteria."""

    def __init__(self, parent: tk.Tk, callback: Callable):
        super().__init__(parent)
        self.callback = callback
        self.title("Filter Unwanted Annotations")
        self.geometry("400x500")

        tk.Label(
            self,
            text="Remove annotations outside these bounds:",
            font=("Arial", 10, "bold"),
        ).pack(pady=10)
        tk.Label(self, text="(Values in % relative to image size)", font=("Arial", 9)).pack(pady=0)

        self.vars: Dict[str, tk.StringVar] = {}
        self.checks: Dict[str, tk.BooleanVar] = {}
        self.entries: Dict[str, tk.Entry] = {}
        self.config_file = CONFIG_DIR / "filter_config.yaml"

        defaults = {
            "max_ratio": 1.2,
            "min_area": 0.05,
            "max_area": 0.6,
            "min_width": 1.0,
            "max_width": 90.0,
            "min_height": 1.0,
            "max_height": 90.0,
        }

        saved_prefs = self.load_prefs()

        criteria = [
            ("Max Aspect Ratio (Long/Short)", "max_ratio"),
            ("Min Area (%)", "min_area"),
            ("Max Area (%)", "max_area"),
            ("Min Width (%)", "min_width"),
            ("Max Width (%)", "max_width"),
            ("Min Height (%)", "min_height"),
            ("Max Height (%)", "max_height"),
        ]

        for label, key in criteria:
            frame = tk.Frame(self)
            frame.pack(fill=tk.X, padx=20, pady=5)

            is_checked = saved_prefs.get(f"{key}_checked", True)
            value = saved_prefs.get(key, defaults[key])

            check_var = tk.BooleanVar(value=is_checked)
            self.checks[key] = check_var
            cb = tk.Checkbutton(
                frame,
                text=label,
                variable=check_var,
                command=lambda k=key: self.toggle_entry(k),  # type: ignore[misc]
            )
            cb.pack(side=tk.LEFT)

            var = tk.StringVar(value=str(value))
            self.vars[key] = var
            entry = tk.Entry(frame, textvariable=var, width=8)
            entry.pack(side=tk.RIGHT)
            self.entries[key] = entry

            entry.config(state=tk.NORMAL if is_checked else tk.DISABLED)

        tk.Button(self, text="Apply Filter", command=self.apply, bg="#ffcccc").pack(
            pady=20, fill=tk.X, padx=50
        )

    def load_prefs(self) -> Dict:
        if self.config_file.exists():
            try:
                with open(self.config_file, "r") as f:
                    data = yaml.safe_load(f) or {}
                return data
            except Exception as e:
                print(f"Error loading prefs: {e}")
        return {}

    def save_prefs(self):
        prefs = {}
        for key, var in self.vars.items():
            prefs[key] = float(var.get()) if var.get() else 0.0
            prefs[f"{key}_checked"] = self.checks[key].get()

        try:
            self.config_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.config_file, "w") as f:
                yaml.dump(prefs, f, default_flow_style=False)
        except Exception as e:
            print(f"Error saving prefs: {e}")

    def toggle_entry(self, key: str) -> None:
        state: str = "normal" if self.checks[key].get() else "disabled"
        self.entries[key].config(state=state)  # type: ignore[call-overload]

    def apply(self):
        try:
            values = {}
            for k, v in self.vars.items():
                if self.checks[k].get():
                    values[k] = float(v.get())
                else:
                    values[k] = None

            self.save_prefs()
            self.callback(values)
            self.destroy()
        except ValueError:
            messagebox.showerror(
                "Error", "Invalid values. Please enter numbers for enabled filters."
            )


class ImageAnnotator:
    """
    GUI application for reviewing and editing polygon annotations.

    Features:
        - Undo/Redo support
        - Annotation filtering
        - Property display
        - Batch operations
    """

    def __init__(self, root: tk.Tk, return_callback: Optional[Callable] = None):
        self.root = root
        self.return_callback = return_callback
        self.root.title("Palm Oil Fruit Annotation Checker")
        self.root.geometry("1400x900")

        self.image_folder = ""
        self.label_folder = ""
        self.image_list: List[str] = []
        self.current_index = -1
        self.original_image: Optional[Image.Image] = None
        self.display_image: Optional[ImageTk.PhotoImage] = None
        self.scale = 1.0
        self.offset_x = 0
        self.offset_y = 0

        self.annotations: List[Dict[str, Any]] = []
        self.selected_annotation_index = -1
        self.hovered_annotation_index = -1

        self.undo_stack: List[List[Dict]] = []
        self.redo_stack: List[List[Dict]] = []

        self.show_tooltip = True

        self.app_config_file = CONFIG_DIR / "annotator_config.yaml"

        self.setup_ui()
        self.setup_menu()
        self.root.bind("<Delete>", self.delete_selected)
        self.root.bind("<Control-z>", self.undo)
        self.root.bind("<Control-y>", self.redo)
        self.root.bind("<Left>", lambda e: self.prev_image())
        self.root.bind("<Right>", lambda e: self.next_image())

        self.root.after(100, self.check_recent_session)

    def setup_menu(self):
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Open Image Folder", command=self.select_folder)
        file_menu.add_command(
            label="Open Recent Session",
            command=lambda: self.check_recent_session(show_error=True),
        )
        file_menu.add_command(
            label="Select Label Folder (Optional)", command=self.select_label_folder
        )
        file_menu.add_command(label="Save Annotations", command=self.save_annotations)
        file_menu.add_separator()
        if self.return_callback:
            file_menu.add_command(label="Back to Main Menu", command=self.return_to_main)
        file_menu.add_command(label="Exit", command=self.root.quit)

        edit_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Edit", menu=edit_menu)
        edit_menu.add_command(label="Undo", command=self.undo, accelerator="Ctrl+Z")
        edit_menu.add_command(label="Redo", command=self.redo, accelerator="Ctrl+Y")

        view_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="View", menu=view_menu)
        view_menu.add_command(label="Refresh", command=self.load_image)

    def setup_ui(self):
        toolbar = tk.Frame(self.root, bd=1, relief=tk.RAISED)
        toolbar.pack(side=tk.TOP, fill=tk.X)

        tk.Button(toolbar, text="Open Images", command=self.select_folder).pack(
            side=tk.LEFT, padx=2, pady=2
        )
        tk.Button(toolbar, text="Set Label Folder", command=self.select_label_folder).pack(
            side=tk.LEFT, padx=2, pady=2
        )
        tk.Button(toolbar, text="Save", command=self.save_annotations).pack(
            side=tk.LEFT, padx=10, pady=2
        )

        tk.Button(toolbar, text="Filter Unwanted", command=self.open_filter_dialog).pack(
            side=tk.LEFT, padx=2, pady=2
        )

        tk.Button(toolbar, text="Undo", command=self.undo).pack(side=tk.LEFT, padx=5, pady=2)
        tk.Button(toolbar, text="Redo", command=self.redo).pack(side=tk.LEFT, padx=2, pady=2)

        tk.Frame(toolbar, width=20).pack(side=tk.LEFT)

        tk.Button(toolbar, text="<< Prev", command=self.prev_image).pack(
            side=tk.LEFT, padx=2, pady=2
        )
        tk.Button(toolbar, text="Next >>", command=self.next_image).pack(
            side=tk.LEFT, padx=2, pady=2
        )

        self.lbl_info = tk.Label(toolbar, text="No Image Loaded", font=("Arial", 10, "bold"))
        self.lbl_info.pack(side=tk.LEFT, padx=20)

        self.main_paned = tk.PanedWindow(self.root, orient=tk.HORIZONTAL, sashwidth=5, bg="gray")
        self.main_paned.pack(fill=tk.BOTH, expand=True)

        self.canvas_frame = tk.Frame(self.main_paned, bg="grey")
        self.main_paned.add(self.canvas_frame, minsize=800, stretch="always")

        self.canvas = tk.Canvas(self.canvas_frame, bg="black", cursor="cross")
        self.canvas.pack(fill=tk.BOTH, expand=True)

        self.canvas.bind("<Button-1>", self.on_canvas_click)
        self.canvas.bind("<Motion>", self.on_mouse_move)

        self.tooltip = tk.Label(self.canvas, text="", bg="yellow", fg="black", padx=3, pady=1)
        self.tooltip.place(x=-100, y=-100)

        self.sidebar = tk.Frame(self.main_paned, width=250, bg="lightgrey", padx=5, pady=5)
        self.main_paned.add(self.sidebar, minsize=200, stretch="never")

        self.props_frame = tk.LabelFrame(
            self.sidebar,
            text="Selected Properties",
            font=("Arial", 10, "bold"),
            bg="lightgrey",
        )
        self.props_frame.pack(fill=tk.X, padx=5, pady=5)

        self.lbl_prop_area = tk.Label(
            self.props_frame, text="Area: N/A", bg="lightgrey", anchor=tk.W
        )
        self.lbl_prop_area.pack(fill=tk.X)
        self.lbl_prop_ratio = tk.Label(
            self.props_frame, text="Ratio: N/A", bg="lightgrey", anchor=tk.W
        )
        self.lbl_prop_ratio.pack(fill=tk.X)
        self.lbl_prop_wh = tk.Label(self.props_frame, text="W/H: N/A", bg="lightgrey", anchor=tk.W)
        self.lbl_prop_wh.pack(fill=tk.X)

        tk.Label(
            self.sidebar,
            text="Annotations List",
            font=("Arial", 12, "bold"),
            bg="lightgrey",
        ).pack(pady=(0, 5))

        self.listbox_frame = tk.Frame(self.sidebar)
        self.listbox_frame.pack(fill=tk.BOTH, expand=True)

        self.scrollbar = tk.Scrollbar(self.listbox_frame, orient=tk.VERTICAL)
        self.lb_annotations = tk.Listbox(
            self.listbox_frame,
            yscrollcommand=self.scrollbar.set,
            selectmode=tk.SINGLE,
            font=("Arial", 10),
        )
        self.scrollbar.config(command=self.lb_annotations.yview)

        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.lb_annotations.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.lb_annotations.bind("<<ListboxSelect>>", self.on_listbox_select)

        tk.Button(
            self.sidebar,
            text="Delete Selected",
            command=self.delete_selected,
            bg="#ffcccc",
            height=2,
        ).pack(fill=tk.X, pady=10)

        self.status_bar = tk.Label(self.root, text="Ready", relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def return_to_main(self):
        self.cleanup()
        if self.return_callback:
            self.return_callback()
        else:
            self.root.quit()

    def cleanup(self):
        try:
            self.root.unbind("<Delete>")
            self.root.unbind("<Left>")
            self.root.unbind("<Right>")
        except Exception as e:
            print(f"Error unbinding: {e}")

    def load_app_config(self) -> Dict:
        if self.app_config_file.exists():
            try:
                with open(self.app_config_file, "r") as f:
                    data = yaml.safe_load(f) or {}
                return data
            except Exception as e:
                print(f"Error loading app config: {e}")
        return {}

    def save_app_config(self):
        config = {
            "recent_image_folder": self.image_folder,
            "recent_label_folder": self.label_folder,
        }
        try:
            self.app_config_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.app_config_file, "w") as f:
                yaml.dump(config, f, default_flow_style=False)
        except Exception as e:
            print(f"Error saving app config: {e}")

    def check_recent_session(self, show_error: bool = False):
        config = self.load_app_config()
        img_dir = config.get("recent_image_folder")
        lbl_dir = config.get("recent_label_folder")

        found = False
        if img_dir and os.path.exists(img_dir):
            found = True
            msg = f"Found recent session:\n\nImages: {img_dir}\n"
            if lbl_dir and os.path.exists(lbl_dir):
                msg += f"Labels: {lbl_dir}\n"
            msg += "\nDo you want to restore this session?"

            if messagebox.askyesno("Restore Recent Session", msg):
                self.select_folder(img_dir)
                if lbl_dir and os.path.exists(lbl_dir):
                    self.select_label_folder(lbl_dir)

        if not found and show_error:
            messagebox.showinfo("Recent Session", "No recent session found.")

    def select_folder(self, folder_path: Optional[str] = None) -> None:
        if folder_path:
            folder = folder_path
        else:
            folder = filedialog.askdirectory(title="Select Image Folder")

        if folder:
            self.image_folder = folder
            self.save_app_config()
            extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
            self.image_list = []
            for ext in extensions:
                self.image_list.extend(glob.glob(os.path.join(folder, ext)))
            self.image_list.sort()

            if self.image_list:
                self.current_index = 0
                self.load_image()
            else:
                messagebox.showinfo("Info", "No images found in folder.")

    def select_label_folder(self, folder_path: Optional[str] = None) -> None:
        if folder_path:
            folder = folder_path
        else:
            folder = filedialog.askdirectory(title="Select Label Folder (contains .txt files)")

        if folder:
            self.label_folder = folder
            self.save_app_config()
            if self.image_list:
                self.load_image()

    def load_image(self):
        if not self.image_list:
            return

        img_path = self.image_list[self.current_index]
        self.original_image = Image.open(img_path)

        self.lbl_info.config(
            text=f"{os.path.basename(img_path)} ({self.current_index + 1}/{len(self.image_list)})"
        )

        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        if cw < 10:
            cw, ch = 800, 600

        iw, ih = self.original_image.size
        self.scale = min(cw / iw, ch / ih) * 0.95

        nw, nh = int(iw * self.scale), int(ih * self.scale)
        resized = self.original_image.resize((nw, nh), Image.Resampling.LANCZOS)
        self.display_image = ImageTk.PhotoImage(resized)

        self.offset_x = (cw - nw) // 2
        self.offset_y = (ch - nh) // 2

        self.load_annotations_from_file(img_path)

        self.update_listbox()
        self.update_properties_panel()
        self.redraw()

    def redraw(self):
        self.canvas.delete("all")
        if self.display_image:
            self.canvas.create_image(
                self.offset_x, self.offset_y, anchor=tk.NW, image=self.display_image
            )

        for idx, ann in enumerate(self.annotations):
            pts = self.transform_points_to_canvas(ann["points"])

            color = "red"
            width = 2
            if idx == self.selected_annotation_index:
                color = "green"
                width = 3
            elif idx == self.hovered_annotation_index:
                color = "yellow"
                width = 2

            if len(pts) > 1:
                flat_pts = [c for p in pts for c in p]
                self.canvas.create_polygon(
                    flat_pts, outline=color, fill="", width=width, tags=f"ann_{idx}"
                )

                if idx == self.selected_annotation_index:
                    for px, py in pts:
                        self.canvas.create_oval(
                            px - 3, py - 3, px + 3, py + 3, fill=color, outline=color
                        )

    def update_listbox(self):
        self.lb_annotations.delete(0, tk.END)
        for idx, ann in enumerate(self.annotations):
            self.lb_annotations.insert(
                tk.END, f"ID {idx}: Class {ann['class_id']} ({len(ann['points'])} pts)"
            )

        if self.selected_annotation_index != -1:
            self.lb_annotations.selection_set(self.selected_annotation_index)
            self.lb_annotations.see(self.selected_annotation_index)

    def on_listbox_select(self, event):
        selection = self.lb_annotations.curselection()
        if selection:
            idx = selection[0]
            self.selected_annotation_index = idx
            self.update_properties_panel()
            self.redraw()
        else:
            self.selected_annotation_index = -1
            self.update_properties_panel()
            self.redraw()

    def update_properties_panel(self):
        if self.selected_annotation_index == -1 or not self.original_image:
            self.lbl_prop_area.config(text="Area: N/A")
            self.lbl_prop_ratio.config(text="Ratio: N/A")
            self.lbl_prop_wh.config(text="W/H: N/A")
            return

        try:
            ann = self.annotations[self.selected_annotation_index]
            points = ann["points"]
            if not points:
                return

            img_w, img_h = self.original_image.size
            img_area = img_w * img_h

            np_points = np.array(points, dtype=np.float32)

            x_coords = np_points[:, 0]
            y_coords = np_points[:, 1]
            w = np.max(x_coords) - np.min(x_coords)
            h = np.max(y_coords) - np.min(y_coords)
            area = cv2.contourArea(np_points)

            aspect_ratio = max(w, h) / min(w, h) if min(w, h) > 0 else 0
            area_pct = (area / img_area) * 100
            w_pct = (w / img_w) * 100
            h_pct = (h / img_h) * 100

            self.lbl_prop_area.config(text=f"Area: {area_pct:.2f}%")
            self.lbl_prop_ratio.config(text=f"Ratio: {aspect_ratio:.2f}")
            self.lbl_prop_wh.config(text=f"W: {w_pct:.1f}%  H: {h_pct:.1f}%")

        except Exception as e:
            print(f"Error calc props: {e}")

    def transform_points_to_canvas(
        self, points: List[Tuple[float, float]]
    ) -> List[Tuple[float, float]]:
        return [(x * self.scale + self.offset_x, y * self.scale + self.offset_y) for x, y in points]

    def transform_canvas_to_image(self, x: float, y: float) -> Tuple[float, float]:
        return ((x - self.offset_x) / self.scale, (y - self.offset_y) / self.scale)

    def is_point_inside_polygon(
        self, x: float, y: float, poly_points: List[Tuple[float, float]]
    ) -> bool:
        contour = np.array(poly_points, dtype=np.float32)
        result = cv2.pointPolygonTest(contour, (x, y), False)
        return result >= 0

    def on_canvas_click(self, event):
        if not self.display_image:
            return

        img_x, img_y = self.transform_canvas_to_image(event.x, event.y)
        clicked_idx = -1

        for idx in range(len(self.annotations) - 1, -1, -1):
            ann = self.annotations[idx]
            if self.is_point_inside_polygon(img_x, img_y, ann["points"]):
                clicked_idx = idx
                break

        self.selected_annotation_index = clicked_idx
        self.update_listbox()
        self.update_properties_panel()
        self.redraw()

    def on_mouse_move(self, event):
        if not self.display_image:
            return

        img_x, img_y = self.transform_canvas_to_image(event.x, event.y)
        hovered = -1

        for idx in range(len(self.annotations) - 1, -1, -1):
            ann = self.annotations[idx]
            if self.is_point_inside_polygon(img_x, img_y, ann["points"]):
                hovered = idx
                break

        self.hovered_annotation_index = hovered

        if hovered != -1:
            self.tooltip.place(x=event.x + 15, y=event.y)
            self.tooltip.config(text=f"ID: {hovered}")
            self.redraw()
        else:
            self.tooltip.place(x=-100, y=-100)
            if self.hovered_annotation_index != -1:
                self.hovered_annotation_index = -1
                self.redraw()

    def save_state(self):
        current_state = []
        for ann in self.annotations:
            current_state.append(ann.copy())

        self.undo_stack.append(current_state)
        self.redo_stack.clear()

        if len(self.undo_stack) > 50:
            self.undo_stack.pop(0)

    def undo(self, event=None):
        if not self.undo_stack:
            self.status_bar.config(text="Nothing to undo.")
            return

        current_state = []
        for ann in self.annotations:
            current_state.append(ann.copy())
        self.redo_stack.append(current_state)

        self.annotations = self.undo_stack.pop()
        self.selected_annotation_index = -1
        self.update_listbox()
        self.update_properties_panel()
        self.redraw()
        self.status_bar.config(text="Undo successful.")

    def redo(self, event=None):
        if not self.redo_stack:
            self.status_bar.config(text="Nothing to redo.")
            return

        current_state = []
        for ann in self.annotations:
            current_state.append(ann.copy())
        self.undo_stack.append(current_state)

        self.annotations = self.redo_stack.pop()
        self.selected_annotation_index = -1
        self.update_listbox()
        self.update_properties_panel()
        self.redraw()
        self.status_bar.config(text="Redo successful.")

    def delete_selected(self, event=None):
        if self.selected_annotation_index != -1:
            self.save_state()
            del self.annotations[self.selected_annotation_index]
            self.selected_annotation_index = -1
            self.save_annotations()
            self.update_listbox()
            self.update_properties_panel()
            self.redraw()
            self.status_bar.config(text="Annotation deleted. Remember to Save.")

    def next_image(self):
        if self.current_index < len(self.image_list) - 1:
            self.current_index += 1
            self.load_image()

    def prev_image(self):
        if self.current_index > 0:
            self.current_index -= 1
            self.load_image()

    def get_annotation_path(self, img_path: str) -> str:
        base_name = os.path.splitext(os.path.basename(img_path))[0]

        if self.label_folder:
            return os.path.join(self.label_folder, base_name + ".txt")

        parent = os.path.dirname(img_path)
        if os.path.basename(parent) == "images":
            grandparent = os.path.dirname(parent)
            labels_dir = os.path.join(grandparent, "labels")
            if os.path.exists(labels_dir):
                return os.path.join(labels_dir, base_name + ".txt")

        return os.path.splitext(img_path)[0] + ".txt"

    def load_annotations_from_file(self, img_path: str):
        self.annotations = []
        self.undo_stack.clear()
        self.redo_stack.clear()

        label_path = self.get_annotation_path(img_path)

        if os.path.exists(label_path):
            try:
                with open(label_path, "r") as f:
                    lines = f.readlines()

                if self.original_image is None:
                    return
                img_w, img_h = self.original_image.size

                for line in lines:
                    parts = list(map(float, line.strip().split()))
                    if len(parts) < 3:
                        continue

                    class_id = int(parts[0])
                    coords = parts[1:]

                    points = []
                    for i in range(0, len(coords), 2):
                        if i + 1 < len(coords):
                            x = coords[i] * img_w
                            y = coords[i + 1] * img_h
                            points.append((x, y))

                    self.annotations.append(
                        {"type": "polygon", "points": points, "class_id": class_id}
                    )
                self.status_bar.config(
                    text=f"Loaded {len(self.annotations)} annotations from {os.path.basename(label_path)}"
                )
            except Exception as e:
                print(f"Error loading labels: {e}")
                self.status_bar.config(text=f"Error loading labels")
        else:
            self.status_bar.config(text="No labels found")

        self.selected_annotation_index = -1

    def save_annotations(self):
        if not self.image_list:
            return

        img_path = self.image_list[self.current_index]
        if self.label_folder:
            label_path = os.path.join(
                self.label_folder,
                os.path.splitext(os.path.basename(img_path))[0] + ".txt",
            )
            os.makedirs(self.label_folder, exist_ok=True)
        else:
            label_path = self.get_annotation_path(img_path)

        try:
            if self.original_image is None:
                return
            img_w, img_h = self.original_image.size
            with open(label_path, "w") as f:
                for ann in self.annotations:
                    class_id = ann["class_id"]
                    points = ann["points"]

                    norm_points = []
                    for pt in points:
                        norm_points.append(f"{pt[0] / img_w:.6f}")
                        norm_points.append(f"{pt[1] / img_h:.6f}")

                    line = f"{class_id} " + " ".join(norm_points)
                    f.write(line + "\n")

            self.status_bar.config(text=f"Saved to {label_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save: {e}")

    def open_filter_dialog(self):
        FilterDialog(self.root, self.run_filter)

    def _calculate_annotation_metrics(
        self, ann: Dict, img_w: int, img_h: int, img_area: int
    ) -> Optional[Dict]:
        points = ann["points"]
        if not points:
            return None

        np_points = np.array(points, dtype=np.float32)

        x_coords = np_points[:, 0]
        y_coords = np_points[:, 1]
        w = float(np.max(x_coords)) - np.min(x_coords)
        h = float(np.max(y_coords)) - np.min(y_coords)
        area = cv2.contourArea(np_points)

        aspect_ratio = max(w, h) / min(w, h) if min(w, h) > 0 else 9999

        return {
            "area": (area / img_area) * 100,
            "width": (w / img_w) * 100,
            "height": (h / img_h) * 100,
            "ratio": aspect_ratio,
        }

    def _should_remove_annotation(self, metrics: Optional[Dict], criteria: Dict) -> bool:
        if metrics is None:
            return False

        filter_rules = [
            ("area", "min_area", "max_area"),
            ("width", "min_width", "max_width"),
            ("height", "min_height", "max_height"),
            ("ratio", None, "max_ratio"),
        ]

        for metric_name, min_key, max_key in filter_rules:
            value = metrics[metric_name]

            if min_key and criteria.get(min_key) is not None:
                if value < criteria[min_key]:
                    return True

            if max_key and criteria.get(max_key) is not None:
                if value > criteria[max_key]:
                    return True

        return False

    def run_filter(self, criteria: Dict):
        if not self.original_image or not self.annotations:
            return

        img_w, img_h = self.original_image.size
        img_area = img_w * img_h

        to_remove = []
        for idx, ann in enumerate(self.annotations):
            metrics = self._calculate_annotation_metrics(ann, img_w, img_h, img_area)
            if self._should_remove_annotation(metrics, criteria):
                to_remove.append(idx)

        if to_remove:
            confirm = messagebox.askyesno(
                "Confirm Filter",
                f"Found {len(to_remove)} annotations to remove.\nIDs: {to_remove}\nProceed?",
            )
            if confirm:
                self.save_state()
                for idx in sorted(to_remove, reverse=True):
                    del self.annotations[idx]

                self.selected_annotation_index = -1
                self.update_listbox()
                self.update_properties_panel()
                self.redraw()
                self.status_bar.config(text=f"Removed {len(to_remove)} annotations.")
        else:
            messagebox.showinfo("Filter", "No annotations matched the removal criteria.")


def main() -> None:
    """Entry point for the palm-annotate command."""
    root = tk.Tk()
    app = ImageAnnotator(root)
    root.mainloop()


if __name__ == "__main__":
    main()
