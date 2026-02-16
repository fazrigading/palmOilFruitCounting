import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
from PIL import Image, ImageTk, ImageDraw
import os
import glob
import json
import cv2
import numpy as np

class ImageAnnotator:
    def __init__(self, root, return_callback=None):
        self.root = root
        self.return_callback = return_callback
        self.root.title("Palm Oil Fruit Annotator")
        self.root.geometry("1200x800")
        
        # Data
        self.image_folder = ""
        self.image_list = []
        self.current_index = -1
        self.original_image = None
        self.display_image = None
        self.scale = 1.0
        self.offset_x = 0
        self.offset_y = 0
        
        # Annotation Data
        self.annotations = [] # List of dicts: {'type': 'polygon', 'points': [(x,y), ...], 'label': 'fruit'}
        self.current_polygon = [] # List of points for currently drawing polygon
        self.selected_annotation_index = -1
        self.hovered_annotation_index = -1
        
        # UI State
        self.mode = "view" # 'view', 'draw_polygon'
        self.show_tooltip = True
        
        self.setup_ui()
        self.setup_menu()
        self.root.bind("<Delete>", self.delete_selected)
        self.root.bind("<Escape>", self.cancel_draw)

    def setup_menu(self):
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Open Folder", command=self.select_folder)
        file_menu.add_command(label="Save Annotations", command=self.save_annotations)
        file_menu.add_separator()
        if self.return_callback:
             file_menu.add_command(label="Back to Main Menu", command=self.return_to_main)
        file_menu.add_command(label="Exit", command=self.root.quit)
        
        tools_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Tools", menu=tools_menu)
        tools_menu.add_command(label="Polygon Tool", command=lambda: self.set_mode("draw_polygon"))
        tools_menu.add_command(label="Bounding Box (Placeholder)", command=self.bbox_placeholder)

    def setup_ui(self):
        # Toolbar
        toolbar = tk.Frame(self.root, bd=1, relief=tk.RAISED)
        toolbar.pack(side=tk.TOP, fill=tk.X)
        
        tk.Button(toolbar, text="Open Folder", command=self.select_folder).pack(side=tk.LEFT, padx=2, pady=2)
        tk.Button(toolbar, text="Prev", command=self.prev_image).pack(side=tk.LEFT, padx=2, pady=2)
        tk.Button(toolbar, text="Next", command=self.next_image).pack(side=tk.LEFT, padx=2, pady=2)
        
        self.lbl_info = tk.Label(toolbar, text="No Image")
        self.lbl_info.pack(side=tk.LEFT, padx=10)
        
        tk.Button(toolbar, text="Polygon Mode", command=lambda: self.set_mode("draw_polygon")).pack(side=tk.LEFT, padx=10)
        tk.Button(toolbar, text="Delete Selected", command=self.delete_selected).pack(side=tk.LEFT, padx=2)
        
        self.status_label = tk.Label(toolbar, text="Ready", relief=tk.SUNKEN, anchor=tk.W)
        self.status_label.pack(side=tk.RIGHT, padx=5)

        # Canvas
        self.canvas_frame = tk.Frame(self.root, bg="grey")
        self.canvas_frame.pack(fill=tk.BOTH, expand=True)
        
        self.canvas = tk.Canvas(self.canvas_frame, bg="black", cursor="cross")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        self.canvas.bind("<Button-1>", self.on_click)
        self.canvas.bind("<Motion>", self.on_mouse_move)
        self.canvas.bind("<Button-3>", self.complete_polygon) # Right click to close polygon
        
        # Tooltip Label (Floating)
        self.tooltip = tk.Label(self.canvas, text="", bg="yellow", fg="black", padx=3, pady=1)
        self.tooltip.place(x=-100, y=-100) # Hide initially

    def return_to_main(self):
        self.cleanup()
        if self.return_callback:
            self.return_callback()
        else:
            self.root.quit()

    def cleanup(self):
        try:
            self.root.unbind("<Delete>")
            self.root.unbind("<Escape>")
        except Exception as e:
            print(f"Error unbinding: {e}")

    def select_folder(self):
        folder = filedialog.askdirectory()
        if folder:
            self.image_folder = folder
            extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
            self.image_list = []
            for ext in extensions:
                self.image_list.extend(glob.glob(os.path.join(folder, ext)))
            self.image_list.sort()
            
            if self.image_list:
                self.current_index = 0
                self.load_image()
            else:
                messagebox.showinfo("Info", "No images found in folder.")

    def load_image(self):
        if not self.image_list: return
        
        img_path = self.image_list[self.current_index]
        self.original_image = Image.open(img_path)
        
        # Resize logic
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        if cw < 10: cw, ch = 800, 600 # Fallback
        
        iw, ih = self.original_image.size
        self.scale = min(cw/iw, ch/ih) * 0.9 # 90% fit
        
        nw, nh = int(iw * self.scale), int(ih * self.scale)
        resized = self.original_image.resize((nw, nh), Image.Resampling.LANCZOS)
        self.display_image = ImageTk.PhotoImage(resized)
        
        self.offset_x = (cw - nw) // 2
        self.offset_y = (ch - nh) // 2
        
        self.lbl_info.config(text=f"{os.path.basename(img_path)} ({self.current_index+1}/{len(self.image_list)})")
        
        # Load existing annotations if any
        self.load_annotations_from_file(img_path)
        
        self.redraw()

    def redraw(self):
        self.canvas.delete("all")
        if self.display_image:
            self.canvas.create_image(self.offset_x, self.offset_y, anchor=tk.NW, image=self.display_image)
            
        # Draw existing polygons
        for idx, ann in enumerate(self.annotations):
            if ann['type'] == 'polygon':
                pts = self.transform_points_to_canvas(ann['points'])
                color = "green" if idx == self.selected_annotation_index else "red"
                if idx == self.hovered_annotation_index: color = "yellow"
                
                if len(pts) > 1:
                    # Flatten list for create_polygon
                    flat_pts = [c for p in pts for c in p]
                    self.canvas.create_polygon(flat_pts, outline=color, fill="", width=2, tags=f"ann_{idx}")
                    # Draw vertices
                    for px, py in pts:
                        self.canvas.create_oval(px-2, py-2, px+2, py+2, fill=color, outline=color)

        # Draw current polygon in progress
        if self.current_polygon:
            pts = self.transform_points_to_canvas(self.current_polygon)
            flat_pts = [c for p in pts for c in p]
            self.canvas.create_line(flat_pts, fill="cyan", width=2)
            for px, py in pts:
                self.canvas.create_oval(px-2, py-2, px+2, py+2, fill="cyan", outline="cyan")

    def transform_points_to_canvas(self, points):
        return [(x * self.scale + self.offset_x, y * self.scale + self.offset_y) for x, y in points]

    def transform_canvas_to_image(self, x, y):
        return ((x - self.offset_x) / self.scale, (y - self.offset_y) / self.scale)

    def is_point_inside_polygon(self, x, y, poly_points):
        # Using OpenCV for point polygon test
        contour = np.array(poly_points, dtype=np.float32)
        # PointPolygonTest returns +ve if inside, -ve if outside, 0 if on edge
        result = cv2.pointPolygonTest(contour, (x, y), False)
        return result >= 0

    def on_click(self, event):
        if not self.display_image: return
        
        img_x, img_y = self.transform_canvas_to_image(event.x, event.y)
        
        if self.mode == "draw_polygon":
            self.current_polygon.append((img_x, img_y))
            self.redraw()
        elif self.mode == "view":
            clicked_idx = -1
            # Check if clicked inside any polygon
            # Iterate in reverse to select top-most if overlapping
            for idx in range(len(self.annotations) - 1, -1, -1):
                ann = self.annotations[idx]
                if self.is_point_inside_polygon(img_x, img_y, ann['points']):
                    clicked_idx = idx
                    break
            
            self.selected_annotation_index = clicked_idx
            self.redraw()

    def on_mouse_move(self, event):
        if not self.display_image: return
        
        img_x, img_y = self.transform_canvas_to_image(event.x, event.y)
        hovered = -1
        
        for idx in range(len(self.annotations) - 1, -1, -1):
            ann = self.annotations[idx]
            if self.is_point_inside_polygon(img_x, img_y, ann['points']):
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

    def complete_polygon(self, event):
        if self.mode == "draw_polygon" and len(self.current_polygon) > 2:
            self.annotations.append({
                'type': 'polygon',
                'points': self.current_polygon,
                'label': 'fruit'
            })
            self.current_polygon = []
            self.mode = "view"
            self.status_label.config(text="Polygon saved. Switch to view mode.")
            self.save_annotations() # Auto-save
            self.redraw()

    def cancel_draw(self, event):
        self.current_polygon = []
        self.mode = "view"
        self.redraw()

    def delete_selected(self, event=None):
        if self.selected_annotation_index != -1:
            del self.annotations[self.selected_annotation_index]
            self.selected_annotation_index = -1
            self.save_annotations()
            self.redraw()

    def set_mode(self, mode):
        self.mode = mode
        self.status_label.config(text=f"Mode: {mode}")

    def bbox_placeholder(self):
        messagebox.showinfo("Not Implemented", "Bounding Box annotation is coming soon!")

    def next_image(self):
        if self.current_index < len(self.image_list) - 1:
            self.save_annotations()
            self.current_index += 1
            self.load_image()

    def prev_image(self):
        if self.current_index > 0:
            self.save_annotations()
            self.current_index -= 1
            self.load_image()

    def get_annotation_path(self, img_path):
        base, _ = os.path.splitext(img_path)
        return base + ".json"

    def save_annotations(self):
        if not self.image_list: return
        img_path = self.image_list[self.current_index]
        json_path = self.get_annotation_path(img_path)
        
        with open(json_path, 'w') as f:
            json.dump(self.annotations, f, indent=4)
        print(f"Saved annotations to {json_path}")

    def load_annotations_from_file(self, img_path):
        json_path = self.get_annotation_path(img_path)
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                self.annotations = json.load(f)
        else:
            self.annotations = []
        self.selected_annotation_index = -1

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageAnnotator(root)
    root.mainloop()