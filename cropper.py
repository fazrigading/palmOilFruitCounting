import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk, ImageDraw
import os
import glob

__author__ = "Fazri Gading"
__copyright__ = "Copyright 2026, FGDX"
__version__ = "1.0.0"
__email__ = "fazrigading@gmail.com"
__status__ = "Production"

class ImageCropper:
    def __init__(self, root):
        self.root = root
        self.root.title("Gading's Batch Image Cropper")
        self.root.geometry("1000x800")

        self.image_folder = ""
        self.image_list = []
        self.current_index = -1
        self.original_image = None
        self.display_image = None
        self.scale = 1.0
        
        # Crop variables
        self.crop_rect_id = None
        self.start_x = None
        self.start_y = None
        self.rect_coords = [0, 0, 100, 100] # [x1, y1, x2, y2] in canvas space
        self.drag_mode = None # 'move' or 'resize'
        
        self.author_info = {
            "name": "Fazri Gading",
            "version": "1.0.0",
            "description": "Gading's Batch Image Cropper (FGDX)",
            "contact": "fazrigading@gmail.com"
        }
        
        self.setup_ui()
        self.setup_menu()

    def setup_menu(self):
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="About", command=self.show_about)

    def show_about(self):
        about_text = f"{self.author_info['description']}\n\n" \
                     f"Author: {self.author_info['name']}\n" \
                     f"Version: {self.author_info['version']}\n" \
                     f"Contact: {self.author_info['contact']}"
        messagebox.showinfo("About", about_text)

    def setup_ui(self):
        # Top Panel: Folder Selection
        top_frame = tk.Frame(self.root, pady=10)
        top_frame.pack(side=tk.TOP, fill=tk.X)
        
        tk.Button(top_frame, text="Select Image Folder", command=self.select_folder).pack(side=tk.LEFT, padx=10)
        self.folder_label = tk.Label(top_frame, text="No folder selected", fg="gray")
        self.folder_label.pack(side=tk.LEFT, padx=10)

        # Config Panel
        config_frame = tk.Frame(self.root, pady=5)
        config_frame.pack(side=tk.TOP, fill=tk.X)
        
        tk.Label(config_frame, text="Aspect Ratio:").pack(side=tk.LEFT, padx=5)
        self.aspect_var = tk.StringVar(value="Free")
        aspect_options = ["Free", "1:1", "2:3", "3:4", "4:5","9:16", "3:2", "4:3", "5:4", "16:9"]
        self.aspect_menu = ttk.Combobox(config_frame, textvariable=self.aspect_var, values=aspect_options, width=10)
        self.aspect_menu.pack(side=tk.LEFT, padx=5)
        self.aspect_menu.bind("<<ComboboxSelected>>", self.on_config_change)

        tk.Label(config_frame, text="Min Width:").pack(side=tk.LEFT, padx=5)
        self.min_w_ent = tk.Entry(config_frame, width=5)
        self.min_w_ent.insert(0, "640")
        self.min_w_ent.pack(side=tk.LEFT, padx=5)

        tk.Label(config_frame, text="Min Height:").pack(side=tk.LEFT, padx=5)
        self.min_h_ent = tk.Entry(config_frame, width=5)
        self.min_h_ent.insert(0, "640")
        self.min_h_ent.pack(side=tk.LEFT, padx=5)

        # Main Preview Area
        self.preview_container = tk.Frame(self.root, bg="black")
        self.preview_container.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)

        self.canvas = tk.Canvas(self.preview_container, bg="gray20", highlightthickness=0)
        self.canvas.pack(expand=True, fill=tk.BOTH)
        
        self.canvas.bind("<ButtonPress-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        
        # Status Label (Cropped ✅)
        self.status_label = tk.Label(self.root, text="Cropped ✅", bg="green", fg="white", font=("Arial", 12, "bold"), pady=5)
        # We'll place it using place() when needed

        # Bottom Panel: Navigation
        bottom_frame = tk.Frame(self.root, pady=10)
        bottom_frame.pack(side=tk.BOTTOM, fill=tk.X)

        self.btn_prev = tk.Button(bottom_frame, text="<< Previous", command=self.prev_image, state=tk.DISABLED)
        self.btn_prev.pack(side=tk.LEFT, padx=20)

        self.info_label = tk.Label(bottom_frame, text="0 / 0")
        self.info_label.pack(side=tk.LEFT, expand=True)

        self.btn_crop = tk.Button(bottom_frame, text="Crop & Save", command=self.save_crop, bg="blue", fg="white", font=("Arial", 10, "bold"), state=tk.DISABLED)
        self.btn_crop.pack(side=tk.LEFT, padx=10)

        self.btn_next = tk.Button(bottom_frame, text="Next >>", command=self.next_image, state=tk.DISABLED)
        self.btn_next.pack(side=tk.RIGHT, padx=20)

    def select_folder(self):
        folder = filedialog.askdirectory()
        if folder:
            self.image_folder = folder
            self.folder_label.config(text=folder, fg="black")
            
            extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
            self.image_list = []
            for ext in extensions:
                self.image_list.extend(glob.glob(os.path.join(folder, ext)))
            
            self.image_list.sort()
            
            if self.image_list:
                self.current_index = 0
                self.load_image()
                self.update_nav_buttons()
            else:
                messagebox.showinfo("No Images", "No images found in the selected folder.")

    def load_image(self):
        if not self.image_list: return
        
        img_path = self.image_list[self.current_index]
        self.original_image = Image.open(img_path)
        
        # Scale image to fit canvas
        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()
        
        if canvas_w < 10 or canvas_h < 10: # Window not fully rendered yet
            self.root.update()
            canvas_w = self.canvas.winfo_width()
            canvas_h = self.canvas.winfo_height()

        img_w, img_h = self.original_image.size
        ratio = min(canvas_w/img_w, canvas_h/img_h)
        self.scale = ratio
        
        new_w = int(img_w * ratio)
        new_h = int(img_h * ratio)
        
        resized_img = self.original_image.resize((new_w, new_h), Image.Resampling.LANCZOS)
        self.display_image = ImageTk.PhotoImage(resized_img)
        
        self.canvas.delete("all")
        # Center image
        self.offset_x = (canvas_w - new_w) // 2
        self.offset_y = (canvas_h - new_h) // 2
        self.canvas.create_image(self.offset_x, self.offset_y, anchor=tk.NW, image=self.display_image)
        
        # Reset crop selection to fit aspect ratio if not Free
        aspect = self.aspect_var.get()
        if aspect == "Free":
            self.rect_coords = [self.offset_x, self.offset_y, self.offset_x + new_w, self.offset_y + new_h]
        else:
            aw, ah = map(int, aspect.split(':'))
            target_ratio = aw / ah
            
            img_ratio = new_w / new_h
            
            if img_ratio > target_ratio:
                # Image is wider than target ratio
                rect_h = new_h
                rect_w = new_h * target_ratio
            else:
                # Image is taller than target ratio
                rect_w = new_w
                rect_h = new_w / target_ratio
                
            # Center the rectangle
            x1 = self.offset_x + (new_w - rect_w) / 2
            y1 = self.offset_y + (new_h - rect_h) / 2
            self.rect_coords = [x1, y1, x1 + rect_w, y1 + rect_h]

        self.draw_crop_rect()
        
        self.info_label.config(text=f"{self.current_index + 1} / {len(self.image_list)}")
        self.check_if_cropped()

    def check_if_cropped(self):
        base_name = os.path.basename(self.image_list[self.current_index])
        cropped_path = os.path.join(self.image_folder, "cropped", base_name)
        if os.path.exists(cropped_path):
            self.status_label.place(relx=0.5, y=50, anchor=tk.CENTER)
        else:
            self.status_label.place_forget()

    def draw_crop_rect(self):
        if self.crop_rect_id:
            self.canvas.delete(self.crop_rect_id)
        
        x1, y1, x2, y2 = self.rect_coords
        self.crop_rect_id = self.canvas.create_rectangle(x1, y1, x2, y2, outline="yellow", width=2, dash=(4, 4))
        
        # Add handles at all corners
        self.canvas.delete("handle")
        h_size = 5
        corners = [(x1, y1), (x2, y1), (x1, y2), (x2, y2)]
        for cx, cy in corners:
            self.canvas.create_rectangle(cx-h_size, cy-h_size, cx+h_size, cy+h_size, fill="yellow", tags="handle")

    def on_press(self, event):
        x, y = event.x, event.y
        x1, y1, x2, y2 = self.rect_coords
        threshold = 10
        
        # Check corners for resizing
        if abs(x - x1) < threshold and abs(y - y1) < threshold:
            self.drag_mode = 'resize_nw'
        elif abs(x - x2) < threshold and abs(y - y1) < threshold:
            self.drag_mode = 'resize_ne'
        elif abs(x - x1) < threshold and abs(y - y2) < threshold:
            self.drag_mode = 'resize_sw'
        elif abs(x - x2) < threshold and abs(y - y2) < threshold:
            self.drag_mode = 'resize_se'
        elif x1 < x < x2 and y1 < y < y2:
            self.drag_mode = 'move'
            self.start_x = x
            self.start_y = y
        else:
            self.drag_mode = None

    def on_drag(self, event):
        if not self.drag_mode: return
        
        x, y = event.x, event.y
        x1, y1, x2, y2 = self.rect_coords
        
        img_w, img_h = self.display_image.width(), self.display_image.height()
        max_x = self.offset_x + img_w
        max_y = self.offset_y + img_h
        
        if self.drag_mode == 'move':
            dx = x - self.start_x
            dy = y - self.start_y
            
            new_x1 = max(self.offset_x, min(x1 + dx, max_x - (x2 - x1)))
            new_y1 = max(self.offset_y, min(y1 + dy, max_y - (y2 - y1)))
            w, h = x2 - x1, y2 - y1
            
            self.rect_coords = [new_x1, new_y1, new_x1 + w, new_y1 + h]
            self.start_x = x
            self.start_y = y
            
        elif self.drag_mode.startswith('resize'):
            aspect = self.aspect_var.get()
            target_ratio = None
            if aspect != "Free":
                aw, ah = map(int, aspect.split(':'))
                target_ratio = aw / ah

            # Current fixed points based on which corner is dragged
            if self.drag_mode == 'resize_se':
                new_x2 = max(x1 + 20, min(x, max_x))
                new_y2 = max(y1 + 20, min(y, max_y))
                if target_ratio:
                    # Adjust y2 to match aspect ratio
                    h = (new_x2 - x1) / target_ratio
                    if y1 + h <= max_y: new_y2 = y1 + h
                    else:
                        new_y2 = max_y
                        new_x2 = x1 + (new_y2 - y1) * target_ratio
                self.rect_coords = [x1, y1, new_x2, new_y2]
                
            elif self.drag_mode == 'resize_nw':
                new_x1 = max(self.offset_x, min(x, x2 - 20))
                new_y1 = max(self.offset_y, min(y, y2 - 20))
                if target_ratio:
                    h = (x2 - new_x1) / target_ratio
                    if y2 - h >= self.offset_y: new_y1 = y2 - h
                    else:
                        new_y1 = self.offset_y
                        new_x1 = x2 - (y2 - new_y1) * target_ratio
                self.rect_coords = [new_x1, new_y1, x2, y2]
                
            elif self.drag_mode == 'resize_ne':
                new_x2 = max(x1 + 20, min(x, max_x))
                new_y1 = max(self.offset_y, min(y, y2 - 20))
                if target_ratio:
                    h = (new_x2 - x1) / target_ratio
                    if y2 - h >= self.offset_y: new_y1 = y2 - h
                    else:
                        new_y1 = self.offset_y
                        new_x2 = x1 + (y2 - new_y1) * target_ratio
                self.rect_coords = [x1, new_y1, new_x2, y2]
                
            elif self.drag_mode == 'resize_sw':
                new_x1 = max(self.offset_x, min(x, x2 - 20))
                new_y2 = max(y1 + 20, min(y, max_y))
                if target_ratio:
                    h = (x2 - new_x1) / target_ratio
                    if y1 + h <= max_y: new_y2 = y1 + h
                    else:
                        new_y2 = max_y
                        new_x1 = x2 - (new_y2 - y1) * target_ratio
                self.rect_coords = [new_x1, y1, x2, new_y2]
            
        self.draw_crop_rect()

    def on_config_change(self, event):
        # When aspect ratio changes, reset selection to fit ratio
        self.load_image()

    def save_crop(self):
        if not self.original_image: return
        
        x1, y1, x2, y2 = self.rect_coords
        
        # Convert canvas coords to original image coords
        orig_x1 = int((x1 - self.offset_x) / self.scale)
        orig_y1 = int((y1 - self.offset_y) / self.scale)
        orig_x2 = int((x2 - self.offset_x) / self.scale)
        orig_y2 = int((y2 - self.offset_y) / self.scale)
        
        cropped_img = self.original_image.crop((orig_x1, orig_y1, orig_x2, orig_y2))
        
        # Resize if min dimensions specified
        try:
            tw = int(self.min_w_ent.get())
            th = int(self.min_h_ent.get())
            cropped_img = cropped_img.resize((tw, th), Image.Resampling.LANCZOS)
        except:
            pass
            
        output_dir = os.path.join(self.image_folder, "cropped")
        os.makedirs(output_dir, exist_ok=True)
        
        base_name = os.path.basename(self.image_list[self.current_index])
        save_path = os.path.join(output_dir, base_name)
        cropped_img.save(save_path)
        
        self.status_label.place(relx=0.5, y=50, anchor=tk.CENTER)
        print(f"Saved: {save_path}")

    def next_image(self):
        if self.current_index < len(self.image_list) - 1:
            self.current_index += 1
            self.load_image()
            self.update_nav_buttons()

    def prev_image(self):
        if self.current_index > 0:
            self.current_index -= 1
            self.load_image()
            self.update_nav_buttons()

    def update_nav_buttons(self):
        self.btn_prev.config(state=tk.NORMAL if self.current_index > 0 else tk.DISABLED)
        self.btn_next.config(state=tk.NORMAL if self.current_index < len(self.image_list) - 1 else tk.DISABLED)
        self.btn_crop.config(state=tk.NORMAL)

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageCropper(root)
    root.mainloop()
