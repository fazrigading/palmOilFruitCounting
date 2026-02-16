import tkinter as tk
from tkinter import messagebox
import sys
import os

# Add the current directory to sys.path to ensure we can import the module
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from gui_tools.cropper import ImageCropper
    from gui_tools.annotator import ImageAnnotator
except ImportError as e:
    # If running from a different directory, try to adjust
    try:
        sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'gui_tools'))
        from cropper import ImageCropper
        from annotator import ImageAnnotator
    except ImportError:
        print(f"Error importing tools: {e}")
        sys.exit(1)

class MainLauncher:
    def __init__(self, root):
        self.root = root
        self.root.geometry("400x300")
        self.show_main_menu()

    def clear_root(self):
        for widget in self.root.winfo_children():
            widget.destroy()
        # Reset menu
        self.root.config(menu=tk.Menu(self.root))

    def show_main_menu(self):
        self.clear_root()
        self.root.title("Palm Oil Fruit Counting Tools")
        self.root.geometry("400x350")
        
        # Header
        header = tk.Label(self.root, text="Palm Oil Tools", font=("Arial", 20, "bold"))
        header.pack(pady=20)
        
        # Buttons
        btn_frame = tk.Frame(self.root)
        btn_frame.pack(expand=True, fill=tk.BOTH, padx=50)
        
        tk.Button(btn_frame, text="Image Cropper", font=("Arial", 12), 
                  command=self.launch_cropper, height=2).pack(fill=tk.X, pady=10)
        
        tk.Button(btn_frame, text="Image Annotator", font=("Arial", 12), 
                  command=self.launch_annotator, height=2).pack(fill=tk.X, pady=10)
        
        tk.Button(btn_frame, text="Exit", font=("Arial", 12), 
                  command=self.root.quit, height=2, bg="#ffcccc").pack(fill=tk.X, pady=20)
        
        # Footer
        tk.Label(self.root, text="v1.0.0 | FGDX", fg="gray").pack(side=tk.BOTTOM, pady=5)

    def launch_cropper(self):
        self.clear_root()
        try:
            ImageCropper(self.root, return_callback=self.show_main_menu)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to launch Cropper: {e}")
            self.show_main_menu()

    def launch_annotator(self):
        self.clear_root()
        try:
            ImageAnnotator(self.root, return_callback=self.show_main_menu)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to launch Annotator: {e}")
            self.show_main_menu()

if __name__ == "__main__":
    root = tk.Tk()
    app = MainLauncher(root)
    root.mainloop()
