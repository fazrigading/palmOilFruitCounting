#!/usr/bin/env python3
"""CLI entry point for GUI cropper tool."""

import tkinter as tk
from palm_oil_counting.gui import ImageCropper


def main():
    root = tk.Tk()
    app = ImageCropper(root)
    root.mainloop()


if __name__ == "__main__":
    main()
