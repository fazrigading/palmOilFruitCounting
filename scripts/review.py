#!/usr/bin/env python3
"""CLI entry point for GUI annotation reviewer tool."""

import tkinter as tk
from palm_oil_counting.gui import ImageAnnotator


def main():
    root = tk.Tk()
    app = ImageAnnotator(root)
    root.mainloop()


if __name__ == "__main__":
    main()
