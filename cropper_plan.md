# Plan: Batch Image Cropper (Tkinter)

## 1. Overview
A GUI-based tool to batch-process image cropping. Users can select a folder, define aspect ratios/dimensions, and manually adjust crop areas for each image before saving.

## 2. Technical Stack
- **Language:** Python 3
- **GUI Framework:** Tkinter
- **Image Processing:** Pillow (PIL)
- **File I/O:** `os`, `glob`, `pathlib`

## 3. Application Flow
1.  **Source Selection:** User selects a directory containing images.
2.  **Configuration:** 
    - Set Target Aspect Ratio (e.g., 1:1, 4:3, 16:9, or Free).
    - Monitor current crop dimensions in original image pixels.
3.  **Image Navigation:**
    - Load images from the folder into a list.
    - Display the first image with a crop overlay.
    - **Next/Previous** buttons to navigate.
4.  **Cropping Interface:**
    - **Canvas:** Displays the image resized to fit the screen.
    - **Selection Overlay:** A semi-transparent or dashed rectangle.
    - **Interaction:** Drag to move, handles to resize (optionally maintaining aspect ratio).
    - **Dynamic Feedback:** Real-time display of crop width and height in actual pixels.
5.  **Saving:**
    - `Crop` button saves the current selection to a `cropped` sub-folder using the original image's resolution (no forced resizing).
    - Visual feedback: "Cropped ✅" label with a green background appears if the current image has already been processed.

## 4. UI Layout
- **Top Bar:** Folder path display and "Select Folder" button.
- **Side/Top Panel:** Aspect Ratio dropdown, Crop Dimension display (Width x Height).
- **Center:** Main Canvas for image preview and crop interaction.
- **Bottom Bar:** 
    - [Previous] [Current/Total] [Next]
    - [Crop & Save]
- **Overlay:** "Cropped ✅" status indicator.

## 5. Development Steps
1.  Initialize Tkinter window and basic layout.
2.  Implement folder selection and image loading logic.
3.  Create the Canvas-based cropping logic (rectangle drawing/dragging).
4.  Add aspect ratio constraint logic.
5.  Implement the save functionality (mapping canvas coordinates back to original image size).
6.  Add the "Cropped" status indicator and navigation state management.
