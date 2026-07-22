#!/usr/bin/env python3
"""
Fast-SCNN Inference Diagnostics Desktop Viewer.
A fast, offline Tkinter-based desktop application to inspect segmentation outputs.

Usage
-----
# Default directory (inference_results/)
python gui_viewer.py

# Custom directory
python gui_viewer.py --dir results/
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional

import tkinter as tk
from tkinter import messagebox, ttk

# Import Pillow for robust image scaling and format support
try:
    from PIL import Image, ImageTk
except ImportError:
    # Tkinter popup fallback if Pillow is missing
    root = tk.Tk()
    root.withdraw()
    messagebox.showerror(
        "Missing Dependency",
        "The GUI viewer requires the Pillow package for image processing.\n"
        "Please install it using:\n\n   pip install Pillow"
    )
    sys.exit(1)

# File suffixes definitions and descriptive display names
SUFFIX_MAP = {
    "_overlay.jpg": "Prediction Overlay",
    "_comparison.jpg": "Error Diagnostic (TP/FP/FN)",
    "_prob.jpg": "Saliency Heatmap",
    "_binary.png": "Binary Mask (0-255)",
    "_class.png": "Class Mask (0-1)",
    "_prob_gray.png": "Grayscale Probability",
    "_merged.jpg": "Merged Collage"
}


class DiagnosticsViewerApp:
    def __init__(self, root: tk.Tk, dir_path: Path) -> None:
        self.root = root
        self.dir_path = dir_path
        self.groups: List[Dict] = []
        self.current_idx = -1
        
        # Keep references to ImageTk objects to prevent garbage collection
        self.image_refs: List[ImageTk.PhotoImage] = []
        self.resize_job: Optional[str] = None
        self.last_w = 0
        self.last_h = 0

        self.setup_window()
        self.setup_styles()
        self.build_ui()
        self.scan_directory()
        
        # Bind keyboard shortcuts
        self.root.bind("<Left>", lambda e: self.navigate(-1))
        self.root.bind("<Right>", lambda e: self.navigate(1))
        self.root.bind("<Escape>", lambda e: self.root.quit())
        
        # Load first item
        if self.groups:
            self.select_index(0)
        else:
            self.show_empty_message()

    def setup_window(self) -> None:
        self.root.title("Fast-SCNN Diagnostics Viewer")
        self.root.geometry("1300x800")
        self.root.configure(bg="#0f172a")  # Dark slate background
        self.root.rowconfigure(2, weight=1)
        self.root.columnconfigure(0, weight=1)

    def setup_styles(self) -> None:
        style = ttk.Style()
        style.theme_use("clam")
        
        # Global styling colors
        style.configure(".", background="#0f172a", foreground="#f1f5f9", fieldbackground="#1e293b")
        style.configure("TLabel", background="#0f172a", foreground="#f1f5f9", font=("Helvetica", 10))
        style.configure("Header.TLabel", font=("Helvetica", 12, "bold"), foreground="#10b981", background="#1e293b")
        style.configure("SubHeader.TLabel", font=("Helvetica", 9), foreground="#94a3b8", background="#1e293b")
        style.configure("ImageTitle.TLabel", font=("Helvetica", 10, "bold"), background="#1e293b", foreground="#10b981")
        
        # Ttk Button
        style.configure("TButton", background="#1e293b", foreground="#f1f5f9", font=("Helvetica", 10, "bold"), borderwidth=1, relief="flat")
        style.map("TButton", background=[("active", "#10b981"), ("pressed", "#059669")], foreground=[("active", "#0f172a")])
        
        # Checkbuttons and Combobox
        style.configure("TCheckbutton", background="#0f172a", foreground="#f1f5f9", font=("Helvetica", 9))
        style.map("TCheckbutton", foreground=[("active", "#10b981")], background=[("active", "#0f172a")])
        
        style.configure("TCombobox", arrowcolor="#f1f5f9")
        style.map("TCombobox", fieldbackground=[("readonly", "#1e293b")], foreground=[("readonly", "#f1f5f9")])

    def build_ui(self) -> None:
        # --- Top Navigation Bar ---
        self.nav_frame = tk.Frame(self.root, bg="#1e293b", bd=0, height=70)
        self.nav_frame.grid(row=0, column=0, sticky="ew", padx=0, pady=0)
        self.nav_frame.columnconfigure(1, weight=1)

        # Logo text
        self.logo_lbl = ttk.Label(self.nav_frame, text=" FAST-SCNN DIAGNOSTICS", style="Header.TLabel")
        self.logo_lbl.grid(row=0, column=0, padx=20, pady=(10, 2), sticky="w")
        self.sub_lbl = ttk.Label(self.nav_frame, text="  Offline Desktop Debugging App", style="SubHeader.TLabel")
        self.sub_lbl.grid(row=1, column=0, padx=20, pady=(0, 10), sticky="w")

        # Core navigation controls
        self.ctrl_frame = tk.Frame(self.nav_frame, bg="#1e293b")
        self.ctrl_frame.grid(row=0, column=1, rowspan=2, padx=20, sticky="e")

        self.prev_btn = ttk.Button(self.ctrl_frame, text=" ◀  Prev ", command=lambda: self.navigate(-1))
        self.prev_btn.pack(side="left", padx=5)

        self.select_var = tk.StringVar()
        self.dropdown = ttk.Combobox(self.ctrl_frame, textvariable=self.select_var, state="readonly", width=35)
        self.dropdown.pack(side="left", padx=5)
        self.dropdown.bind("<<ComboboxSelected>>", self.on_dropdown_change)

        self.next_btn = ttk.Button(self.ctrl_frame, text=" Next  ▶ ", command=lambda: self.navigate(1))
        self.next_btn.pack(side="left", padx=5)

        # --- Active Layer Config Panel ---
        self.toggle_frame = tk.Frame(self.root, bg="#0f172a", bd=0)
        self.toggle_frame.grid(row=1, column=0, sticky="ew", padx=20, pady=10)
        
        lbl = ttk.Label(self.toggle_frame, text="ACTIVE LAYERS: ", font=("Helvetica", 9, "bold"), foreground="#94a3b8")
        lbl.pack(side="left", marginRight=10, padx=(0, 10))

        # Checkbutton state variables
        self.toggles: Dict[str, tk.BooleanVar] = {}
        defaults = {"overlay_jpg": True, "comparison_jpg": True, "prob_jpg": True}
        
        for k, display_name in SUFFIX_MAP.items():
            key = k.lstrip("_").replace(".", "_")
            var = tk.BooleanVar(value=defaults.get(key, False))
            self.toggles[key] = var
            cb = ttk.Checkbutton(
                self.toggle_frame, text=display_name, variable=var,
                command=self.update_display,
            )
            cb.pack(side="left", padx=10)

        # --- Main Grid Display Area ---
        self.grid_frame = tk.Frame(self.root, bg="#0f172a")
        self.grid_frame.grid(row=2, column=0, sticky="nsew", padx=20, pady=(5, 20))
        self.grid_frame.bind("<Configure>", self.on_resize)

    def scan_directory(self) -> None:
        """Scan directory and group outputs by image name stem."""
        if not self.dir_path.exists():
            return
            
        groups: Dict[str, Dict[str, Path]] = {}
        for f in self.dir_path.iterdir():
            if f.is_dir() or f.name.startswith("."):
                continue
                
            matched = False
            for suffix in SUFFIX_MAP.keys():
                if f.name.endswith(suffix):
                    stem = f.name[:-len(suffix)]
                    if stem not in groups:
                        groups[stem] = {}
                    groups[stem][suffix] = f
                    matched = True
                    break
            
            # Unrecognized images fallback
            if not matched and f.suffix.lower() in {".png", ".jpg", ".jpeg"}:
                stem = f.stem
                if stem not in groups:
                    groups[stem] = {}
                groups[stem][f.suffix] = f
                
        # Sort and store
        for stem in sorted(groups.keys()):
            file_list = {}
            for suffix, path in groups[stem].items():
                key = suffix.lstrip("_").replace(".", "_")
                file_list[key] = path
            self.groups.append({
                "stem": stem,
                "files": file_list
            })
            
        # Update dropdown list
        self.dropdown["values"] = [f"{idx+1}. {g['stem']}" for idx, g in enumerate(self.groups)]

    def show_empty_message(self) -> None:
        empty_lbl = ttk.Label(
            self.grid_frame,
            text=f"No diagnostics images found in:\n{self.dir_path.resolve()}\n\nPlease run inference.py first.",
            justify="center", font=("Helvetica", 12)
        )
        empty_lbl.pack(expand=True)
        self.prev_btn.configure(state="disabled")
        self.next_btn.configure(state="disabled")

    def select_index(self, idx: int) -> None:
        if idx < 0 or idx >= len(self.groups):
            return
        self.current_idx = idx
        
        # Sync navigation widgets
        self.dropdown.current(idx)
        self.prev_btn.configure(state="disabled" if idx == 0 else "normal")
        self.next_btn.configure(state="disabled" if idx == len(self.groups) - 1 else "normal")
        
        self.update_display()

    def on_dropdown_change(self, event) -> None:
        idx = self.dropdown.current()
        if idx != -1:
            self.select_index(idx)

    def navigate(self, direction: int) -> None:
        target = self.current_idx + direction
        if 0 <= target < len(self.groups):
            self.select_index(target)

    def on_resize(self, event) -> None:
        # Throttled redraw check
        w, h = event.width, event.height
        if abs(w - self.last_w) > 8 or abs(h - self.last_h) > 8:
            self.last_w = w
            self.last_h = h
            if self.resize_job:
                self.root.after_cancel(self.resize_job)
            self.resize_job = self.root.after(100, self.update_display)

    def update_display(self) -> None:
        # Cancel any pending resize callbacks
        self.resize_job = None
        
        # Clear main grid area
        for widget in self.grid_frame.winfo_children():
            widget.destroy()
            
        if self.current_idx == -1 or not self.groups:
            return

        group = self.groups[self.current_idx]
        active_items = []
        
        # Find which checked items exist for this group
        for key, var in self.toggles.items():
            if var.get() and key in group["files"]:
                active_items.append((SUFFIX_MAP.get(f"_{key.replace('_', '.')}", "Image"), group["files"][key]))
                
        n = len(active_items)
        if n == 0:
            lbl = ttk.Label(self.grid_frame, text="No active layers selected. Please toggle at least one checkbox above.")
            lbl.pack(expand=True)
            return

        # Determine grid dimensions
        if n <= 3:
            cols = n
            rows = 1
        elif n == 4:
            cols = 2
            rows = 2
        else:
            cols = 3
            rows = 2

        # Configure weights for uniform grid resizing
        for i in range(cols):
            self.grid_frame.columnconfigure(i, weight=1)
        for i in range(rows):
            self.grid_frame.rowconfigure(i, weight=1)

        # Get grid container boundaries
        grid_w = self.grid_frame.winfo_width()
        grid_h = self.grid_frame.winfo_height()
        
        # Fallback to sensible initial values if grid layout has not fully rendered
        if grid_w < 100:
            grid_w = self.root.winfo_width() - 40
        if grid_h < 100:
            grid_h = self.root.winfo_height() - 140

        # Calculate max size per cell (subtract spacing)
        cell_w = (grid_w - (cols - 1) * 15) // cols
        cell_h = (grid_h - (rows - 1) * 35) // rows

        self.image_refs.clear()

        # Render visible images inside cells
        for idx, (title, img_path) in enumerate(active_items):
            r = idx // cols
            c = idx % cols

            # Card structure
            cell_card = tk.Frame(self.grid_frame, bg="#1e293b", bd=1, relief="solid", highlightthickness=0)
            cell_card.grid(row=r, column=c, sticky="nsew", padx=6, pady=6)
            cell_card.rowconfigure(1, weight=1)
            cell_card.columnconfigure(0, weight=1)

            # Label Header
            header = ttk.Label(cell_card, text=f"  {title}", style="ImageTitle.TLabel", anchor="w")
            header.grid(row=0, column=0, sticky="ew", pady=4)

            # Load and scale image
            try:
                pil_img = Image.open(img_path)
                img_w, img_h = pil_img.size
                
                # Fit aspect ratio
                ratio = min(cell_w / img_w, (cell_h - 30) / img_h)
                new_w = max(int(img_w * ratio), 1)
                new_h = max(int(img_h * ratio), 1)
                
                # Resize using Lanczos filter
                scaled_pil = pil_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
                photo_img = ImageTk.PhotoImage(scaled_pil)
                self.image_refs.append(photo_img)

                # Image frame container
                img_lbl = tk.Label(cell_card, image=photo_img, bg="#0b0f19", bd=0)
                img_lbl.grid(row=1, column=0, sticky="nsew")
            except Exception as e:
                err_lbl = ttk.Label(cell_card, text=f"Load Error:\n{e}", justify="center")
                err_lbl.grid(row=1, column=0, sticky="nsew")


def main() -> None:
    p = argparse.ArgumentParser(description="Start Tkinter Results Viewer")
    p.add_argument("--dir", type=str, default="inference_results",
                   help="Directory containing inference output files")
    args = p.parse_args()

    dir_path = Path(args.dir)
    
    root = tk.Tk()
    
    # Initialize the Tkinter app
    app = DiagnosticsViewerApp(root, dir_path)
    
    try:
        root.mainloop()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
