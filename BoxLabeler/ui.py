# This file is part of BoxLabeler.
# 
# BoxLabeler is licensed under the MIT License.
# To view a copy of this license, visit https://opensource.org/licenses/MIT
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import os
import json
import random
import tkinter as tk
from tkinter import filedialog, messagebox, font, ttk
from PIL import Image, ImageTk
import cv2
import threading

import datetime

from BoxLabeler.annotations.image_annotation import ImageAnnotation
from BoxLabeler.annotations.bounding_box import BoundingBox
from BoxLabeler.exporters import get_exporter
from BoxLabeler.models.yolov8_import import YoloV8ImportModel

class ObjectDetectionLabeler:
    def __init__(self, master):
        self.master = master
        self.master.title("Object Detection Labeler")
        
        # Initialize variables
        self.label_colors = {}
        self.filter_mode = "All"
        self.current_image = None
        self.original_image = None  # Original image storage
        self.scaled_image = None
        self.photo = None
        self.image_list = []
        self.filtered_image_list = []
        self.current_image_index = 0
        self.annotations = {}
        self.current_bbox = None
        self.bbox_items = []
        self.history = []
        self.edit_mode = False  # Control Edit mode
        self.zoom_level = 1.0    # Current zoom level (used for display)
        self.user_zoom_level = 1.0  # Zoom level set by user
        self.image_scale = 1.0    # Scaling factor
        
        # Variables to track resizing and moving
        self.resizing = False
        self.moving = False
        self.resize_corner = None  # Which corner is being resized
        self.move_bbox_index = None  # Index of bbox being moved
        
        self.selected_bbox_index = None  # Initialize selected_bbox_index
        
        self.yolov8_model = YoloV8ImportModel()
        self.current_model = None  # Initialize current_model
        
        # Position of the image on the canvas
        self.image_x = 0
        self.image_y = 0

        # Control Variables
        self.auto_resize = tk.BooleanVar(value=True)  # Default to checked
        self.auto_next = tk.BooleanVar()

        # Variables for Auto Predict
        self.auto_predict_thread = None
        self.auto_predict_cancel_flag = False

        # Dictionary to store color images for Treeview
        self.color_images = {}

        # Setup UI
        self.setup_menu()
        self.setup_ui()
        
        # Bind keyboard shortcuts
        self.bind_shortcuts()

    # ==================== UI Setup ==================== #
    def setup_menu(self):
        menu = tk.Menu(self.master)
        self.master.config(menu=menu)
        
        self.create_file_menu(menu)
        self.create_view_menu(menu)
        self.create_edit_menu(menu)
        self.create_import_menu(menu)
        self.create_info_menu(menu)

    def create_file_menu(self, parent_menu):
        file_menu = tk.Menu(parent_menu, tearoff=0)
        parent_menu.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Open Directory", command=self.open_directory)
        file_menu.add_command(label="Load Annotations", command=self.load_annotations)
        file_menu.add_command(label="Save Annotations", command=self.save_annotations_auto)
        export_menu = tk.Menu(file_menu, tearoff=0)
        file_menu.add_cascade(label="Export", menu=export_menu)
        export_options = ["COCO", "Dataset_COCO", "TFRecord", "YOLO v8", "Pascal VOC", "Excel"]
        export_formats = ["coco", "dataset_coco", "tfrecord", "yolov8", "pascal_voc", "excel"]
        for fmt, label in zip(export_formats, export_options):
            export_menu.add_command(label=f"Export to {label}", command=lambda f=fmt: self.export(f))
      
    def create_view_menu(self, parent_menu):
        view_menu = tk.Menu(parent_menu, tearoff=0)
        parent_menu.add_cascade(label="View", menu=view_menu)
        view_modes = ["All Images", "Unlabeled Images", "Labeled Images"]
        modes = ["All", "Unlabeled", "Labeled"]
        for mode_label, mode in zip(view_modes, modes):
            view_menu.add_command(label=mode_label, command=lambda m=mode: self.set_filter_mode(m))

    def create_edit_menu(self, parent_menu):
        edit_menu = tk.Menu(parent_menu, tearoff=0)
        parent_menu.add_cascade(label="Edit", menu=edit_menu)
        edit_menu.add_command(label="Toggle Edit Mode", command=self.toggle_edit_mode)
        edit_menu.add_separator()
        edit_menu.add_command(label="++Zoom In++", command=lambda: self.zoom(1.2))
        edit_menu.add_command(label="--Zoom Out--", command=lambda: self.zoom(0.8))
        edit_menu.add_command(label="Reset Zoom", command=self.reset_zoom)

    def create_import_menu(self, parent_menu):
        import_menu = tk.Menu(parent_menu, tearoff=0)
        import_menu.add_command(label="YOLO_v8", command=self.import_yolov8_model)
        parent_menu.add_cascade(label="Import", menu=import_menu)

    def create_info_menu(self, parent_menu):
        info_menu = tk.Menu(parent_menu, tearoff=0)
        parent_menu.add_cascade(label="Info", menu=info_menu)
        info_menu.add_command(label="About", command=self.show_about)
        info_menu.add_command(label="Shortcuts", command=self.show_shortcuts)
        
            
    def setup_ui(self):
        # Main frame
        main_frame = tk.Frame(self.master)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Canvas and Scrollbars
        self.setup_canvas(main_frame)
        
        # Control Frame
        self.setup_controls(main_frame)

        # Add Auto Predict Button
        self.setup_auto_predict_button(main_frame)

        # Bind window resize event
        self.master.bind("<Configure>", self.on_window_resize)

    def setup_canvas(self, parent):
        canvas_frame = tk.Frame(parent)
        canvas_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Canvas Area
        canvas_area = tk.Frame(canvas_frame)
        canvas_area.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(canvas_area, cursor="cross", bg='#949494')
        self.canvas.pack(expand=True, fill=tk.BOTH, side=tk.LEFT)

        # Scrollbars
        self.h_scroll = tk.Scrollbar(canvas_area, orient=tk.HORIZONTAL, command=self.canvas.xview)
        self.h_scroll.pack(side=tk.BOTTOM, fill=tk.X)
        self.v_scroll = tk.Scrollbar(canvas_area, orient=tk.VERTICAL, command=self.canvas.yview)
        self.v_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.canvas.configure(xscrollcommand=self.h_scroll.set, yscrollcommand=self.v_scroll.set)

        # Bind canvas events
        self.canvas.bind('<Motion>', self.show_crosshair)
        self.canvas.bind("<ButtonPress-1>", self.on_mouse_down)
        self.canvas.bind("<B1-Motion>", self.on_mouse_move)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_up)
        self.canvas.bind("<Button-3>", self.on_right_click)  # Right-click to edit label

        # Label List Panel using ttk.Treeview for optimization
        self.label_list_frame = tk.Frame(canvas_frame, width=150)
        self.label_list_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0))

        # Label List Header
        tk.Label(self.label_list_frame, text="Labels:", font=('Arial', 12, 'bold')).pack(pady=5)

        # Treeview for label items
        self.label_tree = ttk.Treeview(
            self.label_list_frame, 
            columns=("Label",), 
            show='tree headings',  # Show both tree and headings
            selectmode="browse",
            height=20
        )
        self.label_tree.heading("#0", text="Color")  # Tree column header
        self.label_tree.heading("Label", text="Label")  # Second column header
        self.label_tree.column("#0", width=50, anchor='center')  # Tree column width for color
        self.label_tree.column("Label", width=140, anchor='w')  # Label column width
        self.label_tree.pack(fill=tk.BOTH, expand=True)

        # Bind double-click to copy label
        self.label_tree.bind("<Double-1>", self.on_label_double_click)

        # Initialize the label list
        self.refresh_label_list()

    def on_label_double_click(self, event):
        """Handle double-click on a label to copy it."""
        selected_item = self.label_tree.selection()
        if selected_item:
            label = self.label_tree.item(selected_item, 'values')[0]
            self.copy_label(label)

    def refresh_label_list(self):
        """Refresh the label list panel with current labels and their colors using Treeview."""
        # Clear existing items
        for item in self.label_tree.get_children():
            self.label_tree.delete(item)

        # Sort labels alphabetically for better organization
        sorted_labels = sorted(self.label_colors.keys())

        # Clear previous color images to free memory
        self.color_images.clear()

        for label in sorted_labels:
            color = self.label_colors[label]
            # Create a small colored square image
            color_image = Image.new('RGB', (20, 20), color)
            photo = ImageTk.PhotoImage(color_image)
            self.color_images[label] = photo  # Prevent garbage collection

            # Insert item with image in the tree column and label in the 'Label' column
            self.label_tree.insert('', 'end', text='', image=photo, values=(label,))

    def copy_label(self, label):
        """Copy the selected label to the label entry field."""
        self.label_entry.delete(0, tk.END)
        self.label_entry.insert(0, label)

    def setup_controls(self, parent):
        control_frame = tk.Frame(parent)
        control_frame.pack(fill=tk.X, padx=10, pady=10)
        
        self.setup_label_entry(control_frame)
        self.setup_buttons(control_frame)
        self.setup_indicators(control_frame)
        self.setup_extra_controls(control_frame)

    def setup_label_entry(self, parent):
        label_frame = tk.Frame(parent)
        label_frame.pack(side=tk.TOP, fill=tk.X)
        tk.Label(label_frame, text="Type Label:").pack(side=tk.LEFT)
        self.label_entry = tk.Entry(label_frame)
        self.label_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.label_entry.bind("<Return>", self.save_label)  # Save label on Enter

    def setup_buttons(self, parent):
        button_frame = tk.Frame(parent)
        button_frame.pack(side=tk.TOP, pady=5)
        
        buttons = [
            ("Detect & Predict", self.predict),
            ("<<Previous Image<<", self.prev_image),
            (">>Next Image>>", self.next_image),
            ("Delete BBox", self.delete_bbox),
            ("Delete Image", self.delete_image),
            ("++Zoom In++", lambda: self.zoom(1.2)),
            ("--Zoom Out--", lambda: self.zoom(0.8)),
            ("Edit Mode", self.toggle_edit_mode)
        ]
        
        for i, (text, cmd) in enumerate(buttons):
            button = tk.Button(button_frame, text=text, command=cmd)
            button.grid(row=0, column=i, padx=5)
        
        # Center the buttons by configuring grid weights
        total_buttons = len(buttons)
        for i in range(total_buttons):
            button_frame.grid_columnconfigure(i, weight=1)
        button_frame.grid_columnconfigure(total_buttons, weight=1)  # Extra weight for centering

    def setup_indicators(self, parent):
        self.filter_indicator = tk.Label(parent, text="Filter: All Images")
        self.filter_indicator.pack(anchor='center', pady=2)
        
        self.label_count_label = tk.Label(parent, text="Label counts: ")
        self.label_count_label.pack(anchor='center', pady=2)
        
        self.image_counter = tk.Label(parent, text="Image: 0 / 0")
        self.image_counter.pack(anchor='center', pady=2)

    def setup_extra_controls(self, parent):
        # Auto-next Checkbox
        self.auto_next_checkbox = tk.Checkbutton(parent, text="Auto-next after bbox", variable=self.auto_next)
        self.auto_next_checkbox.pack(anchor='center', pady=2)
        
        # Auto Resize Checkbox
        self.auto_resize_checkbox = tk.Checkbutton(
            parent, 
            text="Auto Resize", 
            variable=self.auto_resize, 
            command=self.on_auto_resize_toggle
        )
        self.auto_resize_checkbox.pack(anchor='center', pady=2)

    def setup_auto_predict_button(self, parent):
        """Add the Auto Predict button to the control frame."""
        auto_predict_frame = tk.Frame(parent)
        auto_predict_frame.pack(side=tk.TOP, pady=10)

        self.auto_predict_button = tk.Button(
            auto_predict_frame, 
            text="Auto Predict", 
            command=self.auto_predict
        )
        self.auto_predict_button.pack()

    def bind_shortcuts(self):
        # Keyboard shortcuts
        self.master.bind_all("<Control-z>", lambda event: self.undo())
        self.master.bind_all("<Control-Z>", lambda event: self.undo())
        self.master.bind_all("<Tab>", lambda event: self.next_image())
        self.master.bind_all("<Shift-Tab>", lambda event: self.prev_image())
        self.master.bind_all("<Alt-KeyPress>", self.handle_alt_shortcuts)
        self.master.bind_all("<Control-plus>", lambda event: self.zoom(1.2))
        self.master.bind_all("<Control-minus>", lambda event: self.zoom(0.8))
        self.master.bind_all("<Control-0>", lambda event: self.reset_zoom())

    # ==================== Event Handlers ==================== #
    def handle_alt_shortcuts(self, event):
        key = event.char.lower()
        shortcuts = {
            'd': self.predict,
            'r': self.delete_image,
            'b': self.delete_bbox,
            'w': self.clear_and_focus_label
        }
        action = shortcuts.get(key)
        if action:
            action()

    def show_crosshair(self, event=None):
        if self.scaled_image:
            # Calculate crosshair position relative to image
            mouse_x = event.x
            mouse_y = event.y

            # Check if mouse is over the image
            if self.image_x <= mouse_x <= self.image_x + self.scaled_image.width and \
               self.image_y <= mouse_y <= self.image_y + self.scaled_image.height:
                cross_x = mouse_x
                cross_y = mouse_y

                # Update crosshair position
                if hasattr(self, 'crosshair_h'):
                    self.canvas.coords(
                        self.crosshair_h, 
                        self.image_x, cross_y, 
                        self.image_x + self.scaled_image.width, cross_y
                    )
                else:
                    self.crosshair_h = self.canvas.create_line(
                        self.image_x, cross_y, 
                        self.image_x + self.scaled_image.width, cross_y, 
                        fill='blue', dash=(2, 2), tags="crosshair_h"
                    )

                if hasattr(self, 'crosshair_v'):
                    self.canvas.coords(
                        self.crosshair_v, 
                        cross_x, self.image_y, 
                        cross_x, self.image_y + self.scaled_image.height
                    )
                else:
                    self.crosshair_v = self.canvas.create_line(
                        cross_x, self.image_y, 
                        cross_x, self.image_y + self.scaled_image.height, 
                        fill='blue', dash=(2, 2), tags="crosshair_v"
                    )
            else:
                # If mouse is outside the image, hide crosshair
                if hasattr(self, 'crosshair_h'):
                    self.canvas.coords(self.crosshair_h, 0, 0, 0, 0)
                if hasattr(self, 'crosshair_v'):
                    self.canvas.coords(self.crosshair_v, 0, 0, 0, 0)

    def on_mouse_down(self, event):
        image_rel_x, image_rel_y = self.get_image_relative_coords(event.x, event.y)
        
        if self.edit_mode:
            self.handle_edit_mode_mouse_down(event, image_rel_x, image_rel_y)
        else:
            self.start_annotation(event, image_rel_x, image_rel_y)

    def handle_edit_mode_mouse_down(self, event, img_x, img_y):
        clicked_items = self.canvas.find_overlapping(event.x, event.y, event.x, event.y)
        for item in clicked_items:
            tags = self.canvas.gettags(item)
            if "resize_handle" in tags:
                self.initiate_resize(event, tags)
                return
            elif "edge_" in tags:
                self.initiate_move(event, tags)
                return
            elif "move_handle" in tags:
                self.initiate_move(event, tags)
                return
            elif "bbox" in tags:
                self.initiate_move(event, tags)
                return

    def initiate_resize(self, event, tags):
        for tag in tags:
            if tag.startswith("resize_handle_"):
                parts = tag.split("_")
                bbox_index = int(parts[2])
                corner = parts[3]  # e.g., 'tl', 'tr', 'bl', 'br'
                self.selected_bbox_index = bbox_index
                self.resize_corner = corner
                self.resizing = True
                self.resize_start_x, self.resize_start_y = self.get_image_relative_coords(event.x, event.y)
                orig_bbox = self.annotations[self.current_image_path()].bboxes[bbox_index]
                self.original_bbox = BoundingBox(orig_bbox.x, orig_bbox.y, orig_bbox.w, orig_bbox.h, orig_bbox.category_id)
                break

    def initiate_move(self, event, tags):
        for tag in tags:
            if tag.startswith("edge_") or tag.startswith("move_handle_") or tag.startswith("bbox_"):
                try:
                    if tag.startswith("edge_"):
                        # Format: edge_left_bbox_0
                        bbox_index = int(tag.split("_")[3])
                    elif tag.startswith("move_handle_"):
                        # Format: move_handle_bbox_0
                        bbox_index = int(tag.split("_")[2])
                    elif tag.startswith("bbox_"):
                        # Format: bbox_0
                        bbox_index = int(tag.split("_")[1])
                    else:
                        continue
                    self.selected_bbox_index = bbox_index
                    self.moving = True
                    self.move_start_x, self.move_start_y = self.get_image_relative_coords(event.x, event.y)
                    orig_bbox = self.annotations[self.current_image_path()].bboxes[bbox_index]
                    self.original_bbox = BoundingBox(orig_bbox.x, orig_bbox.y, orig_bbox.w, orig_bbox.h, orig_bbox.category_id)
                    self.move_bbox_index = bbox_index
                    break
                except (IndexError, ValueError):
                    continue

    def start_annotation(self, event, img_x, img_y):
        if 0 <= img_x <= self.scaled_image.width and 0 <= img_y <= self.scaled_image.height:
            self.start_x = img_x
            self.start_y = img_y
            self.current_bbox = self.canvas.create_rectangle(
                self.image_x + self.start_x, self.image_y + self.start_y, 
                self.image_x + self.start_x, self.image_y + self.start_y, 
                outline='blue', width=2, tags="current_bbox"
            )

    def on_mouse_move(self, event):
        image_rel_x, image_rel_y = self.get_image_relative_coords(event.x, event.y)
        
        if self.edit_mode:
            self.handle_edit_mode_mouse_move(image_rel_x, image_rel_y)
        elif self.current_bbox:
            self.update_annotation_bbox(event, image_rel_x, image_rel_y)

    def handle_edit_mode_mouse_move(self, img_x, img_y):
        if self.resizing and self.selected_bbox_index is not None:
            self.resize_bbox(img_x, img_y)
        elif self.moving and self.selected_bbox_index is not None:
            self.move_bbox(img_x, img_y)

    def resize_bbox(self, img_x, img_y):
        dx = (img_x - self.resize_start_x) / self.zoom_level
        dy = (img_y - self.resize_start_y) / self.zoom_level
        bbox = self.annotations[self.current_image_path()].bboxes[self.selected_bbox_index]
        
        # Adjust based on which corner is being resized
        if self.resize_corner == 'tl':
            bbox.x = max(0, self.original_bbox.x + dx)
            bbox.y = max(0, self.original_bbox.y + dy)
            bbox.w = max(1, self.original_bbox.w - dx)
            bbox.h = max(1, self.original_bbox.h - dy)
        elif self.resize_corner == 'tr':
            bbox.y = max(0, self.original_bbox.y + dy)
            bbox.w = max(1, self.original_bbox.w + dx)
            bbox.h = max(1, self.original_bbox.h - dy)
        elif self.resize_corner == 'bl':
            bbox.x = max(0, self.original_bbox.x + dx)
            bbox.w = max(1, self.original_bbox.w - dx)
            bbox.h = max(1, self.original_bbox.h + dy)
        elif self.resize_corner == 'br':
            bbox.w = max(1, self.original_bbox.w + dx)
            bbox.h = max(1, self.original_bbox.h + dy)
        
        self.display_image()

    def move_bbox(self, img_x, img_y):
        dx = (img_x - self.move_start_x) / self.zoom_level
        dy = (img_y - self.move_start_y) / self.zoom_level
        bbox = self.annotations[self.current_image_path()].bboxes[self.selected_bbox_index]
        bbox.x = max(0, min(self.original_bbox.x + dx, self.original_image.width - bbox.w))
        bbox.y = max(0, min(self.original_bbox.y + dy, self.original_image.height - bbox.h))
        self.display_image()

    def update_annotation_bbox(self, event, img_x, img_y):
        cur_x = max(0, min(img_x, self.scaled_image.width))
        cur_y = max(0, min(img_y, self.scaled_image.height))
        self.canvas.coords(
            self.current_bbox, 
            self.image_x + self.start_x, self.image_y + self.start_y, 
            self.image_x + cur_x, self.image_y + cur_y
        )
        # Do not call self.display_image() here

    def on_mouse_up(self, event):
        image_rel_x, image_rel_y = self.get_image_relative_coords(event.x, event.y)
        
        if self.edit_mode:
            self.handle_edit_mode_mouse_up()
        elif self.current_bbox:
            self.finalize_annotation(event, image_rel_x, image_rel_y)

    def handle_edit_mode_mouse_up(self):
        if self.resizing:
            self.history.append((
                'resize_bbox', 
                self.current_image_path(), 
                self.selected_bbox_index, 
                self.original_bbox
            ))
            self.resizing = False
            self.selected_bbox_index = None
            self.resize_corner = None
        elif self.moving:
            self.history.append((
                'move_bbox', 
                self.current_image_path(), 
                self.selected_bbox_index, 
                self.original_bbox
            ))
            self.moving = False
            self.selected_bbox_index = None
            self.move_bbox_index = None

    def finalize_annotation(self, event, img_x, img_y):
        end_x = max(0, min(img_x, self.scaled_image.width))
        end_y = max(0, min(img_y, self.scaled_image.height))
        label = self.label_entry.get().strip()
        if label:
            x = min(self.start_x, end_x) / self.zoom_level
            y = min(self.start_y, end_y) / self.zoom_level
            w = abs(end_x - self.start_x) / self.zoom_level
            h = abs(end_y - self.start_y) / self.zoom_level
            image_path = self.current_image_path()
            new_bbox = BoundingBox(x, y, w, h, label)
            self.annotations.setdefault(image_path, ImageAnnotation(image_path)).add_bbox(new_bbox)
            
            color = self.get_color_for_label(label)
            self.canvas.itemconfig(self.current_bbox, outline=color)
            
            self.history.append(('add', image_path, new_bbox))
            self.update_label_counts()
            self.apply_filter()
            
            if self.auto_next.get():
                self.next_image()
        else:
            if self.current_bbox:
                self.canvas.delete(self.current_bbox)
        self.current_bbox = None

    def on_right_click(self, event):
        if self.edit_mode:
            clicked_items = self.canvas.find_overlapping(event.x, event.y, event.x, event.y)
            for item in clicked_items:
                tags = self.canvas.gettags(item)
                if "bbox_label" in tags or "bbox" in tags:
                    bbox_index = self.extract_bbox_index(tags)
                    if bbox_index is not None:
                        bbox = self.annotations[self.current_image_path()].bboxes[bbox_index]
                        self.show_label_edit_menu(event, bbox_index, bbox)
                        return

    def extract_bbox_index(self, tags):
        for tag in tags:
            if tag.startswith("bbox_label_"):
                try:
                    return int(tag.split("_")[2])
                except ValueError:
                    continue  
            elif tag.startswith("bbox_"):
                try:
                    return int(tag.split("_")[1])
                except ValueError:
                    continue  
        return None

    def show_label_edit_menu(self, event, bbox_index, bbox):
        menu = tk.Menu(self.master, tearoff=0)
        menu.add_command(label="Edit Label", command=lambda: self.edit_label(bbox_index, bbox))
        menu.add_command(label="Delete Bounding Box & Label", command=lambda: self.delete_specific_bbox(bbox_index))
        menu.post(event.x_root, event.y_root)

    def edit_label(self, bbox_index, bbox):
        self.label_entry.delete(0, tk.END)
        self.label_entry.insert(0, bbox.category_id)
        self.label_entry.focus_set()
        self.label_entry.bind(
            "<Return>", 
            lambda event, idx=bbox_index, b=bbox: self.save_label(idx, b)
        )

    def clear_and_focus_label(self):
        """Clear the label entry and set focus to it."""
        self.label_entry.delete(0, tk.END)
        self.label_entry.focus_set()

    # ==================== Image Handling ==================== #
    def open_directory(self):
        directory = filedialog.askdirectory()
        if directory:
            self.load_images_from_directory(directory)
            if self.image_list:
                self.current_image_index = 0
                self.set_filter_mode("All")
                self.load_image()
                self.update_ui()
            else:
                messagebox.showinfo("Info", "No images found in the selected directory.")

    def load_images_from_directory(self, directory):
        self.image_list = sorted([
            os.path.abspath(os.path.join(directory, f)) 
            for f in os.listdir(directory) 
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])
        self.filtered_image_list = self.image_list.copy()

    def load_image(self):
        if 0 <= self.current_image_index < len(self.filtered_image_list):
            image_path = self.current_image_path()
            try:
                self.original_image = Image.open(image_path).convert("RGB")  # Original image storage
            except Exception as e:
                messagebox.showerror("Error", f"Cannot load image:\n{e}")
                return

            self.user_zoom_level = 1.0  # Reset user zoom level when loading a new image
            self.display_image()  # Use the stored image for display
            self.update_ui()  # Update UI after displaying the image
        else:
            self.clear_canvas()

    def clear_canvas(self):
        self.canvas.delete("all")
        self.canvas.create_text(
            self.canvas.winfo_width() // 2, 
            self.canvas.winfo_height() // 2,
            text="No image to display", 
            font=("Arial", 20)
        )
        self.image_counter.config(text="Image: 0 / 0")
        # Hide crosshair when no image
        if hasattr(self, 'crosshair_h'):
            self.canvas.delete(self.crosshair_h)
        if hasattr(self, 'crosshair_v'):
            self.canvas.delete(self.crosshair_v)

    def set_filter_mode(self, mode):
        self.filter_mode = mode
        self.filter_indicator.config(text=f"Filter: {mode} Images")
        self.apply_filter()

    def apply_filter(self):
        self.image_list = [img for img in self.image_list if os.path.exists(img)]
        
        if self.filter_mode == "All":
            self.filtered_image_list = self.image_list.copy()
        elif self.filter_mode == "Unlabeled":
            self.filtered_image_list = [
                img for img in self.image_list 
                if img not in self.annotations or not self.annotations[img].bboxes
            ]
        elif self.filter_mode == "Labeled":
            self.filtered_image_list = [
                img for img in self.image_list 
                if img in self.annotations and self.annotations[img].bboxes
            ]
        
        self.current_image_index = min(max(self.current_image_index, 0), len(self.filtered_image_list) - 1) if self.filtered_image_list else 0
        self.load_image()
        
        self.update_ui()

    def update_ui(self):
        self.update_image_counter()
        self.update_label_counts()
        self.filter_indicator.config(text=f"Filter: {self.filter_mode} Images")

    def current_image_path(self):
        return self.filtered_image_list[self.current_image_index]
    def display_image(self):
        if not self.original_image:
            return

        self.canvas.update_idletasks()  # Update canvas size first
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()

        width, height = self.original_image.size

        scale_factor = self.calculate_scale_factor(width, height, canvas_width, canvas_height)
        self.zoom_level = scale_factor

        new_size = (max(int(width * self.zoom_level), 1), max(int(height * self.zoom_level), 1))  # Ensure size >= 1
        self.scaled_image = self.original_image.resize(new_size, Image.LANCZOS)
        self.photo = ImageTk.PhotoImage(self.scaled_image)

        self.canvas.delete("all")

        image_width, image_height = self.scaled_image.size
        self.image_x = max((canvas_width - image_width) // 2, 0)
        self.image_y = max((canvas_height - image_height) // 2, 0)

        self.canvas.create_image(self.image_x, self.image_y, anchor=tk.NW, image=self.photo)
        self.canvas.config(scrollregion=self.canvas.bbox(tk.ALL))

        self.draw_existing_bboxes()

        # Recreate crosshair after displaying image
        if hasattr(self, 'crosshair_h'):
            self.canvas.delete(self.crosshair_h)
            delattr(self, 'crosshair_h')
        if hasattr(self, 'crosshair_v'):
            self.canvas.delete(self.crosshair_v)
            delattr(self, 'crosshair_v')

    def calculate_scale_factor(self, img_w, img_h, canvas_w, canvas_h):
        if self.auto_resize.get():
            return min(canvas_w / img_w, canvas_h / img_h, self.user_zoom_level)
        else:
            return self.user_zoom_level

    def draw_existing_bboxes(self):
        self.canvas.delete("bbox")
        self.canvas.delete("bbox_label_bg")
        self.canvas.delete("bbox_label")
        self.canvas.delete("resize_handle")
        self.canvas.delete("internal_lines")
        self.canvas.delete("move_handle")
        self.canvas.delete("bbox_interior")
        self.bbox_items = []
        image_path = self.current_image_path()
        if image_path in self.annotations:
            for i, bbox in enumerate(self.annotations[image_path].bboxes):
                self.draw_bbox(bbox, i)

    def draw_bbox(self, bbox, index):
        x = bbox.x * self.zoom_level + self.image_x
        y = bbox.y * self.zoom_level + self.image_y
        w = bbox.w * self.zoom_level
        h = bbox.h * self.zoom_level
        color = self.get_color_for_label(bbox.category_id)
        
        # Determine outline width based on mode
        outline_width = 4 if self.edit_mode else 2
        
        # Draw bounding box
        bbox_item = self.canvas.create_rectangle(
            x, y, x + w, y + h, 
            outline=color, width=outline_width, tags=("bbox", f"bbox_{index}")
        )
        
        self.bbox_items.append(bbox_item)
        self.draw_bbox_label(x, y, bbox.category_id, color, index)
        # Draw other elements
        if self.edit_mode:
            self.draw_internal_lines(x, y, w, h, index)  # You may remove this if not needed
            self.draw_bbox_label(x, y, bbox.category_id, color, index)
            self.draw_resize_handles(x, y, w, h, index)
            self.draw_interior_overlay(x, y, w, h, index)
            self.bind_edge_cursors(x, y, w, h, index)

    def draw_internal_lines(
        self, x, y, w, h, index,
        draw_diagonals=True,
        draw_vertical=True,
        draw_horizontal=True,
        draw_oval=True
    ):
        # Vẽ các đường ngang nội bộ
        if draw_horizontal:
            num_horizontal_lines = 3  # Số lượng đường ngang
            spacing_horizontal = h / (num_horizontal_lines + 1)
            
            for i in range(1, num_horizontal_lines + 1):
                start_x = x  # Bắt đầu từ bên trái
                start_y = y + i * spacing_horizontal  # Điểm bắt đầu trên cạnh bên trái
                end_x = x + w  # Kết thúc ở bên phải
                end_y = start_y  # Đảm bảo đường kẻ nằm ngang
                self.canvas.create_line(
                    start_x, start_y, end_x, end_y,
                    fill='gray', dash=(2, 1), tags=("horizontal_lines", f"horizontal_lines_{index}")
                )
        
        # Vẽ các đường dọc nội bộ
        if draw_vertical:
            num_vertical_lines = 3  # Số lượng đường dọc
            spacing_vertical = w / (num_vertical_lines + 1)
            
            for i in range(1, num_vertical_lines + 1):
                start_x = x + i * spacing_vertical  # Điểm bắt đầu trên cạnh trên
                start_y = y  # Bắt đầu từ trên
                end_x = start_x  # Đảm bảo đường kẻ nằm dọc
                end_y = y + h  # Kết thúc ở dưới
                self.canvas.create_line(
                    start_x, start_y, end_x, end_y,
                    fill='gray', dash=(2, 1), tags=("vertical_lines", f"vertical_lines_{index}")
                )
        
        # Vẽ các đường chéo
        if draw_diagonals:
            # Đường chéo từ góc trên bên trái đến góc dưới bên phải
            self.canvas.create_line(
                x, y, x + w, y + h,
                fill='gray', dash=(2, 1), tags=("diagonal_lines", f"diagonal_lines_{index}")
            )
            # Đường chéo từ góc trên bên phải đến góc dưới bên trái
            self.canvas.create_line(
                x + w, y, x, y + h,
                fill='gray', dash=(2, 1), tags=("diagonal_lines", f"diagonal_lines_{index}")
            )
        
        # Vẽ oval với bounding box tương ứng
        if draw_oval:
            self.canvas.create_oval(
                x, y, x + w, y + h,
                outline='gray', width=2, tags=("oval", f"oval_{index}")
            )


    def draw_bbox_label(self, x, y, label, color, index):
        fnt = font.Font(family='Arial', size=12, weight='bold')
        text_width = fnt.measure(label)
        padding = 10  # Padding in pixels
        
        # Label background
        self.canvas.create_rectangle(
            x, y - 20, x + 5 + text_width + padding, y, 
            fill=color, outline=color, tags=("bbox_label_bg", f"bbox_label_bg_{index}")
        )
        
        # Label text
        self.canvas.create_text(
            x + 5, y - 10,
            text=label,
            fill='white',
            font=('Arial', 12, 'bold'),
            anchor='w',
            tags=("bbox_label", f"bbox_label_{index}")
        )

    def draw_resize_handles(self, x, y, w, h, index):
        handle_size = 6
        corners = [
            (x, y, 'tl'),  # top-left
            (x + w, y, 'tr'),  # top-right
            (x, y + h, 'bl'),  # bottom-left
            (x + w, y + h, 'br')  # bottom-right
        ]
        for cx, cy, corner in corners:
            handle = self.canvas.create_rectangle(
                cx - handle_size, cy - handle_size,
                cx + handle_size, cy + handle_size,
                fill='white', outline='black', 
                tags=("resize_handle", f"resize_handle_{index}_{corner}")
            )
            # Bind cursor changes to handles
            self.canvas.tag_bind(handle, "<Enter>", lambda e, c=corner: self.change_cursor_on_handle(c))
            self.canvas.tag_bind(handle, "<Leave>", lambda e: self.reset_cursor())

    def draw_interior_overlay(self, x, y, w, h, index):
        """Draw an invisible rectangle over the interior for detecting hover and clicks."""
        interior = self.canvas.create_rectangle(
            x + 5, y + 5, x + w - 5, y + h - 5,
            outline='', fill='', tags=("bbox_interior", f"bbox_interior_{index}")
        )
        # Bind cursor changes and movement to interior
        self.canvas.tag_bind(interior, "<Enter>", lambda e: self.canvas.config(cursor="fleur"))
        self.canvas.tag_bind(interior, "<Leave>", lambda e: self.reset_cursor())
        self.canvas.tag_bind(interior, "<ButtonPress-1>", lambda e, idx=index: self.initiate_move_bbox(idx))
        self.canvas.tag_bind(interior, "<B1-Motion>", lambda e: self.perform_move(e))
        self.canvas.tag_bind(interior, "<ButtonRelease-1>", lambda e: self.finish_move())

    def initiate_move_bbox(self, index):
        self.moving = True
        self.move_bbox_index = index
        bbox = self.annotations[self.current_image_path()].bboxes[index]
        self.original_bbox = BoundingBox(bbox.x, bbox.y, bbox.w, bbox.h, bbox.category_id)
        # Record the starting position
        self.move_start_x = None
        self.move_start_y = None

    def perform_move(self, event):
        if not self.moving or self.move_bbox_index is None:
            return
        if self.move_start_x is None and self.move_start_y is None:
            self.move_start_x, self.move_start_y = self.get_image_relative_coords(event.x, event.y)
            return
        current_x, current_y = self.get_image_relative_coords(event.x, event.y)
        dx = current_x - self.move_start_x
        dy = current_y - self.move_start_y
        bbox = self.annotations[self.current_image_path()].bboxes[self.move_bbox_index]
        new_x = max(0, min(self.original_bbox.x + dx / self.zoom_level, self.original_image.width - bbox.w))
        new_y = max(0, min(self.original_bbox.y + dy / self.zoom_level, self.original_image.height - bbox.h))
        bbox.x = new_x
        bbox.y = new_y
        self.display_image()

    def finish_move(self):
        if self.moving and self.move_bbox_index is not None:
            self.history.append((
                'move_bbox', 
                self.current_image_path(), 
                self.move_bbox_index, 
                self.original_bbox
            ))
        self.moving = False
        self.move_bbox_index = None
        self.move_start_x = None
        self.move_start_y = None

    def change_cursor_on_handle(self, corner):
        """Change cursor based on the corner."""
        cursors = {
            'tl': 'top_left_corner',
            'tr': 'top_right_corner',
            'bl': 'bottom_left_corner',
            'br': 'bottom_right_corner'
        }
        cursor = cursors.get(corner, 'arrow')
        self.canvas.config(cursor=cursor)

    def reset_cursor(self):
        """Reset cursor to default."""
        if self.edit_mode:
            self.canvas.config(cursor="cross")
        else:
            self.canvas.config(cursor="cross")

    def bind_edge_cursors(self, x, y, w, h, index):
        edge_thickness = 10  # Thickness for edge detection
        
        # Define edges: left, right, top, bottom
        edges = {
            'left': (x, y, x + edge_thickness, y + h),
            'right': (x + w - edge_thickness, y, x + w, y + h),
            'top': (x, y, x + w, y + edge_thickness),
            'bottom': (x, y + h - edge_thickness, x + w, y + h)
        }
        
        for edge, coords in edges.items():
            edge_item = self.canvas.create_rectangle(
                *coords,
                outline='', fill='', tags=(f"edge_{edge}_bbox_{index}",)
            )
            # Bind cursor changes
            if edge in ['left', 'right']:
                self.canvas.tag_bind(edge_item, "<Enter>", lambda e: self.canvas.config(cursor="fleur"))
                self.canvas.tag_bind(edge_item, "<Leave>", lambda e: self.reset_cursor())
            elif edge in ['top', 'bottom']:
                self.canvas.tag_bind(edge_item, "<Enter>", lambda e: self.canvas.config(cursor="fleur"))
                self.canvas.tag_bind(edge_item, "<Leave>", lambda e: self.reset_cursor())

    # ==================== Control Callbacks ==================== #
    def on_auto_resize_toggle(self):
        self.display_image()

    def on_window_resize(self, event):
        # Avoid calling display_image multiple times during resizing
        if self.auto_resize.get():
            self.display_image()

    def toggle_edit_mode(self):
        self.edit_mode = not self.edit_mode
        self.display_image()  # Refresh to show/hide resize handles

    def zoom(self, factor):
        self.user_zoom_level *= factor
        self.display_image()

    def reset_zoom(self):
        self.user_zoom_level = 1.0
        self.display_image()

    # ==================== Annotation Handling ==================== #
    def save_label(self, bbox_index=None, bbox=None):
        if bbox_index is not None and bbox is not None:
            self.update_existing_bbox_label(bbox_index, bbox)
        else:
            self.save_new_annotation()

    def update_existing_bbox_label(self, bbox_index, bbox):
        new_label = self.label_entry.get().strip()
        if new_label:
            old_label = bbox.category_id
            bbox.category_id = new_label
            self.history.append((
                'edit_label', 
                self.current_image_path(), 
                bbox_index, 
                old_label
            ))
            self.update_label_counts()
            self.display_image()
        self.label_entry.delete(0, tk.END)
    def save_new_annotation(self):
        new_label = self.label_entry.get().strip()
        if new_label and self.current_bbox:
            x1, y1, x2, y2 = self.canvas.coords(self.current_bbox)
            x = min(x1 - self.image_x, x2 - self.image_x) / self.zoom_level
            y = min(y1 - self.image_y, y2 - self.image_y) / self.zoom_level
            w = abs(x2 - x1) / self.zoom_level
            h = abs(y2 - y1) / self.zoom_level
            image_path = self.current_image_path()
            new_bbox = BoundingBox(x, y, w, h, new_label)
            self.annotations.setdefault(image_path, ImageAnnotation(image_path)).add_bbox(new_bbox)
            
            color = self.get_color_for_label(new_label)
            self.canvas.itemconfig(self.current_bbox, outline=color)
            
            self.history.append(('add', image_path, new_bbox))
            self.update_label_counts()
            self.apply_filter()
            
            if self.auto_next.get():
                self.next_image()
        else:
            if self.current_bbox:
                self.canvas.delete(self.current_bbox)
        self.current_bbox = None
    def undo(self):
        if not self.history:
            messagebox.showinfo("Info", "No actions to undo.")
            return
        
        action = self.history.pop()
        action_type = action[0]
        try:
            if action_type == 'add':
                _, image_path, bbox = action
                if image_path in self.annotations:
                    self.annotations[image_path].bboxes.remove(bbox)
            elif action_type in ['move_bbox', 'resize_bbox']:
                _, image_path, bbox_index, original_bbox = action
                if image_path in self.annotations and 0 <= bbox_index < len(self.annotations[image_path].bboxes):
                    self.annotations[image_path].bboxes[bbox_index] = original_bbox
            elif action_type == 'edit_label':
                _, image_path, bbox_index, old_label = action
                if image_path in self.annotations and 0 <= bbox_index < len(self.annotations[image_path].bboxes):
                    self.annotations[image_path].bboxes[bbox_index].category_id = old_label
            elif action_type == 'delete_bbox':
                _, image_path, bboxes_to_restore = action
                self.annotations[image_path].bboxes.extend(bboxes_to_restore)  # Khôi phục lại bbox
            elif action_type == 'delete_image':
                _, image_path, index, annotation = action
                self.image_list.insert(index, image_path)
                self.filtered_image_list.insert(index, image_path)
                self.annotations[image_path] = annotation
                self.current_image_index = index
            self.display_image()
            self.update_label_counts()
            self.apply_filter()
        except Exception as e:
            messagebox.showerror("Error", f"Cannot undo action:\n{e}")
    def delete_bbox(self):
        image_path = self.current_image_path()
        if image_path in self.annotations and self.annotations[image_path].bboxes:
            # Lưu lại tất cả các bounding box trước khi xóa
            bboxes_to_delete = self.annotations[image_path].bboxes.copy()
            
            # Xóa tất cả bounding box
            self.annotations[image_path].bboxes.clear()
            
            # Ghi lại hành động vào lịch sử
            self.history.append(('delete_bbox', image_path, bboxes_to_delete))
            
            # Cập nhật hiển thị
            self.display_image()
            self.update_label_counts()
            self.apply_filter()
        else:
            messagebox.showinfo("Info", "No bounding box to delete.")
    def delete_specific_bbox(self, index):
        image_path = self.current_image_path()
        if image_path in self.annotations and 0 <= index < len(self.annotations[image_path].bboxes):
            deleted_bbox = self.annotations[image_path].bboxes.pop(index)
            self.history.append(('delete_bbox', image_path, index, deleted_bbox))
            self.display_image()
            self.update_label_counts()
            self.apply_filter()
        else:
            messagebox.showinfo("Info", "Bounding box not found at the specified index.")
    def delete_image(self):
        image_path = self.current_image_path()
        if messagebox.askyesno("Confirm Delete", f"Are you sure you want to delete {os.path.basename(image_path)}?"):
            try:
                os.remove(image_path)
                annotation = self.annotations.pop(image_path, None)
                self.history.append(('delete_image', image_path, self.current_image_index, annotation))
                self.image_list.remove(image_path)
                self.filtered_image_list.remove(image_path)
                self.apply_filter()
                if self.filtered_image_list:
                    self.current_image_index = min(self.current_image_index, len(self.filtered_image_list) - 1)
                    self.load_image()
                else:
                    self.clear_canvas()
                    messagebox.showinfo("Info", "No more images to display.")
            except OSError as e:
                messagebox.showerror("Error", f"Error deleting file: {e}")

    # ==================== Navigation ==================== #
    def next_image(self):
        if self.current_image_index < len(self.filtered_image_list) - 1:
            self.current_image_index += 1
            self.load_image()
        else:
            messagebox.showinfo("Info", "This is the last image.")

    def prev_image(self):
        if self.current_image_index > 0:
            self.current_image_index -= 1
            self.load_image()
        else:
            messagebox.showinfo("Info", "This is the first image.")

    # ==================== Label Handling ==================== #
    def count_labels(self):
        label_counts = {}
        for annotation in self.annotations.values():
            for bbox in annotation.bboxes:
                label_counts[bbox.category_id] = label_counts.get(bbox.category_id, 0) + 1
        return label_counts

    def update_label_counts(self):
        label_counts = self.count_labels()
        
        # Remove labels with zero count from label_colors
        labels_to_remove = [label for label in self.label_colors if label_counts.get(label, 0) == 0]
        for label in labels_to_remove:
            del self.label_colors[label]
        
        # Refresh the label list
        self.refresh_label_list()
        
        if label_counts:
            count_text = ", ".join([f"{label}: {count}" for label, count in label_counts.items()])
        else:
            count_text = "None"
        self.label_count_label.config(text=f"Label counts: {count_text}")

    def get_color_for_label(self, label):
        """Retrieve or assign a color for the given label."""
        if label not in self.label_colors:
            # Generate a random color for a new label
            color = "#{:06x}".format(random.randint(0, 0xFFFFFF))
            self.label_colors[label] = color
            self.refresh_label_list()  # Refresh the label list when a new label is added
        return self.label_colors[label]

    # ==================== Import/Export ==================== #
    def load_annotations(self):
        file_path = filedialog.askopenfilename(filetypes=[("JSON files", "*.json")])
        if file_path:
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
            except Exception as e:
                messagebox.showerror("Error", f"Cannot load annotations:\n{e}")
                return
            
            self.parse_coco_annotations(data)
            self.apply_filter()
            messagebox.showinfo("Success", "Annotations loaded successfully.")

    def parse_coco_annotations(self, data):
        self.annotations = {}
        image_map = {img['id']: img['file_name'] for img in data.get('images', [])}
        category_names = {cat['id']: cat['name'] for cat in data.get('categories', [])}
        
        for ann in data.get('annotations', []):
            image_id = ann['image_id']
            file_name = image_map.get(image_id)
            if not file_name:
                continue
            full_path = next((img for img in self.image_list if os.path.basename(img) == file_name), None)
            if not full_path:
                continue
            
            x, y, w, h = ann['bbox']
            x, y, w, h = max(0, x), max(0, y), max(1, w), max(1, h)
            category_name = category_names.get(ann['category_id'], "unknown")
            bbox = BoundingBox(x, y, w, h, category_name)
            self.annotations.setdefault(full_path, ImageAnnotation(full_path)).add_bbox(bbox)
            self.get_color_for_label(category_name)

    def save_annotations_auto(self):
        """
        Automatically saves annotations in COCO format with a timestamped filename.
        The file is saved in the directory of the first loaded image.
        """
        if not self.annotations:
            messagebox.showinfo("Info", "No annotations to save.")
            return

        if not self.image_list:
            messagebox.showinfo("Info", "No images loaded to determine save directory.")
            return

        # Generate timestamped filename
        timestamp = datetime.datetime.now().strftime("%H_%M_%d_%m_%Y")
        filename = f"auto_save_{timestamp}.json"

        # Determine the save directory (same as the first image's directory)
        save_dir = os.path.dirname(self.image_list[0])
        file_path = os.path.join(save_dir, filename)

        # Get the COCO exporter
        exporter = get_exporter("coco")

        try:
            exporter.export(self.annotations, file_path)
            messagebox.showinfo("Success", f"Annotations saved to {file_path}.")
        except Exception as e:
            messagebox.showerror("Error", f"Cannot save annotations:\n{e}")

    def export(self, format_):
        if not self.annotations:
            messagebox.showinfo("Info", "No annotations to export.")
            return

        exporter = get_exporter(format_)
        
        try:
            if format_ == "tfrecord":
                self.export_tfrecord(exporter)
            elif format_ in ["yolov8", "pascal_voc", "dataset_coco"]:
                self.export_to_directory(exporter, format_)
            elif format_ == "coco":
                self.export_to_coco(exporter)
            elif format_ == "excel":
                self.export_to_excel(exporter)
            else:
                messagebox.showerror("Error", "Unsupported export format.")
        except Exception as e:
            messagebox.showerror("Error", f"Cannot export annotations:\n{e}")

    def export_tfrecord(self, exporter):
        file_path = filedialog.asksaveasfilename(
            defaultextension=".tfrecord", 
            filetypes=[("TFRecord files", "*.tfrecord")]
        )
        if file_path:
            pbtxt_path = os.path.splitext(file_path)[0] + ".pbtxt"
            exporter.export(self.annotations, file_path, pbtxt_path)
            messagebox.showinfo("Success", f"Exported to TFRecord format and created label map.")

    def export_to_directory(self, exporter, format_):
        output_dir = filedialog.askdirectory(title=f"Select output directory for {format_.upper()}")
        if output_dir:
            exporter.export(self.annotations, output_dir)
            messagebox.showinfo("Success", f"Exported to {format_.upper()} format at {output_dir}.")

    def export_to_coco(self, exporter):
        file_path = filedialog.asksaveasfilename(
            defaultextension=".json", 
            filetypes=[("JSON files", "*.json")]
        )
        if file_path:
            exporter.export(self.annotations, file_path)
            messagebox.showinfo("Success", f"Exported to COCO format at {file_path}.")

    def export_to_excel(self, exporter):
        file_path = filedialog.asksaveasfilename(
            defaultextension=".xlsx", 
            filetypes=[("Excel files", "*.xlsx")]
        )
        if file_path:
            exporter.export(self.annotations, file_path)
            messagebox.showinfo("Success", f"Exported to Excel format at {file_path}.")
    from BoxLabeler.exporters import get_exporter
    
    def export_to_directory(self, exporter, format_):
        output_dir = filedialog.askdirectory(title=f"Select output directory for {format_.upper()}")
        if output_dir:
            exporter.export(self.annotations, output_dir)
            messagebox.showinfo("Success", f"Exported to {format_.upper()} format at {output_dir}.")
            
    def import_yolov8_model(self):
        if self.yolov8_model.import_model():
            messagebox.showinfo("Success", "YOLO model imported successfully.")
            self.current_model = self.yolov8_model
        else:
            messagebox.showerror("Error", "Failed to import YOLO model.")

    def predict(self):
        if not self.original_image:
            messagebox.showwarning("Warning", "Please load an image first.")
            return

        if not self.current_model:
            messagebox.showwarning("Warning", "Please import a model first.")
            return

        image_path = self.current_image_path()
        image = cv2.imread(image_path)
        if image is None:
            messagebox.showerror("Error", f"Cannot read image: {image_path}")
            return
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        try:
            predictions = self.current_model.predict(image_rgb)
        except ValueError as e:
            messagebox.showerror("Error", str(e))
            return

        # Clear existing bounding boxes
        self.annotations[image_path] = ImageAnnotation(image_path)

        # Add predicted bounding boxes
        for pred in predictions:
            bbox = BoundingBox(*pred['bbox'], pred['class'])
            self.annotations[image_path].add_bbox(bbox)

        self.display_image()
        self.update_label_counts()
        self.apply_filter()

    # ==================== Auto Predict ==================== #
    def auto_predict(self):
        if not self.image_list:
            messagebox.showwarning("Warning", "No images loaded to predict.")
            return

        if not self.current_model:
            messagebox.showwarning("Warning", "Please import a model first.")
            return

        # Create a new window for progress
        self.progress_window = tk.Toplevel(self.master)
        self.progress_window.title("Auto Predict")
        self.progress_window.geometry("400x100")
        self.progress_window.grab_set()  # Make the progress window modal

        tk.Label(self.progress_window, text="Auto Predict in progress...").pack(pady=10)

        self.progress_bar = ttk.Progressbar(self.progress_window, orient="horizontal", length=300, mode="determinate")
        self.progress_bar.pack(pady=5)

        self.cancel_button = tk.Button(self.progress_window, text="Cancel", command=self.cancel_auto_predict)
        self.cancel_button.pack(pady=5)

        # Initialize progress variables
        self.progress_bar['maximum'] = len(self.image_list)
        self.auto_predict_cancel_flag = False

        # Start the auto_predict in a separate thread
        self.auto_predict_thread = threading.Thread(target=self.process_auto_predict)
        self.auto_predict_thread.start()

    def process_auto_predict(self):
        """Worker function to process auto prediction."""
        try:
            for idx, image_path in enumerate(self.image_list):
                if self.auto_predict_cancel_flag:
                    break

                # Read and predict
                image = cv2.imread(image_path)
                if image is None:
                    continue
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                try:
                    predictions = self.current_model.predict(image_rgb)
                except ValueError:
                    continue

                # Clear existing bounding boxes
                self.annotations[image_path] = ImageAnnotation(image_path)

                # Add predicted bounding boxes
                for pred in predictions:
                    bbox = BoundingBox(*pred['bbox'], pred['class'])
                    self.annotations[image_path].add_bbox(bbox)

                # Update progress bar
                self.master.after(0, self.update_progress, idx + 1)

            # After processing, export annotations
            if not self.auto_predict_cancel_flag:
                timestamp = datetime.datetime.now().strftime("%H_%M_%d_%m_%Y")
                filename = f"auto_label_{timestamp}.json"
                exporter = get_exporter("coco")
                file_path = os.path.join(os.path.dirname(self.image_list[0]), filename)
                exporter.export(self.annotations, file_path)
                self.master.after(0, lambda: messagebox.showinfo("Success", f"Auto prediction completed and saved to {file_path}"))
            else:
                self.master.after(0, lambda: messagebox.showinfo("Cancelled", "Auto prediction was cancelled."))

        except Exception as e:
            self.master.after(0, lambda: messagebox.showerror("Error", f"An error occurred during auto prediction:\n{e}"))
        finally:
            self.master.after(0, self.finish_auto_predict)

    def update_progress(self, value):
        self.progress_bar['value'] = value

    def cancel_auto_predict(self):
        self.auto_predict_cancel_flag = True
        self.cancel_button.config(state='disabled')

    def finish_auto_predict(self):
        self.progress_window.destroy()
        self.display_image()
        self.update_label_counts()
        self.apply_filter()

    # ==================== Annotation Handling Continued ==================== #
    # (No changes needed here since labels are now dynamically updated)

    # ==================== Utility Methods ==================== #
    def get_image_relative_coords(self, x, y):
        return x - self.image_x, y - self.image_y

    def show_about(self):
        info = (
            "Product Name: BoxLabeler\n"
            "Version: 1.0.0\n"
            "Description: Image annotation tool with bounding boxes.\n"
            "GitHub: @ti014\n"
            "Copyright: © 2024 Phuong Phan Nguyen Mai. All rights reserved.\n"
        )
        messagebox.showinfo("About BoxLabeler", info)

    def show_shortcuts(self):
        shortcuts = (
            "Ctrl + Z/z: Undo last action\n"
            "Tab: Next image\n"
            "Shift + Tab: Previous image\n"
            "Alt + D/d: Detect and Predict objects in the current image\n"
            "Alt + R/r: Delete the current image\n"
            "Alt + B/b: Delete the last bounding box\n"
            "Alt + W/w: Clear the label entry field\n"
            "Ctrl + +: Zoom In\n"
            "Ctrl + -: Zoom Out\n"
            "Ctrl + 0: Reset Zoom"
        )
        messagebox.showinfo("Keyboard Shortcuts", shortcuts)

    def update_image_counter(self):
        self.image_counter.config(
            text=f"Image: {self.current_image_index + 1} / {len(self.filtered_image_list)}"
        )
