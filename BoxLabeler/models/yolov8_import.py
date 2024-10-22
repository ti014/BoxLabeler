from tkinter import filedialog, messagebox
from ultralytics import YOLO
import numpy as np
import torch
import torchvision.ops

class YoloV8ImportModel:
    def __init__(self):
        self.model = None
        self.model_path = None

    def import_model(self):
        self.model_path = filedialog.askopenfilename(filetypes=[("YOLO model", "*.pt")])
        if self.model_path:
            try:
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                self.model = YOLO(self.model_path).to(device)
                return True
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load YOLO model:\n{e}")
        return False

    def non_max_suppression(self, boxes, scores, iou_threshold):
        """
        Perform Non-Maximum Suppression (NMS) on the bounding boxes.

        Args:
            boxes (List[List[float]]): List of bounding boxes [x1, y1, x2, y2].
            scores (List[float]): List of confidence scores for each bounding box.
            iou_threshold (float): IoU threshold for NMS.

        Returns:
            List[int]: Indices of bounding boxes to keep.
        """
        if len(boxes) == 0:
            return []

        # Convert to torch tensors for faster computation
        boxes = torch.tensor(boxes)
        scores = torch.tensor(scores)

        # Use torchvision's built-in NMS function
        keep = torchvision.ops.nms(boxes, scores, iou_threshold)

        return keep.tolist()

    def predict(self, image, iou_threshold=0.5, conf_threshold=0.5):
        """
        Perform prediction on the given image and return all detected bboxes.

        Args:
            image (numpy.ndarray): The input image in RGB format.
            iou_threshold (float): IoU threshold for NMS.
            conf_threshold (float): Confidence threshold for filtering predictions.

        Returns:
            List[Dict]: A list of annotations with 'bbox', 'class', and 'confidence' keys.
        """
        if self.model is None:
            raise ValueError("Model not imported. Please import a model first.")

        # Perform prediction using YOLO model
        results = self.model(image, verbose=False)

        # Storage for all detected bboxes
        annotations = []

        # Iterate through each result
        for r in results:
            boxes = r.boxes.xyxy.cpu().numpy()  # Bounding boxes
            scores = r.boxes.conf.cpu().numpy()  # Confidence scores
            classes = r.boxes.cls.cpu().numpy()  # Class IDs

            # Filter out low-confidence predictions
            filtered_indices = [i for i, score in enumerate(scores) if score >= conf_threshold]
            boxes = boxes[filtered_indices]
            scores = scores[filtered_indices]
            classes = classes[filtered_indices]

            # Apply NMS
            keep = self.non_max_suppression(boxes, scores, iou_threshold)

            for idx in keep:
                x1, y1, x2, y2 = boxes[idx]
                x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)
                cls = int(classes[idx])
                class_name = self.model.names[cls]
                confidence = float(scores[idx])

                # Append bbox with confidence
                annotations.append({
                    'bbox': [x1, y1, x2 - x1, y2 - y1],  # [x, y, width, height]
                    'class': class_name,
                    'confidence': confidence
                })

        return annotations