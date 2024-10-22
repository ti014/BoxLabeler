from tkinter import filedialog, messagebox
from ultralytics import YOLO
import numpy as np

class YoloV8ImportModel:
    def __init__(self):
        self.model = None
        self.model_path = None

    def import_model(self):
        self.model_path = filedialog.askopenfilename(filetypes=[("YOLO model", "*.pt")])
        if self.model_path:
            try:
                self.model = YOLO(self.model_path)
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

        # Convert to numpy arrays for easier manipulation
        boxes = np.array(boxes)
        scores = np.array(scores)

        # Compute the area of the bounding boxes
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)

        # Sort the bounding boxes by their scores in descending order
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)

            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)
            inter = w * h

            iou = inter / (areas[i] + areas[order[1:]] - inter)

            inds = np.where(iou <= iou_threshold)[0]
            order = order[inds + 1]

        return keep

    def predict(self, image, iou_threshold=0.5, conf_threshold=0.25):
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
