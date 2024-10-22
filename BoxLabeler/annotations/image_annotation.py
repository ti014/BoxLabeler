class ImageAnnotation:
    def __init__(self, image_path):
        self.image_path = image_path
        self.bboxes = []

    def add_bbox(self, bbox):
        self.bboxes.append(bbox)

    def remove_bbox(self, index):
        if 0 <= index < len(self.bboxes):
            del self.bboxes[index]
