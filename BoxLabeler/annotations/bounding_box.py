class BoundingBox:
    def __init__(self, x, y, w, h, category_id):
        self.x = x  # Top-left x coordinate (relative to original image)
        self.y = y  # Top-left y coordinate (relative to original image)
        self.w = w  # Width of the bounding box
        self.h = h  # Height of the bounding box
        self.category_id = category_id  # Label of the bounding box
