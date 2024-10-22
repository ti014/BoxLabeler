import os
from PIL import Image
from BoxLabeler.exporters.base import Exporter

class YOLOv8Exporter(Exporter):
    def export(self, annotations, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        
        categories = sorted({bbox.category_id for ann in annotations.values() for bbox in ann.bboxes})
        category_to_id = {cat: i for i, cat in enumerate(categories)}
        
        with open(os.path.join(output_dir, 'classes.txt'), 'w') as f:
            for cat in categories:
                f.write(f"{cat}\n")
        
        for image_path, annotation in annotations.items():
            try:
                img = Image.open(image_path)
                img_width, img_height = img.size
            except Exception as e:
                print(f"Error opening image file: {image_path}\n{e}")
                continue
            
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            txt_path = os.path.join(output_dir, f"{base_name}.txt")
            
            with open(txt_path, 'w') as f:
                for bbox in annotation.bboxes:
                    x_center = (bbox.x + bbox.w / 2) / img_width
                    y_center = (bbox.y + bbox.h / 2) / img_height
                    width = bbox.w / img_width
                    height = bbox.h / img_height
                    class_id = category_to_id[bbox.category_id]
                    
                    f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")
