import os
import json
from PIL import Image
from BoxLabeler.annotations.image_annotation import ImageAnnotation
from BoxLabeler.exporters.base import Exporter

class COCOExporter(Exporter):
    def export(self, annotations, output_path):
        coco_format = {
            "images": [],
            "annotations": [],
            "categories": []
        }
        
        category_dict = {}
        annotation_id = 0

        for image_path, annotation in annotations.items():
            if not os.path.isfile(image_path):
                print(f"Warning: Image file not found: {image_path}")
                continue

            try:
                with Image.open(image_path) as img:
                    width, height = img.size
            except Exception as e:
                print(f"Error opening image file: {image_path}\n{e}")
                continue

            image_id = len(coco_format["images"])
            coco_format["images"].append({
                "id": image_id,
                "width": width,
                "height": height,
                "file_name": os.path.basename(image_path)
            })
            
            for bbox in annotation.bboxes:
                if bbox.category_id not in category_dict:
                    category_dict[bbox.category_id] = len(category_dict) + 1
                    coco_format["categories"].append({
                        "id": category_dict[bbox.category_id],
                        "name": bbox.category_id,
                        "supercategory": "none"
                    })

                area = bbox.w * bbox.h
                coco_format["annotations"].append({
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": category_dict[bbox.category_id],
                    "segmentation": [],
                    "area": area,
                    "bbox": [bbox.x, bbox.y, bbox.w, bbox.h],
                    "iscrowd": 0
                })
                annotation_id += 1
        
        with open(output_path, 'w') as f:
            json.dump(coco_format, f, indent=4)
