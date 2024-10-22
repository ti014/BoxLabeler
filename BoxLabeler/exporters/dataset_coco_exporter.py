import os
import json
import shutil
from PIL import Image
from BoxLabeler.exporters.base import Exporter

class DatasetCocoExporter(Exporter):
    def export(self, annotations, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        
        train_images_dir = os.path.join(output_dir,"dataset", "train")
        val_images_dir = os.path.join(output_dir,"dataset", "val")
        annotations_dir = os.path.join(output_dir,"dataset", "annotations")
        os.makedirs(train_images_dir, exist_ok=True)
        os.makedirs(val_images_dir, exist_ok=True)
        os.makedirs(annotations_dir, exist_ok=True)

        coco_format_train = {
            "images": [],
            "annotations": [],
            "categories": []
        }
        coco_format_val = {
            "images": [],
            "annotations": [],
            "categories": []
        }

        category_dict = {}
        annotation_id = 0
        image_id = 0

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

            # Determine if the image should go to train or val
            if image_id % 5 == 0:  # 20% for validation
                images_dir = val_images_dir
                coco_format = coco_format_val
            else:  # 80% for training
                images_dir = train_images_dir
                coco_format = coco_format_train

            # Copy image to the appropriate directory
            shutil.copy(image_path, images_dir)

            image_info = {
                "id": image_id,
                "width": width,
                "height": height,
                "file_name": os.path.basename(image_path)
            }
            coco_format["images"].append(image_info)

            for bbox in annotation.bboxes:
                if bbox.category_id not in category_dict:
                    category_dict[bbox.category_id] = len(category_dict) + 1
                    category_info = {
                        "id": category_dict[bbox.category_id],
                        "name": bbox.category_id,
                        "supercategory": "none"
                    }
                    coco_format_train["categories"].append(category_info)
                    coco_format_val["categories"].append(category_info)

                area = bbox.w * bbox.h
                annotation_info = {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": category_dict[bbox.category_id],
                    "segmentation": [],
                    "area": area,
                    "bbox": [bbox.x, bbox.y, bbox.w, bbox.h],
                    "iscrowd": 0
                }
                coco_format["annotations"].append(annotation_info)
                annotation_id += 1

            image_id += 1

        with open(os.path.join(annotations_dir, 'train.json'), 'w') as f:
            json.dump(coco_format_train, f, indent=4)

        with open(os.path.join(annotations_dir, 'val.json'), 'w') as f:
            json.dump(coco_format_val, f, indent=4)