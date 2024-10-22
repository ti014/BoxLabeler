import os
import pandas as pd
from PIL import Image
from BoxLabeler.exporters.base import Exporter

class ExcelExporter(Exporter):
    def __init__(self):
        self.column_names = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']

    def export(self, annotations, output_path):
        data = []
        for image_path, annotation in annotations.items():
            filename = os.path.basename(image_path)
            
            try:
                with Image.open(image_path) as img:
                    img_width, img_height = img.size
            except FileNotFoundError:
                print(f"Warning: Image file not found: {image_path}")
                continue
            except IOError:
                print(f"Warning: Unable to open image file: {image_path}")
                continue
            
            for bbox in annotation.bboxes:
                xmin, ymin = bbox.x, bbox.y
                xmax, ymax = bbox.x + bbox.w, bbox.y + bbox.h
                class_name = bbox.category_id
                
                data.append([filename, img_width, img_height, class_name, xmin, ymin, xmax, ymax])
        
        df = pd.DataFrame(data, columns=self.column_names)
        df.to_excel(output_path, index=False, engine='openpyxl')
