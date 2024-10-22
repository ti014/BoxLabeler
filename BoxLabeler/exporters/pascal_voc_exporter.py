import os
from PIL import Image
import xml.etree.ElementTree as ET
from BoxLabeler.exporters.base import Exporter

class PascalVOCExporter(Exporter):
    def export(self, annotations, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        
        for image_path, annotation in annotations.items():
            try:
                img = Image.open(image_path)
                img_width, img_height = img.size
            except Exception as e:
                print(f"Error opening image file: {image_path}\n{e}")
                continue
            
            root = ET.Element("annotation")
            ET.SubElement(root, "filename").text = os.path.basename(image_path)
            
            size = ET.SubElement(root, "size")
            ET.SubElement(size, "width").text = str(img_width)
            ET.SubElement(size, "height").text = str(img_height)
            ET.SubElement(size, "depth").text = str(3)  # Assuming RGB images
            
            for bbox in annotation.bboxes:
                obj = ET.SubElement(root, "object")
                ET.SubElement(obj, "name").text = bbox.category_id
                ET.SubElement(obj, "pose").text = "Unspecified"
                ET.SubElement(obj, "truncated").text = "0"
                ET.SubElement(obj, "difficult").text = "0"
                
                bndbox = ET.SubElement(obj, "bndbox")
                ET.SubElement(bndbox, "xmin").text = str(int(bbox.x))
                ET.SubElement(bndbox, "ymin").text = str(int(bbox.y))
                ET.SubElement(bndbox, "xmax").text = str(int(bbox.x + bbox.w))
                ET.SubElement(bndbox, "ymax").text = str(int(bbox.y + bbox.h))
            
            tree = ET.ElementTree(root)
            xml_path = os.path.join(output_dir, os.path.splitext(os.path.basename(image_path))[0] + ".xml")
            tree.write(xml_path, encoding="utf-8", xml_declaration=True)
