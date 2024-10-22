import os
import tensorflow as tf
import io
from PIL import Image
from BoxLabeler.exporters.base import Exporter

class TFRecordExporter(Exporter):
    def export(self, annotations, output_path, pbtxt_path):
        categories = sorted({bbox.category_id for ann in annotations.values() for bbox in ann.bboxes})
        category_to_id = {cat: i + 1 for i, cat in enumerate(categories)}  # IDs start at 1
        
        self.write_pbtxt_file(category_to_id, pbtxt_path)
        
        with tf.io.TFRecordWriter(output_path) as writer:
            for image_path, annotation in annotations.items():
                try:
                    tf_example = self.create_tf_example(image_path, annotation, category_to_id)
                    writer.write(tf_example.SerializeToString())
                except Exception as e:
                    print(f"Error creating TFExample for {image_path}: {e}")

    def create_tf_example(self, image_path, annotation, category_to_id):
        with tf.io.gfile.GFile(image_path, 'rb') as fid:
            encoded_jpg = fid.read()
        encoded_jpg_io = io.BytesIO(encoded_jpg)
        image = Image.open(encoded_jpg_io)
        width, height = image.size

        filename = os.path.basename(image_path).encode('utf8')
        image_format = os.path.splitext(image_path)[1][1:].encode('utf8')  # e.g., 'jpg'

        xmins, xmaxs, ymins, ymaxs = [], [], [], []
        classes_text, classes = [], []

        for bbox in annotation.bboxes:
            xmins.append(bbox.x / width)
            xmaxs.append((bbox.x + bbox.w) / width)
            ymins.append(bbox.y / height)
            ymaxs.append((bbox.y + bbox.h) / height)
            class_name = str(bbox.category_id)
            classes_text.append(class_name.encode('utf8'))
            classes.append(category_to_id[bbox.category_id])

        tf_example = tf.train.Example(features=tf.train.Features(feature={
            'image/height': self.int64_feature(height),
            'image/width': self.int64_feature(width),
            'image/filename': self.bytes_feature(filename),
            'image/source_id': self.bytes_feature(filename),
            'image/encoded': self.bytes_feature(encoded_jpg),
            'image/format': self.bytes_feature(image_format),
            'image/object/bbox/xmin': self.float_list_feature(xmins),
            'image/object/bbox/xmax': self.float_list_feature(xmaxs),
            'image/object/bbox/ymin': self.float_list_feature(ymins),
            'image/object/bbox/ymax': self.float_list_feature(ymaxs),
            'image/object/class/text': self.bytes_list_feature(classes_text),
            'image/object/class/label': self.int64_list_feature(classes),
        }))
        return tf_example

    def write_pbtxt_file(self, category_to_id, pbtxt_path):
        with open(pbtxt_path, 'w') as f:
            for category, id_ in category_to_id.items():
                f.write(f"item {{\n  id: {id_}\n  name: '{category}'\n}}\n")

    def int64_feature(self, value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    def int64_list_feature(self, value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

    def bytes_feature(self, value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def bytes_list_feature(self, value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

    def float_list_feature(self, value):
        return tf.train.Feature(float_list=tf.train.FloatList(value=value))
