from abc import ABC, abstractmethod

class Exporter(ABC):
    @abstractmethod
    def export(self, annotations, output_path, *args, **kwargs):
        """
        Xuất annotations theo định dạng cụ thể.
        
        :param annotations: Dictionary mapping image paths to ImageAnnotation objects.
        :param output_path: Path để lưu file xuất ra hoặc thư mục.
        """
        pass
