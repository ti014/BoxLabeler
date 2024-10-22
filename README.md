# BoxLabeler

## Giới Thiệu

BoxLabeler là một công cụ gán nhãn (label) các đối tượng trong hình ảnh, hỗ trợ các mô hình học máy trong việc nhận diện và phân loại đối tượng. Công cụ này cho phép người dùng thêm, chỉnh sửa, xóa và di chuyển các bounding box (hộp giới hạn) trên hình ảnh, cũng như dự đoán tự động các bounding box sử dụng mô hình YOLOv8.

## Tính Năng

- Gán nhãn thủ công: Thêm, chỉnh sửa, xóa và di chuyển các bounding box.
- Dự đoán tự động: Sử dụng mô hình YOLOv8 để dự đoán các bounding box.
- Quản lý lịch sử: Hỗ trợ chức năng undo (hoàn tác) các hành động.
- Xuất dữ liệu: Hỗ trợ xuất dữ liệu gán nhãn theo các định dạng phổ biến như COCO, Pascal VOC, TFRecord.

## Cài Đặt

### Yêu Cầu Hệ Thống

- Python 3.8 trở lên
- Các thư viện cần thiết (được liệt kê trong `requirements.txt`)

### Cài Đặt Thư Viện

Sử dụng pip để cài đặt các thư viện cần thiết:

```bash
pip install -r requirements.txt
```

## Sử Dụng
### Chạy Ứng Dụng
Chạy tệp main.py để khởi động ứng dụng:

### Gán Nhãn Thủ Công
1. Mở hình ảnh cần gán nhãn.
2. Sử dụng chuột để vẽ các bounding box trên hình ảnh.
3. Nhấn giữ chuột trái và di chuyển để di chuyển bounding box.

### Dự Đoán Tự Động
1. Nhập mô hình YOLOv8.
2. Chọn hình ảnh cần dự đoán.
3. Nhấn nút "Predict" để dự đoán các bounding box.

### Xuất Dữ Liệu
1. Chọn định dạng xuất dữ liệu (COCO, Pascal VOC, TFRecord).
2. Chọn thư mục lưu trữ.
3. Nhấn nút "Export" để xuất dữ liệu.

## Đóng Góp
Tôi rất hoan nghênh các đóng góp từ cộng đồng. Vui lòng tạo pull request hoặc mở issue trên GitHub để thảo luận về các tính năng mới hoặc báo cáo lỗi.

### Giấy Phép
Dự án này được cấp phép theo giấy phép MIT. Xem tệp LICENSE để biết thêm chi tiết.
