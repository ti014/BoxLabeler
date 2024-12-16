@echo off
:: Xác định thư mục hiện tại (nơi file .bat đang được chạy)
set CURRENT_DIR=%~dp0
:: Điều hướng đến thư mục hiện tại
cd /d %CURRENT_DIR%
:: Kích hoạt virtual environment
call .venv\Scripts\activate
:: Chạy script Python
python .\main.py
:: Giữ cửa sổ mở nếu cần xem kết quả
pause
