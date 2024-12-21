import base64

# Đường dẫn tới tệp ảnh
file_path = 'generated_img_1_0.png'

# Đọc tệp ảnh ở chế độ nhị phân và chuyển đổi thành Base64
with open(file_path, "rb") as image_file:
    # Chuyển ảnh sang Base64
    base64_string = base64.b64encode(image_file.read()).decode('utf-8')

# In kết quả Base64
print(base64_string)

# # Nếu muốn lưu chuỗi Base64 vào file:
# output_path = './output.txt'
# with open(output_path, 'w') as output_file:
#     output_file.write(base64_string)