import base64

# Chuỗi Base64 cần chuyển đổi
base64_string = "iVBORw0KGgoAAAANSUhEUgAAABwAAAAcCAAAAABXZoBIAAACCUlEQVR4nEWQvW8TQRTE573d29uL7dhORIwTCWQSBYFRFCASBQU0NHwI0dABDdT8pUghEm2kUDgkzgckF/s+dvdR3F3yutmfZkY7tOPPKTEigRGKMrBdHr96uqSR/dnb17/yQKSipDtY66vArdX7jwaGACNLBzoTiIQyuzidDnvWhHnQigBwHCuN6gLmZ5Fu91dGd7sKAMhEotlXVHyW5mZttDG0BABgV3C7tkoo0wu3sNxLuHqgqxmP0FA/OzlOc99oOr/ktwBABCCU/w72D08zqWk653e4vlCcHx5PL/11FI8BUCNLHxlqnFbp5MYJsutb6z1Vq+6CrlwkACh+8HqrY+oc6SQaAITAArKbn570NDUl1tYLEUMtbn9+2dc3JdrUsRzb4YsPD7sKkMAEACE021Ky+vzDuM0SQqiDfREaaJY3blsW7xxEVb8spIJEUasTIYgvHCsRAqR0qKFKliIK3ueFI0RMEHGq7qRoQc4muZtOTjPujzcXNbzYhIQAmMVBy/SKydHMC5uVN1/uGT872K1gFKvChVCvyp3330fKTX9wNV6eZqVvFg9XexOnItvSAgAijQsAoIarVgcTMwAovh4UAMz42yhmiGhEJcVwABFEBKQ6j78+syR+NtfY/ml68xysTMzOiWrd2bnlCy+zv5l2H9PcMFh00op86djS0W46aNPs98l/W/nivvraIPUAAAAASUVORK5CYII="

# Đường dẫn lưu file ảnh kết quả
output_path = './output_image.jpg'

# Giải mã chuỗi Base64 và ghi ra file ảnh
with open(output_path, "wb") as output_file:
    # Giải mã Base64 về nhị phân
    image_data = base64.b64decode(base64_string)
    output_file.write(image_data)

# print(f"File ảnh đã được lưu tại: {output_path}")
