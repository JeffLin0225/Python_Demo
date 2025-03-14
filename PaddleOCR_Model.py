from paddleocr import PaddleOCR
from PIL import Image, ImageDraw

# 初始化 PaddleOCR
ocr = PaddleOCR(show_log=False,use_angle_cls=True, lang='ch')

# 指定圖片路徑
image_path = '/Users/linjiaxian/Desktop/789.jpg'

# 執行 OCR
result = ocr.ocr(image_path, cls=True)[0]

# 定義判斷水平對齊的函數
def is_horizontal(box):
    x_coords = [point[0] for point in box]
    y_coords = [point[1] for point in box]
    x_range = max(x_coords) - min(x_coords)
    y_range = max(y_coords) - min(y_coords)
    return x_range > 1.5 * y_range  # 可根據需要調整閾值

# 提取水平對齊的文字
horizontal_texts = []
confidence_threshold = 0.7  # 置信度閾值
for line in result:
    box = line[0]
    text = line[1][0]
    confidence = line[1][1]
    if is_horizontal(box) and confidence > confidence_threshold:
        # 計算 y_center 作為分行依據
        y_coords = [point[1] for point in box]
        y_center = (min(y_coords) + max(y_coords)) / 2
        x_left = min([point[0] for point in box])
        horizontal_texts.append((text, y_center, x_left))

# 按 y_center 座標分行
row_threshold = 10  # 增加行間距閾值
rows = []
current_row = []
previous_y = None
for text, y_center, x_left in sorted(horizontal_texts, key=lambda x: x[1]):
    if previous_y is None or abs(y_center - previous_y) < row_threshold:
        current_row.append((text, x_left))
    else:
        rows.append(current_row)
        current_row = [(text, x_left)]
    previous_y = y_center
if current_row:
    rows.append(current_row)

# 輸出並合併每行文字
print("表格中的水平文字內容（按行彙整）：")
for row in rows:
    row.sort(key=lambda x: x[1])  # 按 x_left 排序
    row_texts = [text for text, _ in row]
    print(" ".join(row_texts))

# 繪製標註框並保存圖片
image = Image.open(image_path).convert('RGB')
draw = ImageDraw.Draw(image)
for line in result:
    box = line[0]
    if is_horizontal(box):
        draw.polygon([tuple(point) for point in box], outline='blue', width=2)
    else:
        draw.polygon([tuple(point) for point in box], outline='red', width=2)
# image.save('標註圖片_有水平文字.jpg')
# print("標註圖片已保存至：標註圖片_有水平文字.jpg")