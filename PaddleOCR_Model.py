from paddleocr import PaddleOCR
from PIL import Image, ImageDraw

# 初始化 PaddleOCR
ocr = PaddleOCR(use_angle_cls=True, lang="ch")

# 指定圖片路徑
img_path = '/Users/linjiaxian/Desktop/cut.png'  # 請替換成您的圖片路徑

# 執行 OCR
result = ocr.ocr(img_path, cls=True)

# 載入圖片
image = Image.open(img_path).convert('RGB')
draw = ImageDraw.Draw(image)

# 繪製標註框
for line in result[0]:
    box = line[0]
    # 繪製紅色矩形框
    draw.polygon([tuple(point) for point in box], outline='red', width=2)

# 保存圖片
image.save('標註圖片_無文字.jpg')
print("標註圖片（無文字）已保存至：標註圖片_無文字.jpg")