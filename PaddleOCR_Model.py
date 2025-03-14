from paddleocr import PaddleOCR

# 初始化 PaddleOCR
ocr = PaddleOCR(use_angle_cls=True, lang='ch')

# 指定圖片路徑
image_path = '/Users/linjiaxian/Desktop/123.png' 

# 執行 OCR
result = ocr.ocr(image_path, cls=True)

# 提取第一張圖片的檢測結果
texts = [line[1][0] for line in result[0] if line[1][1] >  0.9]  
print("所有文字內容：")
for text in texts:
    print(text)


#生成檢測圖片範圍
from PIL import Image, ImageDraw

# 初始化 PaddleOCR

# 執行 OCR
result = ocr.ocr(image_path, cls=True)

# 載入圖片
image = Image.open(image_path).convert('RGB')
draw = ImageDraw.Draw(image)

# 繪製標註框
for line in result[0]:
    box = line[0]
    # 繪製紅色矩形框
    draw.polygon([tuple(point) for point in box], outline='red', width=2)

# 保存圖片
# image.save('標註圖片_無文字.jpg')
print("標註圖片（無文字）已保存至：標註圖片_無文字.jpg")


# print("==============")   #印全部內容
# for text in result:
#     print(text)