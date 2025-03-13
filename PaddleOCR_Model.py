from paddleocr import PaddleOCR

# 初始化 PaddleOCR
ocr = PaddleOCR(use_angle_cls=True, lang='ch')

# 指定圖片路徑
image_path = '/Users/linjiaxian/Desktop/cut.png'  # 確認這是正確路徑

# 執行 OCR
result = ocr.ocr(image_path, cls=True)

# 提取第一張圖片的檢測結果
texts = [line[1][0] for line in result[0] if line[1][1] >  0.9]  # 注意這裡使用 result[0]
print("所有文字內容：")
for text in texts:
    print(text)

# print("==============")


# for text in result:
#     print(text)