from PIL import Image
import numpy as np
from paddleocr import PaddleOCR

# 初始化 PaddleOCR（假設在全局範圍內）
ocr = PaddleOCR(show_log=False, use_angle_cls=True, lang='ch')

def process_ocr_from_local(image_path):
    """
    從本地路徑讀取圖片並進行 OCR 處理
    返回值 (tuple): (success: bool, texts: list)
    """
    try:
        # 嘗試開啟圖片
        image = Image.open(image_path).convert("RGB")

        # 將 PIL 圖像轉換為 numpy 陣列
        image_np = np.array(image)

        # 使用 OCR 處理圖片
        result = ocr.ocr(image_np, cls=True)

        # 檢查 OCR 結果是否有效
        if result is None or result == [None]:
            print("圖片中沒有可辨識的文字, return False, []")
            return False, []

        # 檢查 result[0] 是否存在且可迭代
        if not result or not result[0]:
            print("OCR 結果為空或無效, return False, []")
            return False, []

        # 定義區域和對應閾值的函數
        def get_threshold(x, y):
            if y < 200: return 0.8
            elif y > 800: return 0.7
            elif x < 200: return 0.85
            elif x > 800: return 0.9
            else: return 0.75

        filtered_texts = []
        # 處理每個文字塊
        for line in result[0]:
            box = line[0]          # 文字塊的座標
            text = line[1][0]      # 識別出的文字內容
            confidence = line[1][1] # 置信度分數
            
            # 計算文字塊的中心座標
            x_center = sum([point[0] for point in box]) / 4
            y_center = sum([point[1] for point in box]) / 4
            
            # 根據中心座標獲取該區域的閾值
            threshold = get_threshold(x_center, y_center)
            
            # 篩選：只保留置信度高於閾值的文字
            if confidence > threshold:
                filtered_texts.append(text)

        if filtered_texts:
            print("篩選後的文字:", filtered_texts)  # 檢查最終結果
            return True, filtered_texts
        else:
            print("無符合閾值的文字, return False, []")
            return False, []

    except Exception as e:
        # 錯誤記錄：打印具體錯誤
        print(f"OCR 失敗，發生錯誤：{str(e)}")
        return False, []

# 測試本地圖片
image_path = "/Users/linjiaxian/Desktop/789.jpg"  # 替換為您的本地圖片路徑
success, texts = process_ocr_from_local(image_path)
print(f"最終結果: success={success}, texts={texts}")