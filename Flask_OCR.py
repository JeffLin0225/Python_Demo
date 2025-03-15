from flask import Flask, request, jsonify
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
from googletrans import Translator
from paddleocr import PaddleOCR
import numpy as np

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10MB 限制



# 初始化 PaddleOCR（假設在全局範圍內）
ocr = PaddleOCR(show_log=False, use_angle_cls=True, lang='ch')

def process_ocr(image_path):

    try:
        result = ocr.ocr(image_path, cls=True)
        # print("OCR raw result:", result)  # 調試：檢查原始結果

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
            
            # print(f"Text: {text}, Confidence: {confidence}, Threshold: {threshold}")  # 調試：檢查每塊文字
            
            # 篩選：只保留置信度高於閾值的文字
            if confidence > threshold:
                filtered_texts.append(text)
        
        print("篩選後的文字:", filtered_texts)  # 檢查最終結果
        if filtered_texts:
            return True, filtered_texts
        else:
            print("無符合閾值的文字, return False, []")
            return False, []

    except FileNotFoundError:
        print("檔案不存在, return False, []")
        return False, []
    except Exception as e:
        print(f"OCR 失敗，發生錯誤：{str(e)}, return False, []")
        return False, []



@app.route('/caption', methods=['POST'])
def generate_caption():
    if 'image' not in request.files:
        return jsonify({"error": "請上傳圖片"}), 400
    
    file = request.files['image']
    try:
        image = Image.open(file).convert("RGB")
        image_np = np.array(image)

        success, texts = process_ocr(image_np)

    except Exception as e:
        return jsonify({"error": f"圖片處理失敗：{str(e)}"}), 400



    return jsonify({
        "description_en": texts,
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)