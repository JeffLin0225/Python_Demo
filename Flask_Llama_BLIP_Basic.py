from flask import Flask, request, jsonify
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
from googletrans import Translator
import requests  # 新增這行

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10MB 限制

translator = Translator()
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base",
    use_safetensors=True
)

# ollama API 設定
OLLAMA_API_URL = "http://localhost:11434/api/generate"

def ask_llama(description, question):
    # 設計 prompt，讓 Llama 3.2 根據描述回答問題
    prompt = f"圖片描述：{description}\n問題：{question}\n只能用繁體中文自然回答問題。"
    try:
        response = requests.post(
            OLLAMA_API_URL,
            json={
                "model": "llama3.2:3B",
                "prompt": prompt,
                "stream": False
            }
        )
        response.raise_for_status()
        return response.json()["response"].strip()
    except requests.RequestException as e:
        return f"錯誤：無法連接到 Llama 3.2 - {str(e)}"
    
def blip_analyze(image):
    #BLIP
    inputs = processor(images=image, return_tensors="pt")
    generated_ids = model.generate(**inputs)
    description = processor.decode(generated_ids[0], skip_special_tokens=True)
    description_tw = translator.translate(description, dest="zh-tw").text  # 同步翻譯
    return description_tw 


@app.route('/ask', methods=['POST'])
def generate_caption():
    if 'image' not in request.files or 'question' not in request.form:
        return jsonify({"error": "請上傳圖片並輸入問題"}), 400
    
    file = request.files['image']
    question = request.form['question']

    try:
        image = Image.open(file.stream).convert("RGB")
    except Exception as e:
        return jsonify({"error": f"圖片處理失敗：{str(e)}"}), 400

    blip_description = blip_analyze(image)

    answer = ask_llama(blip_description, question)

    return jsonify({
        "description": blip_description,
        "answer": answer
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)