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

def ask_llama(question, description = None ):
    if description:
        prompt = f"圖片描述：{description}\n問題：{question}\n只能用繁體中文自然回答問題。"
    else:
        prompt = f"問題：{question}\n只能用繁體中文自然回答問題。"
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
    image_file = request.files.get('image')
    question = request.form.get('question')

    if not image_file and not question:
        return jsonify({"error" : "請輸入文字或是圖片"}),400

    blipDescription = None
    answer = None 

    if image_file: 
        try:
            image = Image.open(image_file.stream).convert("RGB")
            blipDescription = blip_analyze(image)
        except Exception as e:
            return jsonify({"error": f"圖片處理失敗：{str(e)}"}), 400
    if question:
        if blipDescription:
            answer = ask_llama( question , blipDescription )
        else:
            answer = ask_llama(question)
    
    return jsonify(
        {
            # "Description" : blipDescription if blipDescription else "無圖片描述" , 
            "answer" : answer if answer else blipDescription
        }
    )

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)