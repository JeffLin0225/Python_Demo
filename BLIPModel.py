from flask import Flask, request, jsonify
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
from googletrans import Translator
import io

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10MB 限制

translator = Translator()
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base",
    use_safetensors=True
)

@app.route('/caption', methods=['POST'])
def generate_caption():
    if 'image' not in request.files:
        return jsonify({"error": "請上傳圖片"}), 400
    
    file = request.files['image']
    try:
        image = Image.open(file.stream).convert("RGB")
    except Exception as e:
        return jsonify({"error": f"圖片處理失敗：{str(e)}"}), 400

    inputs = processor(images=image, return_tensors="pt")
    generated_ids = model.generate(**inputs)
    caption = processor.decode(generated_ids[0], skip_special_tokens=True)
    caption_tw = translator.translate(caption, dest="zh-tw").text  # 同步翻譯

    return jsonify({
        "description_en": caption,
        "description_tw": caption_tw
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)