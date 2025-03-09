import asyncio
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
from googletrans import Translator

# 初始化翻譯器
translator = Translator()

# 加載模型，使用 Safetensors 格式
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base",
    use_safetensors=True
)

# 處理圖片
image = Image.open("/Users/linjiaxian/Desktop/image.jpg").convert("RGB")
inputs = processor(images=image, return_tensors="pt")
generated_ids = model.generate(**inputs)
caption = processor.decode(generated_ids[0], skip_special_tokens=True)

# 異步翻譯函數
async def translate_text(text):
    return await translator.translate(text, dest="zh-tw")

# 運行翻譯
caption_tw = asyncio.run(translate_text(caption)).text

# 輸出
print("生成的描述（英文）：", caption)
print("生成的描述（繁體中文）：", caption_tw)

# 釋放記憶體
del model
del processor
print("記憶體已釋放")