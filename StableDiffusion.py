from diffusers import StableDiffusionPipeline
import torch
import gc

# 載入模型
pipe = StableDiffusionPipeline.from_pretrained(
    "/Users/linjiaxian/ProgramServ/AI_Model_Lib/sd15_diffusers",
    low_cpu_mem_usage=True #優化 CPU 記憶體使用，減少占用。
)
pipe = pipe.to("mps") #模型移動到 MPS 計算設備，使用 macOS 系統上的 Metal Performance Shaders (MPS)。
pipe.enable_attention_slicing() #啟用注意力切片技術，以提高生成效率。

# 生成圖像
prompt = "a gray cat with subtle purple fur and green eyes sitting on a tree branch, realistic style with a slight fantasy touch, lush green forest background with dense foliage, soft natural lighting, detailed fur texture, mossy tree branch"
negative_prompt = "cartoonish, blurry, low detail, abstract, unrealistic"
image = pipe(prompt, height=448, width=448, num_inference_steps=20, negative_prompt=negative_prompt).images[0]
image.save("output3.png")

# 清理記憶體
gc.collect() # 調用垃圾回收機制，釋放未使用的物件。
torch.mps.empty_cache() # 清除 Metal Performance Shaders 的內存快取，進一步釋放資源。
