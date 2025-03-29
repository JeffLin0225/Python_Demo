from diffusers import StableDiffusionPipeline

'''＊檔案目的＊： 
    對（下載的模型）轉換diffusers 生成可用的格式  '''
'''＊只要執行一次即可＊   '''

# 從單一檔案載入模型
pipe = StableDiffusionPipeline.from_single_file(
    "/Users/linjiaxian/ProgramServ/AI_Model_Lib/v1-5-pruned-emaonly.safetensors",
)

# 儲存為 Diffusers 格式
pipe.save_pretrained("/Users/linjiaxian/ProgramServ/AI_Model_Lib/sd15_diffusers")