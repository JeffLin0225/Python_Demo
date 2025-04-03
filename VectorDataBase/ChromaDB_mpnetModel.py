import chromadb
from chromadb.utils import embedding_functions
"""
使用向量資料庫，使用較強模型 (缺點:400MB 初始化時間較久)： all-mpnet-base-v2 模型 
"""
custom_embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-mpnet-base-v2")

""" 初始化 ChromaDB 客戶端 """
chroma_client = chromadb.Client()

""" 創建集合，並指定使用自定義嵌入函數 ，選用該模型"""
# documents 主要內容 [必填] (會被轉為向量)
# metadatas 補充內容 [選填] (不會被轉為向量) [key:value]
# ids 主鍵 [必填]
testCollect = chroma_client.create_collection(
    name="testCollect",
    embedding_function=custom_embedding_function
)

""" 添加數據 """
testCollect.add(
    documents=["get_weather: 獲取城市天氣", "search_web: 搜索網頁", "get_location: 獲取城市位置"],
    metadatas=[{"name": "get_weather", "description": "獲取城市天氣"},
               {"name": "search_web", "description": "搜索網頁"},
               {"name": "get_location", "description": "獲取城市位置"}],
    ids=["fun1", "fun2", "fun3"],
)

""" 查詢 """
results = testCollect.query(
    query_texts=["台灣在哪裡？"],
    n_results=1  # 找 Top-1
)

""" 逐項印出結果 """
print("IDs:", results["ids"])
print("Documents:", results["documents"])
print("Metadatas:", results["metadatas"])
print("Distances:", results["distances"])