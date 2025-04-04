import chromadb
"""
使用向量資料庫，原廠預設使用： all-MiniLM-L6-v2 模型 
"""

""" 初始化 ChromaDB 客戶端 [記憶體模式]"""
# chroma_client = chromadb.Client()

chroma_client = chromadb.PersistentClient("./chroma_data")

""" 創建集合 """
testCollect = chroma_client.get_or_create_collection(
    name="testCollect",
)

""" 添加數據 """
"""
    documents 主要內容 [必填] (會被轉為向量)
    metadatas 補充內容 [選填] (不會被轉為向量) [key:value]
    ids 主鍵 [必填]
"""
if testCollect.count() == 0 :
    testCollect.add(
      documents=["get_weather: 獲取城市天氣", "search_web: 搜索網頁", "get_location: 獲取城市位置"],
      metadatas=[
          {"name": "get_weather", "description": "獲取城市天氣"},
          {"name": "search_web", "description": "搜索網頁"},
          {"name": "get_location", "description": "獲取城市位置"}
      ],
      ids=["fun1", "fun2", "fun3"],
    )
    print("collection 裡面還沒有資料")
else :
    print("collection 已經有資料了")

""" 查詢 """
results = testCollect.query(
    query_texts=["ˇ中國在哪邊？"],
    n_results=1,  # 找 Top-1
)

""" 逐項印出結果 """
print("IDs:", results["ids"])
print("Documents:", results["documents"])
print("Metadatas:", results["metadatas"])
print("Distances:", results["distances"])
print("Embeddings:", results["embeddings"])

""" [查詢 ＊ ] 所有在DB裡面的資料 """
data_in_collection = testCollect.get()
print("所有結果：")
print(data_in_collection)