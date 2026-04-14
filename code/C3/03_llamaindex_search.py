from llama_index.core import StorageContext, load_index_from_storage, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# 使用构建索引时相同的 embedding model
Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-zh-v1.5"
)

persist_path = "./llamaindex_index_store"

# 加载存储上下文
storage_context = StorageContext.from_defaults(persist_dir=persist_path)

# 加载索引
index = load_index_from_storage(storage_context)

question = "LlamaIndex是什么"

# 获取检索器做相似性搜索
retriever = index.as_retriever(search_kwargs={"k":1})
results = retriever.retrieve(question)

# result = query_engine.query(question)

for i, result in enumerate(results):
    print(f"相似度分数{result.score}")
    print(result.text)


# 输出
# 相似度分数0.6837823735997418
# LlamaIndex是一个用于构建和查询私有或领域特定数据的框架。
# 相似度分数0.32171357976269044
# 它提供了数据连接、索引和查询接口等工具。
# llamaindex score_threshold默认为0.2