from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document

embedding = OllamaEmbeddings(model="nomic-embed-text")

# local persistent store
#client = QdrantClient(path="data/qdrant_db")

docs = [
    Document(page_content="Ncell follows a hybrid work model...", metadata={"source": "Ncell_Policy.txt"}),
    Document(page_content="Hello this is Ramesh...", metadata={"source": "Ramesh_info.txt"}),
]

# ✅ Correct: Do not pass `client=`
qdrant_store = QdrantVectorStore.from_documents(
    documents=docs,
    embedding=embedding,
    path="data/qdrant_db",           # use path instead of client
    collection_name="IT-BSS_Docs",
)

print("✅ Local Qdrant vector store created successfully!")
