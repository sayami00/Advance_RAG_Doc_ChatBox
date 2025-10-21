import atexit
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from utils.config_loader import load_config
from utils.model_loader import ModelLoader

class Retriever:
    def __init__(self):
        self.model_loader = ModelLoader()
        self.config = load_config()
        self.client = QdrantClient(path="data/qdrant_db")  # Local persistent Qdrant store
        self.vstores = {}
        self.retrievers = {}  # ✅ Add this back (you commented it out earlier)

    def load_retriever_qdrant(self, collection_name: str = "defaultdb"):
        """Load a retriever backed by QdrantVectorStore"""
        embedding_fn = self.model_loader.load_embeddings()

        qdrant_store = QdrantVectorStore(
            client=self.client,
            collection_name=collection_name,
            embedding=embedding_fn,  # ✅ FIXED: correct argument name is 'embedding'
        )

        retriever = qdrant_store.as_retriever(
            search_kwargs={
                "k": self.config.get("retriever", {}).get("top_k", 5),
                "score_threshold": 0.3
            }
        )

        self.retrievers[collection_name] = retriever
        print(f"✅ Retriever loaded successfully for collection: {collection_name}")
        return retriever

    def call_retriever(self, query, collection_name):
        retriever = self.load_retriever_qdrant(collection_name)
        docs = retriever.invoke(query)  # ✅ Modern LC 0.3+ API
        if not docs or all(d.score < 0.3 for d in docs):
            print("⚠️ No relevant documents found for this query.")
            return []
        # if not docs:
        #     print("⚠️ No relevant documents found.")
        else:
            for idx, doc in enumerate(docs, 1):
                print(f"\n--- Document {idx} ---")
                print("Content:")
                print(doc.page_content.strip())
                print("Metadata:", doc.metadata)

        return docs

    def close_qdrant_client(self):
        """Gracefully close Qdrant connection before interpreter shutdown"""
        try:
            self.client.close()
            print("🧹 Qdrant client closed cleanly.")
        except Exception:
            pass


if __name__ == "__main__":
    user_query = "Hi"
    print(f"useruer={user_query}")
    retriever_obj = Retriever()
    atexit.register(retriever_obj.close_qdrant_client)
    retriever_obj.call_retriever(user_query, "IT-BSS_Docs")
