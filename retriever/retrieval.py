import atexit
from typing import List
from langchain_chroma import Chroma
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from utils.model_loader import ModelLoader


class Retriever:
    def __init__(self, use_qdrant: bool = True):
        """
        Initialize Retriever.

        Args:
            use_qdrant: If True, use Qdrant; otherwise fallback to Chroma.
        """
        self.use_qdrant = use_qdrant
        self.model_loader = ModelLoader()
        self.embedding = self.model_loader.load_embeddings()
        self.retrievers = {}  # cache: collection_name -> retriever
        self.vstores = {}     # cache for Chroma vector stores

        # ---------------- Qdrant setup ----------------
        if self.use_qdrant:
            try:
                # ✅ Connect to Qdrant HTTP server
                self.client = QdrantClient(url="http://localhost:6333")
                print("✅ Connected to Qdrant server at http://localhost:6333")
                atexit.register(self.close_qdrant_client)
            except Exception as e:
                print(f"❌ Failed to connect to Qdrant server: {e}")
                self.client = None
        else:
            self.client = None
            print("⚙️ Using Chroma local vector store")

    # ---------------- Qdrant ----------------
    def load_retriever_qdrant(self, collection_name: str = "defaultdb", top_k: int = 5, score_threshold: float = 0.3):
        """
        🔧 FIXED: Load retriever with proper search_type for score tracking
        """
        # 🔧 Don't cache if we want different parameters
        cache_key = f"{collection_name}_{top_k}_{score_threshold}"
        if cache_key in self.retrievers:
            return self.retrievers[cache_key]

        # ✅ Ensure collection exists
        try:
            collections = [c.name for c in self.client.get_collections().collections]
            if collection_name not in collections:
                print(f"ℹ️ Creating new Qdrant collection: {collection_name}")
                self.client.create_collection(
                    collection_name=collection_name,
                    vectors_config={"size": self.embedding.dimension, "distance": "Cosine"}
                )
        except Exception as e:
            print(f"⚠️ Could not verify/create collection: {e}")

        # ✅ Create Qdrant vector store
        qdrant_store = QdrantVectorStore(
            client=self.client,
            collection_name=collection_name,
            embedding=self.embedding
        )

        # 🔧 KEY FIX: Must set search_type to get scores!
        retriever = qdrant_store.as_retriever(
            search_type="similarity_score_threshold",  # ⚠️ THIS IS CRITICAL
            search_kwargs={
                "k": top_k,
                "score_threshold": score_threshold
            }
        )

        self.retrievers[cache_key] = retriever
        print(f"🔍 Qdrant retriever initialized for collection: {collection_name} (k={top_k}, threshold={score_threshold})")
        return retriever

    # ---------------- Chroma ----------------
    def load_retriever_chroma(self, collection_name: str = "defaultdb", top_k: int = 5):
        """
        Load retriever for a Chroma collection.
        """
        if collection_name in self.retrievers:
            return self.retrievers[collection_name]

        if collection_name not in self.vstores:
            self.vstores[collection_name] = Chroma(
                collection_name=collection_name,
                persist_directory="data/Chroma",
                embedding_function=self.embedding,
            )

        vstore = self.vstores[collection_name]
        retriever = vstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": top_k}
        )
        self.retrievers[collection_name] = retriever
        print(f"🗂️ Chroma retriever initialized for collection: {collection_name}")
        return retriever

    # ---------------- Retrieve ----------------
    def call_retriever(
        self,
        query: str,
        collection_name: str = "defaultdb",
        top_k: int = 5,
        score_threshold: float = 0.3
    ) -> List:
        """
        Retrieve relevant documents for a query.
        """
        if self.use_qdrant:
            retriever = self.load_retriever_qdrant(collection_name, top_k, score_threshold)
        else:
            retriever = self.load_retriever_chroma(collection_name, top_k)

        # Retrieve documents
        docs = retriever.invoke(query)

        # 🔧 REMOVED: Don't filter again - the retriever already filtered by threshold!
        # The search_type="similarity_score_threshold" handles this

        # Logging results
        if not docs:
            print("⚠️ No relevant documents found.")
        else:
            print(f"✅ Retrieved {len(docs)} document(s) for query: '{query}'")
            for i, doc in enumerate(docs, 1):
                # 🔧 Try to extract score from metadata or attribute
                score = "N/A"
                if hasattr(doc, 'metadata') and isinstance(doc.metadata, dict):
                    score = doc.metadata.get('score', getattr(doc, 'score', 'N/A'))
                elif hasattr(doc, 'score'):
                    score = doc.score
                
                print(f"\n--- Document {i} ---")
                print(f"Score: {score}")
                print("Content:", doc.page_content.strip()[:200], "...")
                print("Metadata:", doc.metadata)

        return docs

    # ---------------- Advanced Retrieval Methods ----------------
    def call_retriever_mmr(
        self,
        query: str,
        collection_name: str = "defaultdb",
        top_k: int = 5,
        fetch_k: int = 20,
        lambda_mult: float = 0.5
    ) -> List:
        """
        🆕 Retrieve using MMR (Maximal Marginal Relevance) for diversity
        
        Args:
            query: Search query
            collection_name: Collection to search
            top_k: Number of final results
            fetch_k: Number of candidates to fetch before MMR
            lambda_mult: 0=diversity, 1=relevance
        """
        if self.use_qdrant:
            qdrant_store = QdrantVectorStore(
                client=self.client,
                collection_name=collection_name,
                embedding=self.embedding
            )
            
            retriever = qdrant_store.as_retriever(
                search_type="mmr",
                search_kwargs={
                    "k": top_k,
                    "fetch_k": fetch_k,
                    "lambda_mult": lambda_mult
                }
            )
            
            docs = retriever.invoke(query)
            print(f"🔄 MMR Retrieved {len(docs)} diverse document(s)")
            return docs
        else:
            print("⚠️ MMR only supported with Qdrant")
            return self.call_retriever(query, collection_name, top_k)

    # ---------------- Cleanup ----------------
    def close_qdrant_client(self):
        """
        Gracefully close the Qdrant client connection.
        """
        if hasattr(self, "client") and self.client:
            try:
                self.client.close()
                print("🧹 Qdrant client closed cleanly.")
            except Exception as e:
                print(f"⚠️ Error closing Qdrant client: {e}")


# ---------------- Example usage ----------------
if __name__ == "__main__":
    retriever = Retriever(use_qdrant=True)
    
    print("\n" + "="*60)
    print("Testing Standard Retrieval with Score Threshold")
    print("="*60)
    query = "What is the IT-BSS team?"
    docs = retriever.call_retriever(query, "IT-BSS_Docs", top_k=5, score_threshold=0.3)
    
    print("\n" + "="*60)
    print("Testing MMR Retrieval (Diverse Results)")
    print("="*60)
    docs_mmr = retriever.call_retriever_mmr(query, "IT-BSS_Docs", top_k=3, fetch_k=10)