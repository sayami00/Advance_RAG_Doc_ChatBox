import os
import shutil
from typing import List
from fastapi import UploadFile
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader, Docx2txtLoader, UnstructuredExcelLoader,TextLoader
)

from logger import GLOBAL_LOGGER as log
from exception.custom_exception import ChatBoxException
from utils.model_loader import ModelLoader
from langchain_chroma import Chroma
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http import models
import hashlib

# 🔧 FastAPI File Adapter
class FastAPIFileAdapter:
    """Adapter for FastAPI's UploadFile object."""
    def __init__(self, uf: UploadFile):
        self._uf = uf
        self.name = uf.filename

    def getbuffer(self) -> bytes:
        self._uf.file.seek(0)
        return self._uf.file.read()


# File Handling
class DocHandler:
    def __init__(self):
        self.temp_data_dir = os.getenv("TEMP_DATA_STORAGE_PATH", "data/tempfile")
        self.data_dir = os.getenv("DATA_STORAGE_PATH", "data/files")
        os.makedirs(self.temp_data_dir, exist_ok=True)
        os.makedirs(self.data_dir, exist_ok=True)
        log.info("DocHandler initialized", temp_data_dir=self.temp_data_dir)

    async def temp_save_file(self, uploaded_file, department_path: str) -> str:
        """Save uploaded file temporarily."""
        try:
            filename = os.path.basename(uploaded_file.name)
            if not filename.lower().endswith((".pdf", ".docx", ".xlsx", ".txt", ".csv")):
                raise ValueError(f"File type not allowed: {filename}")

            save_dir = os.path.join(self.temp_data_dir, department_path)
            os.makedirs(save_dir, exist_ok=True)

            save_path = os.path.join(save_dir, filename)
            with open(save_path, "wb") as f:
                content = uploaded_file.read() if hasattr(uploaded_file, "read") else uploaded_file.getbuffer()
                f.write(content)

            log.info("File saved successfully", file=filename, save_path=save_path)
            return save_path

        except Exception as e:
            log.error("Failed to save file", error=str(e))
            raise ChatBoxException(f"Failed to save file: {str(e)}", e) from e

    async def save_file(self,temp_saved_path,uploaded_file, department_path) -> str:
            save_dir = os.path.join(self.data_dir, department_path)
            os.makedirs(save_dir, exist_ok=True)
            destinationpath = os.path.join(save_dir, uploaded_file)
            shutil.move(temp_saved_path, destinationpath)
            log.info(f"File moved to processed: {destinationpath}")
            return destinationpath

# Chroma DB Manager
class ChromaManager:
    def __init__(self):
        self.chroma_path = os.getenv("CHROMA_PATH", "data/chroma")
        self.model_loader = ModelLoader()
        self.embeddings = self.model_loader.load_embeddings()

    def get_db(self, collection_name: str) -> Chroma:
        """Return a Chroma DB instance with embeddings."""
        return Chroma(
            collection_name=collection_name,
            persist_directory=self.chroma_path,
            embedding_function=self.embeddings,
        )


# Document Processor
class DocumentProcessor:
    def __init__(self, data_source_path: str = "data/tempfile", chroma_path: str = "data/chroma"):
        self.data_source_path = data_source_path
        self.chroma_path = chroma_path
        self.chroma_manager = ChromaManager()

    def duplicate_validation(self, file_path: str, collection_name: str):
        db = self.chroma_manager.get_db(collection_name)
        full_path = os.path.join(self.data_source_path, file_path)
        return db.get(where={"source": full_path})

    def load_document(self, file_path: str) -> List[Document]:
        """Load a document based on its file extension."""
        full_path = os.path.abspath(file_path)
        log.info("Loading document", path=full_path)

        if file_path.endswith(".pdf"):
            loader = PyPDFLoader(full_path)
        elif file_path.endswith(".docx"):
            loader = Docx2txtLoader(full_path)
        elif file_path.endswith(".xlsx"):
            loader = UnstructuredExcelLoader(full_path)
        elif file_path.endswith(".txt"):
            loader = TextLoader(full_path)
        else:
            raise ValueError("Unsupported file type")

        return loader.load()

    def split_documents(self, documents: List[Document]) -> List[Document]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            is_separator_regex=False,
        )
        return splitter.split_documents(documents)

    def add_to_vectorestore(self, chunks: List[Document], collection_name: str):
        db = self.chroma_manager.get_db(collection_name)
        chunks = self.calculate_chunk_ids(chunks)

        existing_items = db.get(include=[])
        existing_ids = set(existing_items["ids"])
        log.info("Existing documents in DB", count=len(existing_ids))

        new_chunks = [c for c in chunks if c.metadata["id"] not in existing_ids]

        if new_chunks:
            new_ids = [c.metadata["id"] for c in new_chunks]
            db.add_documents(new_chunks, ids=new_ids)
            log.info("Added new chunks to DB", count=len(new_chunks))
        else:
            log.info("No new documents to add")

    @staticmethod
    def calculate_chunk_ids(chunks: List[Document]) -> List[Document]:
        last_page_id = None
        current_chunk_index = 0

        for chunk in chunks:
            source = chunk.metadata.get("source")
            page = chunk.metadata.get("page")
            current_page_id = f"{source}:{page}"

            if current_page_id == last_page_id:
                current_chunk_index += 1
            else:
                current_chunk_index = 0

            chunk.metadata["id"] = f"{current_page_id}:{current_chunk_index}"
            last_page_id = current_page_id

        return chunks

    def clear_database(self):
        if os.path.exists(self.chroma_path):
            shutil.rmtree(self.chroma_path)
            log.warning("Chroma DB cleared")
        else:
            log.info("Chroma DB already clean")

    def process_local_document(self, file_path: str, collection_name: str):
        documents = self.load_document(file_path)
        chunks = self.split_documents(documents)
        self.add_to_chroma(chunks, collection_name)

    def process_web_document(self, documents: List[Document], collection_name: str):
        chunks = self.split_documents(documents)
        self.add_to_chroma(chunks, collection_name)




# -------------------------------
# Qdrant Manager
# -------------------------------
class QdrantManager:
    def __init__(self, url: str = "http://localhost:6333", collection_path: str = None):
        self.url = url
        self.client = QdrantClient(url=self.url)
        self.embeddings = ModelLoader().load_embeddings()
        self.collection_path = collection_path  # (optional future control)
        
    def ensure_collection(self, collection_name: str, vector_size: int = 768):
        """Create the collection if it doesn't exist."""
        collections = [c.name for c in self.client.get_collections().collections]
        if collection_name not in collections:
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE)
            )
            print(f"✅ Created Qdrant collection: {collection_name}")

    def get_store(self, collection_name: str):
        """Get (or create) a vector store handle."""
        self.ensure_collection(collection_name)
        return QdrantVectorStore(
            client=self.client,
            collection_name=collection_name,
            embedding=self.embeddings
        )

# -------------------------------
# Document Processor
# -------------------------------
class DocumentProcessor_Qdrant:
    def __init__(self, data_source_path: str = "data/tempfile"):
        self.data_source_path = data_source_path
        # Connect to Qdrant server
        self.qdrant_manager = QdrantManager(url="http://localhost:6333")
    
    def duplicate_validation(self, file_path: str, collection_name: str):
        """Check if a document already exists in Qdrant by 'source' metadata."""
        store = self.qdrant_manager.get_store(collection_name)
        full_path = os.path.join(self.data_source_path, file_path)

        # Scroll without extra arguments
        existing_points, _ = store.client.scroll(
            collection_name=collection_name,
            limit=1000
        )

        # Check if any point has the same source
        for point in existing_points:
            if point.payload.get("source") == full_path:
                return True
        return False

    def load_document(self, file_path: str) -> List[Document]:
        full_path = os.path.abspath(file_path)
        log.info(f"Loading document: {full_path}")

        if file_path.endswith(".pdf"):
            loader = PyPDFLoader(full_path)
        elif file_path.endswith(".docx"):
            loader = Docx2txtLoader(full_path)
        elif file_path.endswith(".xlsx"):
            loader = UnstructuredExcelLoader(full_path)
        elif file_path.endswith(".txt"):
            loader = TextLoader(full_path)
        else:
            raise ValueError(f"Unsupported file type: {file_path}")

        return loader.load()

    def split_documents(self, documents: List[Document]) -> List[Document]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            is_separator_regex=False,
        )
        return splitter.split_documents(documents)


    @staticmethod
    def calculate_chunk_ids(chunks: List[Document]) -> List[Document]:
        for chunk in chunks:
            # Use file path + page + chunk index to generate a unique hash
            source = chunk.metadata.get("source", "unknown_source")
            page = chunk.metadata.get("page")
            page_str = str(page) if page is not None else "0"

            # Create a short hash for the point ID
            unique_string = f"{source}:{page_str}:{chunk.page_content[:50]}"  # first 50 chars
            point_id = hashlib.md5(unique_string.encode("utf-8")).hexdigest()

            chunk.metadata["id"] = point_id

        return chunks



    def add_to_vectorestore(self, chunks: List[Document], collection_name: str):
        store = self.qdrant_manager.get_store(collection_name)
        chunks = self.calculate_chunk_ids(chunks)

        # Scroll without include_payload
        existing_points, _ = store.client.scroll(
            collection_name=collection_name,
            limit=1000
        )

        existing_ids = {point.id for point in existing_points}
        log.info(f"Existing documents in Qdrant: {len(existing_ids)}")

        new_chunks = [c for c in chunks if c.metadata["id"] not in existing_ids]

        if not new_chunks:
            log.info("No new chunks to add.")
            return

        texts = [c.page_content for c in new_chunks]
        metadatas = [c.metadata for c in new_chunks]
        ids = [c.metadata["id"] for c in new_chunks]

        store.add_texts(texts=texts, metadatas=metadatas, ids=ids)
        log.info(f"Added {len(new_chunks)} new chunks to collection '{collection_name}'")

    def clear_database(self):
        log.warning(
            "Clearing Qdrant collections must be done via server API. Local folder not used in server mode."
        )

    def process_local_document(self, file_path: str, collection_name: str):
        documents = self.load_document(file_path)
        chunks = self.split_documents(documents)
        self.add_to_qdrant(chunks, collection_name)

    def process_web_document(self, documents: List[Document], collection_name: str):
        chunks = self.split_documents(documents)
        self.add_to_qdrant(chunks, collection_name)