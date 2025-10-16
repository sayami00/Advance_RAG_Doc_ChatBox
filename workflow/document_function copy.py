    
from typing import Iterable, List, Optional, Dict, Any
import os
from logger import GLOBAL_LOGGER as log
from exception.custom_exception import ChatBoxException
from fastapi import UploadFile
import argparse
import os
import shutil
from langchain_community.document_loaders import PyPDFLoader, PyPDFDirectoryLoader, Docx2txtLoader, UnstructuredExcelLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from utils.model_loader import ModelLoader
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

# ---------- Helpers ----------
class FastAPIFileAdapter:
    """Adapt FastAPI UploadFile -> .name + .getbuffer() API"""
    def __init__(self, uf: UploadFile):
        self._uf = uf
        self.name = uf.filename
    def getbuffer(self) -> bytes:
        self._uf.file.seek(0)
        return self._uf.file.read()
    


class DocHandler:
    """
    PDF save + read (page-wise) for analysis.
    """
    def __init__(self):
        self.temp_data_dir = os.getenv("TEMP_DATA_STORAGE_PATH")
        self.data_dir = os.getenv("DATA_STORAGE_PATH")
        os.makedirs(self.temp_data_dir, exist_ok=True)
        os.makedirs(self.data_dir, exist_ok=True)
        log.info("DocHandler initialized",temp_data_dir=self.temp_data_dir)

    async def temp_save_pdf(self, uploaded_file, departmenttemppath: str) -> str:
        try:
            filename = os.path.basename(uploaded_file.name)
            print(f"{filename} -->{departmenttemppath}")
            if not filename.lower().endswith((".pdf", ".docx", ".xlsx", ".txt", ".csv")):
                raise ValueError(status_code=400, detail=f"File type not allowed: {filename}")
            # Build save path
            temp_file_path = os.path.join(self.temp_data_dir, departmenttemppath)
            print(temp_file_path)
            os.makedirs(temp_file_path, exist_ok=True)

            save_path = os.path.join(temp_file_path, filename)
            with open(save_path, "wb") as f:
                if hasattr(uploaded_file, "read"):
                    f.write(uploaded_file.read())
                else:
                    f.write(uploaded_file.getbuffer())
            # #file_path = os.path.join(departmenttemppath, file.filename)
            # content = await uploaded_file.read()
            # with open(uploaded_file, "wb") as f:
            #     f.write(content)
            log.info("Doc saved successfully", file=filename, save_path=save_path)
            return save_path            
        except Exception as e:
            log.error("Failed to save PDF", error=str(e))
            raise ChatBoxException(f"Failed to save PDF: {str(e)}", e) from e        


class ChromaManager:
    def __init__(self):
        self.CHROMA_PATH = os.environ.get("CHROMA_PATH", "data/chroma")
        self.model_loader = ModelLoader()
        self.emb = self.model_loader.load_embeddings()

    def chroma_db(self,collection_name: str ):
        return Chroma(collection_name=collection_name, persist_directory=self.CHROMA_PATH, embedding_function=self.emb)


class DocumentProcessor:
    def __init__(self, data_source_path="data/tempfile", chroma_path="data/chroma"):
        self.DATA_SOURCE_PATH = data_source_path
        self.CHROMA_PATH = chroma_path

    def duplicate_validation(self, file_path: str, collection_name: str):
        db = ChromaManager.chroma_db(collection_name)
        full_path = os.path.join(self.DATA_SOURCE_PATH, file_path)
        existing = db.get(where={"source": full_path})
        return existing

    def load_document(self, file_path: str) -> list[Document]:
        """Load a document from path based on file extension."""
        full_path = os.path.abspath(file_path)
        print(f"Loading file: {full_path}")

        if file_path.endswith(".pdf"):
            loader = PyPDFLoader(full_path)
        elif file_path.endswith(".docx"):
            loader = Docx2txtLoader(full_path)
        elif file_path.endswith(".xlsx"):
            loader = UnstructuredExcelLoader(full_path)
        else:
            raise ValueError("Unsupported file type")
        return loader.load()

    def split_documents(self, documents: list[Document]) -> list[Document]:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            is_separator_regex=False,
        )
        return text_splitter.split_documents(documents)

    def add_to_chroma(self, chunks: list[Document], collection_name: str):
        db = get_chroma_db(collection_name)
        chunks_with_ids = self.calculate_chunk_ids(chunks)

        for chunk in chunks:
            print(f"Chunk Page Sample: {chunk.metadata['id']}\n{chunk.page_content}\n\n")

        existing_items = db.get(include=[])
        existing_ids = set(existing_items["ids"])
        print(f"Number of existing documents in DB: {len(existing_ids)}")

        new_chunks = [chunk for chunk in chunks_with_ids if chunk.metadata["id"] not in existing_ids]

        if new_chunks:
            print(f"👉 Adding new documents: {len(new_chunks)}")
            new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
            db.add_documents(new_chunks, ids=new_chunk_ids)
        else:
            print("✅ No new documents to add")

    def calculate_chunk_ids(self, chunks: list[Document]) -> list[Document]:
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

            chunk_id = f"{current_page_id}:{current_chunk_index}"
            last_page_id = current_page_id
            chunk.metadata["id"] = chunk_id

        return chunks

    def clear_database(self):
        if os.path.exists(self.CHROMA_PATH):
            shutil.rmtree(self.CHROMA_PATH)
            print("🗑️  Chroma DB cleared.")
        else:
            print("✅ Chroma DB is already clean.")

    def process_local_document(self, file_path: str, collection_name: str):
        documents = self.load_document(file_path)
        chunks = self.split_documents(documents)
        self.add_to_chroma(chunks, collection_name)

    def process_web_document(self, documents: list[Document], collection_name: str):
        chunks = self.split_documents(documents)
        self.add_to_chroma(chunks, collection_name)
