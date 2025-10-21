import os
from langchain_astradb import AstraDBVectorStore
from langchain_chroma import Chroma
from utils.config_loader import load_config
from utils.model_loader import ModelLoader
from dotenv import load_dotenv
from langchain.retrievers.document_compressors import LLMChainFilter
from langchain.retrievers import ContextualCompressionRetriever
# Add the project root to the Python path for direct script execution
# import sys
# import pathlib
# project_root = pathlib.Path(__file__).resolve().parents[2]
# sys.path.insert(0, str(project_root))


class Retriever:
    def __init__(self):
        self.model_loader = ModelLoader()
        self.config = load_config()
        self.path_to_chroma_db = self.config["chroma_db"]["CHROMA_PATH"] if "chroma_db" in self.config else "data/Chroma"
        self.retrievers: dict[str, ContextualCompressionRetriever] = {}
        self.vstores: dict[str, Chroma] = {}

    def load_retriever(self, collection_name: str = "defaultdb") -> ContextualCompressionRetriever:
        # Return existing retriever if available
        if collection_name in self.retrievers:
            return self.retrievers[collection_name]

        # Create vector store if needed
        if collection_name not in self.vstores:
            self.vstores[collection_name] = Chroma(
                collection_name=collection_name,
                persist_directory=self.path_to_chroma_db,
                embedding_function=self.model_loader.load_embeddings(),
            )
            
        vstore = self.vstores[collection_name]
        doc_count = vstore._collection.count()
        print(f"Number of documents in collection '{collection_name}': {doc_count}")
        # Create retriever
        #top_k = self.config.get("retriever", {}).get("top_k", 3)
        top_k = self.config["retriever"]["top_k"] if "retriever" in self.config else 3
        use_compression = self.config["retriever"].get("use_llm_compression", False)

        if use_compression and doc_count > 3:
            mmr_retriever = vstore.as_retriever(
                search_type="mmr",
                search_kwargs={
                    "k": top_k,
                    "fetch_k": max(top_k * 3, 20),
                    "lambda_mult": 0.65,
                    "score_threshold": 0.5
                }
            )
            llm = self.model_loader.load_llm()
            compressor = LLMChainFilter.from_llm(llm)
            retriever_instance = ContextualCompressionRetriever(
                base_compressor=compressor,
                base_retriever=mmr_retriever
            )
        else:
            print("Using basic similarity retriever (doc count <= 10)")
            base_retriever = vstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": top_k}
            )
            retriever_instance=base_retriever
        print(f"Retriever loaded successfully for collection: {collection_name}")


        print(retriever_instance)
        # Store retriever for future use
        self.retrievers[collection_name] = retriever_instance

        return retriever_instance
            
    def call_retriever(self,query,collection_name):
        """_summary_
        """
        retriever=self.load_retriever(collection_name=collection_name)
        output=retriever.invoke(query)
        return output
    
if __name__=='__main__':
    user_query = "Do you have any information about ramesh"
    
    retriever_obj = Retriever()
    
    retrieved_docs = retriever_obj.call_retriever(user_query,"IT-BSS_Docs")
    print(retrieved_docs)
    for idx, doc in enumerate(retrieved_docs, 1):
        print(f"\n--- Document {idx} ---")
        print("Content:")
        print(doc.page_content)
        print("Metadata:", doc.metadata)



    # def _format_docs(docs) -> str:
    #     if not docs:
    #         return "No relevant documents found."
    #     formatted_chunks = []
    #     for d in docs:
    #         meta = d.metadata or {}
    #         formatted = (
    #             f"Title: {meta.get('product_title', 'N/A')}\n"
    #             f"Price: {meta.get('price', 'N/A')}\n"
    #             f"Rating: {meta.get('rating', 'N/A')}\n"
    #             f"Reviews:\n{d.page_content.strip()}"
    #         )
    #         formatted_chunks.append(formatted)
    #     return "\n\n---\n\n".join(formatted_chunks)
    
    # retrieved_contexts = [_format_docs(doc) for doc in retrieved_docs]
    
    # #this is not an actual output this have been written to test the pipeline
    # response="iphone 16 plus, iphone 16, iphone 15 are best phones under 1,00,000 INR."

    


    
    
    
    # for idx, doc in enumerate(results, 1):
    #     print(f"Result {idx}: {doc.page_content}\nMetadata: {doc.metadata}\n")