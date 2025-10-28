
from typing import TypedDict, Annotated,Sequence,Literal
from langgraph.graph import StateGraph, END,START
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage,BaseMessage,SystemMessage
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.checkpoint.sqlite import SqliteSaver
from typing import List, Optional, Dict, Any
from datetime import datetime
import sqlite3
from utils.model_loader import ModelLoader
from retriever.retrieval import Retriever
from prompt_lib.prompts import PROMPT_REGISTRY, PromptType
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3
from pydantic import BaseModel, EmailStr
from typing import Optional,List
from langchain_core.runnables import RunnableConfig



class AgenticRag:

    class BasicChatState(TypedDict): 
        #messages: Annotated[list, add_messages]
        messages: Annotated[Sequence[BaseMessage], add_messages]


    def __init__(self):
        self.retriever_obj = Retriever()
        self.model_loader = ModelLoader()
        self.llm = self.model_loader.load_llm()
        self.sqlite_conn = sqlite3.connect("database/checkpoint.db", check_same_thread=False)
        self.checkpointer = SqliteSaver(self.sqlite_conn)
        self.workflow = self._build_workflow()
        self.app = self.workflow.compile(checkpointer=self.checkpointer)

    # ---------- Helpers ----------
    def _format_docs(self, docs) -> str:
        if not docs:
            return "No relevant documents found."

        formatted_docs = []
        source_list = []  # To collect source information
        
        for d in docs:
            meta = d.metadata or {}
            formatted = (
                f"Source: {meta.get('source', 'N/A')}\n"
                f"page:\n{d.page_content.strip()}"
            )
            formatted_docs.append(formatted)
        return "\n\n---\n\n".join(formatted_docs) 


    
    def _vector_retriever(self, state: BasicChatState, config: RunnableConfig = None):
        
        print("--- RETRIEVER ---")
        query = state["messages"][-1].content
        collection_name = config.get("configurable", {}).get("collection_name", "defaultdb")
        print(f"under vector retriever Using collection: {query}--> {collection_name}")
        retriever = self.retriever_obj.load_retriever(collection_name=collection_name)
        docs = retriever.invoke(query)
        context = self._format_docs(docs)
        
        # ✅ Print raw retrieved docs
        print(f"Retrieved {len(docs)} document(s):")
        for i, doc in enumerate(docs):
            print(f"\n--- Document {i+1} ---")
            print(f"Content:\n{doc.page_content}")
            if hasattr(doc, 'metadata'):
                print(f"Metadata: {doc.metadata}")

        return {"messages": [SystemMessage(content=context)]}    
    
    def _generate(self, state: BasicChatState):
        print("--- GENERATE ---")

        question = None
        context = None

        for msg in state["messages"]:
            if isinstance(msg, HumanMessage):
                question = msg.content
            elif isinstance(msg, SystemMessage):
                context = msg.content

        print(f"context --> {context[:120] if context else 'None'}")
        print(f"question --> {question}")

        prompt = ChatPromptTemplate.from_template(
            PROMPT_REGISTRY[PromptType.DOC_BOT].template
        )
        chain = prompt | self.llm | StrOutputParser()
        response = chain.invoke({"context": context, "question": question})
        return {"messages": [AIMessage(content=response)]}


    def _rewrite(self, state: BasicChatState):
        print("--- REWRITE ---")
        question = state["messages"][0].content
        new_q = self.llm.invoke(
            [HumanMessage(content=f"Rewrite the query to be clearer: {question}")]
        )
        return {"messages": [HumanMessage(content=new_q.content)]}
        

    def _build_workflow(self):
        # Create the graph

        workflow = StateGraph(self.BasicChatState)
        workflow.add_node("Retriever", self._vector_retriever)
        workflow.add_node("Generator", self._generate)
        workflow.add_edge(START, "Retriever")

        workflow.add_edge("Retriever", "Generator")
        workflow.add_edge("Generator", END)

        return workflow

    def _is_allowed_query(self, query: str) -> bool:
        disallowed = [
            "ignore instructions",
            "system prompt",
            "politics",
            "religion",
            "hack",
            "password",
            "song",
            "love",
        ]
        return not any(bad_word in query.lower() for bad_word in disallowed)



    # ---------- Public Run ----------
    def run(self, query: str,thread_id: str = "default_thread", collection_name : str = "defaultdb") -> str:
        """Run the workflow for a given query and return the final answer."""
        print(f" my collection name is inside run  : {collection_name}")


        if not self._is_allowed_query(query):
            return "This question is outside the scope of the available documents."
        config = {
            "configurable": {
                "thread_id": thread_id,
                "collection_name": collection_name
            }
        }
        result = self.app.invoke(
            {"messages": [HumanMessage(content=query)]},
            config=config
        )
        return result["messages"][-1].content
    
    def get_chat_messages_from_langgraph(self,chat_id: str) -> List[Any]:
        """
        Get chat messages from LangGraph checkpointer for a specific chat_id
        
        Args:
            chat_id: The thread_id/chat_id to retrieve messages for
            
        Returns:
            List of LangGraph message objects
        """
                # Get state from LangGraph checkpointer using the same pattern as your existing code
        state = self.app.get_state(config={'configurable': {'thread_id': chat_id}})
            
            # Extract messages from state, return empty list if not found
        messages = state.values.get('messages', [])
            
        return messages
