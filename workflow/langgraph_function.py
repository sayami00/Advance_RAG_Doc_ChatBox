
from typing import TypedDict, Annotated,Sequence,Literal
from langgraph.graph import StateGraph, END,START
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage,BaseMessage
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.checkpoint.sqlite import SqliteSaver
from typing import List, Optional, Dict, Any
from datetime import datetime
from langchain.schema.runnable import RunnableConfig
import sqlite3
from utils.model_loader import ModelLoader
from retriever.retrieval import Retriever
from prompt_lib.prompts import PROMPT_REGISTRY, PromptType
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3
from pydantic import BaseModel, EmailStr
from typing import Optional,List



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
                f"Source: {meta.get('source_file', 'N/A')}\n"
                f"page:\n{d.page_content.strip()}"
            )
            formatted_docs.append(formatted)
        return "\n\n---\n\n".join(formatted_docs) 


    # ---------- Nodes ----------
    def _ai_assistant(self, state: BasicChatState):
        print("--- CALL ASSISTANT ---")
        messages = state["messages"]
        last_message = messages[-1].content

        if any(word in last_message.lower() for word in ["Source", "page"]):
            return {"messages": [HumanMessage(content="TOOL: retriever")]}
        else:
            prompt = ChatPromptTemplate.from_template(
                "You are a helpful assistant. Answer the user directly.\n\nQuestion: {question}\nAnswer:"
            )
            chain = prompt | self.llm | StrOutputParser()
            response = chain.invoke({"question": last_message})
            return {"messages": [HumanMessage(content=response)]}
    
    def _vector_retriever(self, state: BasicChatState):
        
        print("--- RETRIEVER ---")
        query = state["messages"][-1].content
        retriever = self.retriever_obj.load_retriever()
        docs = retriever.invoke(query)
        context = self._format_docs(docs)
        return {"messages": [HumanMessage(content=context)]}    

    def _grade_documents(self, state: BasicChatState) -> Literal["generator", "rewriter"]:
        print("--- GRADER ---")
        question = state["messages"][0].content
        docs = state["messages"][-1].content

        prompt = PromptTemplate(
            template="""You are a grader. Question: {question}\nDocs: {docs}\n
            Are docs relevant to the question? Answer yes or no.""",
            input_variables=["question", "docs"],
        )
        chain = prompt | self.llm | StrOutputParser()
        score = chain.invoke({"question": question, "docs": docs})
        return "generator" if "yes" in score.lower() else "rewriter"
    
    def _generate(self, state: BasicChatState):
        print("--- GENERATE ---")
        question = state["messages"][0].content
        docs = state["messages"][-1].content
        prompt = ChatPromptTemplate.from_template(
            PROMPT_REGISTRY[PromptType.DOC_BOT].template
        )
        chain = prompt | self.llm | StrOutputParser()
        response = chain.invoke({"context": docs, "question": question})
        return {"messages": [HumanMessage(content=response)]}

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
        workflow.add_node("Assistant", self._ai_assistant)
        workflow.add_node("Retriever", self._vector_retriever)
        workflow.add_node("Generator", self._generate)
        workflow.add_node("Rewriter", self._rewrite)

        workflow.add_edge(START, "Assistant")
        workflow.add_conditional_edges(
            "Assistant",
            lambda state: "Retriever" if "TOOL" in state["messages"][-1].content else END,
            {"Retriever": "Retriever", END: END},
        )
        workflow.add_conditional_edges(
            "Retriever",
            self._grade_documents,
            {"generator": "Generator", "rewriter": "Rewriter"},
        )
        workflow.add_edge("Generator", END)
        workflow.add_edge("Rewriter", "Assistant")
        return workflow

    # ---------- Public Run ----------
    def run(self, query: str,thread_id: str = "default_thread", collection_name : str = "defaultdb") -> str:
        """Run the workflow for a given query and return the final answer."""
        print(f" my collection name is inside run  : {collection_name}")

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


        # graph = StateGraph(self.BasicChatState)
        # graph.add_node("Assistant", self._ai_assistant)
        # graph.add_node("Retriever", self._vector_retriever)
        # graph.add_node("Generator", self._generate)
        # graph.add_node("Rewriter", self._rewrite)

        # # Add the chatbot node
        # graph.add_node("chatbot", chatbot)
        # # Set up the flow
        # graph.set_entry_point("chatbot")
        # graph.add_edge("chatbot", END)
        # return graph

















# from pydantic import BaseModel

# # Pydantic models for the API response
# class ChatMessage(BaseModel):  # Renamed to avoid conflict with LangGraph Message
#     id: str
#     role: str  # "user" or "assistant"
#     content: str
#     timestamp: datetime

# class ChatMessagesResponse(BaseModel):
#     chat_id: str
#     messages: List[ChatMessage]
#     total_messages: int

    


# def chatbot(state: BasicChatState, config: RunnableConfig = None): 
#     # Get the last human message
#     last_message = state["messages"][-1]
    
#     # Extract the content from the message to query RAG
#     if hasattr(last_message, 'content'):
#         query_text = last_message.content
#     else:
#         query_text = str(last_message)

#     # Get collection_name from config
#     collection_name = config.get("configurable", {}).get("collection_name", "default_collection")


#     # Query the RAG system
#     #rag_response = query_rag_langgraph(query_text)

#     rag_response = query_rag_langgraph(query_text, collection_name=collection_name)
    
#     # Create an AI message from the RAG response
#     ai_message = AIMessage(content=rag_response.response_text)
    
#     return {
#        "messages": [ai_message]
#     }

# # Initialize memory
# sqlite_conn = sqlite3.connect("database/checkpoint.db", check_same_thread=False)
# checkpointer = SqliteSaver(sqlite_conn)

# # Create the graph
# graph = StateGraph(BasicChatState)

# # Add the chatbot node
# graph.add_node("chatbot", chatbot)

# # Set up the flow
# graph.set_entry_point("chatbot")
# graph.add_edge("chatbot", END)

# # Compile the graph
# chabotapp = graph.compile(checkpointer=checkpointer)



# def get_chat_messages_from_langgraphbak(chat_id: str) -> List[Any]:
#     """
#     Get chat messages from LangGraph checkpointer for a specific chat_id
    
#     Args:
#         chat_id: The thread_id/chat_id to retrieve messages for
        
#     Returns:
#         List of LangGraph message objects
#     """
#     # Get state from LangGraph checkpointer using the same pattern as your existing code
#     state = chabotapp.get_state(config={'configurable': {'thread_id': chat_id}})
        
#     # Extract messages from state, return empty list if not found
#     messages = state.values.get('messages', [])
#     print("checkpoint message :" ,messages)        
#     return messages

# def get_chat_messages_from_langgraph(chat_id: str) -> List[Any]:
#     """
#     Get chat messages from LangGraph checkpointer for a specific chat_id
    
#     Args:
#         chat_id: The thread_id/chat_id to retrieve messages for
        
#     Returns:
#         List of LangGraph message objects
#     """
#             # Get state from LangGraph checkpointer using the same pattern as your existing code
#     state = chabotapp.get_state(config={'configurable': {'thread_id': chat_id}})
        
#         # Extract messages from state, return empty list if not found
#     messages = state.values.get('messages', [])
        
#     return messages




# def convert_langgraph_messages_to_format(langgraph_messages: List[Any]) -> List[ChatMessage]:
#     """
#     Convert LangGraph checkpointer message format to ChatMessage format
    
#     Args:
#         langgraph_messages: List of HumanMessage/AIMessage objects from LangGraph
        
#     Returns:
#         List of ChatMessage objects in desired format
#     """
#     messages = []
    
#     for msg in langgraph_messages:
#         try:
#             # Determine the role based on message type
#             message_type = msg.__class__.__name__
            
#             if message_type == 'HumanMessage':
#                 role = "user"
#             elif message_type == 'AIMessage':
#                 role = "assistant"
#             elif message_type == 'SystemMessage':
#                 role = "system"
#             else:
#                 # Handle any other message types
#                 role = "system"
            
#             # Extract content safely
#             content = getattr(msg, 'content', '')
#             if not isinstance(content, str):
#                 content = str(content)
            
#             # Extract ID safely
#             message_id = getattr(msg, 'id', '')
#             if not isinstance(message_id, str):
#                 message_id = str(message_id)
            
#             # Extract timestamp if available in additional_kwargs or use current time
#             timestamp = datetime.now()
            
#             # Check if there's a timestamp in additional_kwargs or response_metadata
#             if hasattr(msg, 'additional_kwargs') and msg.additional_kwargs:
#                 if 'timestamp' in msg.additional_kwargs:
#                     try:
#                         timestamp = datetime.fromisoformat(str(msg.additional_kwargs['timestamp']))
#                     except:
#                         pass
            
#             if hasattr(msg, 'response_metadata') and msg.response_metadata:
#                 if 'timestamp' in msg.response_metadata:
#                     try:
#                         timestamp = datetime.fromisoformat(str(msg.response_metadata['timestamp']))
#                     except:
#                         pass
            
#             # Create ChatMessage object using dict to avoid Pydantic conflicts
#             message_dict = {
#                 "id": message_id,
#                 "role": role,
#                 "content": content,
#                 "timestamp": timestamp
#             }
            
#             # Create Pydantic model from dict
#             message = ChatMessage(**message_dict)
#             messages.append(message)
            
#         except Exception as e:
#             print(f"Error processing message {msg}: {e}")
#             continue
    
#     return messages



# def convert_langgraph_messages_to_formatbak(langgraph_messages: List[Any]) -> List[ChatMessage]:
#     """
#     Convert LangGraph checkpointer message format to Message format
    
#     Args:
#         langgraph_messages: List of HumanMessage/AIMessage objects from LangGraph
        
#     Returns:
#         List of Message objects in desired format
#     """
#     messages = []
    
#     for msg in langgraph_messages:
#         # Determine the role based on message type
#         message_type = msg.__class__.__name__
        
#         if message_type == 'HumanMessage':
#             role = "user"
#         elif message_type == 'AIMessage':
#             role = "assistant"
#         elif message_type == 'SystemMessage':
#             role = "system"
#         else:
#             # Handle any other message types
#             role = "system"
        
#         # Extract timestamp if available in additional_kwargs or use current time
#         timestamp = datetime.now()
        
#         # Check if there's a timestamp in additional_kwargs or response_metadata
#         if hasattr(msg, 'additional_kwargs') and msg.additional_kwargs:
#             if 'timestamp' in msg.additional_kwargs:
#                 try:
#                     timestamp = datetime.fromisoformat(str(msg.additional_kwargs['timestamp']))
#                 except:
#                     pass
        
#         if hasattr(msg, 'response_metadata') and msg.response_metadata:
#             if 'timestamp' in msg.response_metadata:
#                 try:
#                     timestamp = datetime.fromisoformat(str(msg.response_metadata['timestamp']))
#                 except:
#                     pass
        
#         # Create Message object
#         message = Message(
#             id=str(msg.id),
#             role=role,
#             content=msg.content,
#             timestamp=timestamp
#         )
        
#         messages.append(message)
    
#     return messages