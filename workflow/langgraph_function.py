from typing import TypedDict, Sequence, Annotated, List, Any
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_core.runnables import RunnableConfig
from utils.model_loader import ModelLoader
from retriever.retrieval import Retriever
from prompt_lib.prompts import PROMPT_REGISTRY, PromptType
import sqlite3


class AgenticRAG:

    class BasicChatState(TypedDict):
        messages: Annotated[Sequence[BaseMessage], add_messages]
        context: str  # 🔧 Add explicit context field

    def __init__(self):
        self.retriever_obj = Retriever(use_qdrant=True)
        self.model_loader = ModelLoader()
        self.llm = self.model_loader.load_llm()

        self.sqlite_conn = sqlite3.connect("database/checkpoint.db", check_same_thread=False)
        self.checkpointer = SqliteSaver(self.sqlite_conn)

        self.workflow = self._build_workflow()
        self.app = self.workflow.compile(checkpointer=self.checkpointer)

    def _format_docs(self, docs) -> str:
        """Format retrieved docs into a string for prompt."""
        if not docs:
            return ""
        formatted_docs = []
        for d in docs:
            meta = getattr(d, "metadata", {}) or {}
            formatted_docs.append(
                f"Source: {meta.get('source', 'N/A')}\nPage:\n{d.page_content.strip()}"
            )
        return "\n\n---\n\n".join(formatted_docs)

    def _rewrite_query(self, question: str, chat_history: str) -> str:
        """🔧 FIXED: Rewrite follow-up questions to be standalone"""
        if not chat_history or len(question.split()) > 8:
            # Don't rewrite if no history or question is already detailed
            return question
        
        rewrite_prompt = f"""Rewrite this follow-up question to be a standalone question that includes necessary context.

Conversation history:
{chat_history}

Follow-up question: {question}

Standalone question (be concise):"""
        
        try:
            response = self.llm.invoke(rewrite_prompt)
            
            # 🔧 Extract string content from AIMessage
            if hasattr(response, 'content'):
                rewritten = response.content.strip()
            elif isinstance(response, str):
                rewritten = response.strip()
            else:
                rewritten = str(response).strip()
            
            # 🔧 Clean up any extra formatting
            # Remove quotes if the LLM added them
            if rewritten.startswith('"') and rewritten.endswith('"'):
                rewritten = rewritten[1:-1]
            elif rewritten.startswith("'") and rewritten.endswith("'"):
                rewritten = rewritten[1:-1]
            
            print(f"🔄 Query rewritten: '{question}' → '{rewritten}'")
            return rewritten
        except Exception as e:
            print(f"⚠️ Query rewrite failed: {e}, using original query")
            return question  # Fallback to original

    def _extract_score(self, doc) -> float:
        """🔧 Extract relevance score from document (handles multiple storage locations)"""
        # Try multiple locations where score might be stored
        
        # 1. Direct attribute
        if hasattr(doc, 'score') and doc.score is not None:
            return float(doc.score)
        
        # 2. Metadata dict
        if hasattr(doc, 'metadata') and isinstance(doc.metadata, dict):
            if 'score' in doc.metadata:
                return float(doc.metadata['score'])
            # Sometimes stored as _score or relevance_score
            if '_score' in doc.metadata:
                return float(doc.metadata['_score'])
            if 'relevance_score' in doc.metadata:
                return float(doc.metadata['relevance_score'])
        
        # 3. No score found
        return None

    def _vector_retriever(self, state: BasicChatState, config: RunnableConfig = None):
        """🔧 FIXED: Proper retriever configuration with score tracking"""
        print("--- RETRIEVER ---")
        
        # Get original query
        query = state["messages"][-1].content
        
        # 🔧 Optional: Rewrite query using chat history for follow-up questions
        chat_messages = [msg for msg in state["messages"][:-1] 
                        if isinstance(msg, (HumanMessage, AIMessage))]
        if len(chat_messages) >= 2:
            # Build mini history for context
            recent_history = f"User: {chat_messages[-2].content}\nAssistant: {chat_messages[-1].content}"
            query = self._rewrite_query(query, recent_history)
        
        collection_name = config.get("configurable", {}).get("collection_name", "defaultdb")
        print(f"Using collection: {collection_name} for query: {query}")


        # 🔧 Load retriever with proper score threshold (0.5 = moderate relevance)
        retriever = self.retriever_obj.load_retriever_chroma(
        #retriever = self.retriever_obj.load_retriever_qdrant(
            collection_name=collection_name,
            top_k=5,
            #score_threshold=0.3  # 🔧 Lowered to 0.3 for testing - adjust as needed
        )
        
        # 🔧 Retriever now automatically filters by score_threshold
        docs = retriever.invoke(query)

        if not docs:
            print(f"⚠️ No relevant documents found for query: '{query}' (score < 0.3)")
            return {"context": ""}

        context = self._format_docs(docs)

        print(f"Retrieved {len(docs)} relevant document(s)")
        for i, doc in enumerate(docs):
            score = self._extract_score(doc)
            score_display = f"{score:.3f}" if score is not None else "N/A"
            
            print(f"\n--- Document {i+1} ---")
            print(f"Score: {score_display}")
            print(f"Content:\n{doc.page_content[:200]}...")
            if hasattr(doc, "metadata"):
                # Print metadata without score fields to reduce clutter
                meta_clean = {k: v for k, v in doc.metadata.items() 
                             if k not in ['score', '_score', 'relevance_score']}
                if meta_clean:
                    print(f"Metadata: {meta_clean}")

        return {"context": context}

    def _generate(self, state: BasicChatState):
        """🔧 FIXED: Better context/history handling"""
        print("--- GENERATE ---")

        # 🔧 Get context from state field, not from messages
        context = state.get("context", "").strip()
        
        # Get only Human and AI messages for history
        chat_messages = [msg for msg in state["messages"] 
                        if isinstance(msg, (HumanMessage, AIMessage))]
        
        # Latest question is the last HumanMessage
        question = chat_messages[-1].content.strip() if chat_messages else ""

        # 🔧 Build history from last N exchanges (excluding current question)
        history_pairs = []
        i = len(chat_messages) - 2  # Start before the current question
        pair_count = 0
        
        while i >= 0 and pair_count < 5:  # Last 5 exchanges
            if i > 0 and isinstance(chat_messages[i], AIMessage) and isinstance(chat_messages[i-1], HumanMessage):
                history_pairs.insert(0, f"User: {chat_messages[i-1].content.strip()}\nAssistant: {chat_messages[i].content.strip()}")
                pair_count += 1
                i -= 2
            else:
                i -= 1

        chat_history = "\n\n".join(history_pairs)

        # Debug info
        print(f"Context length: {len(context)} chars")
        print(f"Chat history pairs: {len(history_pairs)}")
        print(f"Question: {question}")

        # 🔧 Handle no relevant context
        if not context:
            print(f"⚠️ No relevant context for query: '{question}'")
            return {
                "messages": [
                    AIMessage(content="I don't have enough information in the documents to answer this question.")
                ]
            }

        # 🔧 Improved prompt emphasizing question focus
        prompt_text = """You are a helpful assistant that answers questions based on provided context.

IMPORTANT: Answer the specific question asked by the user. Do not just summarize the context.

Context from documents:
{context}

Previous conversation:
{chat_history}

Current question: {question}

Provide a direct answer to the question. If the context doesn't contain the answer, say so clearly."""

        prompt = ChatPromptTemplate.from_template(prompt_text)
        chain = prompt | self.llm | StrOutputParser()

        response = chain.invoke({
            "context": context,
            "chat_history": chat_history,
            "question": question
        })

        print(f"AI Response: {response[:100]}...")  # 🔧 Log response preview
        return {"messages": [AIMessage(content=response)]}

    def _build_workflow(self):
        workflow = StateGraph(self.BasicChatState)
        workflow.add_node("Retriever", self._vector_retriever)
        workflow.add_node("Generator", self._generate)
        workflow.add_edge(START, "Retriever")
        workflow.add_edge("Retriever", "Generator")
        workflow.add_edge("Generator", END)
        return workflow

    def run(self, query: str, thread_id: str = "default_thread", collection_name: str = "defaultdb") -> str:
        """Run the workflow for a query."""
        config = {"configurable": {"thread_id": thread_id, "collection_name": collection_name}}
        result = self.app.invoke({"messages": [HumanMessage(content=query)]}, config=config)
        return result["messages"][-1].content

    def get_chat_messages_from_langgraph(self, chat_id: str) -> List[Any]:
        """Get chat messages from LangGraph checkpointer for a specific chat_id"""
        state = self.app.get_state(config={'configurable': {'thread_id': chat_id}})
        messages = state.values.get('messages', [])
        return messages