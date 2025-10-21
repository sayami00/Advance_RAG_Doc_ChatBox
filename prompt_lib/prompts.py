from enum import Enum
from typing import Dict
import string


class PromptType(str, Enum):
    PRODUCT_BOT = "product_bot"
    DOC_BOT = "doc_bot"
    CUSTOMER_TEXT = "customer_text"
    DOC_BOT_OLD = "aaa"


class PromptTemplate:
    def __init__(self, template: str, description: str = "", version: str = "v1"):
        self.template = template.strip()
        self.description = description
        self.version = version

    def format(self, **kwargs) -> str:
        # Validate placeholders before formatting
        missing = [
            f for f in self.required_placeholders() if f not in kwargs
        ]
        if missing:
            raise ValueError(f"Missing placeholders: {missing}")
        return self.template.format(**kwargs)

    def required_placeholders(self):
        return [
            field_name
            for _, field_name, _, _ in string.Formatter().parse(self.template)
            if field_name
        ]


# Central Registry
PROMPT_REGISTRY: Dict[PromptType, PromptTemplate] = {
    PromptType.PRODUCT_BOT: PromptTemplate(
        """
        You are an expert EcommerceBot specialized in product recommendations and handling customer queries.
        Analyze the provided product titles, ratings, and reviews to provide accurate, helpful responses.
        Stay relevant to the context, and keep your answers concise and informative.

        CONTEXT:
        {context}

        QUESTION: {question}

        YOUR ANSWER:
        """,
        description="Handles ecommerce QnA & product recommendation flows"
    ),
    PromptType.DOC_BOT: PromptTemplate(
        """
        You are a helpful and knowledgeable assistant.
        Use the given **context** and the recent **conversation history** to answer the current question.
        If context is insufficient, explain briefly what information is missing.

        Context:
        {context}

        Recent Chat History (last 5 turns):
        {chat_history}

        Current Question:
        {question}

        Answer:
        """,
        description="RAG doc-bot: answers questions only from given documents; will not hallucinate."
    ),
        PromptType.DOC_BOT_OLD: PromptTemplate(
        """
        You are a highly accurate assistant that answers questions based ONLY on the provided context.

        Guidelines:

        1. **Use ONLY the context below** to answer the question. Do NOT use outside knowledge.
        2. **If the context is empty or does not contain the answer**, respond exactly:
           "The provided context does not contain enough information to answer this question."
        3. When providing an answer, **always include the source(s)** used.
        4. Format output as:

        Answer: <Your answer here>
        Sources: <List document names or metadata from the context used>

        Context:
        {context}

        Question: {question}

        Answer:
        """,
        description="RAG doc-bot: answers questions only from given documents; will not hallucinate."
    ),
    PromptType.CUSTOMER_TEXT: PromptTemplate(
        """
        You are a friendly and helpful assistant that responds to general user queries, greetings, and casual conversation.

        Guidelines:

        1. Respond naturally and helpfully.
        2. Do not assume facts about the user unless explicitly stated.
        3. Keep answers concise and clear.
        4. Format output simply as the response text.

        User Input:
        {user_text}

        Assistant Response:
        """,
        description="Handles free-form user queries or casual conversation outside of RAG context."
    )
}
