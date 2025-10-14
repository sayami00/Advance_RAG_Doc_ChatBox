import os
import sqlite3
import uuid
from datetime import datetime
from typing import Dict, List, Optional,Any
from pydantic import BaseModel
from fastapi import HTTPException
from contextlib import contextmanager
from workflow.responsemodel import responsemodel

class ChatSummary(BaseModel):
    id: str
    title: str
    created_at: datetime
    updated_at: datetime


class ChatDBOperation:
    def __init__(self, db_path: str = "database/chatbox_users.db"):
        self.db_path = db_path
        if not os.path.exists(self.db_path):
            raise FileNotFoundError(f"Database file not found at: {self.db_path}")

    @contextmanager
    def get_db_connection(self):
        """Context manager for DB connection."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def get_collection(self,ragtype: str, user_department: str) -> str:
        """
        Decide which vectorstore collection to use based on ragtype and user department.
        """
        # Mapping logic
        if ragtype == "Mydepartment":
                return f"{user_department}_Docs"  # fallback per department
        
        elif ragtype == "ITServicesGlobal":
            return "GlobalITServicesDB"
        
        elif ragtype == "GPTs":
            return "GPTCollection"
        
        # Default fallback
        return "GeneralCollection"

    def get_user_info(self, user_name: str) -> Optional[str]:
        """Return department for a user (or None if not found)."""
        with self.get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT department FROM users WHERE username = ?",
                (user_name,)
            )
            result = cursor.fetchone()
            return result["department"] if result else None

    def save_conversation(self, chat_id: str, user_name: str, title: str, session_id: str) -> str:
        """Create new conversation record if it doesn't exist."""
        with self.get_db_connection() as conn:
            cursor = conn.cursor()

            cursor.execute(
                "SELECT chat_id FROM chat_conversations WHERE chat_id = ?",
                (chat_id,)
            )
            result = cursor.fetchone()

            if result:
                return result["chat_id"]

            cursor.execute(
                "INSERT INTO chat_conversations (chat_id, user_name, session_id, title) VALUES (?, ?, ?, ?)",
                (chat_id, user_name, session_id, title)
            )
            conn.commit()
            return chat_id

    def get_conversation_history(self, user_name: str, limit: int = 10, offset: int = 0) -> List[ChatSummary]:
        """Return recent chat history for a user."""
        try:
            with self.get_db_connection() as conn:
                cursor = conn.cursor()

                query = """
                    SELECT chat_id, title, created_at, updated_at
                    FROM chat_conversations
                    WHERE user_name = ?
                    ORDER BY updated_at DESC
                    LIMIT ? OFFSET ?
                """

                cursor.execute(query, (user_name, limit, offset))
                rows = cursor.fetchall()

                chat_history = []

                for row in rows:
                    created_at = self._parse_datetime(row["created_at"])
                    updated_at = self._parse_datetime(row["updated_at"])

                    chat_summary = ChatSummary(
                        id=row["chat_id"],
                        title=row["title"] or f"Chat {row['chat_id'][:8]}",
                        created_at=created_at,
                        updated_at=updated_at,
                    )
                    chat_history.append(chat_summary)

                return chat_history

        except sqlite3.Error as e:
            raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error loading chat history: {str(e)}")

    

    def convert_langgraph_messages_to_format(self,langgraph_messages: List[Any]) -> List[responsemodel.ChatMessage]:
        """
        Convert LangGraph checkpointer message format to ChatMessage format
        
        Args:
            langgraph_messages: List of HumanMessage/AIMessage objects from LangGraph
            
        Returns:
            List of ChatMessage objects in desired format
        """
        messages = []
        
        for msg in langgraph_messages:
            try:
                # Determine the role based on message type
                message_type = msg.__class__.__name__
                
                if message_type == 'HumanMessage':
                    role = "user"
                elif message_type == 'AIMessage':
                    role = "assistant"
                elif message_type == 'SystemMessage':
                    role = "system"
                else:
                    # Handle any other message types
                    role = "system"
                
                # Extract content safely
                content = getattr(msg, 'content', '')
                if not isinstance(content, str):
                    content = str(content)
                
                # Extract ID safely
                message_id = getattr(msg, 'id', '')
                if not isinstance(message_id, str):
                    message_id = str(message_id)
                
                # Extract timestamp if available in additional_kwargs or use current time
                timestamp = datetime.now()
                
                # Check if there's a timestamp in additional_kwargs or response_metadata
                if hasattr(msg, 'additional_kwargs') and msg.additional_kwargs:
                    if 'timestamp' in msg.additional_kwargs:
                        try:
                            timestamp = datetime.fromisoformat(str(msg.additional_kwargs['timestamp']))
                        except:
                            pass
                
                if hasattr(msg, 'response_metadata') and msg.response_metadata:
                    if 'timestamp' in msg.response_metadata:
                        try:
                            timestamp = datetime.fromisoformat(str(msg.response_metadata['timestamp']))
                        except:
                            pass
                
                # Create ChatMessage object using dict to avoid Pydantic conflicts
                message_dict = {
                    "id": message_id,
                    "role": role,
                    "content": content,
                    "timestamp": timestamp
                }
                
                # Create Pydantic model from dict
                message = responsemodel.ChatMessage(**message_dict)
                messages.append(message)
                
            except Exception as e:
                print(f"Error processing message {msg}: {e}")
                continue
        
        return messages

    @staticmethod
    def _parse_datetime(value) -> datetime:
        """Helper to safely parse datetime strings or return now."""
        if isinstance(value, str):
            try:
                return datetime.fromisoformat(value.replace('Z', '+00:00'))
            except Exception:
                return datetime.now()
        return value if isinstance(value, datetime) else datetime.now()
