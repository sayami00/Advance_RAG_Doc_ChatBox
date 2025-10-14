from pydantic import BaseModel,EmailStr
from typing import Optional,List,Dict
from datetime import datetime

class responsemodel:
    class ChatResponse(BaseModel):
        success: bool
        message: str
        response: str
        chat_id: str
        timestamp: str
        session_id: str
        sources: Optional[List[str]] = None
        conversation_id: Optional[str] = None

    class ChatRequest(BaseModel):
        query: str
        chat_id: Optional[str] = None
        user_id: Optional[str] = None
        session_id: Optional[str] = None
        email: Optional[EmailStr] = None
        use_rag: Optional[bool] = True
        ragtype:str

    class SubmitRequest(BaseModel):
        requesttext:str

    class FileNameRequest(BaseModel):
        filename: str

    # Pydantic models for request/response
    class UserRegistration(BaseModel):
        email: EmailStr
        department: str
        username: str = None  # Optional username

    class SuccessResponse(BaseModel):
        success: bool
        message: str
        user_id: int = None


    class UserResponse(BaseModel):
        id: int
        email: str
        department: str
        username: str = None
        is_active: bool
        created_at: datetime
        
        class Config:
            from_attributes = True


    class ResponseModel(BaseModel):
        success: bool
        message: str
        data: Optional[Dict] = None

    class OTPRequest(BaseModel):
        email: EmailStr


    class SessionResponse(BaseModel):
        success: bool
        message: str
        session_token: str
        user_email: str
        expires_at: str

    class OTPVerification(BaseModel):
        email: EmailStr
        otp: str

    class ChatMessage(BaseModel):  # Renamed to avoid conflict with LangGraph Message
        id: str
        role: str  # "user" or "assistant"
        content: str
        timestamp: datetime