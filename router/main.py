import uvicorn
from fastapi import FastAPI, Request, Form,Depends,HTTPException,UploadFile,File
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from langchain_core.messages import HumanMessage
from fastapi.responses import FileResponse
#from workflow.langgraph_function import AgenticRAG
from workflow.langgraph_function_guardrails import AgenticRAG
from workflow.responsemodel import responsemodel
from utils.db_loader import UserRepository
from utils.db_loader import get_db
from sqlalchemy.orm import Session
from workflow.otp_function import OtpOperation
from workflow.chat_function import ChatDBOperation
from typing import Optional,List,Any
from datetime import datetime,timedelta
import os
from logger import GLOBAL_LOGGER as log
from workflow.document_function import DocHandler,FastAPIFileAdapter,DocumentProcessor,DocumentProcessor_Qdrant


import os
import sys


# Global OTP handler instance
otp_ops = OtpOperation()
chat_ops = ChatDBOperation()
#chatbot_app = AgenticRAG()
chatbot_app = AgenticRAG(use_guardrails=True, strict_mode=False)

app = FastAPI()
# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)


# ---------- FastAPI Endpoints ----------
# Mount /Rag to serve your simple HTML app
app.mount("/chatbox", StaticFiles(directory="static/chatbox", html=True), name="chatbox")
# Optional: Redirect /Rag to the index.html file
@app.get("/chatbox", include_in_schema=False)
async def serve_newrag_index():
    return FileResponse("static/chatbox/mainpage.html")

@app.get("/")
def index():
    return {"Ncell":"Ncell AI platform"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "message": "FastAPI OLLAMA backend is running"}

@app.post("/register-user", response_model=responsemodel.SuccessResponse)
async def register_user(
    user_data: responsemodel.UserRegistration,
    db: Session = Depends(get_db)
):
    user_repo = UserRepository(db)
    return user_repo.register_user(user_data)

@app.get("/users", response_model=list[responsemodel.UserResponse])
async def get_all_users(db: Session = Depends(get_db)):
    """Get all registered users (for admin purposes)"""
    user_repo = UserRepository(db)
    users = UserRepository.fetch_all_user()
    return users

@app.get("/users/{email}", response_model=responsemodel.UserResponse)
async def get_user(email: str, db: Session = Depends(get_db)):
    """Get specific user by ID"""
    user_repo = UserRepository(db)
    user=user_repo.fetch_user(email)
    return user



@app.post("/request-otp", response_model=responsemodel.ResponseModel)
async def request_otp(request: responsemodel.OTPRequest):
    """Generate and send OTP to the provided email"""
    try:
        print(f"try ti get otp for {request.email}")
        # Clean up expired data
        otp_ops.cleanup_expired_data()
        email = request.email.lower()
        # Generate OTP
        otp = otp_ops.generate_otp()
        print(f" my generated otp is : {otp}")
        # Set expiration time (5 minutes from now)
        expiration_time = datetime.now() + timedelta(minutes=5)
        
        # Store OTP in database
        otp_ops.store_otp(email, otp, expiration_time)
        
        # Send email (for development, print the OTP)
        if os.getenv("DEVELOPMENT_MODE", "true").lower() == "true":
            print(f"Development Mode: OTP for {email} is {otp}")
            email_sent = True
        else:
            print(f"Development Mode: OTP for {email} is {otp}")
            email_sent = OtpOperation.send_email_otp(email, otp)
        
        if email_sent:
            return responsemodel.ResponseModel(
                success=True,
                message="OTP sent successfully to your email"
            )
        else:
            raise HTTPException(status_code=500, detail="Failed to send email")
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/verify-otp", response_model=responsemodel.SessionResponse)
async def verify_otp(request: responsemodel.OTPVerification):
    """Verify OTP and create a session for the user"""
    try:
        email = request.email.lower()
        provided_otp = request.otp.strip()

        # Retrieve OTP data
        otp_data = otp_ops.get_otp_data(email)
        if not otp_data:
            raise HTTPException(
                status_code=400,
                detail="No OTP found. Please request a new OTP."
            )

        # Expired OTP check
        if datetime.now() > otp_data["expires_at"]:
            otp_ops.delete_otp(email)
            raise HTTPException(
                status_code=400,
                detail="OTP has expired. Please request a new one."
            )

        # Too many attempts
        if otp_data["attempts"] >= 3:
            otp_ops.delete_otp(email)
            raise HTTPException(
                status_code=400,
                detail="Maximum OTP attempts exceeded. Request a new OTP."
            )

        # Correct OTP
        if provided_otp == otp_data["otp"]:
            otp_ops.delete_otp(email)
            session_data = otp_ops.create_user_session_db(email)

            return responsemodel.SessionResponse(
                success=True,
                message="OTP verified successfully. Session created.",
                session_token=session_data["jwt_token"],
                user_email=email,
                expires_at=session_data["expires_at"].isoformat()
            )

        # Incorrect OTP: increment attempt counter
        attempts = otp_ops.increment_otp_attempts(email)
        remaining = 3 - attempts

        if remaining > 0:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid OTP. {remaining} attempt(s) remaining."
            )
        else:
            otp_ops.delete_otp(email)
            raise HTTPException(
                status_code=400,
                detail="Invalid OTP. Maximum attempts exceeded."
            )

    except HTTPException:
        raise  # Let FastAPI handle known exceptions
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")


@app.get("/chat-history/{username}")
async def get_chat_history(
    username: str,
    limit: int = 50,
    offset: int = 0
):
    print(f"Getting chat history for session_id: {username}")
    """Get chat history for a session from SQLite database"""

    chat_history=chat_ops.get_conversation_history(username)
    print(chat_history)
    print(username)
    return chat_history[offset:offset + limit]

@app.get("/chat-history/history/{chat_id}" )
async def get_conversation_detail(
    chat_id: str,
    db: Session = Depends(get_db)
):
    """
    Get chat messages directly from LangGraph without database verification
    """
    print(f"Getting messages (direct) for chat_id: {chat_id}")
    
    try:
        # Get messages from LangGraph checkpointer
        langgraph_messages = chatbot_app.get_chat_messages_from_langgraph(chat_id)
        print("output formateed_message",langgraph_messages)
        # Convert to desired format
        formatted_messages = chat_ops.convert_langgraph_messages_to_format(langgraph_messages)
        print("formateed_message-----:",formatted_messages)
        
        # # Create response
        # response = ChatMessagesResponse(
        #     chat_id=chat_id,
        #     messages=formatted_messages,
        #     total_messages=len(formatted_messages)
        # )
        
        print(f"Found {len(formatted_messages)} messages for chat {chat_id}")
        return formatted_messages
        
    except Exception as e:
        print(f"Error getting chat messages: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error loading chat messages: {str(e)}"
        )


@app.post("/chat_langgraph", response_model=responsemodel.ChatResponse)
async def chat_query_langgraph(
    request: responsemodel.ChatRequest
    #,
    #session_id: str = Depends(verify_session)
):
    # print(request.query)
    # print(request.session_id)
    # print(request.chat_id)
    # print(request.email)
    # print(request.use_rag)
    # print(request.user_id)
    # print(request.ragtype)

    #setragtype(request.ragtype)
    # Configuration for conversation thread
    userdepartment=chat_ops.get_user_info(request.user_id)

    collection_name=chat_ops.get_collection(request.ragtype,userdepartment)
    result = chatbot_app.run(query=request.query, thread_id=request.chat_id,  collection_name=collection_name )

    #result = chabotapp.invoke({"messages": [HumanMessage(content=request.query)]},config=config)

    # Extract and print the AI response
    ai_response = result  # result is the final content from run()
    print("AI: " + ai_response)   
    chat_ops.save_conversation(request.chat_id,request.user_id,request.query,request.session_id)
    return responsemodel.ChatResponse(
        success=True,
        message="Chat processed successfully",
        response=ai_response,
        chat_id=str(request.chat_id),  # convert to str
        timestamp=datetime.now().isoformat(),
        session_id=request.session_id or ""  # ensure not None
    )


@app.post("/uploadprocess")
async def upload_process(
    current_user: str = Form(...),
    files: Optional[List[UploadFile]] = File(None),
    urls: Optional[List[str]] = Form(None),
):
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded.")

    user_department = chat_ops.get_user_info(current_user)
    dh = DocHandler()
    if user_department == "IT-GLobal":
        collection_name = chat_ops.get_collection("IT-GLobal", user_department)
        #dp= DocumentProcessor_Qdrant()
        dp = DocumentProcessor()   

    else:
        collection_name = chat_ops.get_collection("Mydepartment", user_department)
        #dp = DocumentProcessor() 
        dp= DocumentProcessor_Qdrant()
  
    department_temp_path = user_department.replace(" ", "")
    log.info(f"User '{current_user}' from {user_department} trying to load doc in collection {collection_name}.")
    processed_files = []

    for file in files:
        log.info(f"📥 Received file: {file.filename}")
        try:
            # Save file temporarily
            temp_saved_path = await dh.temp_save_file(FastAPIFileAdapter(file), department_temp_path)
            
            # Check for duplication
            existing_doc = dp.duplicate_validation(file.filename, collection_name)
            if existing_doc and len(existing_doc["ids"]) > 0:
                log.info(f"⚠️ File '{file.filename}' already exists in ChromaDB. Skipping upload.")
                os.remove(temp_saved_path)
                continue

            # Process document
            log.info(f"🔍 No duplicate found. Processing file: {file.filename}")
            documents = dp.load_document(temp_saved_path)
            chunks = dp.split_documents(documents)
            dp.add_to_vectorestore(chunks, collection_name)
            log.info(f"✅ File '{file.filename}' processed and added to ChromaDB.")

            # Optional: move file to permanent storage (implement save_file if needed)
            saved_path = await dh.save_file(temp_saved_path, file.filename, department_temp_path)
            if saved_path:
                processed_files.append(file.filename)

        except Exception as e:
            log.exception(f"❌ Error processing file '{file.filename}': {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to process '{file.filename}': {str(e)}")

    return {
        "status": "success",
        "current_user": current_user,
        "files_processed": processed_files,
        "urls_received": urls or [],
    }





if __name__ == "__main__":
    #
    port = 3000 
    addr = "127.0.0.1" 

    print (f"Running fastAPI server on port {port}")
    uvicorn.run("main:app",host =addr,port=port)

