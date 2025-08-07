from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import google.generativeai as genai
import os
from dotenv import load_dotenv
from typing import Dict, List, Optional
import time
import json
from collections import defaultdict
from datetime import datetime
import uuid

load_dotenv()

app = FastAPI(title="AskMe Bot API - Enhanced", version="2.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure Google Gemini
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model = genai.GenerativeModel('gemini-1.5-flash')

# Enhanced storage
user_requests = defaultdict(list)
user_conversations = defaultdict(list)
user_analytics = defaultdict(dict)
MAX_REQUESTS_PER_SESSION = 100 
TIME_WINDOW = 3600  # 1 hour
MAX_CONVERSATION_HISTORY = 50

class ChatRequest(BaseModel):
    message: str
    session_id: str
    conversation_mode: bool = Field(default=True, description="Include conversation context")
    creativity_level: float = Field(default=0.7, ge=0.0, le=2.0, description="Temperature for AI responses")

class ChatResponse(BaseModel):
    response: str
    remaining_requests: int
    session_id: str
    message_id: str
    timestamp: str
    conversation_count: int
    estimated_tokens: int

class ConversationExport(BaseModel):
    session_id: str
    messages: List[dict]
    created_at: str
    total_messages: int

class UserAnalytics(BaseModel):
    total_queries: int
    session_start_time: str
    most_common_topics: List[str]
    average_response_time: float

# Enhanced rate limiting
def check_rate_limit(session_id: str) -> bool:
    current_time = time.time()
    user_requests[session_id] = [
        req_time for req_time in user_requests[session_id] 
        if current_time - req_time < TIME_WINDOW
    ]
    return len(user_requests[session_id]) < MAX_REQUESTS_PER_SESSION

def add_request(session_id: str):
    user_requests[session_id].append(time.time())
    # Update analytics
    if session_id not in user_analytics:
        user_analytics[session_id] = {
            'total_queries': 0,
            'session_start_time': datetime.now().isoformat(),
            'response_times': [],
            'topics': []
        }
    user_analytics[session_id]['total_queries'] += 1

def get_remaining_requests(session_id: str) -> int:
    return MAX_REQUESTS_PER_SESSION - len(user_requests[session_id])

def get_conversation_context(session_id: str, max_context: int = 6) -> str:
    if session_id not in user_conversations:
        return ""
    
    recent_messages = user_conversations[session_id][-max_context:]
    context_parts = []
    
    for msg in recent_messages:
        role = "Human" if msg['role'] == 'user' else "Assistant"
        context_parts.append(f"{role}: {msg['content'][:200]}")  # Limit context length
    
    return "\n".join(context_parts) if context_parts else ""

def add_to_conversation(session_id: str, role: str, content: str, message_id: str = None):
    message = {
        'id': message_id or str(uuid.uuid4()),
        'role': role,
        'content': content,
        'timestamp': datetime.now().isoformat(),
    }
    
    if session_id not in user_conversations:
        user_conversations[session_id] = []
    
    user_conversations[session_id].append(message)
    
    # Keep only recent messages
    if len(user_conversations[session_id]) > MAX_CONVERSATION_HISTORY:
        user_conversations[session_id] = user_conversations[session_id][-MAX_CONVERSATION_HISTORY:]

def estimate_tokens(text: str) -> int:
    # Simple token estimation (roughly 4 characters per token)
    return len(text) // 4

@app.get("/")
async def root():
    return {
        "message": "🤖 Enhanced AskMe Bot API is running!",
        "status": "healthy",
        "model": "gemini-1.5-flash",
        "features": [
            "conversation_context", 
            "rate_limiting", 
            "analytics", 
            "export_conversations",
            "creative_responses"
        ]
    }

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    start_time = time.time()
    
    try:
        if not check_rate_limit(request.session_id):
            raise HTTPException(
                status_code=429, 
                detail=f"Rate limit exceeded. Maximum {MAX_REQUESTS_PER_SESSION} requests per session. Try again in an hour."
            )
        
        add_request(request.session_id)
        message_id = str(uuid.uuid4())
        
        # Build enhanced prompt
        base_prompt = "Answer like you're a helpful assistant."
        
        if request.conversation_mode:
            context = get_conversation_context(request.session_id)
            if context:
                prompt = f"{base_prompt}\n\nPrevious conversation:\n{context}\n\nCurrent question: {request.message}"
            else:
                prompt = f"{base_prompt} {request.message}"
        else:
            prompt = f"{base_prompt} {request.message}"
        
        # Generate response with custom creativity
        generation_config = genai.types.GenerationConfig(
            candidate_count=1,
            max_output_tokens=10000,
            temperature=request.creativity_level,
        )
        
        response = model.generate_content(prompt, generation_config=generation_config)
        ai_response = response.text.strip() if response.text else "I apologize, but I couldn't generate a response. Please try again."
        
        # Add to conversation
        add_to_conversation(request.session_id, "user", request.message, message_id)
        add_to_conversation(request.session_id, "assistant", ai_response)
        
        # Update analytics
        response_time = time.time() - start_time
        user_analytics[request.session_id]['response_times'].append(response_time)
        
        remaining = get_remaining_requests(request.session_id)
        conversation_count = len(user_conversations[request.session_id])
        
        return ChatResponse(
            response=ai_response,
            remaining_requests=remaining,
            session_id=request.session_id,
            message_id=message_id,
            timestamp=datetime.now().isoformat(),
            conversation_count=conversation_count,
            estimated_tokens=estimate_tokens(ai_response)
        )
        
    except Exception as e:
        error_message = str(e)
        print(f"API Error: {error_message}")
        
        if "API_KEY_INVALID" in error_message or "invalid API key" in error_message.lower():
            raise HTTPException(status_code=401, detail="Invalid Google API key")
        elif "QUOTA_EXCEEDED" in error_message or "quota" in error_message.lower():
            raise HTTPException(status_code=429, detail="Google API quota exceeded")
        elif "SAFETY" in error_message or "blocked" in error_message.lower():
            raise HTTPException(status_code=400, detail="Content blocked by safety filters")
        else:
            raise HTTPException(status_code=500, detail=f"Google API error: {error_message}")

@app.get("/conversation/{session_id}/export", response_model=ConversationExport)
async def export_conversation(session_id: str):
    """Export conversation history"""
    if session_id not in user_conversations:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    return ConversationExport(
        session_id=session_id,
        messages=user_conversations[session_id],
        created_at=datetime.now().isoformat(),
        total_messages=len(user_conversations[session_id])
    )

@app.get("/analytics/{session_id}", response_model=UserAnalytics)
async def get_user_analytics(session_id: str):
    """Get user analytics for session"""
    if session_id not in user_analytics:
        raise HTTPException(status_code=404, detail="Analytics not found")
    
    analytics = user_analytics[session_id]
    avg_response_time = sum(analytics.get('response_times', [0])) / max(len(analytics.get('response_times', [1])), 1)
    
    return UserAnalytics(
        total_queries=analytics['total_queries'],
        session_start_time=analytics['session_start_time'],
        most_common_topics=analytics.get('topics', [])[:5],
        average_response_time=round(avg_response_time, 2)
    )

@app.delete("/conversation/{session_id}")
async def clear_conversation(session_id: str):
    """Clear conversation history"""
    if session_id in user_conversations:
        del user_conversations[session_id]
    if session_id in user_analytics:
        del user_analytics[session_id]
    return {"message": "Conversation and analytics cleared successfully"}

@app.get("/health")
async def health_check():
    total_sessions = len(user_conversations)
    total_messages = sum(len(conv) for conv in user_conversations.values())
    
    return {
        "status": "healthy",
        "service": "Enhanced AskMe Bot API",
        "model": "gemini-1.5-flash",
        "active_sessions": total_sessions,
        "total_messages": total_messages,
        "uptime": datetime.now().isoformat()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
