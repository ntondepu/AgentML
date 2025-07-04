"""
AI Chatbot API endpoints for the AutoML Distributed Platform.
"""

from fastapi import APIRouter, HTTPException, Depends, FastAPI, Request, Header
from pydantic import BaseModel
from typing import Dict, List, Optional, Any
import asyncio
import logging
import requests
import faiss
import numpy as np
from prometheus_client import make_asgi_app, Counter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
# from openai import OpenAI  # Uncomment if using OpenAI

from .bot import ChatbotManager
from .models import ChatMessage, ChatSession, ChatResponse

logger = logging.getLogger(__name__)

# Create router
chatbot_router = APIRouter()

# Global chatbot manager instance
chatbot_manager = None

# Prometheus metrics
chat_counter = Counter('chatbot_messages', 'Number of chatbot messages processed')

# FAISS vector DB setup (in-memory, for demo)
dim = 384  # Example embedding size
index = faiss.IndexFlatL2(dim)
vector_store = []  # Store (text, vector) tuples

# TODO: Replace with your OpenAI or Hugging Face API key
OPENAI_API_KEY = "sk-..."

API_KEY = "demo-key-123"

def api_key_auth(x_api_key: str = Header(...)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")

class ChatRequest(BaseModel):
    user: str
    message: str


async def get_chatbot_manager() -> ChatbotManager:
    """Get the chatbot manager instance."""
    global chatbot_manager
    if chatbot_manager is None:
        chatbot_manager = ChatbotManager()
        await chatbot_manager.initialize()
    return chatbot_manager


@chatbot_router.get("/")
async def get_chatbot_status():
    """Get chatbot status."""
    return {
        "status": "running",
        "component": "AI Chatbot",
        "features": [
            "Natural language queries about ML models",
            "Platform status and metrics",
            "Distributed systems simulation control",
            "Multi-turn conversation support",
            "Vector-based context search"
        ]
    }


@chatbot_router.post("/chat")
async def chat(
    message: ChatMessage,
    manager: ChatbotManager = Depends(get_chatbot_manager)
) -> ChatResponse:
    """Send a message to the chatbot."""
    try:
        response = await manager.process_message(message)
        return response
    except Exception as e:
        logger.error(f"Error processing chat message: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@chatbot_router.get("/sessions")
async def list_sessions(
    manager: ChatbotManager = Depends(get_chatbot_manager)
) -> List[ChatSession]:
    """List all chat sessions."""
    try:
        return await manager.list_sessions()
    except Exception as e:
        logger.error(f"Error listing sessions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@chatbot_router.get("/sessions/{session_id}")
async def get_session(
    session_id: str,
    manager: ChatbotManager = Depends(get_chatbot_manager)
) -> ChatSession:
    """Get a specific chat session."""
    try:
        session = await manager.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        return session
    except Exception as e:
        logger.error(f"Error getting session {session_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@chatbot_router.post("/sessions")
async def create_session(
    session_data: Dict[str, Any],
    manager: ChatbotManager = Depends(get_chatbot_manager)
) -> ChatSession:
    """Create a new chat session."""
    try:
        session = await manager.create_session(session_data)
        return session
    except Exception as e:
        logger.error(f"Error creating session: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@chatbot_router.delete("/sessions/{session_id}")
async def delete_session(
    session_id: str,
    manager: ChatbotManager = Depends(get_chatbot_manager)
) -> Dict[str, str]:
    """Delete a chat session."""
    try:
        await manager.delete_session(session_id)
        return {"status": "deleted", "session_id": session_id}
    except Exception as e:
        logger.error(f"Error deleting session {session_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@chatbot_router.get("/sessions/{session_id}/messages")
async def get_session_messages(
    session_id: str,
    manager: ChatbotManager = Depends(get_chatbot_manager)
) -> List[ChatMessage]:
    """Get messages from a session."""
    try:
        messages = await manager.get_session_messages(session_id)
        return messages
    except Exception as e:
        logger.error(f"Error getting messages for session {session_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@chatbot_router.post("/sessions/{session_id}/clear")
async def clear_session(
    session_id: str,
    manager: ChatbotManager = Depends(get_chatbot_manager)
) -> Dict[str, str]:
    """Clear messages from a session."""
    try:
        await manager.clear_session(session_id)
        return {"status": "cleared", "session_id": session_id}
    except Exception as e:
        logger.error(f"Error clearing session {session_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@chatbot_router.get("/capabilities")
async def get_capabilities(
    manager: ChatbotManager = Depends(get_chatbot_manager)
) -> Dict[str, Any]:
    """Get chatbot capabilities."""
    try:
        return await manager.get_capabilities()
    except Exception as e:
        logger.error(f"Error getting capabilities: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@chatbot_router.post("/train")
async def train_chatbot(
    training_data: Dict[str, Any],
    manager: ChatbotManager = Depends(get_chatbot_manager)
) -> Dict[str, str]:
    """Train/update the chatbot with new data."""
    try:
        job_id = await manager.train(training_data)
        return {"status": "training_started", "job_id": job_id}
    except Exception as e:
        logger.error(f"Error training chatbot: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@chatbot_router.get("/knowledge")
async def get_knowledge_base(
    manager: ChatbotManager = Depends(get_chatbot_manager)
) -> Dict[str, Any]:
    """Get knowledge base information."""
    try:
        return await manager.get_knowledge_base_info()
    except Exception as e:
        logger.error(f"Error getting knowledge base: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@chatbot_router.post("/knowledge/update")
async def update_knowledge_base(
    knowledge_data: Dict[str, Any],
    manager: ChatbotManager = Depends(get_chatbot_manager)
) -> Dict[str, str]:
    """Update knowledge base with new information."""
    try:
        await manager.update_knowledge_base(knowledge_data)
        return {"status": "updated"}
    except Exception as e:
        logger.error(f"Error updating knowledge base: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@chatbot_router.post("/analyze")
async def analyze_conversation(
    session_id: str,
    manager: ChatbotManager = Depends(get_chatbot_manager)
) -> Dict[str, Any]:
    """Analyze a conversation for insights."""
    try:
        analysis = await manager.analyze_conversation(session_id)
        return analysis
    except Exception as e:
        logger.error(f"Error analyzing conversation {session_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@chatbot_router.get("/metrics")
async def get_chatbot_metrics(
    manager: ChatbotManager = Depends(get_chatbot_manager)
) -> Dict[str, Any]:
    """Get chatbot performance metrics."""
    try:
        return await manager.get_metrics()
    except Exception as e:
        logger.error(f"Error getting chatbot metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@chatbot_router.post("/feedback")
async def submit_feedback(
    feedback_data: Dict[str, Any],
    manager: ChatbotManager = Depends(get_chatbot_manager)
) -> Dict[str, str]:
    """Submit feedback about chatbot responses."""
    try:
        await manager.submit_feedback(feedback_data)
        return {"status": "feedback_received"}
    except Exception as e:
        logger.error(f"Error submitting feedback: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Quick command endpoints
@chatbot_router.post("/commands/ml/status")
async def get_ml_status_command(
    manager: ChatbotManager = Depends(get_chatbot_manager)
) -> Dict[str, Any]:
    """Get ML pipeline status via command."""
    try:
        return await manager.execute_command("ml_status")
    except Exception as e:
        logger.error(f"Error executing ML status command: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@chatbot_router.post("/commands/distributed/status")
async def get_distributed_status_command(
    manager: ChatbotManager = Depends(get_chatbot_manager)
) -> Dict[str, Any]:
    """Get distributed simulation status via command."""
    try:
        return await manager.execute_command("distributed_status")
    except Exception as e:
        logger.error(f"Error executing distributed status command: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@chatbot_router.post("/commands/platform/health")
async def get_platform_health_command(
    manager: ChatbotManager = Depends(get_chatbot_manager)
) -> Dict[str, Any]:
    """Get platform health via command."""
    try:
        return await manager.execute_command("platform_health")
    except Exception as e:
        logger.error(f"Error executing platform health command: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@chatbot_router.post("/chat")
def chat_endpoint(req: ChatRequest, dep=Depends(api_key_auth)):
    chat_counter.inc()
    # TODO: Replace with real embedding (OpenAI/HF) and LLM call
    user_vector = np.random.rand(dim).astype('float32')
    index.add(np.expand_dims(user_vector, 0))
    vector_store.append((req.message, user_vector))
    # TODO: Use OpenAI/HF to generate response
    response = f"Echo: {req.message} (context size: {len(vector_store)})"
    return {"response": response}


@chatbot_router.get("/model-info")
def model_info(dep=Depends(api_key_auth)):
    # Call ML pipeline API for model info
    try:
        r = requests.get("http://ml-pipeline:8000/latest-accuracy")
        return r.json()
    except Exception as e:
        return {"error": str(e)}


@chatbot_router.get("/cluster-status")
def cluster_status(dep=Depends(api_key_auth)):
    # Call Raft API for cluster state
    try:
        r = requests.get("http://raft-simulator:8000/api/raft/cluster-state")
        return r.json()
    except Exception as e:
        return {"error": str(e)}
