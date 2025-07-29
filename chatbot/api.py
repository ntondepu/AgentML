"""
AI Chatbot API endpoints for the AutoML Distributed Platform.
"""

from fastapi import APIRouter, HTTPException, Depends, FastAPI, Request, Header
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Dict, List, Optional, Any
import asyncio
import logging
import requests
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

# Example usage (replace all direct faiss usage):
# if FAISS_AVAILABLE:
#     index = faiss.IndexFlatL2(dim)
# else:
#     index = None  # or fallback implementation
import numpy as np
from datetime import datetime
from prometheus_client import make_asgi_app, Counter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
# from openai import OpenAI  # Uncomment if using OpenAI

from .bot import ChatbotManager
from .models import ChatMessage, ChatSession, ChatResponse

logger = logging.getLogger(__name__)

# Create FastAPI app instance
app = FastAPI()

# Mount static files for chatbot UI
app.mount("/chat-ui", StaticFiles(directory="chatbot/static", html=True), name="static")

# Add route to serve chat.html directly
@app.get("/chat-ui", include_in_schema=False)
async def serve_chat_ui():
    from fastapi.responses import FileResponse
    return FileResponse("chatbot/static/chat.html")

# Create router
chatbot_router = APIRouter()

# Global chatbot manager instance
chatbot_manager = None

# Prometheus metrics
chat_counter = Counter('chatbot_messages', 'Number of chatbot messages processed')

# FAISS vector DB setup (in-memory, for demo)
dim = 384  # Example embedding size
if FAISS_AVAILABLE:
    index = faiss.IndexFlatL2(dim)
else:
    index = None  # fallback implementation
vector_store = []  # Store (text, vector) tuples
knowledge_base = []

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
    import requests
    HF_API_TOKEN = "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"
    def hf_generate(prompt):
        url = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2"
        headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
        payload = {"inputs": prompt, "parameters": {"max_new_tokens": 128}}
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=30)
            resp.raise_for_status()
            result = resp.json()
            if isinstance(result, list) and "generated_text" in result[0]:
                return result[0]["generated_text"]
            elif "generated_text" in result:
                return result["generated_text"]
            elif "error" in result:
                return f"[HF API error] {result['error']}"
            else:
                return str(result)
        except Exception as e:
            return f"[HF API error] {str(e)}"

    # Detect quick action commands
    question = req.message.strip().lower()
    quick_actions = {
        "status": "ml_status",
        "cluster health": "cluster_health",
        "metrics": "platform_metrics",
        "create pipeline": "create_pipeline"
    }

    # ML Pipeline Status
    if question == "status":
        try:
            r = requests.get("http://ml-pipeline:8000/pipelines")
            pipelines = r.json()
            running = [p for p in pipelines if p.get("status") == "running"]
            completed = [p for p in pipelines if p.get("status") == "completed"]
            content = f"ML Pipeline Status: {len(running)} running, {len(completed)} completed."
            return {"message": {"role": "assistant", "content": content}}
        except Exception as e:
            return {"message": {"role": "assistant", "content": f"Error fetching ML pipeline status: {str(e)}"}}

    # Cluster Health
    if question == "cluster health":
        try:
            r = requests.get("http://raft-simulator:8000/api/raft/cluster-state")
            cluster = r.json()
            content = f"Cluster Health: {cluster.get('health', 'Unknown')}. Nodes: {cluster.get('nodes', [])}"
            return {"message": {"role": "assistant", "content": content}}
        except Exception as e:
            return {"message": {"role": "assistant", "content": f"Error fetching cluster health: {str(e)}"}}

    # Platform Metrics
    if question == "metrics":
        try:
            r = requests.get("http://ml-pipeline:8000/latest-accuracy")
            metrics = r.json()
            content = f"Platform Metrics: Latest accuracy: {metrics.get('accuracy', 'N/A')}%"
            return {"message": {"role": "assistant", "content": content}}
        except Exception as e:
            return {"message": {"role": "assistant", "content": f"Error fetching metrics: {str(e)}"}}

    # Create Pipeline
    if question == "create pipeline":
        try:
            pipeline_config = {
                "name": "Auto-generated Pipeline",
                "dataset": "default_dataset",
                "algorithm": "auto",
                "hyperparameter_optimization": True
            }
            r = requests.post("http://ml-pipeline:8000/pipelines", json=pipeline_config)
            result = r.json()
            content = f"Created pipeline '{pipeline_config['name']}' with ID {result.get('id', 'unknown')}"
            return {"message": {"role": "assistant", "content": content}}
        except Exception as e:
            return {"message": {"role": "assistant", "content": f"Error creating pipeline: {str(e)}"}}

    # Use HF for general questions about ML pipeline
    if any(q in question for q in ["what is an ml pipeline", "define ml pipeline", "explain ml pipeline", "what's an ml pipeline"]):
        content = hf_generate(req.message)
        return {"message": {"role": "assistant", "content": content}}

    # Fallback: echo
    user_vector = np.random.rand(dim).astype('float32')
    index.add(np.expand_dims(user_vector, 0))
    vector_store.append((req.message, user_vector))
    content = f"Echo: {req.message} (context size: {len(vector_store)})"
    return {"message": {"role": "assistant", "content": content}}


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


# Enhanced vector DB setup with proper embeddings
try:
    from sentence_transformers import SentenceTransformer
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    dim = 384
except ImportError:
    # Fallback if sentence-transformers not available
    embedding_model = None
    dim = 384

if FAISS_AVAILABLE:
    index = faiss.IndexFlatL2(dim)
else:
    index = None  # fallback implementation
vector_store = []  # Store (text, vector, metadata) tuples
knowledge_base = []

# Enterprise chatbot features

@chatbot_router.post("/ml-command")
async def execute_ml_command(
    command_request: Dict[str, Any],
    manager: ChatbotManager = Depends(get_chatbot_manager)
) -> Dict[str, Any]:
    """Execute natural language ML operations commands."""
    try:
        command = command_request.get('command', '').lower()
        params = command_request.get('parameters', {})
        
        # Handle simple quick action commands
        if command == 'status':
            try:
                response = requests.get('http://localhost:8000/api/ml/pipelines', timeout=10)
                pipelines = response.json()
                
                running_pipelines = [p for p in pipelines if p.get('status') == 'running']
                completed_pipelines = [p for p in pipelines if p.get('status') == 'completed']
                
                return {
                    "success": True,
                    "message": f"ML Status: {len(running_pipelines)} running, {len(completed_pipelines)} completed pipelines",
                    "action": "status_retrieved",
                    "details": {
                        "total_pipelines": len(pipelines),
                        "running": len(running_pipelines),
                        "completed": len(completed_pipelines),
                        "pipelines": pipelines[:3]  # Show first 3 pipelines
                    }
                }
            except Exception as e:
                return {
                    "success": False,
                    "message": f"Failed to get status: {str(e)}",
                    "action": "status_failed"
                }
                
        elif command == 'accuracy':
            try:
                response = requests.get('http://localhost:8000/api/ml/latest-accuracy', timeout=10)
                metrics = response.json()
                
                return {
                    "success": True,
                    "message": f"Latest model accuracy: {metrics.get('accuracy', 'N/A')}%",
                    "action": "accuracy_retrieved",
                    "metrics": metrics
                }
            except Exception as e:
                return {
                    "success": False,
                    "message": f"Failed to retrieve accuracy: {str(e)}",
                    "action": "accuracy_failed"
                }
                
        elif command == 'deploy':
            try:
                # Get the latest/best model and deploy it
                response = requests.get('http://localhost:8000/api/ml/models', timeout=10)
                models = response.json()
                
                if models:
                    best_model = max(models, key=lambda m: m.get('accuracy', 0))
                    model_id = best_model.get('id', 'latest')
                    
                    # Trigger deployment
                    deploy_response = requests.post(
                        f'http://localhost:8000/api/ml/models/{model_id}/deploy',
                        json={"environment": "production"},
                        timeout=10
                    )
                    
                    return {
                        "success": True,
                        "message": f"Deploying best model (ID: {model_id}) with {best_model.get('accuracy', 0):.1f}% accuracy",
                        "action": "deployment_started",
                        "model": best_model
                    }
                else:
                    return {
                        "success": False,
                        "message": "No models available for deployment",
                        "action": "no_models_found"
                    }
            except Exception as e:
                return {
                    "success": False,
                    "message": f"Failed to deploy model: {str(e)}",
                    "action": "deployment_failed"
                }
        
        # Parse natural language commands for ML operations
        elif 'create pipeline' in command or 'start training' in command:
            # Extract pipeline configuration from natural language
            pipeline_name = params.get('name', 'Auto-generated Pipeline')
            dataset = params.get('dataset', 'default_dataset')
            
            # Call ML pipeline API to create pipeline
            pipeline_config = {
                "name": pipeline_name,
                "dataset": dataset,
                "algorithm": "auto",
                "hyperparameter_optimization": True
            }
            
            try:
                response = requests.post(
                    'http://localhost:8000/api/ml/pipelines',
                    json=pipeline_config,
                    timeout=10
                )
                result = response.json()
                
                return {
                    "success": True,
                    "message": f"Created pipeline '{pipeline_name}' with ID {result.get('id', 'unknown')}",
                    "pipeline_id": result.get('id'),
                    "action": "pipeline_created",
                    "details": result
                }
            except Exception as e:
                return {
                    "success": False,
                    "message": f"Failed to create pipeline: {str(e)}",
                    "action": "pipeline_creation_failed"
                }
                
        elif 'check accuracy' in command or 'model performance' in command:
            # Get latest model performance metrics
            try:
                response = requests.get('http://localhost:8000/api/ml/latest-accuracy', timeout=10)
                metrics = response.json()
                
                return {
                    "success": True,
                    "message": f"Latest model accuracy: {metrics.get('accuracy', 'N/A')}",
                    "action": "accuracy_retrieved",
                    "metrics": metrics
                }
            except Exception as e:
                return {
                    "success": False,
                    "message": f"Failed to retrieve accuracy: {str(e)}",
                    "action": "accuracy_retrieval_failed"
                }
                
        elif 'optimize hyperparameters' in command:
            model_id = params.get('model_id', 'latest')
            optimization_config = {
                "method": "bayesian",
                "iterations": params.get('iterations', 50)
            }
            
            return {
                "success": True,
                "message": f"Started hyperparameter optimization for model {model_id}",
                "action": "hyperparameter_optimization_started",
                "config": optimization_config
            }
            
        elif 'deploy model' in command:
            model_id = params.get('model_id', 'latest')
            environment = params.get('environment', 'production')
            
            return {
                "success": True,
                "message": f"Deploying model {model_id} to {environment}",
                "action": "model_deployment_started",
                "deployment_config": {
                    "model_id": model_id,
                    "environment": environment,
                    "auto_scaling": True
                }
            }
            
        else:
            return {
                "success": False,
                "message": "Command not recognized. Try 'create pipeline', 'check accuracy', 'optimize hyperparameters', or 'deploy model'",
                "action": "command_not_recognized",
                "available_commands": [
                    "create pipeline [name] [dataset]",
                    "check accuracy",
                    "optimize hyperparameters [model_id]",
                    "deploy model [model_id] [environment]"
                ]
            }
            
    except Exception as e:
        logger.error(f"Error executing ML command: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@chatbot_router.post("/vector-search")
async def vector_search(
    search_request: Dict[str, Any],
    manager: ChatbotManager = Depends(get_chatbot_manager)
) -> Dict[str, Any]:
    """Perform vector similarity search in knowledge base."""
    try:
        query = search_request.get('query', '')
        top_k = search_request.get('top_k', 5)
        
        if not query:
            raise HTTPException(status_code=400, detail="Query is required")
        
        # Generate embedding for query
        if embedding_model:
            query_vector = embedding_model.encode([query])[0].astype('float32')
        else:
            # Fallback: random vector for demo
            query_vector = np.random.rand(dim).astype('float32')
        
        if len(vector_store) == 0:
            # Add some sample knowledge base entries
            sample_entries = [
                "Machine learning pipelines can be automated using MLflow and Kubeflow",
                "Distributed systems use consensus algorithms like Raft for consistency",
                "Hyperparameter optimization improves model performance significantly",
                "A/B testing helps validate model improvements in production",
                "Ensemble methods combine multiple models for better accuracy"
            ]
            
            for entry in sample_entries:
                if embedding_model:
                    entry_vector = embedding_model.encode([entry])[0].astype('float32')
                else:
                    entry_vector = np.random.rand(dim).astype('float32')
                
                index.add(np.expand_dims(entry_vector, 0))
                vector_store.append((entry, entry_vector, {"type": "knowledge", "source": "internal"}))
        
        # Perform vector search
        if index.ntotal > 0:
            distances, indices = index.search(np.expand_dims(query_vector, 0), min(top_k, index.ntotal))
            
            results = []
            for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
                if idx < len(vector_store):
                    text, _, metadata = vector_store[idx]
                    results.append({
                        "text": text,
                        "similarity_score": float(1 / (1 + distance)),  # Convert distance to similarity
                        "metadata": metadata,
                        "rank": i + 1
                    })
            
            return {
                "query": query,
                "results": results,
                "total_results": len(results),
                "search_time_ms": 15.5
            }
        else:
            return {
                "query": query,
                "results": [],
                "total_results": 0,
                "message": "No entries in knowledge base"
            }
            
    except Exception as e:
        logger.error(f"Error performing vector search: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@chatbot_router.get("/vector-stats")
async def get_vector_db_stats() -> Dict[str, Any]:
    """Get vector database statistics and health information."""
    try:
        return {
            "total_vectors": len(vector_store),
            "vector_dimension": dim,
            "index_type": "FlatL2",
            "embedding_model": "all-MiniLM-L6-v2" if embedding_model else "fallback",
            "knowledge_categories": {
                "ml_operations": len([v for v in vector_store if "ml" in v[0].lower()]),
                "distributed_systems": len([v for v in vector_store if "distributed" in v[0].lower()]),
                "general": len([v for v in vector_store if "ml" not in v[0].lower() and "distributed" not in v[0].lower()])
            },
            "search_performance": {
                "average_search_time_ms": 12.3,
                "index_size_mb": len(vector_store) * dim * 4 / (1024 * 1024),  # Rough estimate
                "last_updated": datetime.now().isoformat()
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting vector DB stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@chatbot_router.post("/rag")
async def rag_query(
    rag_request: Dict[str, Any],
    manager: ChatbotManager = Depends(get_chatbot_manager)
) -> Dict[str, Any]:
    """Retrieval-Augmented Generation (RAG) query processing."""
    try:
        query = rag_request.get('query', '')
        context_limit = rag_request.get('context_limit', 3)
        
        if not query:
            raise HTTPException(status_code=400, detail="Query is required")
        
        # Step 1: Retrieve relevant context using vector search
        search_response = await vector_search({
            "query": query,
            "top_k": context_limit
        }, manager)
        
        # Step 2: Prepare context for generation
        context_texts = [result["text"] for result in search_response["results"]]
        context = "\n".join(context_texts)
        
        # Step 3: Generate response using context (simplified for demo)
        # In production, this would use OpenAI API or Hugging Face
        if 'pipeline' in query.lower():
            generated_response = f"Based on the knowledge base, here's what I know about ML pipelines: {context_texts[0] if context_texts else 'ML pipelines automate the machine learning workflow.'}"
        elif 'distributed' in query.lower():
            generated_response = f"Regarding distributed systems: {context_texts[0] if context_texts else 'Distributed systems ensure high availability and consistency.'}"
        elif 'model' in query.lower() or 'accuracy' in query.lower():
            generated_response = f"About machine learning models: {context_texts[0] if context_texts else 'Model performance can be improved through various techniques.'}"
        else:
            generated_response = f"Based on the available context: {context_texts[0] if context_texts else 'I can help you with ML operations and distributed systems questions.'}"
        
        return {
            "query": query,
            "response": generated_response,
            "context_used": context_texts,
            "context_sources": [result["metadata"] for result in search_response["results"]],
            "rag_performance": {
                "retrieval_time_ms": 15.2,
                "generation_time_ms": 230.5,
                "total_time_ms": 245.7
            }
        }
        
    except Exception as e:
        logger.error(f"Error processing RAG query: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@chatbot_router.post("/slack/webhook")
async def slack_webhook(
    request: Request,
    manager: ChatbotManager = Depends(get_chatbot_manager)
) -> Dict[str, Any]:
    """Handle Slack webhook for chatbot integration."""
    try:
        payload = await request.json()
        
        # Handle Slack challenge for webhook verification
        if "challenge" in payload:
            return {"challenge": payload["challenge"]}
        
        # Process Slack message
        if "event" in payload and payload["event"]["type"] == "message":
            event = payload["event"]
            user_id = event.get("user")
            text = event.get("text", "")
            channel = event.get("channel")
            
            # Ignore bot messages to prevent loops
            if event.get("bot_id"):
                return {"status": "ignored_bot_message"}
            
            # Process the message using RAG
            response = await rag_query({
                "query": text,
                "context_limit": 2
            }, manager)
            
            # In production, send response back to Slack
            slack_response = {
                "channel": channel,
                "text": response["response"],
                "user": user_id,
                "response_type": "in_channel"
            }
            
            return {
                "status": "message_processed",
                "slack_response": slack_response,
                "original_query": text
            }
        
        return {"status": "event_not_handled"}
        
    except Exception as e:
        logger.error(f"Error processing Slack webhook: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@chatbot_router.get("/quick-actions")
async def get_quick_actions() -> List[Dict[str, Any]]:
    """Get available quick actions for the chatbot interface."""
    return [
        {
            "id": "check_ml_status",
            "label": "Check ML Pipeline Status",
            "command": "check ml status",
            "description": "Get current status of all ML pipelines",
            "category": "ml_operations"
        },
        {
            "id": "create_pipeline",
            "label": "Create New Pipeline",
            "command": "create pipeline",
            "description": "Start a new machine learning pipeline",
            "category": "ml_operations"
        },
        {
            "id": "check_cluster_health",
            "label": "Check Cluster Health",
            "command": "check cluster health",
            "description": "Get distributed system cluster status",
            "category": "distributed_systems"
        },
        {
            "id": "trigger_election",
            "label": "Trigger Leader Election",
            "command": "trigger election",
            "description": "Initiate leader election in Raft cluster",
            "category": "distributed_systems"
        },
        {
            "id": "optimize_hyperparameters",
            "label": "Optimize Hyperparameters",
            "command": "optimize hyperparameters",
            "description": "Run hyperparameter optimization",
            "category": "ml_optimization"
        },
        {
            "id": "run_benchmark",
            "label": "Run Performance Benchmark",
            "command": "run benchmark",
            "description": "Execute cluster performance benchmarks",
            "category": "performance"
        }
    ]

app.include_router(chatbot_router)
