"""
Pydantic models for the AI Chatbot component.
"""

from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from enum import Enum


class MessageType(str, Enum):
    """Types of chat messages."""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    COMMAND = "command"
    ERROR = "error"


class MessageRole(str, Enum):
    """Message roles for conversation."""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class ChatMessage(BaseModel):
    """Individual chat message model."""
    id: str = Field(..., description="Message identifier")
    session_id: str = Field(..., description="Session identifier")
    role: MessageRole = Field(..., description="Message role")
    content: str = Field(..., description="Message content")
    timestamp: datetime = Field(default_factory=datetime.now, description="Message timestamp")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    class Config:
        schema_extra = {
            "example": {
                "id": "msg_123",
                "session_id": "session_456",
                "role": "user",
                "content": "What is the status of my ML pipeline?",
                "timestamp": "2025-07-03T12:00:00Z",
                "metadata": {"intent": "status_query"}
            }
        }


class ChatResponse(BaseModel):
    """Response from the chatbot."""
    message: ChatMessage = Field(..., description="Response message")
    confidence: float = Field(..., description="Response confidence score")
    intent: Optional[str] = Field(default=None, description="Detected intent")
    entities: List[Dict[str, Any]] = Field(default_factory=list, description="Extracted entities")
    suggestions: List[str] = Field(default_factory=list, description="Suggested follow-up questions")
    actions: List[Dict[str, Any]] = Field(default_factory=list, description="Suggested actions")
    context: Dict[str, Any] = Field(default_factory=dict, description="Conversation context")
    
    class Config:
        schema_extra = {
            "example": {
                "message": {
                    "id": "msg_124",
                    "session_id": "session_456",
                    "role": "assistant",
                    "content": "Your ML pipeline 'Customer Churn Prediction' is currently running. It's at 75% progress and should complete in about 5 minutes.",
                    "timestamp": "2025-07-03T12:00:01Z",
                    "metadata": {"response_type": "status_info"}
                },
                "confidence": 0.92,
                "intent": "ml_pipeline_status",
                "entities": [{"type": "pipeline_name", "value": "Customer Churn Prediction"}],
                "suggestions": ["Show me the training metrics", "Stop the pipeline", "View pipeline logs"],
                "actions": [{"type": "view_pipeline", "pipeline_id": "job_123456"}],
                "context": {"current_pipeline": "job_123456"}
            }
        }


class ChatSession(BaseModel):
    """Chat session model."""
    id: str = Field(..., description="Session identifier")
    user_id: Optional[str] = Field(default=None, description="User identifier")
    title: str = Field(..., description="Session title")
    created_at: datetime = Field(default_factory=datetime.now, description="Session creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.now, description="Last update timestamp")
    message_count: int = Field(default=0, description="Number of messages in session")
    context: Dict[str, Any] = Field(default_factory=dict, description="Session context")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Session metadata")
    active: bool = Field(default=True, description="Whether session is active")
    
    class Config:
        schema_extra = {
            "example": {
                "id": "session_456",
                "user_id": "user_789",
                "title": "ML Pipeline Management",
                "created_at": "2025-07-03T11:00:00Z",
                "updated_at": "2025-07-03T12:00:00Z",
                "message_count": 10,
                "context": {"current_pipeline": "job_123456", "user_preferences": {}},
                "metadata": {"ip_address": "192.168.1.100"},
                "active": True
            }
        }


class ConversationIntent(str, Enum):
    """Conversation intents."""
    GREETING = "greeting"
    GOODBYE = "goodbye"
    HELP = "help"
    
    # ML Pipeline intents
    ML_STATUS = "ml_pipeline_status"
    ML_CREATE = "ml_pipeline_create"
    ML_STOP = "ml_pipeline_stop"
    ML_METRICS = "ml_pipeline_metrics"
    ML_LOGS = "ml_pipeline_logs"
    ML_DEPLOY = "ml_model_deploy"
    ML_PREDICT = "ml_model_predict"
    
    # Distributed System intents
    DISTRIBUTED_STATUS = "distributed_status"
    DISTRIBUTED_NODES = "distributed_nodes"
    DISTRIBUTED_ELECTION = "distributed_election"
    DISTRIBUTED_PARTITION = "distributed_partition"
    DISTRIBUTED_SCENARIO = "distributed_scenario"
    
    # Platform intents
    PLATFORM_HEALTH = "platform_health"
    PLATFORM_METRICS = "platform_metrics"
    PLATFORM_LOGS = "platform_logs"
    
    # General intents
    UNKNOWN = "unknown"
    ERROR = "error"


class Entity(BaseModel):
    """Named entity extracted from text."""
    type: str = Field(..., description="Entity type")
    value: str = Field(..., description="Entity value")
    confidence: float = Field(..., description="Extraction confidence")
    start: int = Field(..., description="Start position in text")
    end: int = Field(..., description="End position in text")
    
    class Config:
        schema_extra = {
            "example": {
                "type": "pipeline_name",
                "value": "Customer Churn Prediction",
                "confidence": 0.95,
                "start": 20,
                "end": 45
            }
        }


class KnowledgeBaseEntry(BaseModel):
    """Knowledge base entry."""
    id: str = Field(..., description="Entry identifier")
    title: str = Field(..., description="Entry title")
    content: str = Field(..., description="Entry content")
    category: str = Field(..., description="Entry category")
    tags: List[str] = Field(default_factory=list, description="Entry tags")
    embedding: Optional[List[float]] = Field(default=None, description="Text embedding")
    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.now, description="Update timestamp")
    
    class Config:
        schema_extra = {
            "example": {
                "id": "kb_001",
                "title": "How to create an ML pipeline",
                "content": "To create an ML pipeline, you can use the /api/ml/pipelines endpoint...",
                "category": "ml_pipeline",
                "tags": ["tutorial", "api", "machine_learning"],
                "created_at": "2025-07-03T10:00:00Z",
                "updated_at": "2025-07-03T10:00:00Z"
            }
        }


class ChatbotMetrics(BaseModel):
    """Chatbot performance metrics."""
    total_sessions: int = Field(default=0, description="Total number of sessions")
    active_sessions: int = Field(default=0, description="Number of active sessions")
    total_messages: int = Field(default=0, description="Total number of messages")
    average_response_time: float = Field(default=0.0, description="Average response time in seconds")
    intent_recognition_accuracy: float = Field(default=0.0, description="Intent recognition accuracy")
    user_satisfaction: float = Field(default=0.0, description="User satisfaction score")
    knowledge_base_size: int = Field(default=0, description="Number of knowledge base entries")
    successful_queries: int = Field(default=0, description="Number of successful queries")
    failed_queries: int = Field(default=0, description="Number of failed queries")
    
    class Config:
        schema_extra = {
            "example": {
                "total_sessions": 150,
                "active_sessions": 12,
                "total_messages": 1500,
                "average_response_time": 0.8,
                "intent_recognition_accuracy": 0.89,
                "user_satisfaction": 4.2,
                "knowledge_base_size": 250,
                "successful_queries": 1350,
                "failed_queries": 150
            }
        }


class CommandResult(BaseModel):
    """Result of a chatbot command execution."""
    command: str = Field(..., description="Command that was executed")
    success: bool = Field(..., description="Whether command was successful")
    result: Dict[str, Any] = Field(..., description="Command result data")
    message: str = Field(..., description="Human-readable message")
    timestamp: datetime = Field(default_factory=datetime.now, description="Execution timestamp")
    
    class Config:
        schema_extra = {
            "example": {
                "command": "ml_pipeline_status",
                "success": True,
                "result": {"pipeline_id": "job_123456", "status": "running", "progress": 75},
                "message": "Pipeline is running at 75% progress",
                "timestamp": "2025-07-03T12:00:00Z"
            }
        }


class FeedbackData(BaseModel):
    """User feedback about chatbot responses."""
    session_id: str = Field(..., description="Session identifier")
    message_id: str = Field(..., description="Message identifier")
    rating: int = Field(..., description="Rating (1-5)")
    comment: Optional[str] = Field(default=None, description="Optional comment")
    helpful: bool = Field(..., description="Whether response was helpful")
    timestamp: datetime = Field(default_factory=datetime.now, description="Feedback timestamp")
    
    class Config:
        schema_extra = {
            "example": {
                "session_id": "session_456",
                "message_id": "msg_124",
                "rating": 4,
                "comment": "Good response but could be more detailed",
                "helpful": True,
                "timestamp": "2025-07-03T12:05:00Z"
            }
        }


class ConversationAnalysis(BaseModel):
    """Analysis of a conversation."""
    session_id: str = Field(..., description="Session identifier")
    sentiment: str = Field(..., description="Overall sentiment")
    topics: List[str] = Field(..., description="Main topics discussed")
    intents: List[str] = Field(..., description="Intents detected")
    entities: List[Entity] = Field(..., description="Entities extracted")
    satisfaction_score: float = Field(..., description="Estimated satisfaction score")
    key_insights: List[str] = Field(..., description="Key insights from conversation")
    recommendations: List[str] = Field(..., description="Recommendations for improvement")
    
    class Config:
        schema_extra = {
            "example": {
                "session_id": "session_456",
                "sentiment": "positive",
                "topics": ["ml_pipeline", "model_deployment", "monitoring"],
                "intents": ["ml_pipeline_status", "ml_model_deploy"],
                "entities": [],
                "satisfaction_score": 4.1,
                "key_insights": ["User is interested in model deployment", "Needs help with monitoring"],
                "recommendations": ["Provide deployment tutorial", "Add monitoring guides"]
            }
        }


class ChatbotCapabilities(BaseModel):
    """Chatbot capabilities and features."""
    supported_intents: List[str] = Field(..., description="Supported conversation intents")
    supported_entities: List[str] = Field(..., description="Supported entity types")
    available_commands: List[str] = Field(..., description="Available commands")
    integration_points: List[str] = Field(..., description="Integration points with platform")
    languages: List[str] = Field(default=["en"], description="Supported languages")
    features: List[str] = Field(..., description="Available features")
    
    class Config:
        schema_extra = {
            "example": {
                "supported_intents": ["ml_pipeline_status", "distributed_status", "platform_health"],
                "supported_entities": ["pipeline_name", "model_id", "node_id"],
                "available_commands": ["status", "help", "logs", "metrics"],
                "integration_points": ["ml_pipeline", "distributed_sim", "monitoring"],
                "languages": ["en"],
                "features": ["intent_recognition", "entity_extraction", "context_awareness"]
            }
        }
