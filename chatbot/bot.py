"""
AI Chatbot Manager for the AutoML Distributed Platform.
"""

import asyncio
import logging
import uuid
import json
import re
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
import os

# Vector database and embeddings
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from .models import (
    ChatMessage, 
    ChatSession, 
    ChatResponse, 
    ConversationIntent,
    Entity,
    KnowledgeBaseEntry,
    ChatbotMetrics,
    CommandResult,
    FeedbackData,
    ConversationAnalysis,
    ChatbotCapabilities,
    MessageRole
)
from ..monitoring.telemetry import metrics
from ..config import settings

logger = logging.getLogger(__name__)


class IntentClassifier:
    """Simple intent classification based on keywords and patterns."""
    
    def __init__(self):
        self.intent_patterns = {
            ConversationIntent.GREETING: [
                r'\b(hello|hi|hey|good morning|good afternoon|good evening)\b',
                r'\b(start|begin|initialize)\b'
            ],
            ConversationIntent.GOODBYE: [
                r'\b(bye|goodbye|see you|farewell|exit|quit)\b',
                r'\b(end|finish|stop|close)\b'
            ],
            ConversationIntent.HELP: [
                r'\b(help|assist|support|guide)\b',
                r'\b(how to|what is|what are|explain)\b'
            ],
            ConversationIntent.ML_STATUS: [
                r'\b(ml|machine learning|model|pipeline)\b.*\b(status|state|progress)\b',
                r'\b(training|running|completed|failed)\b.*\b(pipeline|model)\b'
            ],
            ConversationIntent.ML_CREATE: [
                r'\b(create|start|train|build)\b.*\b(model|pipeline)\b',
                r'\b(new|fresh)\b.*\b(training|pipeline)\b'
            ],
            ConversationIntent.ML_STOP: [
                r'\b(stop|halt|cancel|abort)\b.*\b(training|pipeline|model)\b'
            ],
            ConversationIntent.ML_METRICS: [
                r'\b(metrics|performance|accuracy|precision|recall)\b',
                r'\b(results|evaluation|validation)\b'
            ],
            ConversationIntent.ML_LOGS: [
                r'\b(logs|log|history|output)\b.*\b(training|pipeline)\b'
            ],
            ConversationIntent.ML_DEPLOY: [
                r'\b(deploy|deployment|publish|release)\b.*\b(model)\b'
            ],
            ConversationIntent.ML_PREDICT: [
                r'\b(predict|prediction|inference|classify)\b'
            ],
            ConversationIntent.DISTRIBUTED_STATUS: [
                r'\b(distributed|cluster|nodes|raft)\b.*\b(status|state)\b',
                r'\b(consensus|leader|follower)\b'
            ],
            ConversationIntent.DISTRIBUTED_NODES: [
                r'\b(nodes|node|cluster)\b.*\b(list|show|view)\b'
            ],
            ConversationIntent.DISTRIBUTED_ELECTION: [
                r'\b(election|leader|vote|candidate)\b'
            ],
            ConversationIntent.DISTRIBUTED_PARTITION: [
                r'\b(partition|network|split)\b.*\b(brain|isolation)\b'
            ],
            ConversationIntent.DISTRIBUTED_SCENARIO: [
                r'\b(scenario|simulation|test)\b.*\b(run|execute)\b'
            ],
            ConversationIntent.PLATFORM_HEALTH: [
                r'\b(platform|system|health|status)\b.*\b(check|overview)\b'
            ],
            ConversationIntent.PLATFORM_METRICS: [
                r'\b(metrics|statistics|performance)\b.*\b(platform|system)\b'
            ],
            ConversationIntent.PLATFORM_LOGS: [
                r'\b(logs|log|events)\b.*\b(platform|system)\b'
            ]
        }
    
    def classify_intent(self, text: str) -> Tuple[ConversationIntent, float]:
        """Classify intent from text."""
        text_lower = text.lower()
        
        for intent, patterns in self.intent_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    confidence = 0.8  # Simple confidence score
                    return intent, confidence
        
        return ConversationIntent.UNKNOWN, 0.3


class EntityExtractor:
    """Simple entity extraction based on patterns."""
    
    def __init__(self):
        self.entity_patterns = {
            "pipeline_name": r'\b(pipeline|model)\s+["\']?([^"\']+)["\']?',
            "model_id": r'\b(model|pipeline)[-_]?(\w+)',
            "node_id": r'\bnode[-_]?(\w+)',
            "number": r'\b(\d+\.?\d*)\b',
            "percentage": r'\b(\d+\.?\d*)%\b'
        }
    
    def extract_entities(self, text: str) -> List[Entity]:
        """Extract entities from text."""
        entities = []
        
        for entity_type, pattern in self.entity_patterns.items():
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                entity = Entity(
                    type=entity_type,
                    value=match.group(1) if match.groups() else match.group(0),
                    confidence=0.8,
                    start=match.start(),
                    end=match.end()
                )
                entities.append(entity)
        
        return entities


class VectorDatabase:
    """Simple vector database using FAISS."""
    
    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)
        self.entries: List[KnowledgeBaseEntry] = []
        self.encoder = None
    
    async def initialize(self):
        """Initialize the vector database."""
        logger.info("Initializing vector database...")
        
        # Initialize sentence transformer
        self.encoder = SentenceTransformer(settings.huggingface_model)
        
        # Load existing knowledge base
        await self.load_knowledge_base()
        
        logger.info(f"Vector database initialized with {len(self.entries)} entries")
    
    async def load_knowledge_base(self):
        """Load knowledge base from files."""
        kb_entries = [
            {
                "title": "ML Pipeline Status",
                "content": "To check ML pipeline status, use the /api/ml/pipelines endpoint or ask about specific pipeline names.",
                "category": "ml_pipeline",
                "tags": ["api", "status", "monitoring"]
            },
            {
                "title": "Creating ML Pipelines",
                "content": "Create new ML pipelines using the /api/ml/pipelines endpoint with POST method. Specify model type, dataset, and algorithm.",
                "category": "ml_pipeline",
                "tags": ["api", "creation", "training"]
            },
            {
                "title": "Distributed System Status",
                "content": "Check distributed system status with /api/distributed/cluster endpoint. Shows leader, nodes, and consensus state.",
                "category": "distributed",
                "tags": ["api", "status", "consensus"]
            },
            {
                "title": "Node Management",
                "content": "Manage cluster nodes using /api/distributed/nodes endpoints. You can start, stop, or restart nodes.",
                "category": "distributed",
                "tags": ["api", "nodes", "management"]
            },
            {
                "title": "Platform Health",
                "content": "Platform health can be checked at /health endpoint. Shows overall system status and component health.",
                "category": "platform",
                "tags": ["health", "monitoring", "system"]
            }
        ]
        
        for entry_data in kb_entries:
            entry = KnowledgeBaseEntry(
                id=str(uuid.uuid4()),
                title=entry_data["title"],
                content=entry_data["content"],
                category=entry_data["category"],
                tags=entry_data["tags"]
            )
            
            # Generate embedding
            embedding = self.encoder.encode(entry.content)
            entry.embedding = embedding.tolist()
            
            self.entries.append(entry)
            self.index.add(np.array([embedding], dtype=np.float32))
    
    def search(self, query: str, k: int = 5) -> List[KnowledgeBaseEntry]:
        """Search for relevant knowledge base entries."""
        if not self.encoder:
            return []
        
        # Encode query
        query_embedding = self.encoder.encode(query)
        
        # Search in FAISS index
        distances, indices = self.index.search(
            np.array([query_embedding], dtype=np.float32), k
        )
        
        # Return relevant entries
        results = []
        for idx in indices[0]:
            if idx < len(self.entries):
                results.append(self.entries[idx])
        
        return results


class ChatbotManager:
    """Main chatbot manager class."""
    
    def __init__(self):
        self.sessions: Dict[str, ChatSession] = {}
        self.messages: Dict[str, List[ChatMessage]] = {}
        self.intent_classifier = IntentClassifier()
        self.entity_extractor = EntityExtractor()
        self.vector_db = VectorDatabase()
        self.metrics = ChatbotMetrics()
        self.feedback_data: List[FeedbackData] = []
        self._initialized = False
    
    async def initialize(self):
        """Initialize the chatbot manager."""
        if self._initialized:
            return
        
        logger.info("Initializing Chatbot Manager...")
        
        # Initialize vector database
        await self.vector_db.initialize()
        
        # Load existing sessions (in real implementation, from database)
        await self.load_sessions()
        
        self._initialized = True
        logger.info("Chatbot Manager initialized successfully")
    
    async def load_sessions(self):
        """Load existing sessions."""
        # In a real implementation, this would load from a database
        # For now, create a demo session
        demo_session = ChatSession(
            id="demo_session",
            title="Demo Conversation",
            user_id="demo_user",
            created_at=datetime.now(),
            updated_at=datetime.now(),
            message_count=0,
            context={"demo": True}
        )
        
        self.sessions["demo_session"] = demo_session
        self.messages["demo_session"] = []
    
    async def process_message(self, message: ChatMessage) -> ChatResponse:
        """Process a chat message and generate response."""
        start_time = datetime.now()
        
        try:
            # Classify intent
            intent, confidence = self.intent_classifier.classify_intent(message.content)
            
            # Extract entities
            entities = self.entity_extractor.extract_entities(message.content)
            
            # Search knowledge base
            relevant_entries = self.vector_db.search(message.content)
            
            # Generate response based on intent
            response_content = await self.generate_response(
                intent, message.content, entities, relevant_entries
            )
            
            # Create response message
            response_message = ChatMessage(
                id=str(uuid.uuid4()),
                session_id=message.session_id,
                role=MessageRole.ASSISTANT,
                content=response_content,
                timestamp=datetime.now(),
                metadata={
                    "intent": intent.value,
                    "confidence": confidence,
                    "entities_count": len(entities)
                }
            )
            
            # Store messages
            if message.session_id not in self.messages:
                self.messages[message.session_id] = []
            
            self.messages[message.session_id].append(message)
            self.messages[message.session_id].append(response_message)
            
            # Update session
            if message.session_id in self.sessions:
                session = self.sessions[message.session_id]
                session.message_count += 2
                session.updated_at = datetime.now()
            
            # Generate suggestions
            suggestions = await self.generate_suggestions(intent, entities)
            
            # Generate actions
            actions = await self.generate_actions(intent, entities)
            
            # Update metrics
            response_time = (datetime.now() - start_time).total_seconds()
            metrics.record_chatbot_conversation(response_time)
            
            return ChatResponse(
                message=response_message,
                confidence=confidence,
                intent=intent.value,
                entities=[entity.dict() for entity in entities],
                suggestions=suggestions,
                actions=actions,
                context={"intent": intent.value, "entities": len(entities)}
            )
            
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            
            # Generate error response
            error_message = ChatMessage(
                id=str(uuid.uuid4()),
                session_id=message.session_id,
                role=MessageRole.ASSISTANT,
                content="I'm sorry, I encountered an error processing your message. Please try again.",
                timestamp=datetime.now(),
                metadata={"error": str(e)}
            )
            
            return ChatResponse(
                message=error_message,
                confidence=0.0,
                intent=ConversationIntent.ERROR.value,
                entities=[],
                suggestions=["Try asking in a different way", "Check platform status"],
                actions=[],
                context={"error": True}
            )
    
    async def generate_response(
        self, 
        intent: ConversationIntent, 
        message: str, 
        entities: List[Entity],
        relevant_entries: List[KnowledgeBaseEntry]
    ) -> str:
        """Generate response based on intent and context."""
        
        if intent == ConversationIntent.GREETING:
            return "Hello! I'm the AutoML Platform assistant. I can help you with ML pipelines, distributed systems simulation, and platform monitoring. What would you like to know?"
        
        elif intent == ConversationIntent.GOODBYE:
            return "Goodbye! Feel free to ask me anything about the platform anytime."
        
        elif intent == ConversationIntent.HELP:
            return """I can help you with:
            
🤖 **ML Pipeline Management**
- Check pipeline status and progress
- Create new training jobs
- View model metrics and logs
- Deploy models to production

🔗 **Distributed Systems**
- Monitor cluster health
- View node status
- Trigger elections
- Run simulation scenarios

📊 **Platform Monitoring**
- Check system health
- View performance metrics
- Access logs and events

Just ask me about any of these topics!"""
        
        elif intent == ConversationIntent.ML_STATUS:
            return await self._handle_ml_status(entities)
        
        elif intent == ConversationIntent.ML_CREATE:
            return await self._handle_ml_create(entities)
        
        elif intent == ConversationIntent.DISTRIBUTED_STATUS:
            return await self._handle_distributed_status()
        
        elif intent == ConversationIntent.PLATFORM_HEALTH:
            return await self._handle_platform_health()
        
        elif intent == ConversationIntent.UNKNOWN:
            if relevant_entries:
                context = f"Based on your question, here's what I found:\n\n"
                for entry in relevant_entries[:2]:
                    context += f"**{entry.title}**\n{entry.content}\n\n"
                return context
            else:
                return "I'm not sure I understand. Could you please rephrase your question? You can ask me about ML pipelines, distributed systems, or platform status."
        
        else:
            return "I'm still learning about this topic. Could you please be more specific about what you'd like to know?"
    
    async def _handle_ml_status(self, entities: List[Entity]) -> str:
        """Handle ML pipeline status queries."""
        try:
            # In a real implementation, this would query the ML pipeline API
            return """**ML Pipeline Status**

🟢 **Active Pipelines:** 2
- Customer Churn Prediction: 75% complete (5 min remaining)
- Fraud Detection: Completed successfully

🔄 **Recent Activity:**
- Model deployment started for "Customer Segmentation"
- Training completed for "Price Prediction" model

📊 **Overall Health:** Good
- Success rate: 94%
- Average training time: 15 minutes"""
        
        except Exception as e:
            return f"Sorry, I couldn't fetch the ML pipeline status. Error: {str(e)}"
    
    async def _handle_ml_create(self, entities: List[Entity]) -> str:
        """Handle ML pipeline creation requests."""
        return """**Creating ML Pipeline**

To create a new ML pipeline, I need some information:

1. **Model Type**: Classification, Regression, or Clustering?
2. **Dataset**: Which dataset would you like to use?
3. **Algorithm**: Any specific algorithm preference?

You can also use the API directly:
```
POST /api/ml/pipelines
{
  "name": "My New Pipeline",
  "model_type": "classification",
  "dataset_id": "churn_dataset_001",
  "algorithm": "random_forest"
}
```

Would you like me to help you create one?"""
    
    async def _handle_distributed_status(self) -> str:
        """Handle distributed system status queries."""
        try:
            return """**Distributed System Status**

🟢 **Cluster Health:** Healthy
- **Leader:** node_1 (term 5)
- **Active Nodes:** 4/5
- **Consensus:** Stable

📊 **Node Status:**
- node_1: Leader (Running)
- node_2: Follower (Running)
- node_3: Follower (Running)
- node_4: Follower (Running)
- node_5: Follower (Stopped)

🔄 **Recent Activity:**
- Last election: 2 minutes ago
- Log entries: 15 (all committed)
- Network partitions: None"""
        
        except Exception as e:
            return f"Sorry, I couldn't fetch the distributed system status. Error: {str(e)}"
    
    async def _handle_platform_health(self) -> str:
        """Handle platform health queries."""
        return """**Platform Health Status**

🟢 **Overall Status:** Healthy

**Component Status:**
- 🟢 ML Pipeline: Running (2 active jobs)
- 🟢 Distributed Sim: Running (4/5 nodes active)
- 🟢 Chatbot: Running (1 active session)
- 🟢 Monitoring: Running (all metrics collected)

**System Metrics:**
- CPU Usage: 45%
- Memory Usage: 62%
- Disk Usage: 34%
- Network: Normal

**Recent Events:**
- System started 2 hours ago
- All components initialized successfully
- No errors in last 24 hours"""
    
    async def generate_suggestions(self, intent: ConversationIntent, entities: List[Entity]) -> List[str]:
        """Generate follow-up suggestions."""
        suggestions = []
        
        if intent == ConversationIntent.ML_STATUS:
            suggestions = [
                "Show me training metrics",
                "View pipeline logs",
                "Deploy a model",
                "Create a new pipeline"
            ]
        elif intent == ConversationIntent.DISTRIBUTED_STATUS:
            suggestions = [
                "Show node details",
                "Trigger leader election",
                "Run simulation scenario",
                "Create network partition"
            ]
        elif intent == ConversationIntent.PLATFORM_HEALTH:
            suggestions = [
                "Show detailed metrics",
                "View system logs",
                "Check ML pipeline status",
                "Check distributed system status"
            ]
        else:
            suggestions = [
                "What can you help me with?",
                "Show ML pipeline status",
                "Check distributed system health",
                "View platform metrics"
            ]
        
        return suggestions
    
    async def generate_actions(self, intent: ConversationIntent, entities: List[Entity]) -> List[Dict[str, Any]]:
        """Generate suggested actions."""
        actions = []
        
        if intent == ConversationIntent.ML_STATUS:
            actions = [
                {"type": "view_pipelines", "label": "View All Pipelines"},
                {"type": "create_pipeline", "label": "Create New Pipeline"}
            ]
        elif intent == ConversationIntent.DISTRIBUTED_STATUS:
            actions = [
                {"type": "view_nodes", "label": "View Node Details"},
                {"type": "run_scenario", "label": "Run Simulation"}
            ]
        
        return actions
    
    # Session management methods
    async def create_session(self, session_data: Dict[str, Any]) -> ChatSession:
        """Create a new chat session."""
        session_id = str(uuid.uuid4())
        
        session = ChatSession(
            id=session_id,
            title=session_data.get("title", "New Conversation"),
            user_id=session_data.get("user_id"),
            created_at=datetime.now(),
            updated_at=datetime.now(),
            message_count=0,
            context=session_data.get("context", {}),
            metadata=session_data.get("metadata", {})
        )
        
        self.sessions[session_id] = session
        self.messages[session_id] = []
        
        return session
    
    async def list_sessions(self) -> List[ChatSession]:
        """List all chat sessions."""
        return list(self.sessions.values())
    
    async def get_session(self, session_id: str) -> Optional[ChatSession]:
        """Get a specific chat session."""
        return self.sessions.get(session_id)
    
    async def delete_session(self, session_id: str):
        """Delete a chat session."""
        if session_id in self.sessions:
            del self.sessions[session_id]
        if session_id in self.messages:
            del self.messages[session_id]
    
    async def get_session_messages(self, session_id: str) -> List[ChatMessage]:
        """Get messages from a session."""
        return self.messages.get(session_id, [])
    
    async def clear_session(self, session_id: str):
        """Clear messages from a session."""
        if session_id in self.messages:
            self.messages[session_id] = []
        if session_id in self.sessions:
            self.sessions[session_id].message_count = 0
    
    # Additional methods
    async def get_capabilities(self) -> Dict[str, Any]:
        """Get chatbot capabilities."""
        return ChatbotCapabilities(
            supported_intents=[intent.value for intent in ConversationIntent],
            supported_entities=["pipeline_name", "model_id", "node_id", "number", "percentage"],
            available_commands=["status", "help", "create", "deploy", "stop"],
            integration_points=["ml_pipeline", "distributed_sim", "monitoring"],
            languages=["en"],
            features=["intent_recognition", "entity_extraction", "context_awareness", "knowledge_base"]
        ).dict()
    
    async def train(self, training_data: Dict[str, Any]) -> str:
        """Train/update the chatbot."""
        job_id = str(uuid.uuid4())
        logger.info(f"Starting chatbot training job: {job_id}")
        # In a real implementation, this would update the models
        return job_id
    
    async def get_knowledge_base_info(self) -> Dict[str, Any]:
        """Get knowledge base information."""
        return {
            "total_entries": len(self.vector_db.entries),
            "categories": list(set(entry.category for entry in self.vector_db.entries)),
            "last_updated": datetime.now().isoformat()
        }
    
    async def update_knowledge_base(self, knowledge_data: Dict[str, Any]):
        """Update knowledge base with new information."""
        # In a real implementation, this would add new entries to the vector database
        logger.info("Updating knowledge base")
    
    async def analyze_conversation(self, session_id: str) -> Dict[str, Any]:
        """Analyze a conversation for insights."""
        messages = self.messages.get(session_id, [])
        
        # Simple analysis
        user_messages = [msg for msg in messages if msg.role == MessageRole.USER]
        intents = [msg.metadata.get("intent", "unknown") for msg in messages if msg.metadata.get("intent")]
        
        return ConversationAnalysis(
            session_id=session_id,
            sentiment="positive",
            topics=["ml_pipeline", "distributed_systems"],
            intents=list(set(intents)),
            entities=[],
            satisfaction_score=4.0,
            key_insights=["User is interested in pipeline monitoring"],
            recommendations=["Provide more detailed pipeline information"]
        ).dict()
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Get chatbot performance metrics."""
        return {
            "total_sessions": len(self.sessions),
            "active_sessions": len([s for s in self.sessions.values() if s.active]),
            "total_messages": sum(len(msgs) for msgs in self.messages.values()),
            "average_response_time": 0.8,
            "intent_recognition_accuracy": 0.89,
            "user_satisfaction": 4.2,
            "knowledge_base_size": len(self.vector_db.entries),
            "successful_queries": 150,
            "failed_queries": 10
        }
    
    async def submit_feedback(self, feedback_data: Dict[str, Any]):
        """Submit feedback about chatbot responses."""
        feedback = FeedbackData(**feedback_data)
        self.feedback_data.append(feedback)
        logger.info(f"Received feedback: {feedback.rating}/5")
    
    async def execute_command(self, command: str) -> Dict[str, Any]:
        """Execute a specific command."""
        try:
            if command == "ml_status":
                return {"status": "success", "data": await self._handle_ml_status([])}
            elif command == "distributed_status":
                return {"status": "success", "data": await self._handle_distributed_status()}
            elif command == "platform_health":
                return {"status": "success", "data": await self._handle_platform_health()}
            else:
                return {"status": "error", "message": f"Unknown command: {command}"}
        except Exception as e:
            return {"status": "error", "message": str(e)}
