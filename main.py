#!/usr/bin/env python3
"""
AutoML Distributed Platform - Main Application Entry Point

This is the main entry point for the AutoML Distributed Platform that orchestrates:
- ML Pipeline Automation (Kubeflow/MLflow)
- Distributed Systems Simulation (Raft Consensus)
- AI Chatbot Interface (OpenAI/HuggingFace)
"""

import asyncio
import logging
import sys
from pathlib import Path
from typing import Optional

import click
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Add project root to Python path
sys.path.append(str(Path(__file__).parent))

from ml_pipeline.api import ml_router
from distributed_sim.api import distributed_router
from chatbot.api import chatbot_router
from monitoring.telemetry import setup_telemetry
from config import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="AutoML Distributed Platform",
    description="A comprehensive platform for AutoML, distributed systems simulation, and AI chatbot",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup telemetry and monitoring
setup_telemetry(app)

# Include routers
app.include_router(ml_router, prefix="/api/ml", tags=["ML Pipeline"])
app.include_router(distributed_router, prefix="/api/distributed", tags=["Distributed Simulation"])
app.include_router(chatbot_router, prefix="/api/chatbot", tags=["AI Chatbot"])


@app.get("/")
async def root():
    """Root endpoint with platform information."""
    return {
        "message": "AutoML Distributed Platform",
        "version": "1.0.0",
        "components": {
            "ml_pipeline": "ML Pipeline Automation with Kubeflow/MLflow",
            "distributed_sim": "Raft Consensus Simulation",
            "chatbot": "AI Chatbot Interface"
        },
        "endpoints": {
            "docs": "/docs",
            "ml_api": "/api/ml",
            "distributed_api": "/api/distributed",
            "chatbot_api": "/api/chatbot"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": "2025-07-03"}


async def startup_services():
    """Initialize all platform services."""
    logger.info("Starting AutoML Distributed Platform services...")
    
    try:
        # Initialize ML pipeline components
        from ml_pipeline.pipeline import MLPipelineManager
        ml_manager = MLPipelineManager()
        await ml_manager.initialize()
        
        # Initialize distributed simulation
        from distributed_sim.raft import RaftSimulator
        raft_sim = RaftSimulator()
        await raft_sim.initialize()
        
        # Initialize chatbot
        from chatbot.bot import ChatbotManager
        chatbot_manager = ChatbotManager()
        await chatbot_manager.initialize()
        
        logger.info("All services initialized successfully!")
        
    except Exception as e:
        logger.error(f"Failed to initialize services: {e}")
        raise


@app.on_event("startup")
async def on_startup():
    """Application startup event."""
    await startup_services()


@app.on_event("shutdown")
async def on_shutdown():
    """Application shutdown event."""
    logger.info("Shutting down AutoML Distributed Platform...")


@click.command()
@click.option("--host", default="0.0.0.0", help="Host to bind to")
@click.option("--port", default=8000, help="Port to bind to")
@click.option("--reload", is_flag=True, help="Enable auto-reload for development")
@click.option("--workers", default=1, help="Number of worker processes")
def run_server(host: str, port: int, reload: bool, workers: int):
    """Run the AutoML Distributed Platform server."""
    logger.info(f"Starting server on {host}:{port}")
    
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=reload,
        workers=workers if not reload else 1,
        log_level="info"
    )


@click.group()
def cli():
    """AutoML Distributed Platform CLI."""
    pass


@cli.command()
def init():
    """Initialize the platform (setup database, create initial data, etc.)."""
    logger.info("Initializing AutoML Distributed Platform...")
    # Add initialization logic here
    logger.info("Platform initialized successfully!")


@cli.command()
def demo():
    """Run platform demo scenarios."""
    logger.info("Running platform demo...")
    # Add demo logic here
    logger.info("Demo completed!")


# Add CLI commands
cli.add_command(run_server, name="server")


if __name__ == "__main__":
    cli()
