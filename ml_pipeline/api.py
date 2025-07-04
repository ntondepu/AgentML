"""
ML Pipeline API endpoints for the AutoML Distributed Platform.
"""

from fastapi import APIRouter, BackgroundTasks, HTTPException, Depends
from pydantic import BaseModel
from typing import Dict, List, Optional, Any
import asyncio
import logging

from .pipeline import MLPipelineManager
from .models import (
    PipelineJob, 
    PipelineStatus, 
    ModelMetrics,
    TrainingRequest,
    DeploymentRequest
)

logger = logging.getLogger(__name__)

# Create router
ml_router = APIRouter()

# Global pipeline manager instance
pipeline_manager = None


async def get_pipeline_manager() -> MLPipelineManager:
    """Get the ML pipeline manager instance."""
    global pipeline_manager
    if pipeline_manager is None:
        pipeline_manager = MLPipelineManager()
        await pipeline_manager.initialize()
    return pipeline_manager


@ml_router.get("/")
async def get_ml_status():
    """Get ML pipeline status."""
    return {
        "status": "running",
        "component": "ML Pipeline",
        "features": [
            "Data ingestion and preprocessing",
            "Model training and validation",
            "Model deployment automation",
            "Performance monitoring"
        ]
    }


@ml_router.get("/pipelines")
async def list_pipelines(
    manager: MLPipelineManager = Depends(get_pipeline_manager)
) -> List[PipelineJob]:
    """List all ML pipelines."""
    try:
        return await manager.list_pipelines()
    except Exception as e:
        logger.error(f"Error listing pipelines: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@ml_router.post("/pipelines")
async def create_pipeline(
    request: TrainingRequest,
    background_tasks: BackgroundTasks,
    manager: MLPipelineManager = Depends(get_pipeline_manager)
) -> PipelineJob:
    """Create and start a new ML pipeline."""
    try:
        pipeline = await manager.create_pipeline(request)
        background_tasks.add_task(manager.run_pipeline, pipeline.id)
        return pipeline
    except Exception as e:
        logger.error(f"Error creating pipeline: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@ml_router.get("/pipelines/{pipeline_id}")
async def get_pipeline(
    pipeline_id: str,
    manager: MLPipelineManager = Depends(get_pipeline_manager)
) -> PipelineJob:
    """Get pipeline details by ID."""
    try:
        pipeline = await manager.get_pipeline(pipeline_id)
        if not pipeline:
            raise HTTPException(status_code=404, detail="Pipeline not found")
        return pipeline
    except Exception as e:
        logger.error(f"Error getting pipeline {pipeline_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@ml_router.post("/pipelines/{pipeline_id}/stop")
async def stop_pipeline(
    pipeline_id: str,
    manager: MLPipelineManager = Depends(get_pipeline_manager)
) -> Dict[str, str]:
    """Stop a running pipeline."""
    try:
        await manager.stop_pipeline(pipeline_id)
        return {"status": "stopped", "pipeline_id": pipeline_id}
    except Exception as e:
        logger.error(f"Error stopping pipeline {pipeline_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@ml_router.get("/pipelines/{pipeline_id}/logs")
async def get_pipeline_logs(
    pipeline_id: str,
    manager: MLPipelineManager = Depends(get_pipeline_manager)
) -> List[str]:
    """Get pipeline execution logs."""
    try:
        logs = await manager.get_pipeline_logs(pipeline_id)
        return logs
    except Exception as e:
        logger.error(f"Error getting logs for pipeline {pipeline_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@ml_router.get("/models")
async def list_models(
    manager: MLPipelineManager = Depends(get_pipeline_manager)
) -> List[Dict[str, Any]]:
    """List all deployed models."""
    try:
        return await manager.list_models()
    except Exception as e:
        logger.error(f"Error listing models: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@ml_router.post("/models/{model_id}/deploy")
async def deploy_model(
    model_id: str,
    request: DeploymentRequest,
    manager: MLPipelineManager = Depends(get_pipeline_manager)
) -> Dict[str, str]:
    """Deploy a model to production."""
    try:
        deployment = await manager.deploy_model(model_id, request)
        return {"status": "deployed", "deployment_id": deployment.id}
    except Exception as e:
        logger.error(f"Error deploying model {model_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@ml_router.get("/models/{model_id}/metrics")
async def get_model_metrics(
    model_id: str,
    manager: MLPipelineManager = Depends(get_pipeline_manager)
) -> ModelMetrics:
    """Get model performance metrics."""
    try:
        metrics = await manager.get_model_metrics(model_id)
        if not metrics:
            raise HTTPException(status_code=404, detail="Model metrics not found")
        return metrics
    except Exception as e:
        logger.error(f"Error getting metrics for model {model_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@ml_router.post("/models/{model_id}/predict")
async def predict(
    model_id: str,
    data: Dict[str, Any],
    manager: MLPipelineManager = Depends(get_pipeline_manager)
) -> Dict[str, Any]:
    """Make predictions using a deployed model."""
    try:
        prediction = await manager.predict(model_id, data)
        return {"prediction": prediction, "model_id": model_id}
    except Exception as e:
        logger.error(f"Error making prediction with model {model_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@ml_router.get("/data/datasets")
async def list_datasets(
    manager: MLPipelineManager = Depends(get_pipeline_manager)
) -> List[Dict[str, Any]]:
    """List available datasets."""
    try:
        return await manager.list_datasets()
    except Exception as e:
        logger.error(f"Error listing datasets: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@ml_router.post("/data/preprocess")
async def preprocess_data(
    dataset_id: str,
    preprocessing_config: Dict[str, Any],
    manager: MLPipelineManager = Depends(get_pipeline_manager)
) -> Dict[str, str]:
    """Preprocess a dataset."""
    try:
        job_id = await manager.preprocess_data(dataset_id, preprocessing_config)
        return {"status": "started", "job_id": job_id}
    except Exception as e:
        logger.error(f"Error preprocessing dataset {dataset_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@ml_router.get("/experiments")
async def list_experiments(
    manager: MLPipelineManager = Depends(get_pipeline_manager)
) -> List[Dict[str, Any]]:
    """List MLflow experiments."""
    try:
        return await manager.list_experiments()
    except Exception as e:
        logger.error(f"Error listing experiments: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@ml_router.get("/experiments/{experiment_id}/runs")
async def list_experiment_runs(
    experiment_id: str,
    manager: MLPipelineManager = Depends(get_pipeline_manager)
) -> List[Dict[str, Any]]:
    """List runs for an experiment."""
    try:
        return await manager.list_experiment_runs(experiment_id)
    except Exception as e:
        logger.error(f"Error listing runs for experiment {experiment_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))
