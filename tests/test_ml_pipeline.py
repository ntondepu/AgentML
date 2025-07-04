"""
Test suite for ML Pipeline component.
"""

import asyncio
import pytest
from datetime import datetime
from unittest.mock import Mock, patch, AsyncMock

from ml_pipeline.pipeline import MLPipelineManager
from ml_pipeline.models import (
    PipelineJob, 
    PipelineStatus, 
    ModelType, 
    TrainingRequest,
    DeploymentRequest
)


class TestMLPipelineManager:
    """Test cases for MLPipelineManager."""
    
    @pytest.fixture
    async def manager(self):
        """Create ML pipeline manager instance."""
        manager = MLPipelineManager()
        await manager.initialize()
        return manager
    
    @pytest.fixture
    def training_request(self):
        """Create a sample training request."""
        return TrainingRequest(
            name="Test Pipeline",
            model_type=ModelType.CLASSIFICATION,
            dataset_id="test_dataset",
            algorithm="random_forest",
            hyperparameters={"n_estimators": 100},
            validation_split=0.2,
            target_metric="accuracy"
        )
    
    @pytest.fixture
    def deployment_request(self):
        """Create a sample deployment request."""
        return DeploymentRequest(
            name="test-deployment",
            environment="testing",
            replicas=1,
            cpu_limit="500m",
            memory_limit="1Gi",
            auto_scaling=False
        )
    
    async def test_initialize(self, manager):
        """Test manager initialization."""
        assert manager._initialized is True
        assert len(manager.datasets) > 0
        assert "churn_dataset_001" in manager.datasets
    
    async def test_create_pipeline(self, manager, training_request):
        """Test pipeline creation."""
        pipeline = await manager.create_pipeline(training_request)
        
        assert pipeline.name == "Test Pipeline"
        assert pipeline.status == PipelineStatus.PENDING
        assert pipeline.model_type == ModelType.CLASSIFICATION
        assert pipeline.dataset_id == "test_dataset"
        assert pipeline.algorithm == "random_forest"
        assert pipeline.id in manager.jobs
    
    async def test_list_pipelines(self, manager, training_request):
        """Test listing pipelines."""
        # Create a pipeline
        pipeline = await manager.create_pipeline(training_request)
        
        # List pipelines
        pipelines = await manager.list_pipelines()
        
        assert len(pipelines) == 1
        assert pipelines[0].id == pipeline.id
    
    async def test_get_pipeline(self, manager, training_request):
        """Test getting specific pipeline."""
        # Create a pipeline
        pipeline = await manager.create_pipeline(training_request)
        
        # Get pipeline
        retrieved = await manager.get_pipeline(pipeline.id)
        
        assert retrieved is not None
        assert retrieved.id == pipeline.id
        assert retrieved.name == pipeline.name
    
    async def test_get_nonexistent_pipeline(self, manager):
        """Test getting non-existent pipeline."""
        result = await manager.get_pipeline("nonexistent")
        assert result is None
    
    @pytest.mark.asyncio
    async def test_run_pipeline(self, manager, training_request):
        """Test running a pipeline."""
        # Create a pipeline
        pipeline = await manager.create_pipeline(training_request)
        
        # Mock the dataset to avoid file operations
        with patch('pandas.read_csv') as mock_read_csv:
            mock_df = Mock()
            mock_df.drop.return_value = Mock()
            mock_df.__getitem__.return_value = Mock()
            mock_read_csv.return_value = mock_df
            
            # Run pipeline
            await manager.run_pipeline(pipeline.id)
            
            # Check pipeline status
            updated_pipeline = await manager.get_pipeline(pipeline.id)
            assert updated_pipeline.status in [PipelineStatus.COMPLETED, PipelineStatus.FAILED]
    
    async def test_stop_pipeline(self, manager, training_request):
        """Test stopping a pipeline."""
        # Create and start pipeline
        pipeline = await manager.create_pipeline(training_request)
        pipeline.status = PipelineStatus.RUNNING
        pipeline.started_at = datetime.now()
        
        # Stop pipeline
        await manager.stop_pipeline(pipeline.id)
        
        # Check status
        updated_pipeline = await manager.get_pipeline(pipeline.id)
        assert updated_pipeline.status == PipelineStatus.STOPPED
        assert updated_pipeline.completed_at is not None
    
    async def test_get_pipeline_logs(self, manager, training_request):
        """Test getting pipeline logs."""
        # Create a pipeline
        pipeline = await manager.create_pipeline(training_request)
        
        # Get logs
        logs = await manager.get_pipeline_logs(pipeline.id)
        
        assert isinstance(logs, list)
        assert len(logs) > 0
        assert pipeline.name in logs[0]
    
    async def test_list_datasets(self, manager):
        """Test listing datasets."""
        datasets = await manager.list_datasets()
        
        assert isinstance(datasets, list)
        assert len(datasets) > 0
        assert any(d['id'] == 'churn_dataset_001' for d in datasets)
    
    async def test_list_models(self, manager):
        """Test listing models."""
        models = await manager.list_models()
        
        assert isinstance(models, list)
        # Initially empty since no models trained
        assert len(models) == 0
    
    async def test_deploy_model(self, manager, deployment_request):
        """Test model deployment."""
        # Create a mock model
        model_id = "test_model_123"
        manager.models[model_id] = {
            'id': model_id,
            'name': 'Test Model',
            'type': 'classification',
            'created_at': datetime.now()
        }
        
        # Deploy model
        deployment = await manager.deploy_model(model_id, deployment_request)
        
        assert deployment.model_id == model_id
        assert deployment.name == "test-deployment"
        assert deployment.status.value == "deployed"
        assert deployment.id in manager.deployments
    
    async def test_deploy_nonexistent_model(self, manager, deployment_request):
        """Test deploying non-existent model."""
        with pytest.raises(ValueError, match="Model not found"):
            await manager.deploy_model("nonexistent", deployment_request)
    
    async def test_get_model_metrics(self, manager):
        """Test getting model metrics."""
        # Create a mock model with metrics
        model_id = "test_model_123"
        manager.models[model_id] = {
            'id': model_id,
            'name': 'Test Model',
            'type': 'classification',
            'created_at': datetime.now(),
            'metrics': {
                'accuracy': 0.85,
                'precision': 0.82,
                'recall': 0.87,
                'f1_score': 0.84
            }
        }
        
        # Get metrics
        metrics = await manager.get_model_metrics(model_id)
        
        assert metrics is not None
        assert metrics.model_id == model_id
        assert metrics.accuracy == 0.85
        assert metrics.precision == 0.82
        assert metrics.recall == 0.87
        assert metrics.f1_score == 0.84
    
    async def test_get_metrics_nonexistent_model(self, manager):
        """Test getting metrics for non-existent model."""
        result = await manager.get_model_metrics("nonexistent")
        assert result is None
    
    async def test_preprocess_data(self, manager):
        """Test data preprocessing."""
        job_id = await manager.preprocess_data(
            "churn_dataset_001", 
            {"normalize": True, "handle_missing": "mean"}
        )
        
        assert isinstance(job_id, str)
        assert len(job_id) > 0
    
    async def test_list_experiments(self, manager):
        """Test listing MLflow experiments."""
        with patch('mlflow.search_experiments') as mock_search:
            mock_search.return_value = []
            
            experiments = await manager.list_experiments()
            
            assert isinstance(experiments, list)
            mock_search.assert_called_once()
    
    async def test_list_experiment_runs(self, manager):
        """Test listing experiment runs."""
        with patch('mlflow.search_runs') as mock_search:
            mock_search.return_value.empty = True
            mock_search.return_value.to_dict.return_value = []
            
            runs = await manager.list_experiment_runs("test_experiment")
            
            assert isinstance(runs, list)
            mock_search.assert_called_once_with(experiment_ids=["test_experiment"])


@pytest.mark.asyncio
class TestMLPipelineIntegration:
    """Integration tests for ML Pipeline component."""
    
    async def test_full_pipeline_workflow(self):
        """Test complete pipeline workflow."""
        manager = MLPipelineManager()
        await manager.initialize()
        
        # Create training request
        request = TrainingRequest(
            name="Integration Test Pipeline",
            model_type=ModelType.CLASSIFICATION,
            dataset_id="churn_dataset_001",
            algorithm="random_forest"
        )
        
        # Create pipeline
        pipeline = await manager.create_pipeline(request)
        assert pipeline.status == PipelineStatus.PENDING
        
        # Check if pipeline is in the list
        pipelines = await manager.list_pipelines()
        assert len(pipelines) == 1
        assert pipelines[0].id == pipeline.id
        
        # Get pipeline logs
        logs = await manager.get_pipeline_logs(pipeline.id)
        assert len(logs) > 0
        
        # Stop pipeline (simulate stopping before completion)
        await manager.stop_pipeline(pipeline.id)
        updated_pipeline = await manager.get_pipeline(pipeline.id)
        assert updated_pipeline.status == PipelineStatus.STOPPED


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
