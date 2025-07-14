"""
ML Pipeline Manager for the AutoML Distributed Platform.
"""

import asyncio
import logging
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any
import json
import os

import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib

from .models import (
    PipelineJob, 
    PipelineStatus, 
    ModelType,
    ModelMetrics,
    TrainingRequest,
    DeploymentRequest,
    ModelDeployment,
    DeploymentStatus,
    DatasetInfo
)
from monitoring.telemetry import metrics, get_tracer
from config import settings

logger = logging.getLogger(__name__)


class MLPipelineManager:
    """Manages ML pipeline operations including training, deployment, and monitoring."""
    
    def __init__(self):
        self.jobs: Dict[str, PipelineJob] = {}
        self.models: Dict[str, Dict[str, Any]] = {}
        self.deployments: Dict[str, ModelDeployment] = {}
        self.datasets: Dict[str, DatasetInfo] = {}
        self.tracer = get_tracer()
        self._initialized = False
    
    async def initialize(self):
        """Initialize the ML pipeline manager."""
        if self._initialized:
            return
        
        logger.info("Initializing ML Pipeline Manager...")
        
        # Set up MLflow
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        
        # Create default experiment
        try:
            experiment_id = mlflow.create_experiment("AutoML Platform")
            logger.info(f"Created MLflow experiment: {experiment_id}")
        except Exception as e:
            logger.info(f"Using existing MLflow experiment: {e}")
        
        # Generate sample datasets for demo
        await self._create_sample_datasets()
        
        self._initialized = True
        logger.info("ML Pipeline Manager initialized successfully")
    
    async def _create_sample_datasets(self):
        """Create sample datasets for demonstration."""
        # Sample classification dataset
        np.random.seed(42)
        n_samples = 1000
        
        # Customer churn dataset
        data = {
            'customer_id': [f'cust_{i:04d}' for i in range(n_samples)],
            'age': np.random.randint(18, 80, n_samples),
            'tenure': np.random.randint(1, 60, n_samples),
            'monthly_charges': np.random.uniform(20, 120, n_samples),
            'total_charges': np.random.uniform(100, 5000, n_samples),
            'contract_type': np.random.choice(['month_to_month', 'one_year', 'two_year'], n_samples),
            'payment_method': np.random.choice(['credit_card', 'bank_transfer', 'electronic_check'], n_samples),
            'churn': np.random.choice([0, 1], n_samples, p=[0.7, 0.3])
        }
        
        # Save dataset
        df = pd.DataFrame(data)
        dataset_path = os.path.join(settings.data_dir, 'churn_dataset.csv')
        df.to_csv(dataset_path, index=False)
        
        # Register dataset
        dataset_info = DatasetInfo(
            id="churn_dataset_001",
            name="Customer Churn Dataset",
            description="Historical customer data for churn prediction",
            size=os.path.getsize(dataset_path),
            rows=len(df),
            columns=len(df.columns),
            format="csv",
            created_at=datetime.now(),
            schema={col: str(dtype) for col, dtype in df.dtypes.items()}
        )
        
        self.datasets[dataset_info.id] = dataset_info
        logger.info(f"Created sample dataset: {dataset_info.name}")
    
    async def list_pipelines(self) -> List[PipelineJob]:
        """List all pipeline jobs."""
        return list(self.jobs.values())
    
    async def create_pipeline(self, request: TrainingRequest) -> PipelineJob:
        """Create a new ML pipeline job."""
        job_id = str(uuid.uuid4())
        
        job = PipelineJob(
            id=job_id,
            name=request.name,
            status=PipelineStatus.PENDING,
            model_type=request.model_type,
            dataset_id=request.dataset_id,
            algorithm=request.algorithm,
            created_at=datetime.now()
        )
        
        self.jobs[job_id] = job
        logger.info(f"Created pipeline job: {job_id}")
        
        return job
    
    async def get_pipeline(self, pipeline_id: str) -> Optional[PipelineJob]:
        """Get pipeline job by ID."""
        return self.jobs.get(pipeline_id)
    
    async def run_pipeline(self, pipeline_id: str):
        """Run a pipeline job."""
        job = self.jobs.get(pipeline_id)
        if not job:
            logger.error(f"Pipeline job not found: {pipeline_id}")
            return
        
        with self.tracer.start_as_current_span("ml_pipeline_run") as span:
            span.set_attributes({
                "pipeline.id": pipeline_id,
                "pipeline.name": job.name,
                "pipeline.model_type": job.model_type.value
            })
            
            start_time = datetime.now()
            job.status = PipelineStatus.RUNNING
            job.started_at = start_time
            
            try:
                await self._execute_pipeline(job)
                
                job.status = PipelineStatus.COMPLETED
                job.completed_at = datetime.now()
                job.duration = int((job.completed_at - start_time).total_seconds())
                job.progress = 100.0
                
                metrics.record_ml_job("training", "completed", job.duration)
                logger.info(f"Pipeline job completed: {pipeline_id}")
                
            except Exception as e:
                job.status = PipelineStatus.FAILED
                job.error_message = str(e)
                job.completed_at = datetime.now()
                job.duration = int((job.completed_at - start_time).total_seconds())
                
                metrics.record_ml_job("training", "failed", job.duration)
                logger.error(f"Pipeline job failed: {pipeline_id}, error: {e}")
                span.set_attributes({"error": True, "error.message": str(e)})
    
    async def _execute_pipeline(self, job: PipelineJob):
        """Execute the ML pipeline."""
        # Load dataset
        dataset_info = self.datasets.get(job.dataset_id)
        if not dataset_info:
            raise ValueError(f"Dataset not found: {job.dataset_id}")
        
        dataset_path = os.path.join(settings.data_dir, 'churn_dataset.csv')
        df = pd.read_csv(dataset_path)
        
        # Prepare features and target
        X = df.drop(['customer_id', 'churn'], axis=1)
        y = df['churn']
        
        # Encode categorical variables
        label_encoders = {}
        for col in X.select_dtypes(include=['object']).columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])
            label_encoders[col] = le
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        with mlflow.start_run(experiment_id=mlflow.get_experiment_by_name("AutoML Platform").experiment_id):
            if job.algorithm == "random_forest" or job.algorithm == "auto":
                if job.model_type == ModelType.CLASSIFICATION:
                    model = RandomForestClassifier(n_estimators=100, random_state=42)
                else:
                    model = RandomForestRegressor(n_estimators=100, random_state=42)
            else:
                # Default to random forest
                model = RandomForestClassifier(n_estimators=100, random_state=42)
            
            # Simulate training progress
            for i in range(0, 101, 20):
                job.progress = i
                await asyncio.sleep(0.1)  # Simulate training time
            
            # Fit model
            model.fit(X_train_scaled, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test_scaled)
            
            # Calculate metrics
            if job.model_type == ModelType.CLASSIFICATION:
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, average='weighted')
                recall = recall_score(y_test, y_pred, average='weighted')
                f1 = f1_score(y_test, y_pred, average='weighted')
                
                job.metrics = {
                    "accuracy": accuracy,
                    "precision": precision,
                    "recall": recall,
                    "f1_score": f1
                }
                
                # Log to MLflow
                mlflow.log_metrics(job.metrics)
            
            # Save model
            model_id = str(uuid.uuid4())
            model_path = os.path.join(settings.model_dir, f"{model_id}.pkl")
            joblib.dump({
                'model': model,
                'scaler': scaler,
                'label_encoders': label_encoders,
                'feature_names': X.columns.tolist()
            }, model_path)
            
            # Log model to MLflow
            mlflow.sklearn.log_model(
                model, 
                "model", 
                registered_model_name=f"{job.name}_{model_id}"
            )
            
            # Store model information
            self.models[model_id] = {
                'id': model_id,
                'name': job.name,
                'type': job.model_type.value,
                'algorithm': job.algorithm,
                'path': model_path,
                'metrics': job.metrics,
                'created_at': datetime.now(),
                'mlflow_run_id': mlflow.active_run().info.run_id
            }
            
            job.model_id = model_id
    
    async def stop_pipeline(self, pipeline_id: str):
        """Stop a running pipeline."""
        job = self.jobs.get(pipeline_id)
        if job and job.status == PipelineStatus.RUNNING:
            job.status = PipelineStatus.STOPPED
            job.completed_at = datetime.now()
            if job.started_at:
                job.duration = int((job.completed_at - job.started_at).total_seconds())
            logger.info(f"Pipeline stopped: {pipeline_id}")
    
    async def get_pipeline_logs(self, pipeline_id: str) -> List[str]:
        """Get pipeline execution logs."""
        # In a real implementation, this would fetch logs from logging system
        job = self.jobs.get(pipeline_id)
        if not job:
            return []
        
        logs = [
            f"[{job.created_at}] Pipeline created: {job.name}",
            f"[{job.started_at}] Pipeline started",
            f"[{job.started_at}] Loading dataset: {job.dataset_id}",
            f"[{job.started_at}] Preprocessing data...",
            f"[{job.started_at}] Training model with {job.algorithm}...",
        ]
        
        if job.status == PipelineStatus.COMPLETED:
            logs.append(f"[{job.completed_at}] Training completed successfully")
            logs.append(f"[{job.completed_at}] Model saved: {job.model_id}")
        elif job.status == PipelineStatus.FAILED:
            logs.append(f"[{job.completed_at}] Training failed: {job.error_message}")
        
        return logs
    
    async def list_models(self) -> List[Dict[str, Any]]:
        """List all trained models."""
        return list(self.models.values())
    
    async def deploy_model(self, model_id: str, request: DeploymentRequest) -> ModelDeployment:
        """Deploy a model to production."""
        model = self.models.get(model_id)
        if not model:
            raise ValueError(f"Model not found: {model_id}")
        
        deployment_id = str(uuid.uuid4())
        deployment = ModelDeployment(
            id=deployment_id,
            model_id=model_id,
            name=request.name,
            status=DeploymentStatus.DEPLOYING,
            environment=request.environment,
            replicas=request.replicas,
            cpu_limit=request.cpu_limit,
            memory_limit=request.memory_limit,
            auto_scaling=request.auto_scaling,
            created_at=datetime.now()
        )
        
        # Simulate deployment process
        await asyncio.sleep(1)
        
        deployment.status = DeploymentStatus.DEPLOYED
        deployment.endpoint_url = f"https://api.example.com/models/{request.name}/predict"
        deployment.updated_at = datetime.now()
        
        self.deployments[deployment_id] = deployment
        logger.info(f"Model deployed: {model_id} -> {deployment_id}")
        
        return deployment
    
    async def get_model_metrics(self, model_id: str) -> Optional[ModelMetrics]:
        """Get model performance metrics."""
        model = self.models.get(model_id)
        if not model:
            return None
        
        metrics_data = model.get('metrics', {})
        
        return ModelMetrics(
            model_id=model_id,
            accuracy=metrics_data.get('accuracy'),
            precision=metrics_data.get('precision'),
            recall=metrics_data.get('recall'),
            f1_score=metrics_data.get('f1_score'),
            evaluation_date=model['created_at']
        )
    
    async def predict(self, model_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Make predictions using a deployed model."""
        model_info = self.models.get(model_id)
        if not model_info:
            raise ValueError(f"Model not found: {model_id}")
        
        # Load model
        model_data = joblib.load(model_info['path'])
        model = model_data['model']
        scaler = model_data['scaler']
        label_encoders = model_data['label_encoders']
        feature_names = model_data['feature_names']
        
        # Prepare input data
        input_df = pd.DataFrame([data])
        
        # Encode categorical variables
        for col, le in label_encoders.items():
            if col in input_df.columns:
                input_df[col] = le.transform(input_df[col])
        
        # Ensure all features are present
        for feature in feature_names:
            if feature not in input_df.columns:
                input_df[feature] = 0
        
        # Scale features
        input_scaled = scaler.transform(input_df[feature_names])
        
        # Make prediction
        prediction = model.predict(input_scaled)[0]
        
        # Get probability if classification
        probability = None
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(input_scaled)[0]
            probability = float(max(proba))
        
        return {
            "prediction": int(prediction) if isinstance(prediction, (np.integer, np.int64)) else prediction,
            "probability": probability,
            "model_id": model_id,
            "timestamp": datetime.now().isoformat()
        }
    
    async def list_datasets(self) -> List[Dict[str, Any]]:
        """List available datasets."""
        return [dataset.dict() for dataset in self.datasets.values()]
    
    async def preprocess_data(self, dataset_id: str, config: Dict[str, Any]) -> str:
        """Preprocess a dataset."""
        job_id = str(uuid.uuid4())
        # In a real implementation, this would start a preprocessing job
        logger.info(f"Starting data preprocessing job: {job_id}")
        return job_id
    
    async def list_experiments(self) -> List[Dict[str, Any]]:
        """List MLflow experiments."""
        try:
            experiments = mlflow.search_experiments()
            return [
                {
                    "experiment_id": exp.experiment_id,
                    "name": exp.name,
                    "lifecycle_stage": exp.lifecycle_stage,
                    "creation_time": exp.creation_time
                }
                for exp in experiments
            ]
        except Exception as e:
            logger.error(f"Error listing experiments: {e}")
            return []
    
    async def list_experiment_runs(self, experiment_id: str) -> List[Dict[str, Any]]:
        """List runs for an experiment."""
        try:
            runs = mlflow.search_runs(experiment_ids=[experiment_id])
            return runs.to_dict('records') if not runs.empty else []
        except Exception as e:
            logger.error(f"Error listing runs for experiment {experiment_id}: {e}")
            return []
