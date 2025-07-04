"""
Pydantic models for the ML Pipeline component.
"""

from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from enum import Enum
from sklearn.linear_model import LogisticRegression


class PipelineStatus(str, Enum):
    """Pipeline execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    STOPPED = "stopped"


class ModelType(str, Enum):
    """Supported model types."""
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    CLUSTERING = "clustering"
    ANOMALY_DETECTION = "anomaly_detection"
    NATURAL_LANGUAGE_PROCESSING = "nlp"
    COMPUTER_VISION = "computer_vision"


class DeploymentStatus(str, Enum):
    """Model deployment status."""
    PENDING = "pending"
    DEPLOYING = "deploying"
    DEPLOYED = "deployed"
    FAILED = "failed"
    STOPPED = "stopped"


class TrainingRequest(BaseModel):
    """Request model for training a new ML model."""
    name: str = Field(..., description="Name of the training job")
    model_type: ModelType = Field(..., description="Type of ML model to train")
    dataset_id: str = Field(..., description="ID of the dataset to use")
    algorithm: str = Field(default="auto", description="ML algorithm to use")
    hyperparameters: Dict[str, Any] = Field(default_factory=dict, description="Hyperparameters")
    validation_split: float = Field(default=0.2, description="Validation split ratio")
    max_training_time: Optional[int] = Field(default=None, description="Max training time in seconds")
    target_metric: str = Field(default="accuracy", description="Target metric to optimize")
    
    class Config:
        schema_extra = {
            "example": {
                "name": "Customer Churn Prediction",
                "model_type": "classification",
                "dataset_id": "churn_dataset_001",
                "algorithm": "random_forest",
                "hyperparameters": {
                    "n_estimators": 100,
                    "max_depth": 10
                },
                "validation_split": 0.2,
                "target_metric": "f1_score"
            }
        }


class DeploymentRequest(BaseModel):
    """Request model for deploying a trained model."""
    name: str = Field(..., description="Name of the deployment")
    environment: str = Field(default="production", description="Deployment environment")
    replicas: int = Field(default=1, description="Number of replicas")
    cpu_limit: str = Field(default="500m", description="CPU limit")
    memory_limit: str = Field(default="1Gi", description="Memory limit")
    auto_scaling: bool = Field(default=True, description="Enable auto-scaling")
    
    class Config:
        schema_extra = {
            "example": {
                "name": "churn-prediction-v1",
                "environment": "production",
                "replicas": 3,
                "cpu_limit": "1000m",
                "memory_limit": "2Gi",
                "auto_scaling": True
            }
        }


class PipelineJob(BaseModel):
    """ML pipeline job model."""
    id: str = Field(..., description="Unique job identifier")
    name: str = Field(..., description="Job name")
    status: PipelineStatus = Field(..., description="Current job status")
    model_type: ModelType = Field(..., description="Type of ML model")
    dataset_id: str = Field(..., description="Dataset identifier")
    algorithm: str = Field(..., description="ML algorithm used")
    created_at: datetime = Field(..., description="Job creation timestamp")
    started_at: Optional[datetime] = Field(default=None, description="Job start timestamp")
    completed_at: Optional[datetime] = Field(default=None, description="Job completion timestamp")
    duration: Optional[int] = Field(default=None, description="Job duration in seconds")
    progress: float = Field(default=0.0, description="Job progress percentage")
    metrics: Dict[str, float] = Field(default_factory=dict, description="Training metrics")
    model_id: Optional[str] = Field(default=None, description="Resulting model ID")
    error_message: Optional[str] = Field(default=None, description="Error message if failed")
    
    class Config:
        schema_extra = {
            "example": {
                "id": "job_123456",
                "name": "Customer Churn Prediction",
                "status": "completed",
                "model_type": "classification",
                "dataset_id": "churn_dataset_001",
                "algorithm": "random_forest",
                "created_at": "2025-07-03T10:00:00Z",
                "started_at": "2025-07-03T10:01:00Z",
                "completed_at": "2025-07-03T10:15:00Z",
                "duration": 840,
                "progress": 100.0,
                "metrics": {
                    "accuracy": 0.87,
                    "f1_score": 0.85,
                    "precision": 0.88,
                    "recall": 0.82
                },
                "model_id": "model_789012"
            }
        }


class ModelMetrics(BaseModel):
    """Model performance metrics."""
    model_id: str = Field(..., description="Model identifier")
    accuracy: Optional[float] = Field(default=None, description="Model accuracy")
    precision: Optional[float] = Field(default=None, description="Model precision")
    recall: Optional[float] = Field(default=None, description="Model recall")
    f1_score: Optional[float] = Field(default=None, description="Model F1 score")
    auc_roc: Optional[float] = Field(default=None, description="AUC-ROC score")
    mse: Optional[float] = Field(default=None, description="Mean squared error")
    rmse: Optional[float] = Field(default=None, description="Root mean squared error")
    mae: Optional[float] = Field(default=None, description="Mean absolute error")
    r2_score: Optional[float] = Field(default=None, description="R-squared score")
    custom_metrics: Dict[str, float] = Field(default_factory=dict, description="Custom metrics")
    evaluation_date: datetime = Field(..., description="Evaluation timestamp")
    
    class Config:
        schema_extra = {
            "example": {
                "model_id": "model_789012",
                "accuracy": 0.87,
                "precision": 0.88,
                "recall": 0.82,
                "f1_score": 0.85,
                "auc_roc": 0.91,
                "custom_metrics": {
                    "business_impact": 0.73
                },
                "evaluation_date": "2025-07-03T10:15:00Z"
            }
        }


class ModelDeployment(BaseModel):
    """Model deployment information."""
    id: str = Field(..., description="Deployment identifier")
    model_id: str = Field(..., description="Model identifier")
    name: str = Field(..., description="Deployment name")
    status: DeploymentStatus = Field(..., description="Deployment status")
    environment: str = Field(..., description="Deployment environment")
    endpoint_url: Optional[str] = Field(default=None, description="Model endpoint URL")
    replicas: int = Field(..., description="Number of replicas")
    cpu_limit: str = Field(..., description="CPU limit")
    memory_limit: str = Field(..., description="Memory limit")
    auto_scaling: bool = Field(..., description="Auto-scaling enabled")
    created_at: datetime = Field(..., description="Deployment creation timestamp")
    updated_at: Optional[datetime] = Field(default=None, description="Last update timestamp")
    
    class Config:
        schema_extra = {
            "example": {
                "id": "deployment_456789",
                "model_id": "model_789012",
                "name": "churn-prediction-v1",
                "status": "deployed",
                "environment": "production",
                "endpoint_url": "https://api.example.com/models/churn-prediction-v1/predict",
                "replicas": 3,
                "cpu_limit": "1000m",
                "memory_limit": "2Gi",
                "auto_scaling": True,
                "created_at": "2025-07-03T10:20:00Z",
                "updated_at": "2025-07-03T10:25:00Z"
            }
        }


class DatasetInfo(BaseModel):
    """Dataset information."""
    id: str = Field(..., description="Dataset identifier")
    name: str = Field(..., description="Dataset name")
    description: Optional[str] = Field(default=None, description="Dataset description")
    size: int = Field(..., description="Dataset size in bytes")
    rows: int = Field(..., description="Number of rows")
    columns: int = Field(..., description="Number of columns")
    format: str = Field(..., description="Data format (csv, json, parquet, etc.)")
    created_at: datetime = Field(..., description="Dataset creation timestamp")
    updated_at: Optional[datetime] = Field(default=None, description="Last update timestamp")
    schema: Dict[str, str] = Field(default_factory=dict, description="Dataset schema")
    
    class Config:
        schema_extra = {
            "example": {
                "id": "churn_dataset_001",
                "name": "Customer Churn Dataset",
                "description": "Historical customer data for churn prediction",
                "size": 10485760,
                "rows": 50000,
                "columns": 15,
                "format": "csv",
                "created_at": "2025-07-03T09:00:00Z",
                "schema": {
                    "customer_id": "string",
                    "age": "integer",
                    "tenure": "integer",
                    "monthly_charges": "float",
                    "total_charges": "float",
                    "churn": "boolean"
                }
            }
        }


class PredictionRequest(BaseModel):
    """Request model for making predictions."""
    features: Dict[str, Any] = Field(..., description="Feature values for prediction")
    
    class Config:
        schema_extra = {
            "example": {
                "features": {
                    "age": 35,
                    "tenure": 24,
                    "monthly_charges": 79.99,
                    "total_charges": 1919.76,
                    "contract_type": "month_to_month",
                    "payment_method": "credit_card"
                }
            }
        }


class PredictionResponse(BaseModel):
    """Response model for predictions."""
    prediction: Union[str, float, int, bool] = Field(..., description="Prediction result")
    probability: Optional[float] = Field(default=None, description="Prediction probability")
    confidence: Optional[float] = Field(default=None, description="Prediction confidence")
    model_id: str = Field(..., description="Model used for prediction")
    timestamp: datetime = Field(..., description="Prediction timestamp")
    
    class Config:
        schema_extra = {
            "example": {
                "prediction": "churn",
                "probability": 0.73,
                "confidence": 0.68,
                "model_id": "model_789012",
                "timestamp": "2025-07-03T11:00:00Z"
            }
        }


class SimpleClassifier:
    def __init__(self):
        self.model = LogisticRegression()
    def fit(self, X, y):
        self.model.fit(X, y)
    def predict(self, X):
        return self.model.predict(X)
    def score(self, X, y):
        return self.model.score(X, y)
