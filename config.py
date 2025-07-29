"""
Configuration settings for the AutoML Distributed Platform.
"""

import os
from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Application settings
    app_name: str = Field(default="AutoML Distributed Platform", env="APP_NAME")
    debug: bool = Field(default=False, env="DEBUG")
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    
    # Database settings
    database_url: str = Field(
        default="postgresql://agentml:agentml_password@localhost:5432/agentml",
        env="DATABASE_URL"
    )
    
    # Redis settings
    redis_url: str = Field(default="redis://localhost:6379", env="REDIS_URL")
    
    # MLflow settings
    mlflow_tracking_uri: str = Field(
        default="http://localhost:5000",
        env="MLFLOW_TRACKING_URI"
    )
    mlflow_artifact_root: str = Field(
        default="s3://mlflow-artifacts/",
        env="MLFLOW_ARTIFACT_ROOT"
    )
    mlflow_image: str = Field(
        default="mlflow/mlflow:latest",
        env="MLFLOW_IMAGE"
    )
    
    # MinIO settings
    minio_endpoint: str = Field(default="localhost:9000", env="MINIO_ENDPOINT")
    minio_access_key: str = Field(default="minioadmin", env="MINIO_ACCESS_KEY")
    minio_secret_key: str = Field(default="minioadmin", env="MINIO_SECRET_KEY")
    
    # OpenAI settings
    openai_api_key: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-3.5-turbo", env="OPENAI_MODEL")
    
    # Hugging Face settings
    huggingface_token: Optional[str] = Field(default=None, env="HUGGINGFACE_TOKEN")
    huggingface_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        env="HUGGINGFACE_MODEL"
    )
    
    # Monitoring settings
    prometheus_port: int = Field(default=9090, env="PROMETHEUS_PORT")
    grafana_port: int = Field(default=3000, env="GRAFANA_PORT")
    jaeger_endpoint: str = Field(
        default="http://localhost:14268/api/traces",
        env="JAEGER_ENDPOINT"
    )
    
    # Kubernetes settings
    kubernetes_namespace: str = Field(default="agentml", env="KUBERNETES_NAMESPACE")
    kubernetes_config_path: Optional[str] = Field(
        default=None, env="KUBERNETES_CONFIG_PATH"
    )
    
    # Distributed simulation settings
    raft_node_count: int = Field(default=5, env="RAFT_NODE_COUNT")
    raft_heartbeat_interval: float = Field(default=0.5, env="RAFT_HEARTBEAT_INTERVAL")
    raft_election_timeout: float = Field(default=1.5, env="RAFT_ELECTION_TIMEOUT")
    
    # Vector database settings
    vector_db_path: str = Field(default="./data/vector_db", env="VECTOR_DB_PATH")
    embedding_dimension: int = Field(default=384, env="EMBEDDING_DIMENSION")
    
    # Data settings
    data_dir: str = Field(default="./data", env="DATA_DIR")
    model_dir: str = Field(default="./models", env="MODEL_DIR")
    
    # Security settings
    secret_key: str = Field(default="change-me", env="SECRET_KEY")
    access_token_expire_minutes: int = Field(
        default=30, env="ACCESS_TOKEN_EXPIRE_MINUTES"
    )
    sanitize_input: bool = Field(default=True, env="SANITIZE_INPUT")
    enable_access_control: bool = Field(default=False, env="ENABLE_ACCESS_CONTROL")
    allowed_api_keys: Optional[str] = Field(default=None, env="ALLOWED_API_KEYS")

    # Logging and metrics
    log_format: str = Field(default="%(asctime)s %(levelname)s %(name)s %(message)s", env="LOG_FORMAT")
    enable_metrics: bool = Field(default=True, env="ENABLE_METRICS")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False

    def __init__(self, **kwargs):
        """
        Initialize settings and create necessary directories.
        """
        super().__init__(**kwargs)
        Path(self.data_dir).mkdir(parents=True, exist_ok=True)
        Path(self.model_dir).mkdir(parents=True, exist_ok=True)
        Path(self.vector_db_path).mkdir(parents=True, exist_ok=True)


# Global settings instance
settings = Settings()


# Environment-specific configurations
class DevelopmentConfig(Settings):
    """
    Development configuration.
    Enables debug mode and verbose logging.
    """
    debug: bool = True
    log_level: str = "DEBUG"


class ProductionConfig(Settings):
    """
    Production configuration.
    Disables debug mode and sets log level to INFO.
    """
    debug: bool = False
    log_level: str = "INFO"


class TestingConfig(Settings):
    """
    Testing configuration.
    Uses SQLite and enables debug mode for tests.
    """
    debug: bool = True
    log_level: str = "DEBUG"
    database_url: str = "sqlite:///./test.db"


def get_settings() -> Settings:
    """
    Get settings based on environment.
    Returns the appropriate config class for dev, prod, or test.
    """
    env = os.getenv("ENVIRONMENT", "development").lower()
    
    if env == "production":
        return ProductionConfig()
    elif env == "testing":
        return TestingConfig()
    else:
        return DevelopmentConfig()


index = None  # Always fallback in config.py; faiss logic should be handled in chatbot/api.py or bot.py
