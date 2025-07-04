"""
Telemetry and monitoring setup for the AutoML Distributed Platform.
"""

import logging
from typing import Optional

from prometheus_client import Counter, Histogram, Gauge, generate_latest
from opentelemetry import trace
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.resources import Resource

logger = logging.getLogger(__name__)

# Prometheus metrics
REQUEST_COUNT = Counter(
    'http_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint', 'status_code']
)

REQUEST_DURATION = Histogram(
    'http_request_duration_seconds',
    'HTTP request duration in seconds',
    ['method', 'endpoint']
)

ML_JOB_COUNT = Counter(
    'ml_jobs_total',
    'Total ML jobs',
    ['job_type', 'status']
)

ML_JOB_DURATION = Histogram(
    'ml_job_duration_seconds',
    'ML job duration in seconds',
    ['job_type']
)

RAFT_LEADER_ELECTIONS = Counter(
    'raft_leader_elections_total',
    'Total Raft leader elections'
)

RAFT_LOG_ENTRIES = Counter(
    'raft_log_entries_total',
    'Total Raft log entries',
    ['node_id']
)

CHATBOT_CONVERSATIONS = Counter(
    'chatbot_conversations_total',
    'Total chatbot conversations'
)

CHATBOT_RESPONSE_TIME = Histogram(
    'chatbot_response_time_seconds',
    'Chatbot response time in seconds'
)

ACTIVE_NODES = Gauge(
    'distributed_sim_active_nodes',
    'Number of active nodes in distributed simulation'
)


def setup_telemetry(app, jaeger_endpoint: Optional[str] = None):
    """Setup telemetry for the application."""
    
    # Configure OpenTelemetry
    resource = Resource.create({
        "service.name": "agentml-platform",
        "service.version": "1.0.0",
    })
    
    tracer_provider = TracerProvider(resource=resource)
    trace.set_tracer_provider(tracer_provider)
    
    # Setup Jaeger exporter if endpoint is provided
    if jaeger_endpoint:
        jaeger_exporter = JaegerExporter(
            agent_host_name="localhost",
            agent_port=6831,
        )
        span_processor = BatchSpanProcessor(jaeger_exporter)
        tracer_provider.add_span_processor(span_processor)
    
    # Instrument FastAPI
    FastAPIInstrumentor.instrument_app(app, tracer_provider=tracer_provider)
    
    # Add metrics endpoint
    @app.get("/metrics")
    async def get_metrics():
        """Prometheus metrics endpoint."""
        return generate_latest()
    
    logger.info("Telemetry setup completed")


def get_tracer():
    """Get OpenTelemetry tracer."""
    return trace.get_tracer(__name__)


class MetricsCollector:
    """Utility class for collecting custom metrics."""
    
    @staticmethod
    def record_http_request(method: str, endpoint: str, status_code: int, duration: float):
        """Record HTTP request metrics."""
        REQUEST_COUNT.labels(method=method, endpoint=endpoint, status_code=status_code).inc()
        REQUEST_DURATION.labels(method=method, endpoint=endpoint).observe(duration)
    
    @staticmethod
    def record_ml_job(job_type: str, status: str, duration: float):
        """Record ML job metrics."""
        ML_JOB_COUNT.labels(job_type=job_type, status=status).inc()
        ML_JOB_DURATION.labels(job_type=job_type).observe(duration)
    
    @staticmethod
    def record_raft_leader_election():
        """Record Raft leader election."""
        RAFT_LEADER_ELECTIONS.inc()
    
    @staticmethod
    def record_raft_log_entry(node_id: str):
        """Record Raft log entry."""
        RAFT_LOG_ENTRIES.labels(node_id=node_id).inc()
    
    @staticmethod
    def record_chatbot_conversation(response_time: float):
        """Record chatbot conversation."""
        CHATBOT_CONVERSATIONS.inc()
        CHATBOT_RESPONSE_TIME.observe(response_time)
    
    @staticmethod
    def set_active_nodes(count: int):
        """Set active nodes count."""
        ACTIVE_NODES.set(count)


# Global metrics collector instance
metrics = MetricsCollector()
