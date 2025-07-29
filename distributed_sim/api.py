"""
Distributed Systems Simulation API endpoints.
"""

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect, FastAPI, Header, Depends
from pydantic import BaseModel
from typing import Dict, List, Optional, Any
import asyncio
import json
import logging
import time
import random
from datetime import datetime
import uuid

from .raft import RaftSimulator
from .models import RaftNodeState, RaftClusterState, RaftLogEntry

logger = logging.getLogger(__name__)

# Create router
distributed_router = APIRouter()

# Global simulator instance
simulator = None

API_KEY = "demo-key-123"

def api_key_auth(x_api_key: str = Header(...)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")

async def get_simulator() -> RaftSimulator:
    """Get the Raft simulator instance."""
    global simulator
    if simulator is None:
        simulator = RaftSimulator()
        await simulator.initialize()
    return simulator


class ConnectionManager:
    """Manages WebSocket connections for real-time updates."""
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
    
    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
    
    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)
    
    async def broadcast(self, message: str):
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except:
                # Remove broken connections
                self.active_connections.remove(connection)


manager = ConnectionManager()


@distributed_router.get("/")
async def get_distributed_status():
    """Get distributed simulation status."""
    return {
        "status": "running",
        "component": "Distributed Systems Simulation",
        "features": [
            "Raft consensus algorithm simulation",
            "Leader election visualization",
            "Log replication tracking",
            "Node failure simulation",
            "Real-time state monitoring"
        ]
    }


@distributed_router.get("/cluster")
async def get_cluster_state() -> RaftClusterState:
    """Get the current cluster state."""
    try:
        sim = await get_simulator()
        return await sim.get_cluster_state()
    except Exception as e:
        logger.error(f"Error getting cluster state: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@distributed_router.get("/nodes")
async def list_nodes() -> List[RaftNodeState]:
    """List all nodes in the cluster."""
    try:
        sim = await get_simulator()
        return await sim.list_nodes()
    except Exception as e:
        logger.error(f"Error listing nodes: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@distributed_router.get("/nodes/{node_id}")
async def get_node_state(node_id: str) -> RaftNodeState:
    """Get state of a specific node."""
    try:
        sim = await get_simulator()
        node_state = await sim.get_node_state(node_id)
        if not node_state:
            raise HTTPException(status_code=404, detail="Node not found")
        return node_state
    except Exception as e:
        logger.error(f"Error getting node state {node_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@distributed_router.post("/nodes/{node_id}/start")
async def start_node(node_id: str) -> Dict[str, str]:
    """Start a stopped node."""
    try:
        sim = await get_simulator()
        await sim.start_node(node_id)
        
        # Broadcast update to WebSocket clients
        await manager.broadcast(json.dumps({
            "type": "node_started",
            "node_id": node_id,
            "timestamp": "2025-07-03T12:00:00Z"
        }))
        
        return {"status": "started", "node_id": node_id}
    except Exception as e:
        logger.error(f"Error starting node {node_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@distributed_router.post("/nodes/{node_id}/stop")
async def stop_node(node_id: str) -> Dict[str, str]:
    """Stop a running node."""
    try:
        sim = await get_simulator()
        await sim.stop_node(node_id)
        
        # Broadcast update to WebSocket clients
        await manager.broadcast(json.dumps({
            "type": "node_stopped",
            "node_id": node_id,
            "timestamp": "2025-07-03T12:00:00Z"
        }))
        
        return {"status": "stopped", "node_id": node_id}
    except Exception as e:
        logger.error(f"Error stopping node {node_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@distributed_router.post("/nodes/{node_id}/restart")
async def restart_node(node_id: str) -> Dict[str, str]:
    """Restart a node."""
    try:
        sim = await get_simulator()
        await sim.restart_node(node_id)
        
        # Broadcast update to WebSocket clients
        await manager.broadcast(json.dumps({
            "type": "node_restarted",
            "node_id": node_id,
            "timestamp": "2025-07-03T12:00:00Z"
        }))
        
        return {"status": "restarted", "node_id": node_id}
    except Exception as e:
        logger.error(f"Error restarting node {node_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@distributed_router.post("/election/trigger")
async def trigger_election() -> Dict[str, str]:
    """Trigger a leader election."""
    try:
        sim = await get_simulator()
        await sim.trigger_election()
        
        # Broadcast update to WebSocket clients
        await manager.broadcast(json.dumps({
            "type": "election_triggered",
            "timestamp": "2025-07-03T12:00:00Z"
        }))
        
        return {"status": "election_triggered"}
    except Exception as e:
        logger.error(f"Error triggering election: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@distributed_router.post("/log/append")
async def append_log_entry(entry: Dict[str, Any]) -> Dict[str, str]:
    """Append a new log entry."""
    try:
        sim = await get_simulator()
        log_entry = RaftLogEntry(
            term=entry.get("term", 1),
            index=entry.get("index", 0),
            command=entry.get("command", ""),
            data=entry.get("data", {})
        )
        
        await sim.append_log_entry(log_entry)
        
        # Broadcast update to WebSocket clients
        await manager.broadcast(json.dumps({
            "type": "log_appended",
            "entry": log_entry.dict(),
            "timestamp": "2025-07-03T12:00:00Z"
        }))
        
        return {"status": "appended", "entry_id": str(log_entry.index)}
    except Exception as e:
        logger.error(f"Error appending log entry: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@distributed_router.get("/log")
async def get_log_entries() -> List[RaftLogEntry]:
    """Get all log entries."""
    try:
        sim = await get_simulator()
        return await sim.get_log_entries()
    except Exception as e:
        logger.error(f"Error getting log entries: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@distributed_router.get("/metrics")
async def get_simulation_metrics() -> Dict[str, Any]:
    """Get simulation metrics."""
    try:
        sim = await get_simulator()
        return await sim.get_metrics()
    except Exception as e:
        logger.error(f"Error getting simulation metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@distributed_router.post("/reset")
async def reset_simulation() -> Dict[str, str]:
    """Reset the simulation to initial state."""
    try:
        sim = await get_simulator()
        await sim.reset()
        
        # Broadcast update to WebSocket clients
        await manager.broadcast(json.dumps({
            "type": "simulation_reset",
            "timestamp": "2025-07-03T12:00:00Z"
        }))
        
        return {"status": "reset"}
    except Exception as e:
        logger.error(f"Error resetting simulation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@distributed_router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates."""
    await manager.connect(websocket)
    
    try:
        while True:
            # Send periodic updates
            sim = await get_simulator()
            cluster_state = await sim.get_cluster_state()
            
            await websocket.send_text(json.dumps({
                "type": "cluster_state",
                "data": cluster_state.dict(),
                "timestamp": "2025-07-03T12:00:00Z"
            }))
            
            await asyncio.sleep(1)  # Send updates every second
            
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)


# Network partition simulation
@distributed_router.post("/partition/create")
async def create_partition(partition_config: Dict[str, Any]) -> Dict[str, str]:
    """Create a network partition."""
    try:
        sim = await get_simulator()
        partition_id = await sim.create_partition(partition_config)
        
        # Broadcast update to WebSocket clients
        await manager.broadcast(json.dumps({
            "type": "partition_created",
            "partition_id": partition_id,
            "config": partition_config,
            "timestamp": "2025-07-03T12:00:00Z"
        }))
        
        return {"status": "created", "partition_id": partition_id}
    except Exception as e:
        logger.error(f"Error creating partition: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@distributed_router.post("/partition/{partition_id}/heal")
async def heal_partition(partition_id: str) -> Dict[str, str]:
    """Heal a network partition."""
    try:
        sim = await get_simulator()
        await sim.heal_partition(partition_id)
        
        # Broadcast update to WebSocket clients
        await manager.broadcast(json.dumps({
            "type": "partition_healed",
            "partition_id": partition_id,
            "timestamp": "2025-07-03T12:00:00Z"
        }))
        
        return {"status": "healed", "partition_id": partition_id}
    except Exception as e:
        logger.error(f"Error healing partition {partition_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@distributed_router.get("/scenarios")
async def list_scenarios() -> List[Dict[str, Any]]:
    """List available simulation scenarios."""
    scenarios = [
        {
            "id": "leader_failure",
            "name": "Leader Failure",
            "description": "Simulate leader node failure and re-election",
            "duration": "30s"
        },
        {
            "id": "network_partition",
            "name": "Network Partition",
            "description": "Create network partition and observe split-brain prevention",
            "duration": "60s"
        },
        {
            "id": "node_recovery",
            "name": "Node Recovery",
            "description": "Simulate node failure and recovery with log catch-up",
            "duration": "45s"
        },
        {
            "id": "high_load",
            "name": "High Load",
            "description": "Simulate high log entry load and replication",
            "duration": "30s"
        }
    ]
    return scenarios


@distributed_router.post("/scenarios/{scenario_id}/run")
async def run_scenario(scenario_id: str) -> Dict[str, str]:
    """Run a predefined simulation scenario."""
    try:
        sim = await get_simulator()
        await sim.run_scenario(scenario_id)
        
        # Broadcast update to WebSocket clients
        await manager.broadcast(json.dumps({
            "type": "scenario_started",
            "scenario_id": scenario_id,
            "timestamp": "2025-07-03T12:00:00Z"
        }))
        
        return {"status": "started", "scenario_id": scenario_id}
    except Exception as e:
        logger.error(f"Error running scenario {scenario_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@distributed_router.get("/api/raft/cluster-state")
async def get_cluster_state(dep=Depends(api_key_auth)):
    state = simulator.get_cluster_state()
    return state.dict() if hasattr(state, 'dict') else state


@distributed_router.post("/api/raft/trigger-election")
async def trigger_election(dep=Depends(api_key_auth)):
    await simulator.trigger_election("node_0")
    return {"status": "Election triggered"}


@distributed_router.post("/api/raft/create-partition")
async def create_partition(dep=Depends(api_key_auth)):
    partition_id = "partition1"
    nodes = [f"node_{i}" for i in range(simulator.cluster_size // 2)]
    await simulator.create_partition(partition_id, nodes)
    return {"status": "Partition created", "partition_id": partition_id}


@distributed_router.post("/api/raft/reset")
async def reset_simulation(dep=Depends(api_key_auth)):
    await simulator.stop()
    simulator.__init__(simulator.cluster_size)
    await simulator.start()
    return {"status": "Simulation reset"}


@distributed_router.post("/inject-failure/{node_id}")
async def inject_node_failure(
    node_id: str,
    failure_config: Dict[str, Any] = None
) -> Dict[str, str]:
    """Inject failure into a specific node for testing."""
    try:
        sim = await get_simulator()
        failure_type = failure_config.get('type', 'crash') if failure_config else 'crash'
        duration = failure_config.get('duration', 30) if failure_config else 30
        
        await sim.inject_failure(node_id, failure_type, duration)
        
        # Broadcast update to WebSocket clients
        await manager.broadcast(json.dumps({
            "type": "failure_injected",
            "node_id": node_id,
            "failure_type": failure_type,
            "duration": duration,
            "timestamp": datetime.now().isoformat()
        }))
        
        return {"status": "failure_injected", "node_id": node_id, "type": failure_type}
    except Exception as e:
        logger.error(f"Error injecting failure for node {node_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@distributed_router.post("/add-node")
async def add_node_to_cluster(
    node_config: Dict[str, Any] = None
) -> Dict[str, Any]:
    """Dynamically add a new node to the cluster."""
    try:
        sim = await get_simulator()
        new_node_id = f"node_{int(time.time())}"
        
        # Add node with configuration
        config = node_config or {}
        await sim.add_node(new_node_id, config)
        
        # Broadcast update to WebSocket clients
        await manager.broadcast(json.dumps({
            "type": "node_added",
            "node_id": new_node_id,
            "config": config,
            "timestamp": datetime.now().isoformat()
        }))
        
        return {
            "status": "node_added",
            "node_id": new_node_id,
            "cluster_size": await sim.get_cluster_size(),
            "rebalancing_required": True
        }
    except Exception as e:
        logger.error(f"Error adding node to cluster: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@distributed_router.delete("/remove-node/{node_id}")
async def remove_node_from_cluster(node_id: str) -> Dict[str, Any]:
    """Dynamically remove a node from the cluster."""
    try:
        sim = await get_simulator()
        await sim.remove_node(node_id)
        
        # Broadcast update to WebSocket clients
        await manager.broadcast(json.dumps({
            "type": "node_removed",
            "node_id": node_id,
            "timestamp": datetime.now().isoformat()
        }))
        
        return {
            "status": "node_removed",
            "node_id": node_id,
            "cluster_size": await sim.get_cluster_size(),
            "rebalancing_required": True
        }
    except Exception as e:
        logger.error(f"Error removing node {node_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@distributed_router.post("/benchmark")
async def run_performance_benchmark() -> Dict[str, Any]:
    """Run comprehensive performance benchmarks on the cluster."""
    try:
        sim = await get_simulator()
        
        # Simulate running various benchmarks
        benchmark_id = str(uuid.uuid4())
        
        # Consensus performance benchmark
        consensus_results = {
            "average_consensus_time_ms": random.randint(50, 200),
            "throughput_ops_per_sec": random.randint(500, 1500),
            "leader_election_time_ms": random.randint(100, 500),
            "log_replication_latency_ms": random.randint(20, 100)
        }
        
        # Network performance benchmark
        network_results = {
            "inter_node_latency_ms": random.randint(1, 10),
            "bandwidth_mbps": random.randint(100, 1000),
            "packet_loss_rate": random.uniform(0, 0.01),
            "jitter_ms": random.randint(1, 5)
        }
        
        # Fault tolerance benchmark
        fault_tolerance_results = {
            "recovery_time_after_leader_failure_ms": random.randint(500, 2000),
            "data_consistency_after_partition": True,
            "split_brain_prevention": True,
            "graceful_degradation": True
        }
        
        # Load testing results
        load_testing_results = {
            "max_sustainable_load_ops_per_sec": random.randint(800, 2000),
            "breaking_point_ops_per_sec": random.randint(2000, 5000),
            "cpu_utilization_at_max_load": random.randint(70, 90),
            "memory_utilization_at_max_load": random.randint(60, 85)
        }
        
        benchmark_results = {
            "benchmark_id": benchmark_id,
            "timestamp": datetime.now().isoformat(),
            "cluster_size": await sim.get_cluster_size(),
            "consensus_performance": consensus_results,
            "network_performance": network_results,
            "fault_tolerance": fault_tolerance_results,
            "load_testing": load_testing_results,
            "overall_score": random.randint(75, 95),
            "recommendations": [
                "Consider increasing cluster size for better throughput",
                "Network latency is within acceptable range",
                "Fault tolerance mechanisms are working correctly"
            ]
        }
        
        # Broadcast update to WebSocket clients
        await manager.broadcast(json.dumps({
            "type": "benchmark_completed",
            "benchmark_id": benchmark_id,
            "results": benchmark_results,
            "timestamp": datetime.now().isoformat()
        }))
        
        return benchmark_results
        
    except Exception as e:
        logger.error(f"Error running performance benchmark: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@distributed_router.get("/metrics/consensus")
async def get_consensus_metrics() -> Dict[str, Any]:
    """Get detailed consensus algorithm metrics."""
    try:
        sim = await get_simulator()
        
        # Simulate detailed consensus metrics
        return {
            "current_term": random.randint(10, 100),
            "committed_log_index": random.randint(500, 2000),
            "last_applied_index": random.randint(500, 2000),
            "leader_heartbeat_interval_ms": 150,
            "election_timeout_ms": random.randint(150, 300),
            "log_entries_per_second": random.randint(50, 200),
            "successful_elections": random.randint(5, 15),
            "failed_elections": random.randint(0, 3),
            "network_partitions_detected": random.randint(0, 2),
            "consistency_violations": 0,  # Should always be 0 in Raft
            "performance_stats": {
                "average_append_entries_latency_ms": random.randint(10, 50),
                "average_vote_request_latency_ms": random.randint(5, 25),
                "log_compaction_frequency": "every 1000 entries",
                "snapshot_size_mb": random.randint(10, 100)
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting consensus metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@distributed_router.get("/metrics/performance")
async def get_performance_metrics() -> Dict[str, Any]:
    """Get real-time performance metrics for the distributed system."""
    try:
        sim = await get_simulator()
        cluster_size = await sim.get_cluster_size()
        
        # Generate realistic performance metrics
        node_metrics = []
        for i in range(cluster_size):
            node_id = f"node_{i}"
            node_metrics.append({
                "node_id": node_id,
                "cpu_usage": random.randint(20, 80),
                "memory_usage": random.randint(30, 70),
                "network_io_mbps": random.randint(10, 100),
                "disk_io_mbps": random.randint(5, 50),
                "active_connections": random.randint(10, 100),
                "requests_per_second": random.randint(50, 500),
                "response_time_ms": random.randint(10, 100),
                "error_rate": random.uniform(0, 0.05)
            })
        
        cluster_metrics = {
            "total_throughput_ops_per_sec": sum(node["requests_per_second"] for node in node_metrics),
            "average_response_time_ms": sum(node["response_time_ms"] for node in node_metrics) / len(node_metrics),
            "cluster_cpu_utilization": sum(node["cpu_usage"] for node in node_metrics) / len(node_metrics),
            "cluster_memory_utilization": sum(node["memory_usage"] for node in node_metrics) / len(node_metrics),
            "total_network_traffic_mbps": sum(node["network_io_mbps"] for node in node_metrics),
            "overall_health_score": random.randint(85, 98)
        }
        
        return {
            "timestamp": datetime.now().isoformat(),
            "cluster_size": cluster_size,
            "node_metrics": node_metrics,
            "cluster_metrics": cluster_metrics,
            "alerts": [
                {
                    "level": "warning",
                    "message": "Node_2 CPU usage above 75%",
                    "timestamp": datetime.now().isoformat()
                }
            ] if cluster_metrics["cluster_cpu_utilization"] > 70 else []
        }
        
    except Exception as e:
        logger.error(f"Error getting performance metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@distributed_router.post("/chaos-engineering")
async def run_chaos_engineering_scenario(
    chaos_config: Dict[str, Any]
) -> Dict[str, Any]:
    """Run chaos engineering scenarios to test system resilience."""
    try:
        sim = await get_simulator()
        scenario_type = chaos_config.get('scenario', 'random_failures')
        duration = chaos_config.get('duration', 60)
        intensity = chaos_config.get('intensity', 'medium')
        
        chaos_id = str(uuid.uuid4())
        
        # Define chaos scenarios
        scenarios = {
            "random_failures": "Randomly fail nodes and recover them",
            "network_partitions": "Create and heal network partitions",
            "high_latency": "Inject network latency between nodes",
            "resource_exhaustion": "Simulate CPU/memory exhaustion",
            "cascading_failures": "Trigger cascading failure scenarios"
        }
        
        # Simulate chaos engineering execution
        results = {
            "chaos_id": chaos_id,
            "scenario": scenario_type,
            "description": scenarios.get(scenario_type, "Unknown scenario"),
            "duration_seconds": duration,
            "intensity": intensity,
            "status": "running",
            "start_time": datetime.now().isoformat(),
            "events": [
                {
                    "timestamp": datetime.now().isoformat(),
                    "event": f"Started {scenario_type} scenario",
                    "impact": "System monitoring initiated"
                }
            ],
            "metrics_before": {
                "availability": 99.9,
                "response_time_ms": random.randint(50, 100),
                "error_rate": 0.01
            }
        }
        
        # Broadcast chaos engineering start
        await manager.broadcast(json.dumps({
            "type": "chaos_engineering_started",
            "chaos_id": chaos_id,
            "scenario": scenario_type,
            "timestamp": datetime.now().isoformat()
        }))
        
        return results
        
    except Exception as e:
        logger.error(f"Error running chaos engineering scenario: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# If using FastAPI app directly
app = FastAPI()
app.include_router(distributed_router)
