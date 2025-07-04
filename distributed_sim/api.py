"""
Distributed Systems Simulation API endpoints.
"""

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect, FastAPI, Header, Depends
from pydantic import BaseModel
from typing import Dict, List, Optional, Any
import asyncio
import json
import logging

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


# If using FastAPI app directly
app = FastAPI()
app.include_router(distributed_router)
