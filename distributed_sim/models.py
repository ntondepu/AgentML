"""
Pydantic models for the Distributed Systems Simulation component.
"""

from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
from datetime import datetime
from enum import Enum


class NodeStatus(str, Enum):
    """Node status in the cluster."""
    RUNNING = "running"
    STOPPED = "stopped"
    CRASHED = "crashed"
    RECOVERING = "recovering"


class NodeRole(str, Enum):
    """Node role in Raft consensus."""
    FOLLOWER = "follower"
    CANDIDATE = "candidate"
    LEADER = "leader"


class LogEntryType(str, Enum):
    """Types of log entries."""
    COMMAND = "command"
    CONFIGURATION = "configuration"
    HEARTBEAT = "heartbeat"


class RaftLogEntry(BaseModel):
    """Raft log entry model."""
    term: int = Field(..., description="Term when entry was created")
    index: int = Field(..., description="Log index")
    command: str = Field(..., description="Command to execute")
    data: Dict[str, Any] = Field(default_factory=dict, description="Entry data")
    timestamp: datetime = Field(default_factory=datetime.now, description="Entry timestamp")
    committed: bool = Field(default=False, description="Whether entry is committed")
    
    class Config:
        schema_extra = {
            "example": {
                "term": 1,
                "index": 5,
                "command": "set_value",
                "data": {"key": "user_count", "value": 1000},
                "timestamp": "2025-07-03T12:00:00Z",
                "committed": True
            }
        }


class RaftNodeState(BaseModel):
    """State of a Raft node."""
    id: str = Field(..., description="Node identifier")
    role: NodeRole = Field(..., description="Current node role")
    status: NodeStatus = Field(..., description="Node status")
    current_term: int = Field(..., description="Current term")
    voted_for: Optional[str] = Field(default=None, description="Candidate voted for in current term")
    log: List[RaftLogEntry] = Field(default_factory=list, description="Node's log entries")
    commit_index: int = Field(default=0, description="Index of highest log entry known to be committed")
    last_applied: int = Field(default=0, description="Index of highest log entry applied to state machine")
    next_index: Dict[str, int] = Field(default_factory=dict, description="Next log index to send to each follower")
    match_index: Dict[str, int] = Field(default_factory=dict, description="Highest log index known to be replicated")
    last_heartbeat: Optional[datetime] = Field(default=None, description="Last heartbeat timestamp")
    election_timeout: float = Field(default=1.5, description="Election timeout in seconds")
    heartbeat_interval: float = Field(default=0.5, description="Heartbeat interval in seconds")
    
    class Config:
        schema_extra = {
            "example": {
                "id": "node_1",
                "role": "leader",
                "status": "running",
                "current_term": 5,
                "voted_for": None,
                "log": [],
                "commit_index": 10,
                "last_applied": 10,
                "next_index": {"node_2": 11, "node_3": 11},
                "match_index": {"node_2": 10, "node_3": 10},
                "last_heartbeat": "2025-07-03T12:00:00Z",
                "election_timeout": 1.5,
                "heartbeat_interval": 0.5
            }
        }


class RaftClusterState(BaseModel):
    """Overall cluster state."""
    nodes: List[RaftNodeState] = Field(..., description="All nodes in the cluster")
    leader_id: Optional[str] = Field(default=None, description="Current leader node ID")
    current_term: int = Field(..., description="Current cluster term")
    total_nodes: int = Field(..., description="Total number of nodes")
    active_nodes: int = Field(..., description="Number of active nodes")
    log_length: int = Field(..., description="Length of committed log")
    last_election: Optional[datetime] = Field(default=None, description="Last election timestamp")
    cluster_health: str = Field(..., description="Overall cluster health status")
    
    class Config:
        schema_extra = {
            "example": {
                "nodes": [],
                "leader_id": "node_1",
                "current_term": 5,
                "total_nodes": 5,
                "active_nodes": 4,
                "log_length": 15,
                "last_election": "2025-07-03T11:45:00Z",
                "cluster_health": "healthy"
            }
        }


class NetworkPartition(BaseModel):
    """Network partition configuration."""
    id: str = Field(..., description="Partition identifier")
    name: str = Field(..., description="Partition name")
    affected_nodes: List[str] = Field(..., description="Nodes affected by partition")
    partition_type: str = Field(..., description="Type of partition (split, isolate)")
    created_at: datetime = Field(..., description="Partition creation timestamp")
    active: bool = Field(default=True, description="Whether partition is active")
    
    class Config:
        schema_extra = {
            "example": {
                "id": "partition_1",
                "name": "Split Brain Test",
                "affected_nodes": ["node_1", "node_2"],
                "partition_type": "split",
                "created_at": "2025-07-03T12:00:00Z",
                "active": True
            }
        }


class SimulationMetrics(BaseModel):
    """Simulation performance metrics."""
    total_elections: int = Field(default=0, description="Total number of elections")
    successful_elections: int = Field(default=0, description="Number of successful elections")
    failed_elections: int = Field(default=0, description="Number of failed elections")
    total_log_entries: int = Field(default=0, description="Total log entries created")
    committed_entries: int = Field(default=0, description="Number of committed entries")
    average_election_time: float = Field(default=0.0, description="Average election time in seconds")
    average_commit_time: float = Field(default=0.0, description="Average commit time in seconds")
    network_partitions: int = Field(default=0, description="Number of network partitions")
    node_failures: int = Field(default=0, description="Number of node failures")
    uptime_percentage: float = Field(default=100.0, description="Cluster uptime percentage")
    
    class Config:
        schema_extra = {
            "example": {
                "total_elections": 3,
                "successful_elections": 3,
                "failed_elections": 0,
                "total_log_entries": 50,
                "committed_entries": 48,
                "average_election_time": 0.8,
                "average_commit_time": 0.2,
                "network_partitions": 1,
                "node_failures": 2,
                "uptime_percentage": 96.5
            }
        }


class SimulationScenario(BaseModel):
    """Simulation scenario configuration."""
    id: str = Field(..., description="Scenario identifier")
    name: str = Field(..., description="Scenario name")
    description: str = Field(..., description="Scenario description")
    duration: int = Field(..., description="Scenario duration in seconds")
    steps: List[Dict[str, Any]] = Field(..., description="Scenario steps")
    expected_outcomes: List[str] = Field(..., description="Expected outcomes")
    
    class Config:
        schema_extra = {
            "example": {
                "id": "leader_failure",
                "name": "Leader Failure Scenario",
                "description": "Simulate leader node failure and observe re-election",
                "duration": 30,
                "steps": [
                    {"action": "stop_node", "node_id": "leader", "delay": 5},
                    {"action": "wait", "duration": 10},
                    {"action": "start_node", "node_id": "leader", "delay": 15}
                ],
                "expected_outcomes": [
                    "New leader elected within 2 seconds",
                    "No data loss",
                    "Cluster remains available"
                ]
            }
        }


class HeartbeatMessage(BaseModel):
    """Heartbeat message between nodes."""
    from_node: str = Field(..., description="Source node ID")
    to_node: str = Field(..., description="Target node ID")
    term: int = Field(..., description="Leader's term")
    leader_id: str = Field(..., description="Leader's ID")
    prev_log_index: int = Field(..., description="Index of log entry immediately preceding new ones")
    prev_log_term: int = Field(..., description="Term of prev_log_index entry")
    entries: List[RaftLogEntry] = Field(default_factory=list, description="Log entries to store")
    leader_commit: int = Field(..., description="Leader's commit index")
    timestamp: datetime = Field(default_factory=datetime.now, description="Message timestamp")


class VoteRequest(BaseModel):
    """Vote request message."""
    from_node: str = Field(..., description="Candidate's ID")
    to_node: str = Field(..., description="Target node ID")
    term: int = Field(..., description="Candidate's term")
    candidate_id: str = Field(..., description="Candidate requesting vote")
    last_log_index: int = Field(..., description="Index of candidate's last log entry")
    last_log_term: int = Field(..., description="Term of candidate's last log entry")
    timestamp: datetime = Field(default_factory=datetime.now, description="Message timestamp")


class VoteResponse(BaseModel):
    """Vote response message."""
    from_node: str = Field(..., description="Responding node ID")
    to_node: str = Field(..., description="Candidate node ID")
    term: int = Field(..., description="Current term")
    vote_granted: bool = Field(..., description="Whether vote was granted")
    timestamp: datetime = Field(default_factory=datetime.now, description="Message timestamp")


class MessageLog(BaseModel):
    """Log of messages between nodes."""
    id: str = Field(..., description="Message ID")
    message_type: str = Field(..., description="Type of message")
    from_node: str = Field(..., description="Source node")
    to_node: str = Field(..., description="Target node")
    content: Dict[str, Any] = Field(..., description="Message content")
    timestamp: datetime = Field(default_factory=datetime.now, description="Message timestamp")
    delivered: bool = Field(default=True, description="Whether message was delivered")
    
    class Config:
        schema_extra = {
            "example": {
                "id": "msg_123",
                "message_type": "heartbeat",
                "from_node": "node_1",
                "to_node": "node_2",
                "content": {"term": 5, "commit_index": 10},
                "timestamp": "2025-07-03T12:00:00Z",
                "delivered": True
            }
        }
