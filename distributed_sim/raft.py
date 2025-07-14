"""
Raft Consensus Algorithm Simulator for Distributed Systems Education.
"""

import asyncio
import logging
import random
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set
import json

from .models import (
    RaftNodeState, 
    RaftClusterState, 
    RaftLogEntry,
    NodeStatus,
    NodeRole,
    NetworkPartition,
    SimulationMetrics,
    HeartbeatMessage,
    VoteRequest,
    VoteResponse,
    MessageLog
)
from monitoring.telemetry import metrics
from config import settings

logger = logging.getLogger(__name__)


class RaftNode:
    """Individual Raft node implementation."""
    
    def __init__(self, node_id: str, cluster_size: int):
        self.id = node_id
        self.cluster_size = cluster_size
        self.role = NodeRole.FOLLOWER
        self.status = NodeStatus.RUNNING
        self.current_term = 0
        self.voted_for: Optional[str] = None
        self.log: List[RaftLogEntry] = []
        self.commit_index = 0
        self.last_applied = 0
        
        # Leader state
        self.next_index: Dict[str, int] = {}
        self.match_index: Dict[str, int] = {}
        
        # Timing
        self.last_heartbeat = datetime.now()
        self.election_timeout = random.uniform(0.15, 0.3)  # seconds
        self.heartbeat_interval = 0.05  # seconds
        
        # Simulation tracking
        self.message_log: List[MessageLog] = []
        self.votes_received: Set[str] = set()
        self.partitioned_from: Set[str] = set()
        
    def reset_election_timeout(self):
        self.election_timeout = random.uniform(0.15, 0.3)
        self.last_heartbeat = datetime.now()

    def is_election_timeout(self) -> bool:
        return (datetime.now() - self.last_heartbeat).total_seconds() > self.election_timeout

    def start_election(self):
        self.current_term += 1
        self.role = NodeRole.CANDIDATE
        self.voted_for = self.id
        self.votes_received = {self.id}
        self.reset_election_timeout()
        logger.info(f"Node {self.id} starting election for term {self.current_term}")

    def become_leader(self):
        self.role = NodeRole.LEADER
        self.next_index = {node_id: len(self.log) for node_id in self.get_cluster_nodes() if node_id != self.id}
        self.match_index = {node_id: 0 for node_id in self.get_cluster_nodes() if node_id != self.id}
        logger.info(f"Node {self.id} became leader for term {self.current_term}")

    def become_follower(self, term: int):
        self.current_term = term
        self.role = NodeRole.FOLLOWER
        self.voted_for = None
        self.votes_received.clear()
        self.reset_election_timeout()

    def get_cluster_nodes(self) -> List[str]:
        """Get list of all node IDs in cluster."""
        return [f"node_{i}" for i in range(self.cluster_size)]
        
    def append_log_entry(self, entry: RaftLogEntry):
        """Append entry to log."""
        self.log.append(entry)
        
    def handle_vote_request(self, request: VoteRequest) -> VoteResponse:
        """Handle vote request from candidate."""
        vote_granted = False
        
        if request.term > self.current_term:
            self.become_follower(request.term)
            
        if (request.term == self.current_term and 
            (self.voted_for is None or self.voted_for == request.candidate_id)):
            # Check if candidate's log is at least as up-to-date
            last_log_term = self.log[-1].term if self.log else 0
            last_log_index = len(self.log) - 1
            
            if (request.last_log_term > last_log_term or 
                (request.last_log_term == last_log_term and 
                 request.last_log_index >= last_log_index)):
                vote_granted = True
                self.voted_for = request.candidate_id
                self.reset_election_timeout()
                
        return VoteResponse(
            node_id=self.id,
            term=self.current_term,
            vote_granted=vote_granted
        )
        
    def handle_vote_response(self, response: VoteResponse):
        """Handle vote response."""
        if response.term > self.current_term:
            self.become_follower(response.term)
            return
            
        if (self.role == NodeRole.CANDIDATE and 
            response.term == self.current_term and 
            response.vote_granted):
            self.votes_received.add(response.node_id)
            
            # Check if we have majority
            if len(self.votes_received) > self.cluster_size // 2:
                self.become_leader()
                
    def handle_heartbeat(self, message: HeartbeatMessage):
        """Handle heartbeat from leader."""
        if message.term >= self.current_term:
            self.become_follower(message.term)
            self.reset_election_timeout()
            
    def get_state(self) -> RaftNodeState:
        """Get current node state."""
        return RaftNodeState(
            node_id=self.id,
            role=self.role,
            status=self.status,
            current_term=self.current_term,
            voted_for=self.voted_for,
            log_length=len(self.log),
            commit_index=self.commit_index,
            last_applied=self.last_applied,
            is_partitioned=len(self.partitioned_from) > 0
        )


class RaftSimulator:
    """Raft consensus algorithm simulator."""
    
    def __init__(self):
        self.nodes: Dict[str, RaftNode] = {}
        self.partitions: Dict[str, NetworkPartition] = {}
        self.message_log: List[MessageLog] = []
        self.metrics = SimulationMetrics()
        self.running = False
        self.simulation_task: Optional[asyncio.Task] = None
        self._initialized = False
    
    async def initialize(self):
        """Initialize the Raft simulator."""
        if self._initialized:
            return
        
        logger.info("Initializing Raft Simulator...")
        
        # Create nodes
        cluster_size = getattr(settings, 'raft_node_count', 5)
        for i in range(cluster_size):
            node_id = f"node_{i}"
            self.nodes[node_id] = RaftNode(node_id, cluster_size)
        
        # Start simulation
        await self.start_simulation()
        
        self._initialized = True
        logger.info(f"Raft Simulator initialized with {len(self.nodes)} nodes")
    
    async def start_simulation(self):
        """Start the simulation loop."""
        if self.running:
            return
        
        self.running = True
        self.simulation_task = asyncio.create_task(self._simulation_loop())
        logger.info("Raft simulation started")
    
    async def stop_simulation(self):
        """Stop the simulation."""
        self.running = False
        if self.simulation_task:
            self.simulation_task.cancel()
            try:
                await self.simulation_task
            except asyncio.CancelledError:
                pass
        logger.info("Raft simulation stopped")
    
    async def _simulation_loop(self):
        """Main simulation loop."""
        while self.running:
            try:
                # Check for election timeouts
                for node in self.nodes.values():
                    if node.status == NodeStatus.RUNNING:
                        if node.role == NodeRole.FOLLOWER and node.is_election_timeout():
                            await self._start_election(node)
                        elif node.role == NodeRole.CANDIDATE and node.is_election_timeout():
                            await self._start_election(node)  # Restart election
                
                # Send heartbeats from leaders
                for node in self.nodes.values():
                    if node.role == NodeRole.LEADER and node.status == NodeStatus.RUNNING:
                        await self._send_heartbeats(node)
                
                # Update metrics
                await self._update_metrics()
                
                await asyncio.sleep(0.05)  # 50ms simulation tick
                
            except Exception as e:
                logger.error(f"Error in simulation loop: {e}")
                await asyncio.sleep(1)
    
    async def _start_election(self, candidate: RaftNode):
        """Start leader election."""
        candidate.start_election()
        
        # Send vote requests to all other nodes
        for node_id, node in self.nodes.items():
            if node_id != candidate.id and node.status == NodeStatus.RUNNING:
                if not self._is_partitioned(candidate.id, node_id):
                    await self._send_vote_request(candidate, node)
        
        # Check if already won (single node cluster)
        if len(candidate.votes_received) > len(self.nodes) // 2:
            candidate.become_leader()
            await self._log_message("election", candidate.id, "broadcast", {
                "result": "won",
                "term": candidate.current_term,
                "votes": len(candidate.votes_received)
            })
    
    async def _send_vote_request(self, candidate: RaftNode, target: RaftNode):
        """Send vote request to target node."""
        last_log_index = len(candidate.log) - 1
        last_log_term = candidate.log[-1].term if candidate.log else 0

        # Build vote request
        vote_request = VoteRequest(
            term=candidate.current_term,
            candidate_id=candidate.id,
            last_log_term=last_log_term,
            last_log_index=last_log_index
        )
        # Target handles vote request
        vote_response = target.handle_vote_request(vote_request)

        # Log the vote request
        await self._log_message("vote_request", candidate.id, target.id, {
            "term": candidate.current_term,
            "granted": vote_response.vote_granted
        })

        if vote_response.vote_granted:
            candidate.votes_received.add(target.id)
            # Check if candidate has won majority after this vote
            if len(candidate.votes_received) > len(self.nodes) // 2:
                candidate.become_leader()
                await self._log_message("election", candidate.id, "broadcast", {
                    "result": "won",
                    "term": candidate.current_term,
                    "votes": len(candidate.votes_received)
                })
                # Notify other nodes to become followers
                for node in self.nodes.values():
                    if node.id != candidate.id:
                        node.become_follower(candidate.current_term)
    
    async def _send_heartbeats(self, leader: RaftNode):
        """Send heartbeats from leader to all followers."""
        for node_id, node in self.nodes.items():
            if node_id != leader.id and node.status == NodeStatus.RUNNING:
                if not self._is_partitioned(leader.id, node_id):
                    # Send heartbeat
                    node.reset_election_timeout()
                    if node.current_term < leader.current_term:
                        node.become_follower(leader.current_term)
                    
                    await self._log_message("heartbeat", leader.id, node_id, {
                        "term": leader.current_term,
                        "commit_index": leader.commit_index
                    })
    
    def _is_partitioned(self, node1: str, node2: str) -> bool:
        """Check if two nodes are partitioned."""
        for partition in self.partitions.values():
            if partition.active:
                if node1 in partition.nodes and node2 not in partition.nodes:
                    return True
                if node2 in partition.nodes and node1 not in partition.nodes:
                    return True
        return False
    
    async def _log_message(self, msg_type: str, sender: str, receiver: str, data: Dict[str, Any]):
        """Log a message for debugging."""
        message = MessageLog(
            timestamp=datetime.now(),
            message_type=msg_type,
            sender=sender,
            receiver=receiver,
            data=data
        )
        self.message_log.append(message)
        
        # Keep only last 1000 messages
        if len(self.message_log) > 1000:
            self.message_log = self.message_log[-1000:]
    
    async def _update_metrics(self):
        """Update simulation metrics."""
        # Count nodes by role
        leader_count = sum(1 for node in self.nodes.values() if node.role == NodeRole.LEADER)
        candidate_count = sum(1 for node in self.nodes.values() if node.role == NodeRole.CANDIDATE)
        follower_count = sum(1 for node in self.nodes.values() if node.role == NodeRole.FOLLOWER)
        
        # Count running nodes
        running_count = sum(1 for node in self.nodes.values() if node.status == NodeStatus.RUNNING)
        
        # Get current term
        current_term = max(node.current_term for node in self.nodes.values()) if self.nodes else 0
        
        # Update metrics
        self.metrics.leader_count = leader_count
        self.metrics.candidate_count = candidate_count
        self.metrics.follower_count = follower_count
        self.metrics.running_nodes = running_count
        self.metrics.current_term = current_term
        self.metrics.active_partitions = len([p for p in self.partitions.values() if p.active])
    
    def get_cluster_state(self) -> RaftClusterState:
        """Get current cluster state."""
        return RaftClusterState(
            nodes=[node.get_state() for node in self.nodes.values()],
            leader_id=next((node.id for node in self.nodes.values() if node.role == NodeRole.LEADER), None),
            current_term=max(node.current_term for node in self.nodes.values()) if self.nodes else 0,
            message_log=self.message_log[-100:],  # Last 100 messages
            metrics=self.metrics
        )
    
    async def create_partition(self, partition_id: str, nodes: List[str]) -> NetworkPartition:
        """Create a network partition."""
        partition = NetworkPartition(
            id=partition_id,
            nodes=nodes,
            active=True,
            created_at=datetime.now()
        )
        self.partitions[partition_id] = partition
        
        # Update node partition state
        for node_id in nodes:
            if node_id in self.nodes:
                self.nodes[node_id].partitioned_from = set(self.nodes.keys()) - set(nodes)
        
        logger.info(f"Created partition {partition_id} with nodes: {nodes}")
        return partition
    
    async def heal_partition(self, partition_id: str):
        """Heal a network partition."""
        if partition_id in self.partitions:
            partition = self.partitions[partition_id]
            partition.active = False
            
            # Update node partition state
            for node_id in partition.nodes:
                if node_id in self.nodes:
                    self.nodes[node_id].partitioned_from.clear()
            
            logger.info(f"Healed partition {partition_id}")
    
    async def stop_node(self, node_id: str):
        """Stop a node."""
        if node_id in self.nodes:
            self.nodes[node_id].status = NodeStatus.STOPPED
            logger.info(f"Stopped node {node_id}")
    
    async def start_node(self, node_id: str):
        """Start a node."""
        if node_id in self.nodes:
            self.nodes[node_id].status = NodeStatus.RUNNING
            self.nodes[node_id].reset_election_timeout()
            logger.info(f"Started node {node_id}")
    
    async def add_log_entry(self, node_id: str, command: str, data: Dict[str, Any]):
        """Add a log entry to a specific node."""
        if node_id in self.nodes:
            node = self.nodes[node_id]
            if node.role == NodeRole.LEADER:
                entry = RaftLogEntry(
                    term=node.current_term,
                    index=len(node.log) + 1,
                    command=command,
                    data=data,
                    timestamp=datetime.now()
                )
                node.append_log_entry(entry)
                logger.info(f"Added log entry to leader {node_id}: {command}")
                return entry
            else:
                raise ValueError(f"Node {node_id} is not a leader")
        else:
            raise ValueError(f"Node {node_id} not found")


# Global simulator instance
simulator = RaftSimulator()
