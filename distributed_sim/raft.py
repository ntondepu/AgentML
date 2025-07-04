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
from ..monitoring.telemetry import metrics
from ..config import settings

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
        self.election_timeout = random.uniform(150, 300)  # ms
        self.heartbeat_interval = 50  # ms
        
        # Simulation tracking
        self.message_log: List[MessageLog] = []
        self.votes_received: Set[str] = set()
        self.partitioned_from: Set[str] = set()
        
    def reset_election_timeout(self):
        """Reset election timeout with random jitter."""
        self.election_timeout = random.uniform(150, 300)
        self.last_heartbeat = datetime.now()
        
    def is_election_timeout(self) -> bool:
        """Check if election timeout has occurred."""
        return (datetime.now() - self.last_heartbeat).total_seconds() * 1000 > self.election_timeout
        
    def start_election(self):
        """Start a new election."""
        self.current_term += 1
        self.role = NodeRole.CANDIDATE
        self.voted_for = self.id
        self.votes_received = {self.id}
        self.reset_election_timeout()
        
        logger.info(f"Node {self.id} starting election for term {self.current_term}")
        
    def become_leader(self):
        """Transition to leader role."""
        self.role = NodeRole.LEADER
        self.next_index = {node_id: len(self.log) for node_id in self.get_cluster_nodes()}
        self.match_index = {node_id: 0 for node_id in self.get_cluster_nodes()}
        
        logger.info(f"Node {self.id} became leader for term {self.current_term}")
        
    def become_follower(self, term: int):
        """Transition to follower role."""
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
        self.last_heartbeat = datetime.now()
        self.election_timeout = settings.raft_election_timeout
        self.heartbeat_interval = settings.raft_heartbeat_interval
        
        # Election state
        self.votes_received: Set[str] = set()
        self.election_start_time: Optional[datetime] = None
        
        # Network partitions
        self.partitioned_from: Set[str] = set()
        
        # Tasks
        self.election_task: Optional[asyncio.Task] = None
        self.heartbeat_task: Optional[asyncio.Task] = None
    
    def to_state(self) -> RaftNodeState:
        """Convert node to RaftNodeState model."""
        return RaftNodeState(
            id=self.id,
            role=self.role,
            status=self.status,
            current_term=self.current_term,
            voted_for=self.voted_for,
            log=self.log,
            commit_index=self.commit_index,
            last_applied=self.last_applied,
            next_index=self.next_index,
            match_index=self.match_index,
            last_heartbeat=self.last_heartbeat,
            election_timeout=self.election_timeout,
            heartbeat_interval=self.heartbeat_interval
        )
    
    def reset_election_timer(self):
        """Reset the election timeout."""
        self.last_heartbeat = datetime.now()
        # Add randomness to prevent split votes
        self.election_timeout = settings.raft_election_timeout + random.uniform(0, 0.5)
    
    def is_election_timeout(self) -> bool:
        """Check if election timeout has occurred."""
        if self.status != NodeStatus.RUNNING:
            return False
        
        time_since_heartbeat = (datetime.now() - self.last_heartbeat).total_seconds()
        return time_since_heartbeat > self.election_timeout
    
    def become_follower(self, term: int):
        """Transition to follower state."""
        self.role = NodeRole.FOLLOWER
        self.current_term = term
        self.voted_for = None
        self.votes_received.clear()
        self.reset_election_timer()
        
        # Cancel leader tasks
        if self.heartbeat_task:
            self.heartbeat_task.cancel()
            self.heartbeat_task = None
    
    def become_candidate(self):
        """Transition to candidate state."""
        self.role = NodeRole.CANDIDATE
        self.current_term += 1
        self.voted_for = self.id
        self.votes_received = {self.id}
        self.election_start_time = datetime.now()
        self.reset_election_timer()
        
        logger.info(f"Node {self.id} became candidate for term {self.current_term}")
    
    def become_leader(self):
        """Transition to leader state."""
        self.role = NodeRole.LEADER
        self.votes_received.clear()
        self.election_start_time = None
        
        # Initialize leader state
        last_log_index = len(self.log)
        for i in range(self.cluster_size):
            node_id = f"node_{i}"
            if node_id != self.id:
                self.next_index[node_id] = last_log_index + 1
                self.match_index[node_id] = 0
        
        logger.info(f"Node {self.id} became leader for term {self.current_term}")
        metrics.record_raft_leader_election()
    
    def append_entry(self, entry: RaftLogEntry):
        """Append entry to log."""
        entry.term = self.current_term
        entry.index = len(self.log) + 1
        self.log.append(entry)
        metrics.record_raft_log_entry(self.id)
        
        logger.debug(f"Node {self.id} appended entry {entry.index}")
    
    def can_vote_for(self, candidate_id: str, candidate_term: int, last_log_index: int, last_log_term: int) -> bool:
        """Check if can vote for candidate."""
        # Don't vote if already voted in this term
        if self.current_term == candidate_term and self.voted_for is not None:
            return False
        
        # Don't vote for older terms
        if candidate_term < self.current_term:
            return False
        
        # Check if candidate's log is at least as up-to-date
        my_last_log_index = len(self.log)
        my_last_log_term = self.log[-1].term if self.log else 0
        
        if last_log_term > my_last_log_term:
            return True
        if last_log_term == my_last_log_term and last_log_index >= my_last_log_index:
            return True
        
        return False


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
                
                await asyncio.sleep(0.1)  # 100ms simulation tick
                
            except Exception as e:
                logger.error(f"Error in simulation loop: {e}")
                await asyncio.sleep(1)
    
    async def _start_election(self, candidate: RaftNode):
        """Start leader election."""
        candidate.become_candidate()
        
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
        last_log_index = len(candidate.log)
        last_log_term = candidate.log[-1].term if candidate.log else 0
        
        # Check if target can vote
        if target.can_vote_for(candidate.id, candidate.current_term, last_log_index, last_log_term):
            target.voted_for = candidate.id
            target.current_term = candidate.current_term
            candidate.votes_received.add(target.id)
            
            await self._log_message("vote_request", candidate.id, target.id, {
                "term": candidate.current_term,
                "granted": True
            })
            
            # Check if won election
            if len(candidate.votes_received) > len(self.nodes) // 2:
                candidate.become_leader()
                
                # Notify other nodes about new leader
                for node in self.nodes.values():
                    if node.id != candidate.id:
                        node.become_follower(candidate.current_term)
        else:
            await self._log_message("vote_request", candidate.id, target.id, {
                "term": candidate.current_term,
                "granted": False
            })
    
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
        
        # Record metrics
        metrics.record_raft_cluster_state(
            leader_count=leader_count,
            candidate_count=candidate_count,
            follower_count=follower_count,
            current_term=current_term
        )
    
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
        else:
            await self._log_message("vote_request", candidate.id, target.id, {
                "term": candidate.current_term,
                "granted": False
            })
    
    async def _send_heartbeats(self, leader: RaftNode):
        """Send heartbeats to all followers."""
        for node_id, node in self.nodes.items():
            if node_id != leader.id and node.status == NodeStatus.RUNNING:
                if not self._is_partitioned(leader.id, node_id):
                    await self._send_heartbeat(leader, node)
    
    async def _send_heartbeat(self, leader: RaftNode, target: RaftNode):
        """Send heartbeat to target node."""
        # Update target's state
        if target.current_term <= leader.current_term:
            target.become_follower(leader.current_term)
        
        await self._log_message("heartbeat", leader.id, target.id, {
            "term": leader.current_term,
            "commit_index": leader.commit_index
        })
    
    async def _log_message(self, msg_type: str, from_node: str, to_node: str, content: Dict[str, Any]):
        """Log a message between nodes."""
        message = MessageLog(
            id=str(uuid.uuid4()),
            message_type=msg_type,
            from_node=from_node,
            to_node=to_node,
            content=content,
            timestamp=datetime.now(),
            delivered=not self._is_partitioned(from_node, to_node)
        )
        
        self.message_log.append(message)
        
        # Keep only recent messages
        if len(self.message_log) > 1000:
            self.message_log = self.message_log[-500:]
    
    def _is_partitioned(self, node1: str, node2: str) -> bool:
        """Check if two nodes are partitioned."""
        for partition in self.partitions.values():
            if partition.active:
                if node1 in partition.affected_nodes or node2 in partition.affected_nodes:
                    return True
        return False
    
    async def _update_metrics(self):
        """Update simulation metrics."""
        active_nodes = sum(1 for node in self.nodes.values() if node.status == NodeStatus.RUNNING)
        metrics.set_active_nodes(active_nodes)
        
        # Update other metrics
        total_log_entries = sum(len(node.log) for node in self.nodes.values())
        self.metrics.total_log_entries = total_log_entries
        
        # Calculate uptime
        if len(self.nodes) > 0:
            self.metrics.uptime_percentage = (active_nodes / len(self.nodes)) * 100
    
    # Public API methods
    async def get_cluster_state(self) -> RaftClusterState:
        """Get current cluster state."""
        nodes = [node.to_state() for node in self.nodes.values()]
        
        # Find leader
        leader_id = None
        current_term = 0
        for node in self.nodes.values():
            if node.role == NodeRole.LEADER:
                leader_id = node.id
                current_term = node.current_term
                break
        
        active_nodes = sum(1 for node in self.nodes.values() if node.status == NodeStatus.RUNNING)
        
        # Determine cluster health
        health = "healthy"
        if leader_id is None:
            health = "no_leader"
        elif active_nodes < len(self.nodes) // 2 + 1:
            health = "unhealthy"
        
        return RaftClusterState(
            nodes=nodes,
            leader_id=leader_id,
            current_term=current_term,
            total_nodes=len(self.nodes),
            active_nodes=active_nodes,
            log_length=max(len(node.log) for node in self.nodes.values()) if self.nodes else 0,
            last_election=None,  # TODO: track last election time
            cluster_health=health
        )
    
    async def list_nodes(self) -> List[RaftNodeState]:
        """List all nodes."""
        return [node.to_state() for node in self.nodes.values()]
    
    async def get_node_state(self, node_id: str) -> Optional[RaftNodeState]:
        """Get state of specific node."""
        node = self.nodes.get(node_id)
        return node.to_state() if node else None
    
    async def start_node(self, node_id: str):
        """Start a stopped node."""
        node = self.nodes.get(node_id)
        if node:
            node.status = NodeStatus.RUNNING
            node.role = NodeRole.FOLLOWER
            node.reset_election_timer()
            logger.info(f"Started node {node_id}")
    
    async def stop_node(self, node_id: str):
        """Stop a running node."""
        node = self.nodes.get(node_id)
        if node:
            node.status = NodeStatus.STOPPED
            if node.role == NodeRole.LEADER:
                # Trigger new election
                pass
            logger.info(f"Stopped node {node_id}")
            self.metrics.node_failures += 1
    
    async def restart_node(self, node_id: str):
        """Restart a node."""
        await self.stop_node(node_id)
        await asyncio.sleep(1)  # Brief downtime
        await self.start_node(node_id)
    
    async def trigger_election(self):
        """Trigger a new election."""
        # Stop current leader
        for node in self.nodes.values():
            if node.role == NodeRole.LEADER:
                node.become_follower(node.current_term)
                break
        
        # Start election from random node
        candidates = [node for node in self.nodes.values() if node.status == NodeStatus.RUNNING]
        if candidates:
            candidate = random.choice(candidates)
            await self._start_election(candidate)
    
    async def append_log_entry(self, entry: RaftLogEntry):
        """Append log entry through leader."""
        leader = None
        for node in self.nodes.values():
            if node.role == NodeRole.LEADER:
                leader = node
                break
        
        if leader:
            leader.append_entry(entry)
            # In a real implementation, would replicate to followers
            logger.info(f"Appended log entry {entry.index} to leader {leader.id}")
        else:
            raise ValueError("No leader available")
    
    async def get_log_entries(self) -> List[RaftLogEntry]:
        """Get all log entries from leader."""
        for node in self.nodes.values():
            if node.role == NodeRole.LEADER:
                return node.log
        
        # Return from any node if no leader
        if self.nodes:
            return list(self.nodes.values())[0].log
        
        return []
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Get simulation metrics."""
        return self.metrics.dict()
    
    async def reset(self):
        """Reset simulation to initial state."""
        await self.stop_simulation()
        
        # Reset all nodes
        for node in self.nodes.values():
            node.role = NodeRole.FOLLOWER
            node.status = NodeStatus.RUNNING
            node.current_term = 0
            node.voted_for = None
            node.log.clear()
            node.commit_index = 0
            node.last_applied = 0
            node.votes_received.clear()
            node.reset_election_timer()
        
        # Clear partitions
        self.partitions.clear()
        self.message_log.clear()
        
        # Reset metrics
        self.metrics = SimulationMetrics()
        
        await self.start_simulation()
        logger.info("Simulation reset")
    
    async def create_partition(self, config: Dict[str, Any]) -> str:
        """Create network partition."""
        partition_id = str(uuid.uuid4())
        partition = NetworkPartition(
            id=partition_id,
            name=config.get("name", "Network Partition"),
            affected_nodes=config.get("affected_nodes", []),
            partition_type=config.get("partition_type", "isolate"),
            created_at=datetime.now(),
            active=True
        )
        
        self.partitions[partition_id] = partition
        self.metrics.network_partitions += 1
        
        logger.info(f"Created network partition {partition_id}")
        return partition_id
    
    async def heal_partition(self, partition_id: str):
        """Heal network partition."""
        partition = self.partitions.get(partition_id)
        if partition:
            partition.active = False
            logger.info(f"Healed network partition {partition_id}")
    
    async def run_scenario(self, scenario_id: str):
        """Run predefined scenario."""
        scenarios = {
            "leader_failure": self._scenario_leader_failure,
            "network_partition": self._scenario_network_partition,
            "node_recovery": self._scenario_node_recovery,
            "high_load": self._scenario_high_load
        }
        
        scenario_func = scenarios.get(scenario_id)
        if scenario_func:
            await scenario_func()
        else:
            raise ValueError(f"Unknown scenario: {scenario_id}")
    
    async def _scenario_leader_failure(self):
        """Simulate leader failure scenario."""
        # Find current leader
        leader = None
        for node in self.nodes.values():
            if node.role == NodeRole.LEADER:
                leader = node
                break
        
        if leader:
            logger.info(f"Simulating failure of leader {leader.id}")
            await self.stop_node(leader.id)
            await asyncio.sleep(5)  # Let new election happen
            await self.start_node(leader.id)
    
    async def _scenario_network_partition(self):
        """Simulate network partition scenario."""
        # Create partition affecting half the nodes
        affected_nodes = list(self.nodes.keys())[:len(self.nodes)//2]
        
        partition_id = await self.create_partition({
            "name": "Split Brain Test",
            "affected_nodes": affected_nodes,
            "partition_type": "split"
        })
        
        await asyncio.sleep(10)  # Let partition effect be observed
        await self.heal_partition(partition_id)
    
    async def _scenario_node_recovery(self):
        """Simulate node recovery scenario."""
        # Stop random node
        nodes = list(self.nodes.keys())
        if nodes:
            node_id = random.choice(nodes)
            await self.stop_node(node_id)
            await asyncio.sleep(5)
            await self.start_node(node_id)
    
    async def _scenario_high_load(self):
        """Simulate high load scenario."""
        # Generate multiple log entries rapidly
        for i in range(20):
            entry = RaftLogEntry(
                term=1,
                index=i,
                command=f"high_load_command_{i}",
                data={"value": i}
            )
            
            try:
                await self.append_log_entry(entry)
                await asyncio.sleep(0.1)
            except ValueError:
                # No leader available
                break
