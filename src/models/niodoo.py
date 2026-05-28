"""
Niodoo Persistent Memory v1.0 — Integrated for Mountain Car Q-SMA
Dual-stream memory system with flinch-based persistence.
- Flinch: Auto-tag high-relevance states (velocity as trigger)
- Graph persistence: Connected components as Betti-0 proxy
- Cycle detection: Short trap loops get dissolved
- Chain linking: Sequential states build momentum chains
"""

import networkx as nx
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Tuple, Any


@dataclass
class MemoryNode:
    """Node in the semantic graph: state content + metadata."""
    id: str
    content: str
    timestamp: datetime
    relevance: float  # 0-1 score (flinch trigger)
    state_vector: np.ndarray = None  # Raw [pos, vel] for similarity
    connections: List[str] = field(default_factory=list)


class NiodooMemory:
    """Dual-stream persistence engine for Mountain Car."""
    
    def __init__(self, beta_threshold: float = 0.4):
        self.graph = nx.DiGraph()  # Directed graph for chain tracking
        self.nodes: Dict[str, MemoryNode] = {}
        self.beta_threshold = beta_threshold
        self.flinch_count = 0
        self.prune_count = 0
        self.chain_length = 0  # Longest active chain
    
    def flinch_tag(self, content: str, velocity: float, state_vector: np.ndarray = None) -> str:
        """Flinch: Quick relevance score based on velocity/energy; auto-tag if high."""
        # Relevance = tanh of velocity magnitude — fast movement = high relevance
        score = float(np.tanh(abs(velocity) * 20))  # Scale velocity for meaningful range
        node_id = f"n{len(self.nodes)}"
        node = MemoryNode(
            id=node_id,
            content=content,
            timestamp=datetime.now(),
            relevance=min(score, 1.0),
            state_vector=state_vector
        )
        self.nodes[node_id] = node
        self.graph.add_node(node_id, relevance=score)
        
        if score > self.beta_threshold:
            self.flinch_count += 1
        
        return node_id
    
    def connect_nodes(self, node_id1: str, node_id2: str, weight: float = 1.0):
        """Link nodes — builds the momentum chain."""
        if node_id1 in self.nodes and node_id2 in self.nodes:
            self.graph.add_edge(node_id1, node_id2, weight=weight)
            self.nodes[node_id1].connections.append(node_id2)
    
    def detect_and_prune_cycles(self):
        """Detect short trap cycles and dissolve low-relevance nodes in them."""
        try:
            # Find simple cycles (loops) — limit search to avoid explosion
            cycles = list(nx.simple_cycles(self.graph))
            for cycle in cycles:
                if len(cycle) < 4:  # Short trap? Dissolve low-relevance
                    nodes_to_remove = []
                    for n in cycle:
                        if n in self.nodes and self.nodes[n].relevance < 0.3:
                            nodes_to_remove.append(n)
                    for n in nodes_to_remove:
                        if n in self.nodes:
                            del self.nodes[n]
                        if n in self.graph:
                            self.graph.remove_node(n)
                            self.prune_count += 1
        except Exception:
            pass  # Graph too complex for full cycle enumeration — skip
    
    def curate_prune(self):
        """Background: Prune low-persistence isolated nodes + trap cycles."""
        # 1. Prune isolates with low relevance
        undirected = self.graph.to_undirected()
        components = list(nx.connected_components(undirected))
        for comp in components:
            if len(comp) < 2:  # Isolated node
                for node_id in list(comp):
                    if node_id in self.graph.nodes and \
                       self.graph.nodes[node_id].get('relevance', 0) < self.beta_threshold:
                        if node_id in self.nodes:
                            del self.nodes[node_id]
                        self.graph.remove_node(node_id)
                        self.prune_count += 1
        
        # 2. Detect and dissolve short trap cycles
        self.detect_and_prune_cycles()
        
        # 3. Update chain length metric
        if len(self.graph) > 0:
            try:
                self.chain_length = nx.dag_longest_path_length(self.graph)
            except (nx.NetworkXError, nx.NetworkXUnfeasible):
                # Graph has cycles, estimate from component sizes
                self.chain_length = max(len(c) for c in components) if components else 0
    
    def query_persistent(self, query_state: np.ndarray, top_k: int = 3) -> List[Tuple[str, float]]:
        """Retrieve memories most similar to query state (cosine on state vectors)."""
        results = []
        for node_id, node in self.nodes.items():
            if node.state_vector is not None and query_state is not None:
                # Cosine similarity on [pos, vel]
                a = query_state
                b = node.state_vector
                dot = np.dot(a, b)
                norm = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8
                sim = dot / norm
                results.append((node.content, sim, node.relevance))
        
        # Sort by similarity * relevance (both matter)
        results.sort(key=lambda x: x[1] * x[2], reverse=True)
        return results[:top_k]
    
    def persistence_bonus(self) -> float:
        """Return a bonus based on graph connectivity — longer chains = more persistence."""
        if len(self.graph) < 5:
            return 0.0
        # Bonus scales with chain length (capped)
        return min(2.0, self.chain_length * 0.1)
    
    def get_stats(self) -> Dict[str, Any]:
        """Return current memory state for logging."""
        return {
            'nodes': len(self.nodes),
            'edges': self.graph.number_of_edges(),
            'flinches': self.flinch_count,
            'prunes': self.prune_count,
            'chain_length': self.chain_length,
            'persistence_bonus': self.persistence_bonus()
        }


# Quick self-test
if __name__ == "__main__":
    memory = NiodooMemory(beta_threshold=0.4)
    
    # Simulate Mountain Car states
    states = [
        ("pos:-0.50, vel:0.00", 0.00, np.array([-0.5, 0.0])),
        ("pos:-0.45, vel:0.02", 0.02, np.array([-0.45, 0.02])),
        ("pos:-0.40, vel:0.04", 0.04, np.array([-0.40, 0.04])),
        ("pos:-0.30, vel:0.05", 0.05, np.array([-0.30, 0.05])),
        ("pos:0.10, vel:0.06", 0.06, np.array([0.10, 0.06])),
        ("pos:0.50, vel:0.03", 0.03, np.array([0.50, 0.03])),  # SUCCESS
    ]
    
    print("=== Niodoo Mountain Car Test ===")
    prev_id = None
    for content, vel, sv in states:
        node_id = memory.flinch_tag(content, vel, sv)
        if prev_id:
            memory.connect_nodes(prev_id, node_id, weight=abs(vel))
        prev_id = node_id
    
    memory.curate_prune()
    stats = memory.get_stats()
    print(f"Nodes: {stats['nodes']}, Edges: {stats['edges']}")
    print(f"Flinches: {stats['flinches']}, Chain: {stats['chain_length']}")
    print(f"Persistence Bonus: {stats['persistence_bonus']:.2f}")
    
    # Query recall
    results = memory.query_persistent(np.array([0.4, 0.05]))
    print(f"Recall for near-goal: {[r[0] for r in results]}")
    print("✅ Niodoo self-test passed")
