"""
Splat Memory: Gaussian Volumetric Reflex System
=================================================
Experiences crystallize into Gaussian splats in state space.
When the agent enters a region near stored splats, they fire
as reflexes — biasing action selection before conscious thought.

Hot pan → pain splat → hand pulls back automatically.
Microwave timing → pleasure splat → you walk back at the right moment.

No external override. The agent grows its own governor from scar tissue.
"""

import numpy as np
from collections import deque
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Splat:
    """A single Gaussian memory splat in state-action space."""
    center: np.ndarray        # [position, velocity] — WHERE in state space
    action: int               # WHAT was done
    valence: float            # HOW it felt: negative = pain, positive = pleasure
    intensity: float          # HOW STRONG (decays over time, reinforced by repetition)
    radius: float             # HOW FAR the influence reaches
    hits: int = 0             # HOW MANY times this splat has fired
    age: int = 0              # Episodes since creation
    reinforcements: int = 1   # Times this experience was repeated


class SplatMemory:
    """
    Gaussian splat volumetric memory for experiential reflexes.
    
    The agent stores experiences as 3D Gaussians in state space.
    Before acting, nearby splats are retrieved (RAG) and their
    valences modify action priorities — pain repels, pleasure attracts.
    
    Over time: splats decay unless reinforced by repeated experience.
    Similar nearby splats consolidate (merge) during "sleep."
    """

    def __init__(self, 
                 pain_threshold: float = -0.1,      # Energy delta below this = pain (raised from -0.01)
                 pleasure_threshold: float = 0.03,   # Energy delta above this = pleasure
                 default_radius: float = 0.15,       # State-space influence radius
                 max_splats: int = 5000,              # Memory capacity
                 decay_rate: float = 0.998,           # Per-episode intensity decay (slower fade)
                 consolidation_radius: float = 0.08,  # Merge splats closer than this
                 reflex_weight: float = 0.1,          # Base reflex influence (lowered from 0.5)
                 maturation_episodes: int = 50,       # Reflexes don't fire until agent has foundation
                 ):
        self.pain_threshold = pain_threshold
        self.pleasure_threshold = pleasure_threshold
        self.default_radius = default_radius
        self.max_splats = max_splats
        self.decay_rate = decay_rate
        self.consolidation_radius = consolidation_radius
        self.reflex_weight = reflex_weight
        self.maturation_episodes = maturation_episodes
        self.current_episode = 0
        
        self.splats: List[Splat] = []
        
        # Telemetry
        self.total_stored = 0
        self.total_retrievals = 0
        self.total_reflex_fires = 0     # Times a splat actually influenced a decision
        self.pain_splats_created = 0
        self.pleasure_splats_created = 0
        self.consolidations = 0
        self.healed_scars = 0

    def store_experience(self, state: np.ndarray, action: int, 
                         energy_delta: float, reward: float, 
                         success: bool = False):
        """
        Crystallize a significant experience into a splat.
        
        Only stores SIGNIFICANT moments:
          - Pain: energy_delta below threshold (you touched the hot pan)
          - Pleasure: energy_delta above threshold, or success (the food was ready)
          - Episode end success: extra-strong pleasure splat
        
        If a similar splat already exists nearby, REINFORCE it instead of
        creating a new one (the scar deepens, the reflex strengthens).
        """
        # Determine if this experience is worth remembering
        if energy_delta < self.pain_threshold:
            valence = energy_delta * 2.0  # Pain (reduced from 5.0)
            intensity = abs(energy_delta) * 3.0  # Reduced from 10.0
        elif energy_delta > self.pleasure_threshold or success:
            valence = max(energy_delta * 3.0, 0.5 if success else 0.0)
            if success:
                valence += 3.0   # Succeeding is VERY memorable (increased)
                intensity = 8.0  # Stronger than pain
            else:
                intensity = energy_delta * 5.0
        else:
            return  # Unremarkable experience — don't store
        
        # Check for nearby similar splat to reinforce OR heal
        for splat in self.splats:
            # ═══════════════════════════════════════════════════════
            # ACTIVE HEALING (The Cure)
            # If we succeeded in a place where we previously felt pain,
            # we must HEAL the scar. "I faced the fear and won."
            # ═══════════════════════════════════════════════════════
            if valence > 0 and splat.valence < 0:
                dist = np.linalg.norm(state[:2] - splat.center)
                if dist < splat.radius * 1.5:
                    # Redemption! Nuke the pain splat.
                    splat.intensity *= 0.1  # Drastic reduction
                    if splat.intensity < 0.1:
                        splat.intensity = 0  # Mark for removal
                    self.healed_scars += 1
            
            # ═══════════════════════════════════════════════════════
            # REINFORCEMENT & COMPOUND TRAUMA
            # ═══════════════════════════════════════════════════════
            elif splat.action == action:
                dist = np.linalg.norm(state[:2] - splat.center)
                if dist < self.consolidation_radius:
                    if valence < 0:
                        # COMPOUND TRAUMA (The Scar Deepens)
                        # Repeated failure makes the region darker.
                        # Additive intensity (capped) + blend valence.
                        splat.intensity = min(30.0, splat.intensity + intensity * 0.5)
                    else:
                        # Standard reinforcement for pleasure
                        splat.intensity = min(20.0, splat.intensity + intensity * 0.5)
                    
                    splat.valence = splat.valence * 0.7 + valence * 0.3  # Blend
                    splat.reinforcements += 1
                    splat.age = 0  # Reset decay
                    return
        
        # New splat — new memory
        splat = Splat(
            center=state[:2].copy(),
            action=action,
            valence=valence,
            intensity=min(10.0, intensity),
            radius=self.default_radius,
        )
        self.splats.append(splat)
        self.total_stored += 1
        
        if valence < 0:
            self.pain_splats_created += 1
        else:
            self.pleasure_splats_created += 1
        
        # Capacity management: remove weakest splat if over limit
        if len(self.splats) > self.max_splats:
            weakest = min(range(len(self.splats)), 
                         key=lambda i: self.splats[i].intensity)
            self.splats.pop(weakest)

    def query_reflexes(self, state: np.ndarray, n_actions: int = 3) -> np.ndarray:
        """
        Retrieve nearby splats and compute per-action reflex bias.
        Returns: np.array of shape (n_actions,)
        """
        self.total_retrievals += 1
        bias = np.zeros(n_actions)
        
        if self.current_episode < self.maturation_episodes or not self.splats:
            return bias
        
        state_2d = state[:2]
        
        for splat in self.splats:
            dist = np.linalg.norm(state_2d - splat.center)
            if dist > splat.radius * 3:
                continue
            activation = np.exp(-0.5 * (dist / splat.radius) ** 2)
            confidence = min(1.0, splat.reinforcements / 10.0)
            reflex_signal = activation * splat.intensity * splat.valence * confidence
            if abs(reflex_signal) > 0.01:
                bias[splat.action] += reflex_signal * self.reflex_weight
                splat.hits += 1
                self.total_reflex_fires += 1
        
        return bias
    
    def query_familiarity(self, state: np.ndarray) -> float:
        """
        Return familiarity score (0-1) for this state region.
        
        0.0 = completely unknown territory (explore more)
        1.0 = highly familiar from many wins (trust Q-values)
        
        This is the TRUE reflex: in familiar territory, ACT FROM MEMORY.
        In unknown territory, EXPLORE. Like walking through your house
        in the dark — you know where the furniture is.
        """
        if self.current_episode < self.maturation_episodes or not self.splats:
            return 0.0
        
        state_2d = state[:2]
        familiarity = 0.0
        
        for splat in self.splats:
            dist = np.linalg.norm(state_2d - splat.center)
            if dist > splat.radius * 3:
                continue
            activation = np.exp(-0.5 * (dist / splat.radius) ** 2)
            if splat.valence > 0:  # Only positive memories build familiarity
                confidence = min(1.0, splat.reinforcements / 5.0)
                familiarity += activation * splat.intensity * confidence * 0.1
        
        return min(1.0, familiarity)

    def decay_and_consolidate(self):
        """
        Called once per episode. Two operations:
        
        1. DECAY: All splats lose intensity over time.
           Unreinforced memories fade. Reinforced ones persist.
           
        2. CONSOLIDATE: Nearby same-action splats merge.
           Like memory consolidation during sleep — similar
           experiences blur into a single strong reflex.
        """
        self.current_episode += 1
        
        # 1. Asymmetric Decay (The Biological Reality)
        for splat in self.splats:
            if splat.valence > 0:
                # Pleasure fades: "Use it or lose it"
                splat.intensity *= self.decay_rate
            else:
                # Pain persists: "Trauma lasts"
                # Immortal (1.0) or extremely slow decay
                splat.intensity *= 1.0 
            
            splat.age += 1
        
        # Remove dead splats (intensity too low)
        before = len(self.splats)
        self.splats = [s for s in self.splats if s.intensity > 0.01]
        
        # 2. Consolidate nearby same-action splats
        if len(self.splats) > 100 and np.random.rand() < 0.1:
            self._consolidate()
        
    def _consolidate(self):
        """Merge nearby same-action splats into stronger combined memories."""
        merged = set()
        new_splats = []
        
        for i, s1 in enumerate(self.splats):
            if i in merged:
                continue
            
            # Find neighbors to merge with
            group = [s1]
            for j, s2 in enumerate(self.splats):
                if j <= i or j in merged:
                    continue
                if s1.action != s2.action:
                    continue
                    
                dist = np.linalg.norm(s1.center - s2.center)
                if dist < self.consolidation_radius:
                    group.append(s2)
                    merged.add(j)
            
            if len(group) > 1:
                # Merge: weighted average center, sum intensity, blend valence
                total_intensity = sum(s.intensity for s in group)
                new_center = sum(s.center * s.intensity for s in group) / total_intensity
                new_valence = sum(s.valence * s.intensity for s in group) / total_intensity
                new_radius = max(s.radius for s in group) * 1.1  # Slightly wider
                
                consolidated = Splat(
                    center=new_center,
                    action=s1.action,
                    valence=new_valence,
                    intensity=min(20.0, total_intensity * 0.8),  # Some loss in merge
                    radius=new_radius,
                    reinforcements=sum(s.reinforcements for s in group),
                    hits=sum(s.hits for s in group),
                )
                new_splats.append(consolidated)
                self.consolidations += 1
            else:
                new_splats.append(s1)
        
        self.splats = new_splats

    def get_stats(self):
        """Return memory statistics."""
        if not self.splats:
            return {
                'count': 0, 'pain': self.pain_splats_created,
                'pleasure': self.pleasure_splats_created,
                'fires': self.total_reflex_fires,
                'consolidations': self.consolidations,
                'healed': self.healed_scars,
            }
        
        intensities = [s.intensity for s in self.splats]
        valences = [s.valence for s in self.splats]
        pain_count = sum(1 for s in self.splats if s.valence < 0)
        pleasure_count = sum(1 for s in self.splats if s.valence > 0)
        
        return {
            'count': len(self.splats),
            'alive_pain': pain_count,
            'alive_pleasure': pleasure_count,
            'total_pain_created': self.pain_splats_created,
            'total_pleasure_created': self.pleasure_splats_created,
            'mean_intensity': np.mean(intensities),
            'max_intensity': max(intensities),
            'mean_valence': np.mean(valences),
            'fires': self.total_reflex_fires,
            'consolidations': self.consolidations,
            'healed': self.healed_scars,
            'most_reinforced': max((s.reinforcements for s in self.splats), default=0),
        }
