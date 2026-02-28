"""
DaydreamWatcher: Stream 2 — The Conscience
============================================
Runs in background during learning. Observes energy states.
Reshapes the Flux landscape WITHOUT touching act().

The agent doesn't decide to be good — it flows into the
good path because the Watcher already paved the road.

Three operations:
1. Groove Deepening — near victory splats + rising energy → smooth path
2. Debris Clearing — stuck + flat energy → roughen path  
3. Splat Gardening — reinforce/mark high-energy moments
"""

import numpy as np


class DaydreamWatcher:
    """
    Stream 2: The Metacognitive Background Process.
    
    Watches the agent's experience stream in real-time.
    Modifies the Flux matrix (viscosity landscape) — NEVER act().
    
    The agent feels the changed viscosity naturally, like
    walking into a room where someone rearranged the furniture.
    """
    
    def __init__(self, 
                 groove_rate: float = 0.05,      # How fast victory paths smooth
                 debris_rate: float = 0.02,       # How fast stuck paths roughen
                 stagnation_window: int = 20,     # Steps to detect "stuck"
                 energy_slope_threshold: float = 0.001,  # Rising vs flat energy
                 ):
        self.groove_rate = groove_rate
        self.debris_rate = debris_rate
        self.stagnation_window = stagnation_window
        self.energy_slope_threshold = energy_slope_threshold
        
        # Telemetry
        self.groove_deepenings = 0
        self.debris_clearings = 0
        self.garden_reinforcements = 0
        self.total_observations = 0
    
    def observe(self, recent_states, flux, splat_memory, discretize_fn):
        """
        The Daydream. Called every N steps during learning.
        
        Inputs:
            recent_states: list of (state, action, energy_delta, reward)
                           from the last N steps of the current episode
            flux: the agent's Flux matrix (modified IN PLACE)
            splat_memory: SplatMemory instance (read for context, may reinforce)
            discretize_fn: function to convert continuous state → discrete index
        
        Outputs:
            None — modifies flux directly. The Watcher whispers to Memory.
        """
        if len(recent_states) < 3:
            return
        
        self.total_observations += 1
        
        # Extract recent trajectory
        states = [s[0] for s in recent_states]
        actions = [s[1] for s in recent_states]
        energies = [s[2] for s in recent_states]  # energy_delta per step
        
        # ═══════════════════════════════════════════════════════
        # 1. GROOVE DEEPENING: Victory Echo
        #    Near positive splats AND energy rising → smooth the path
        #    "This feels like the time I won. Lean into it."
        # ═══════════════════════════════════════════════════════
        if splat_memory.splats:
            cumulative_energy = sum(energies)
            
            if cumulative_energy > 0:  # Energy is net positive in this window
                for state, action, ed, _ in recent_states:
                    state_2d = state[:2]
                    near_victory = False
                    
                    for splat in splat_memory.splats:
                        if splat.valence <= 0:
                            continue
                        dist = np.linalg.norm(state_2d - splat.center)
                        if dist < splat.radius * 2:
                            activation = np.exp(-0.5 * (dist / splat.radius) ** 2)
                            if activation > 0.3:  # Strong enough proximity
                                near_victory = True
                                break
                    
                    if near_victory and ed > 0:
                        # Smooth this groove — lower viscosity on this path
                        s_idx = discretize_fn(state)
                        flux[s_idx, action] += self.groove_rate
                        flux[s_idx, action] = min(20.0, flux[s_idx, action])
                        self.groove_deepenings += 1
        
        # ═══════════════════════════════════════════════════════
        # 2. DEBRIS CLEARING: Pain Memory + Friction Detection
        #    Yin side: "I've been trapped here before" (pain splats)
        #    + "I'm stuck right now" (position stagnation)
        # ═══════════════════════════════════════════════════════
        if len(recent_states) >= self.stagnation_window:
            # Check position variance — are we going anywhere?
            positions = np.array([s[0][0] for s in recent_states[-self.stagnation_window:]])
            pos_variance = np.var(positions)
            
            # Check energy trend — is energy flat?
            recent_energies = energies[-self.stagnation_window:]
            energy_trend = np.mean(recent_energies)
            
            # Pain splat amplification: if near a pain memory, clear debris harder
            pain_amplifier = 1.0
            if splat_memory.splats:
                current_state = recent_states[-1][0][:2]
                for splat in splat_memory.splats:
                    if splat.valence < 0:  # Pain memory nearby
                        dist = np.linalg.norm(current_state - splat.center)
                        if dist < splat.radius * 2:
                            # "I've been trapped here before" — clear harder
                            activation = np.exp(-0.5 * (dist / splat.radius) ** 2)
                            pain_amplifier += activation * splat.intensity * 2.0
            
            if pos_variance < 0.01 and abs(energy_trend) < self.energy_slope_threshold:
                # We're stuck. Roughen the groove, amplified by pain memory.
                effective_debris = self.debris_rate * pain_amplifier
                for state, action, _, _ in recent_states[-5:]:
                    s_idx = discretize_fn(state)
                    flux[s_idx, action] -= effective_debris
                    flux[s_idx, action] = max(-5.0, flux[s_idx, action])
                    self.debris_clearings += 1
        
        # ═══════════════════════════════════════════════════════
        # 3. SPLAT GARDENING: Memory Curation
        #    Watch for strong energy moments → reinforce nearby splats
        #    The Watcher notices "I've felt this before" and strengthens
        #    the memory. Repeated confirmation = confident reflex.
        # ═══════════════════════════════════════════════════════
        if splat_memory.splats:
            for state, action, energy_delta, _ in recent_states:
                if energy_delta > 0.05:  # Significant energy gain
                    state_2d = state[:2]
                    for splat in splat_memory.splats:
                        if splat.valence <= 0 or splat.action != action:
                            continue
                        dist = np.linalg.norm(state_2d - splat.center)
                        if dist < splat.radius * 1.5:
                            # This moment echoes a previous victory
                            splat.reinforcements += 1
                            splat.intensity = min(10.0, splat.intensity * 1.01)
                            self.garden_reinforcements += 1
    
    def get_stats(self):
        return {
            'observations': self.total_observations,
            'groove_deepenings': self.groove_deepenings,
            'debris_clearings': self.debris_clearings,
            'garden_reinforcements': self.garden_reinforcements,
        }
