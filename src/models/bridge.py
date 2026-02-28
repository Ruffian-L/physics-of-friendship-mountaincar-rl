"""
Bridge Module: Physics Body ↔ Q-SMA Mind
==========================================
Three layers connecting the dead-but-perfect physics reflex
to the alive-but-struggling learning agent.

Layer 1: InstinctSeeder  — Pre-load Q/flux with energy landscape
Layer 2: DreamTeacher    — Feed physics trajectories into dream replay
Layer 3: GovernorGate    — Real-time action validation with handoff curve

The key metric: Governor Handoff Curve
  - Early: high overrides (Body protecting Mind)
  - Mid:   declining (Mind learning)
  - Late:  near-zero (Mind in control, earned autonomy)
"""

import gymnasium as gym
import numpy as np
from collections import deque
import math


# =============================================================================
# LAYER 1: INSTINCT SEEDER — "A deer is born knowing gravity"
# =============================================================================
class InstinctSeeder:
    """
    Runs the physics solver to generate energy landscape knowledge,
    then seeds the agent's Q-table and flux with soft priors.
    
    Does NOT give the agent optimal actions — gives it the LANDSCAPE.
    The agent knows "this region has high energy potential" but must
    figure out HOW to get there.
    """

    def __init__(self, num_seed_episodes=100, instinct_weight=0.3):
        self.num_seed_episodes = num_seed_episodes
        self.instinct_weight = instinct_weight
        self.seed_trajectories = []  # Also used by DreamTeacher

    def generate_seeds(self, agent):
        """
        Run physics solver → collect (state, action, energy_delta) tuples
        → aggregate into Q-table priors.
        """
        env = gym.make('MountainCar-v0')
        
        # Accumulators for Q and flux seeds
        q_accum = {}   # (disc_state, action) → [energy_deltas]
        flux_accum = {}  # (disc_state, action) → count
        
        print(f"  [INSTINCT] Generating seeds from {self.num_seed_episodes} physics episodes...")
        
        for ep in range(self.num_seed_episodes):
            state, _ = env.reset(seed=42 + ep)
            trajectory = []
            
            for step in range(200):
                # Physics reflex: push with velocity (the Body's instinct)
                velocity = state[1]
                action = 2 if velocity >= 0 else 0
                
                next_state, reward, terminated, truncated, _ = env.step(action)
                
                # Compute energy delta (what physics cares about)
                phi_now = math.sin(3 * next_state[0]) + 100 * (next_state[1] ** 2)
                phi_prev = math.sin(3 * state[0]) + 100 * (state[1] ** 2)
                energy_delta = phi_now - phi_prev
                energy = (next_state[0]**2) + (next_state[1]**2)
                
                # Discretize using agent's bins
                disc_s = agent.discretize(state)
                
                # Accumulate
                key = (disc_s, action)
                if key not in q_accum:
                    q_accum[key] = []
                    flux_accum[key] = 0
                q_accum[key].append(energy_delta)
                flux_accum[key] += 1
                
                # Store full trajectory for DreamTeacher
                trajectory.append((state.copy(), action, reward, next_state.copy(), energy))
                
                if terminated or truncated:
                    break
                state = next_state
            
            self.seed_trajectories.append(trajectory)
        
        env.close()
        
        # Build seed arrays matching agent's Q-table shape
        q_seed = np.zeros_like(agent.q_table)
        flux_seed = np.zeros_like(agent.flux)
        
        for (disc_s, action), deltas in q_accum.items():
            # Q-seed: mean energy delta — "how good is this action for building energy?"
            q_seed[disc_s, action] = np.mean(deltas)
        
        max_count = max(flux_accum.values()) if flux_accum else 1
        for (disc_s, action), count in flux_accum.items():
            # Flux-seed: normalized frequency — "how often does the Body visit here?"
            flux_seed[disc_s, action] = (count / max_count) * 5.0  # Scale to meaningful range
        
        # Apply seeds with instinct weight (soft, not absolute)
        agent.seed_from_physics(q_seed, flux_seed, self.instinct_weight)
        
        print(f"  [INSTINCT] Seeded {len(q_accum)} state-action pairs")
        print(f"  [INSTINCT] Q-seed range: [{q_seed.min():.4f}, {q_seed.max():.4f}]")
        print(f"  [INSTINCT] Flux-seed range: [{flux_seed.min():.4f}, {flux_seed.max():.4f}]")
        
        return self.seed_trajectories


# =============================================================================
# LAYER 2: DREAM TEACHER — "AlphaGo learns from its perfect self"
# =============================================================================
class DreamTeacher:
    """
    Loads physics-generated trajectories into the agent's replay buffer.
    The agent replays these during dream cycles — absorbing the FEELING
    of correct energy flow without obeying it directly.
    
    Dreams whisper (low LR), they don't shout.
    TDA still analyzes the agent's REAL behavior, not the teacher's.
    """

    def __init__(self, trajectories, dream_fraction=0.3):
        """
        Args:
            trajectories: list of episodes, each a list of (s, a, r, s', energy)
            dream_fraction: fraction of dream batch that comes from teacher (vs own experience)
        """
        self.trajectories = trajectories
        self.dream_fraction = dream_fraction
        self.teacher_buffer = []
        
        # Flatten all trajectories into a single replay buffer
        for traj in trajectories:
            self.teacher_buffer.extend(traj)
        
        print(f"  [DREAMS] Loaded {len(self.teacher_buffer)} teacher transitions "
              f"from {len(trajectories)} episodes")

    def inject_dreams(self, agent):
        """
        Load a subset of teacher data into the agent's replay buffer.
        Called once at startup — the dreams are always available for replay.
        """
        # Don't flood the buffer — inject a meaningful but not dominant amount
        n_inject = min(len(self.teacher_buffer), 2000)
        indices = np.random.choice(len(self.teacher_buffer), n_inject, replace=False)
        
        for idx in indices:
            agent.replay_buffer.append(self.teacher_buffer[idx])
        
        print(f"  [DREAMS] Injected {n_inject} teacher transitions into replay buffer")

    def dream_with_teacher(self, agent, batch_size=256):
        """
        Enhanced dream cycle: mix teacher data with agent's own experience.
        Returns True if dreaming occurred.
        """
        own_data = list(agent.replay_buffer)
        if len(own_data) < batch_size // 2:
            return False
        
        # Mix: fraction from teacher, rest from own experience
        n_teacher = int(batch_size * self.dream_fraction)
        n_own = batch_size - n_teacher
        
        # Sample from teacher
        if len(self.teacher_buffer) >= n_teacher:
            teacher_batch = [self.teacher_buffer[i] 
                           for i in np.random.choice(len(self.teacher_buffer), n_teacher, replace=False)]
        else:
            teacher_batch = list(self.teacher_buffer)
        
        # Sample from own experience
        own_batch = [own_data[i] 
                    for i in np.random.choice(len(own_data), min(n_own, len(own_data)), replace=False)]
        
        batch = teacher_batch + own_batch
        
        # Dream replay at whisper learning rate
        q_updates = []
        for state, action, reward, next_state, energy in batch:
            s = agent.discretize(state)
            ns = agent.discretize(next_state)
            
            best_next = np.max(agent.q_table[ns])
            td_error = 0.005 * (reward + 0.99 * best_next - agent.q_table[s, action])
            q_updates.append((s, action, td_error))
            
            # Flux: gold-only groove deepening
            if energy > 0.15:
                agent.flux[s, action] += 0.05
                agent.flux[s, action] = min(20.0, agent.flux[s, action])
        
        # Mean-center the gradient (from original dream cycle design)
        if q_updates:
            mean_error = np.mean([e for _, _, e in q_updates])
            for s, action, error in q_updates:
                agent.q_table[s, action] += error - mean_error
        
        return True


# =============================================================================
# LAYER 3: GOVERNOR GATE — "The physics safety net with earned autonomy"
# =============================================================================
class GovernorGate:
    """
    Real-time action validation. Checks the Mind's proposed action
    against the Body's physics instinct.
    
    Override rules:
      - If agent and physics AGREE → agent's action (mind in control)
      - If they DISAGREE and agent has LOW confidence → physics overrides
      - If they DISAGREE and agent has HIGH confidence → let agent try
    
    The confidence threshold LOOSENS over time → the Mind earns autonomy.
    """

    def __init__(self, initial_threshold=0.8, final_threshold=0.1, 
                 warmup_episodes=200, rampdown_episodes=1500):
        self.initial_threshold = initial_threshold
        self.final_threshold = final_threshold
        self.warmup_episodes = warmup_episodes
        self.rampdown_episodes = rampdown_episodes
        
        # Tracking
        self.overrides = []        # (episode, step, agent_action, physics_action, confidence)
        self.agreements = 0
        self.total_decisions = 0
        self.per_episode_overrides = []  # count per episode for handoff curve
        self._current_ep_overrides = 0

    def get_threshold(self, episode):
        """
        Compute current confidence threshold.
        Starts strict (0.8), ramps down to loose (0.1) over training.
        """
        if episode < self.warmup_episodes:
            return self.initial_threshold
        
        progress = min(1.0, (episode - self.warmup_episodes) / 
                       (self.rampdown_episodes - self.warmup_episodes))
        return self.initial_threshold - progress * (self.initial_threshold - self.final_threshold)

    def gate(self, state, agent_action, agent, episode, step):
        """
        Evaluate agent's action against physics instinct.
        Returns (final_action, was_overridden).
        """
        self.total_decisions += 1
        
        # Physics instinct: push with velocity
        velocity = state[1]
        physics_action = 2 if velocity >= 0 else 0
        
        # Agreement check
        if agent_action == physics_action:
            self.agreements += 1
            return agent_action, False
        
        # Disagreement — check agent's confidence
        disc_s = agent.discretize(state)
        q_vals = agent.q_table[disc_s]
        
        # Confidence = how much the chosen action dominates alternatives
        q_max = np.max(q_vals)
        q_second = np.sort(q_vals)[-2] if len(q_vals) > 1 else q_max
        q_spread = q_max - q_second
        
        # Normalize spread relative to Q-table scale
        q_range = np.max(agent.q_table) - np.min(agent.q_table)
        confidence = q_spread / (q_range + 1e-8)
        
        threshold = self.get_threshold(episode)
        
        if confidence < threshold:
            # LOW confidence → physics overrides
            self.overrides.append((episode, step, agent_action, physics_action, confidence))
            self._current_ep_overrides += 1
            
            # FEEDBACK: Teach the Mind WHY physics overrode.
            # Nudge Q toward the physics action so the Mind converges.
            # The nudge decays over time — early correction is strong,
            # late correction is gentle (let the Mind develop its own logic).
            feedback_strength = max(0.05, 0.5 * (1.0 - episode / self.rampdown_episodes))
            agent.q_table[disc_s, physics_action] += feedback_strength
            # Also slightly penalize the rejected action
            agent.q_table[disc_s, agent_action] -= feedback_strength * 0.3
            
            return physics_action, True
        else:
            # HIGH confidence → trust the Mind
            return agent_action, False

    def episode_done(self):
        """Track per-episode override count for handoff curve."""
        self.per_episode_overrides.append(self._current_ep_overrides)
        self._current_ep_overrides = 0

    def get_stats(self):
        agreement_rate = self.agreements / max(1, self.total_decisions) * 100
        override_count = len(self.overrides)
        return {
            'total_decisions': self.total_decisions,
            'agreements': self.agreements,
            'agreement_rate': agreement_rate,
            'overrides': override_count,
            'per_episode_overrides': self.per_episode_overrides,
        }

    def get_handoff_curve(self, window=50):
        """Rolling average of overrides per episode."""
        if len(self.per_episode_overrides) < window:
            return self.per_episode_overrides
        return np.convolve(self.per_episode_overrides, 
                          np.ones(window)/window, mode='valid').tolist()
