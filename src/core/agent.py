import numpy as np
from collections import deque
import random
from models.splat_memory import SplatMemory
from core.watcher import DaydreamWatcher

class QSMA_Agent:
    def __init__(self, state_bins=40, action_size=3):
        self.state_bins = state_bins
        self.action_size = action_size
        self.state_size = state_bins * state_bins
        
        # System 1 & 2
        self.q_table = np.zeros((self.state_size, action_size))
        self.flux = np.zeros((self.state_size, action_size))
        
        # Params
        self.params = {'decay': 0.1, 'epsilon': 0.3, 'attractors': [], 'beta': 1.5}  # beta = flux weight
        self.pain_memory = 0.0  # Tracks recent suffering — amplifies next recovery
        
        # Splat Memory: Gaussian volumetric reflex system
        self.splat_memory = SplatMemory()
        
        # Stream 2: The Daydream Watcher (Conscience, not Governor)
        # Observes energy states in background, reshapes Flux landscape
        self.watcher = DaydreamWatcher()
        self.daydream_interval = 20  # Observe every N steps
        
        # Dream Cycle: Replay buffer for post-episode consolidation
        self.replay_buffer = deque(maxlen=10000)
        self.success_buffer = deque(maxlen=5000)
        self.episode_buffer = []
        
        # Discretization Bins
        self.pos_bins = np.linspace(-1.2, 0.6, state_bins)
        self.vel_bins = np.linspace(-0.07, 0.07, state_bins)
        
        # Splat trajectory tracking (separate from episode_buffer for dream cycle)
        self._episode_energy_states = []

    def discretize(self, state):
        p_idx = np.digitize(state[0], self.pos_bins) - 1
        v_idx = np.digitize(state[1], self.vel_bins) - 1
        p_idx = max(0, min(self.state_bins - 1, p_idx))
        v_idx = max(0, min(self.state_bins - 1, v_idx))
        return p_idx * self.state_bins + v_idx

    def sync(self, controller):
        self.params['decay'] = controller.decay_rate
        self.params['epsilon'] = controller.exploration_rate
        self.params['attractors'] = controller.attractors

    def act(self, state):
        disc_state = self.discretize(state)
        
        # ═══════════════════════════════════════════════════════
        # CLEAN ACTION LOOP — No Spasms. Let the agent surf.
        # The wave is Q-values + Flux + Curiosity.
        # Splats influence DREAMS, not waking decisions.
        # ═══════════════════════════════════════════════════════
        
        # CURIOSITY INJECTION
        curiosity = np.zeros(self.action_size)
        for attr in self.params['attractors']:
            d_pos = state[0] - attr[0]
            d_vel = state[1] - attr[1]
            dist = np.sqrt(d_pos**2 + d_vel**2)
            
            if dist < 0.6: 
                curiosity += 5.0 

        # PHYSICS OF HABIT: Viscosity & Flow
        flux_val = self.flux[disc_state]
        threshold = 2.0
        ease = 1 / (1 + np.exp(-(flux_val - threshold)))
        
        beta = self.params.get('beta', 1.5)
        priority = self.q_table[disc_state] + (ease * beta) + curiosity
        
        if np.random.rand() < self.params['epsilon']:
            return np.random.randint(self.action_size)
        return np.argmax(priority)

    def learn(self, state, action, reward, next_state):
        s = self.discretize(state)
        ns = self.discretize(next_state)
        best_next = np.max(self.q_table[ns])
        
        import math
        phi_now = math.sin(3 * next_state[0]) + 100 * (next_state[1] ** 2)
        phi_prev = math.sin(3 * state[0]) + 100 * (state[1] ** 2)
        energy_delta = phi_now - phi_prev
        
        if energy_delta < 0:
            energy_delta *= 1.15
        
        shaped_reward = reward + energy_delta * 10.0
        self.q_table[s, action] += 0.2 * (shaped_reward + 0.999 * best_next - self.q_table[s, action])
        
        position, velocity = next_state
        energy = (position**2) + (velocity**2)
        
        if energy > 0.1:
            impact = np.log(1 + energy * 100) * 0.5
            self.flux[s, action] += impact
        elif energy < 0.05:
            pain = np.log(1 + (0.05 - energy) * 50) * 0.5
            self.flux[s, action] -= pain
            
        self.flux[s, action] *= (1.0 - self.params['decay'])
        self.flux[s, action] = np.clip(self.flux[s, action], -5.0, 20.0)

        # Track energy deltas for splat crystallization at episode end
        self._episode_energy_states.append((state.copy(), action, energy_delta, reward))

        # ═══════════════════════════════════════════════════════
        # STREAM 2: The Daydream Watcher observes in background
        # Every N steps, it looks at recent energy trajectory
        # and sculpts the Flux landscape. Never touches act().
        # ═══════════════════════════════════════════════════════
        if len(self._episode_energy_states) % self.daydream_interval == 0:
            self.watcher.observe(
                self._episode_energy_states[-self.daydream_interval:],
                self.flux,
                self.splat_memory,
                self.discretize
            )

        return energy

    # =========================================================
    # BRIDGE: Physics Body → Mind injection points
    # =========================================================
    def seed_from_physics(self, q_seed, flux_seed, weight=0.3):
        """
        Layer 1: Instinct Seed.
        Soft-load physics landscape into Q-table and flux.
        The agent knows "this region has energy potential" but
        must learn HOW to exploit it.
        
        weight: how much physics knowledge to inject (0=none, 1=full override)
        """
        self.q_table = self.q_table * (1 - weight) + q_seed * weight
        self.flux = self.flux * (1 - weight) + flux_seed * weight
        print(f"  [AGENT] Q-table seeded (weight={weight:.2f}), "
              f"Q range: [{self.q_table.min():.3f}, {self.q_table.max():.3f}]")

    def load_teacher_dreams(self, trajectories, max_inject=2000):
        """
        Layer 2: Dream Teacher injection.
        Load physics-generated perfect trajectories into replay buffer.
        These get replayed during dream cycles alongside own experience.
        """
        flat = []
        for traj in trajectories:
            flat.extend(traj)
        
        n = min(len(flat), max_inject)
        indices = np.random.choice(len(flat), n, replace=False)
        for idx in indices:
            self.replay_buffer.append(flat[idx])
        
        print(f"  [AGENT] Loaded {n} teacher dreams into replay buffer "
              f"(buffer size: {len(self.replay_buffer)})")

    # =========================================================
    # DREAM CYCLE v5: Holistic consolidation
    # - Mean-center TD errors (normalize gradient, not landscape)
    # - Flux: gold-only groove deepening (energy > 0.15)
    # - Uses general replay_buffer (all transitions)
    # =========================================================
    def remember(self, state, action, reward, next_state, energy):
        """Store a transition in the replay buffer for dream consolidation."""
        self.replay_buffer.append((state, action, reward, next_state, energy))

    def commit_episode(self, success):
        """
        End of episode — crystallize key moments as splats.
        
        SUCCESS (Yang): Store turning points from winning trajectory.
        FAILURE (Yin):  Store stuck-points from losing trajectory.
        
        The Conscience needs BOTH to guide:
          - Yang: "This felt like victory. Dream about it. Deepen the groove."
          - Yin:  "This felt like quicksand. Roughen this path. Try elsewhere."
        """
        if success and len(self._episode_energy_states) > 0:
            # YANG: Store winning trajectory moments as pleasure splats
            for state, action, energy_delta, reward in self._episode_energy_states:
                if energy_delta > 0.02:  # Significant energy gain during victory
                    self.splat_memory.store_experience(
                        state, action, energy_delta, reward, success=True
                    )
            
            # Strong splat at the final state (goal reached!)
            last_state, last_action, _, _ = self._episode_energy_states[-1]
            self.splat_memory.store_experience(
                last_state, last_action, 1.0, 100.0, success=True
            )
        
        elif not success and len(self._episode_energy_states) > 20:
            # YIN: Store the stuck-points as pain splats
            # Find where the agent was TRAPPED — same position, no progress
            # Use the LAST 30% of the trajectory (where timeout actually bit)
            trap_start = len(self._episode_energy_states) * 7 // 10
            trap_states = self._episode_energy_states[trap_start:]
            
            if trap_states:
                # Find the most stagnant region — lowest total energy delta
                positions = np.array([s[0][0] for s in trap_states])
                pos_variance = np.var(positions)
                total_energy = sum(s[2] for s in trap_states)
                
                # Only scar if truly stuck (low variance, low energy gain)
                if pos_variance < 0.02 and total_energy < 0.01:
                    # Sample a few representative stuck-states (not ALL of them)
                    # Too many pain splats = learned helplessness (Phase 1 lesson)
                    indices = np.random.choice(len(trap_states), 
                                               size=min(3, len(trap_states)), 
                                               replace=False)
                    for idx in indices:
                        state, action, energy_delta, reward = trap_states[idx]
                        self.splat_memory.store_experience(
                            state, action, -0.5, reward, success=False
                        )
        
        # Gold replay buffer from episode_buffer (correct 5-tuple format)
        if success and len(self.episode_buffer) > 0:
            self.success_buffer.extend(self.episode_buffer)
        
        self.episode_buffer = []
        self._episode_energy_states = []

    def dream_cycle(self, batch_size=256):
        """
        Holistic dream: mean-centered Q replay + gold flux etching.
        
        THE SKILL (not the Spasm):
        Splat memories weight the replay sampling. Transitions near
        high-intensity splats get replayed more often — the agent
        OBSESSES over its victories and traumas during sleep.
        P(sample) ~ 1 + Intensity(nearby_splat)
        
        This deepens Flux grooves naturally. The next day, the agent
        acts correctly not because something jerked its arm, but
        because the neural superhighway was built overnight.
        """
        if len(self.replay_buffer) < batch_size:
            return False
        
        replay_list = list(self.replay_buffer)
        
        # ═══════════════════════════════════════════════════════
        # SPLAT-WEIGHTED SAMPLING: The Dream Priority
        # Base weight = 1.0 (everyone gets a chance)
        # Near a splat = 1.0 + intensity * activation
        # This makes the agent dream about its victories 10x more
        # ═══════════════════════════════════════════════════════
        if self.splat_memory.splats:
            weights = np.ones(len(replay_list))
            for i, (state, action, reward, next_state, energy) in enumerate(replay_list):
                state_2d = state[:2]
                for splat in self.splat_memory.splats:
                    dist = np.linalg.norm(state_2d - splat.center)
                    if dist < splat.radius * 3:
                        activation = np.exp(-0.5 * (dist / splat.radius) ** 2)
                        # Only positive splats boost replay (dream about victories)
                        if splat.valence > 0:
                            weights[i] += activation * splat.intensity
            
            # Normalize to probabilities
            weights /= weights.sum()
            indices = np.random.choice(len(replay_list), size=batch_size, 
                                       replace=False, p=weights)
            batch = [replay_list[i] for i in indices]
        else:
            batch = random.sample(replay_list, batch_size)
        
        # Phase 1: Collect TD errors for mean-centering
        q_updates = []
        for state, action, reward, next_state, energy in batch:
            s = self.discretize(state)
            ns = self.discretize(next_state)
            
            # Compute TD error at low LR
            best_next = np.max(self.q_table[ns])
            td_error = 0.01 * (reward + 0.99 * best_next - self.q_table[s, action])
            q_updates.append((s, action, td_error))
            
            # Flux: gold-only groove deepening (higher threshold = true gold)
            if energy > 0.15:
                self.flux[s, action] += 0.1
                self.flux[s, action] = min(20.0, self.flux[s, action])
        
        # Phase 2: Mean-center the TD errors, then apply
        if q_updates:
            mean_error = np.mean([e for _, _, e in q_updates])
            for s, action, error in q_updates:
                self.q_table[s, action] += error - mean_error
        
        return True
