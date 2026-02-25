import numpy as np

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
        
        # Discretization Bins
        self.pos_bins = np.linspace(-1.2, 0.6, state_bins)
        self.vel_bins = np.linspace(-0.07, 0.07, state_bins)

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
        
        # CURIOSITY INJECTION
        curiosity = np.zeros(self.action_size)
        for attr in self.params['attractors']:
            d_pos = state[0] - attr[0]
            d_vel = state[1] - attr[1]
            dist = np.sqrt(d_pos**2 + d_vel**2)
            
            if dist < 0.6: 
                curiosity += 5.0 

        # PHYSICS OF HABIT: Viscosity & Flow
        # Flux = Depth of groove.
        # Ease = Sigmoid(Flux - Threshold) -> Non-linear transition to flow
        flux_val = self.flux[disc_state]
        threshold = 2.0  # Mastery threshold
        
        # Sigmoid Ease: Low flux=0 ease, High flux=1 ease. Steep transition.
        ease = 1 / (1 + np.exp(-(flux_val - threshold)))
        
        # Hybrid Decision: Q (Goal) + Ease (Flow) + Curiosity (Novelty)
        beta = self.params.get('beta', 1.5)
        priority = self.q_table[disc_state] + (ease * beta) + curiosity
        
        if np.random.rand() < self.params['epsilon']:
            return np.random.randint(self.action_size)
        return np.argmax(priority)

    def learn(self, state, action, reward, next_state):
        s = self.discretize(state)
        ns = self.discretize(next_state)
        # Q-Learning Update (alpha=0.2, gamma=0.999)
        best_next = np.max(self.q_table[ns])
        
        # Reward shaping: YIN AND YANG (balanced positive & negative)
        # Use ENERGY CHANGE as reward signal: Φ(s') - Φ(s)
        # Energy up = positive force, Energy down = equally negative force
        
        # Potential function: height (sin(3*pos)) + kinetic energy (vel²)
        # sin(3*pos) is the actual height in MountainCar's cosine landscape
        import math
        phi_now = math.sin(3 * next_state[0]) + 100 * (next_state[1] ** 2)
        phi_prev = math.sin(3 * state[0]) + 100 * (state[1] ** 2)
        energy_delta = phi_now - phi_prev  # Positive when gaining, negative when losing
        
        # SLIGHT ASYMMETRY: Losing energy stings 15% more than gaining feels good
        # Prevents complacency near the goal — can't rest, must push through
        if energy_delta < 0:
            energy_delta *= 1.15
        
        shaped_reward = reward + energy_delta * 10.0  # Scale for learning speed
        self.q_table[s, action] += 0.2 * (shaped_reward + 0.999 * best_next - self.q_table[s, action])
        
        # FLUX UPDATE (Yin-Yang Habit System)
        position, velocity = next_state
        energy = (position**2) + (velocity**2)
        
        # LOGARITHMIC FLUX: Amplify small gains, punish stagnation
        # Real muscle memory isn't linear; it's log-scale.
        if energy > 0.1:
            # Impact = log(1 + energy*100) -> 0.1 energy gives ~2.4 impact
            impact = np.log(1 + energy * 100) * 0.5
            self.flux[s, action] += impact
        elif energy < 0.05:
            # Pain = log(1 + missing*50) -> 0.0 energy gives ~1.6 pain
            pain = np.log(1 + (0.05 - energy) * 50) * 0.5
            self.flux[s, action] -= pain
            
        self.flux[s, action] *= (1.0 - self.params['decay'])
        # Cap flux between -5.0 (avoidance) and 20.0 (deep mastery)
        self.flux[s, action] = np.clip(self.flux[s, action], -5.0, 20.0)

        return energy  # Return metric for logging
