import numpy as np

class SteeringController:
    """
    The Actuator: Receives commands from the Topological Brain 
    and physically alters the Agent's parameters.
    """
    def __init__(self):
        self.decay_rate = 0.05     # Start LOW to let habits form first
        self.exploration_rate = 0.1
        self.attractors = []       # List of [pos, vel] targets
        self.spike_cooldown = 0    # Prevent repeated spikes

    def apply_decay_spike(self, intensity):
        """
        Triggered when a LOOP is detected.
        Has a cooldown to prevent runaway spiking.
        """
        if self.spike_cooldown > 0:
            self.spike_cooldown -= 1
            return  # Skip — already recovering from a spike
        
        original = self.decay_rate
        self.decay_rate = min(0.5, self.decay_rate + intensity * 0.3)
        self.spike_cooldown = 10  # Don't spike again for 10 TDA intervals
        
        print(f"⚡ STEERING: Decay Spike! Rate: {original:.2f} -> {self.decay_rate:.2f} (Cooldown: {self.spike_cooldown})")

    def inject_attractor(self, coordinates):
        """
        Triggered when a VOID is detected.
        Only injects if we don't already have this attractor nearby.
        """
        # Don't stack duplicate attractors
        for existing in self.attractors:
            dist = np.sqrt((existing[0] - coordinates[0])**2 + (existing[1] - coordinates[1])**2)
            if dist < 0.2:
                return  # Already have one nearby
        
        self.attractors.append(coordinates)
        self.exploration_rate = min(0.4, self.exploration_rate + 0.1)
        
        print(f"🧲 STEERING: Attractor Injected at Pos={coordinates[0]:.2f}, Vel={coordinates[1]:.2f}")
        print(f"   -> Exploration: {self.exploration_rate:.2f}")

    def normalize(self):
        """Gradually returns parameters to baseline after each TDA cycle."""
        self.decay_rate = max(0.05, self.decay_rate * 0.85)
        self.exploration_rate = max(0.01, self.exploration_rate * 0.90)  # Floor at 1%
        if self.spike_cooldown > 0:
            self.spike_cooldown -= 1
