"""
Niodoo Physics Test Bed — Mountain Car Edition
================================================
This solver uses the EXACT same force vocabulary as Niodoo's LLM physics engine
(main.rs PrincipiaEngine) to solve Mountain Car. Every force concept maps 1:1:

    Niodoo LLM Force          →  Mountain Car Equivalent
    ─────────────────────────────────────────────────────
    Gravity Well (prompt pull) →  Attractor toward goal region
    Repulsion (anti-boring)    →  Push away from recently visited states
    Viscosity (sleepwalk det.) →  Detect low-momentum oscillation traps
    Adrenaline (spike burst)   →  Temporary exploration burst when stuck
    Ghost Vector (injected)    →  Memory of high-energy successful states
    Dynamic Ramp (tokens 0-10) →  Gradual force engagement over first N steps
    Physics Blend              →  Weight of physics forces in action selection

Config params match Niodoo's CLI (--physics-blend, --ghost-gravity, etc.)
Telemetry output matches Niodoo's JSONL format for cross-system comparison.

Author: Niodoo Project (Mountain Car Validation)
"""

import gymnasium as gym
import numpy as np
import json
import os
import datetime
from collections import deque

# =============================================================================
# NIODOO CONFIGURATION — Same parameter names as main.rs CLI
# =============================================================================
CONFIG = {
    "physics_blend":      1.5,    # Force magnitude (higher = more steering)
    "ghost_gravity":      10.0,   # Pull toward ghost vector (injected attractor)
    "repulsion_strength": -0.5,   # Push away from visited states (negative = repel)
    "gravity_well":       0.2,    # Pull toward goal region (high elasticity)
    "temperature":        0.7,    # Decision stochasticity
    "viscosity_threshold": 0.02,  # Momentum below this = "sleepwalking"
    "adrenaline_strength": 5.0,   # Burst force when stuck
    "ramp_start":         4,      # Zero physics for first N steps
    "ramp_end":           10,     # Full physics after N steps
    "decay_lambda":       0.01,   # Memory decay rate
    "seed":               42,
}

# =============================================================================
# TELEMETRY — Per-step force data in Niodoo JSONL format
# =============================================================================
class NiodooTelemetry:
    """Mirrors NiodooLogger from main.rs — writes per-step forces to JSONL."""

    def __init__(self, config, log_dir="logs"):
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.path = os.path.join(log_dir, f"niodoo_mountaincar_{timestamp}.jsonl")
        self.file = open(self.path, 'w')
        self.config = config
        self._log({"_type": "session_start", "config": config})

    def log_step(self, episode, step, state, forces, action, ramp_factor):
        """Log per-step telemetry matching Niodoo's TokenPhysics struct."""
        self._log({
            "_type": "token_physics",
            "episode": episode,
            "step": step,
            "position": float(state[0]),
            "velocity": float(state[1]),
            "gravity_force": float(forces.get("gravity", 0)),
            "ghost_force": float(forces.get("ghost", 0)),
            "repulsion_force": float(forces.get("repulsion", 0)),
            "viscosity_force": float(forces.get("viscosity", 0)),
            "adrenaline_force": float(forces.get("adrenaline", 0)),
            "total_force": float(forces.get("total", 0)),
            "ramp_factor": float(ramp_factor),
            "action": int(action),
        })

    def log_event(self, event_type, detail):
        """Log physics events (adrenaline spikes, viscosity detection, etc.)."""
        self._log({"_type": event_type, "detail": detail})

    def finalize(self, stats):
        self._log({"_type": "session_complete", **stats})
        self.file.close()
        print(f"  [TELEMETRY] Saved to {self.path}")

    def _log(self, data):
        data["_timestamp"] = datetime.datetime.now().isoformat()
        self.file.write(json.dumps(data) + "\n")


# =============================================================================
# NIODOO PHYSICS ENGINE — Mirrors PrincipiaEngine from main.rs
# =============================================================================
class NiodooPhysics:
    """
    Physics engine for Mountain Car using Niodoo's exact force vocabulary.
    
    Force Model:
        F_total = ramp(t) * physics_blend * (
            gravity_well * F_gravity +
            ghost_gravity * F_ghost +
            repulsion_strength * F_repulsion +
            F_viscosity +
            F_adrenaline
        )
    """

    def __init__(self, config):
        self.config = config
        self.rng = np.random.default_rng(config["seed"])

        # State tracking (mirrors sentence_history in main.rs)
        self.visited_states = deque(maxlen=200)   # Recent state history
        self.ghost_memory = deque(maxlen=50)      # Successful high-energy states
        self.step_count = 0
        self.episode_count = 0

        # Adrenaline state
        self.adrenaline_active = False
        self.adrenaline_cooldown = 0
        self.stuck_counter = 0

        # Energy tracking
        self.prev_energy = None
        self.energy_history = deque(maxlen=30)

    def reset_episode(self, episode_num):
        """Reset per-episode state (ghost memory persists across episodes)."""
        self.visited_states.clear()
        self.step_count = 0
        self.episode_count = episode_num
        self.adrenaline_active = False
        self.adrenaline_cooldown = 0
        self.stuck_counter = 0
        self.prev_energy = None
        self.energy_history.clear()

    # ─── FORCE COMPUTATIONS ───────────────────────────────────────────

    def compute_gravity(self, state):
        """
        Gravity Well: Pulls toward the goal region (position ≥ 0.5).
        Mirrors GRAVITY_WELL constant in main.rs.
        
        In Niodoo LLM: pulls token trajectory toward prompt context.
        In Mountain Car: pulls toward the right hill (goal).
        """
        position, velocity = state
        # Force increases as we get closer to goal (elastic pull)
        # But weakens at the bottom of the valley (lets the car build momentum)
        goal_pos = 0.5
        distance_to_goal = goal_pos - position
        
        # Elastic: stronger pull when closer (like rubber band snap-back)
        force = self.config["gravity_well"] * np.tanh(distance_to_goal * 2.0)
        return force

    def compute_repulsion(self, state):
        """
        Repulsion Field: Pushes away from recently visited states.
        Mirrors NIODOO_REPULSION in main.rs (the "Anti-Boring" field).
        
        In Niodoo LLM: prevents token repetition / loops.
        In Mountain Car: prevents getting stuck in the same region.
        """
        if len(self.visited_states) < 5:
            return 0.0

        position, velocity = state
        
        # Count how many recent states are "close" to current
        recent = np.array(list(self.visited_states)[-50:])
        distances = np.sqrt((recent[:, 0] - position)**2 + 
                          (recent[:, 1] - velocity)**2)
        
        # More nearby states = stronger repulsion (negative = push away)
        nearby = np.sum(distances < 0.05)
        density = nearby / len(recent)
        
        # Repulsion is strongest when we're stuck in a dense cluster
        # Direction: push in the direction of velocity (escape the trap)
        if abs(velocity) > 1e-6:
            repulsion_dir = np.sign(velocity)
        else:
            repulsion_dir = 1.0 if position < -0.2 else -1.0
            
        force = self.config["repulsion_strength"] * density * repulsion_dir * (-1)
        return force

    def compute_ghost(self, state):
        """
        Ghost Vector: Injected attractor from memory of successful states.
        Mirrors ghost_gravity and ghost_vector in main.rs.
        
        In Niodoo LLM: pulls toward prompt embedding / injected concepts.
        In Mountain Car: pulls toward remembered high-energy states.
        """
        if len(self.ghost_memory) == 0:
            return 0.0

        position, velocity = state
        
        # Ghost = average of remembered high-energy positions
        ghost_positions = [s[0] for s in self.ghost_memory]
        ghost_center = np.mean(ghost_positions)
        
        # Pull toward ghost center (weighted by ghost_gravity)
        displacement = ghost_center - position
        
        # Decay with distance (avoid pulling when very far)
        force = self.config["ghost_gravity"] * np.tanh(displacement) * 0.01
        return force

    def compute_viscosity(self, state):
        """
        Viscosity Detection: Detects "sleepwalking" — low-energy oscillation.
        Mirrors [VISCOSITY] Sleepwalking Detected! in main.rs telemetry.
        
        In Niodoo LLM: detects when model momentum → 0 (stuck in loop).
        In Mountain Car: detects when car oscillates without gaining energy.
        """
        position, velocity = state
        
        if len(self.energy_history) < 10:
            return 0.0

        # Check if energy is flat (sleepwalking)
        energies = list(self.energy_history)
        energy_range = max(energies[-10:]) - min(energies[-10:])
        
        if energy_range < self.config["viscosity_threshold"]:
            # Sleepwalking detected! Apply viscosity force
            # Push harder in velocity direction to break the equilibrium
            self.stuck_counter += 1
            
            if abs(velocity) > 1e-6:
                force = 0.5 * np.sign(velocity)
            else:
                force = 0.5 * (1 if self.rng.random() > 0.5 else -1)
            return force
        
        self.stuck_counter = max(0, self.stuck_counter - 1)
        return 0.0

    def compute_adrenaline(self, state):
        """
        Adrenaline Spike: Emergency burst when deeply stuck.
        Mirrors [ADRENALINE] SHOT! and [REQUEST: SPIKE] in main.rs.
        
        In Niodoo LLM: model requests physics override when boredom=1.0.
        In Mountain Car: massive force injection to break out of low-energy trap.
        """
        if self.adrenaline_cooldown > 0:
            self.adrenaline_cooldown -= 1
            return 0.0

        # Trigger condition: stuck for too many steps
        if self.stuck_counter > 15:
            self.adrenaline_active = True
            self.adrenaline_cooldown = 20  # Cooldown before next spike
            self.stuck_counter = 0
            
            position, velocity = state
            # Massive push in velocity direction (or random if stationary)
            if abs(velocity) > 1e-6:
                force = self.config["adrenaline_strength"] * np.sign(velocity)
            else:
                force = self.config["adrenaline_strength"] * (1 if self.rng.random() > 0.5 else -1)
            return force
        
        self.adrenaline_active = False
        return 0.0

    def dynamic_ramp(self, step):
        """
        Dynamic Ramp: Gradual force engagement.
        Mirrors NIODOO_RAMP_START/END in main.rs.
        
        In Niodoo LLM: prevents garbage tokens at sentence start.
        In Mountain Car: lets the car establish initial momentum before steering.
        """
        if step < self.config["ramp_start"]:
            return 0.0
        elif step < self.config["ramp_end"]:
            progress = (step - self.config["ramp_start"]) / \
                       (self.config["ramp_end"] - self.config["ramp_start"])
            return progress
        return 1.0

    # ─── MAIN POLICY ─────────────────────────────────────────────────

    def compute_action(self, state, step):
        """
        The Niodoo Policy: Blends all forces to choose an action.
        
        Mirrors the force computation loop in main.rs main() function:
            total_force = ramp * blend * (gravity + ghost + repulsion + ...)
        """
        position, velocity = state
        self.step_count = step

        # Track state
        self.visited_states.append(np.array([position, velocity]))

        # Compute mechanical energy (for viscosity detection)
        energy = self._mechanical_energy(state)
        self.energy_history.append(energy)
        self.prev_energy = energy

        # Store high-energy states as ghost memory
        if energy > -0.05 and position > 0.0:
            self.ghost_memory.append(np.array([position, velocity]))

        # ── Compute all forces ──
        f_gravity    = self.compute_gravity(state)
        f_repulsion  = self.compute_repulsion(state)
        f_ghost      = self.compute_ghost(state)
        f_viscosity  = self.compute_viscosity(state)
        f_adrenaline = self.compute_adrenaline(state)

        # ── Blend forces (mirrors main.rs physics blending) ──
        ramp = self.dynamic_ramp(step)
        blend = self.config["physics_blend"]

        total_force = ramp * blend * (
            f_gravity + f_repulsion + f_ghost + f_viscosity + f_adrenaline
        )

        # ── Force-based decision ──
        # During ramp-up: use energy-pump baseline
        if ramp < 0.5:
            # Fallback: simple energy pump (push with velocity)
            if velocity >= 0:
                action = 2  # right
            else:
                action = 0  # left
        else:
            # Temperature-based stochastic decision
            temp = self.config["temperature"]
            
            # Core: direction of net force determines action
            if total_force > temp * 0.1:
                action = 2  # right (net force pushes right)
            elif total_force < -temp * 0.1:
                action = 0  # left (net force pushes left)
            else:
                # In the "wobble zone" — use velocity-based energy pumping
                # This is the "snap-back" behavior from Niodoo v3.1
                if velocity >= 0:
                    action = 2
                else:
                    action = 0

        # Build forces dict for telemetry
        forces = {
            "gravity": f_gravity,
            "ghost": f_ghost,
            "repulsion": f_repulsion,
            "viscosity": f_viscosity,
            "adrenaline": f_adrenaline,
            "total": total_force,
        }

        return action, forces, ramp

    def _mechanical_energy(self, state):
        """Compute total mechanical energy (KE + PE)."""
        position, velocity = state
        # Mountain Car physics: gravity = 0.0025, PE = cos(3*position)
        kinetic = 0.5 * velocity**2
        potential = 0.0025 * np.cos(3 * position)  # Simplified
        return kinetic + potential


# =============================================================================
# EXPERIMENT RUNNER
# =============================================================================

TOTAL_EPISODES = 2000

def run_experiment(config=None):
    """Run Niodoo physics solver on Mountain Car for TOTAL_EPISODES."""
    if config is None:
        config = CONFIG

    env = gym.make('MountainCar-v0')
    physics = NiodooPhysics(config)
    telemetry = NiodooTelemetry(config)

    success_count = 0
    total_steps = 0
    win_steps = []
    max_positions = []
    adrenaline_events = 0
    viscosity_events = 0

    print("=" * 70)
    print("  NIODOO PHYSICS ENGINE — Mountain Car Test Bed")
    print("=" * 70)
    print(f"  physics_blend:      {config['physics_blend']}")
    print(f"  ghost_gravity:      {config['ghost_gravity']}")
    print(f"  repulsion_strength: {config['repulsion_strength']}")
    print(f"  gravity_well:       {config['gravity_well']}")
    print(f"  temperature:        {config['temperature']}")
    print(f"  seed:               {config['seed']}")
    print(f"  episodes:           {TOTAL_EPISODES}")
    print("=" * 70)

    for episode in range(TOTAL_EPISODES):
        state, _ = env.reset(seed=config["seed"] + episode)
        physics.reset_episode(episode)
        
        episode_max_pos = state[0]
        won = False

        for step in range(200):  # MountainCar-v0 has 200 step limit
            action, forces, ramp = physics.compute_action(state, step)

            # Track events
            if forces["adrenaline"] != 0:
                adrenaline_events += 1
                telemetry.log_event("adrenaline_spike", {
                    "episode": episode, "step": step,
                    "force": float(forces["adrenaline"])
                })
            if forces["viscosity"] != 0:
                viscosity_events += 1

            # Log telemetry (every 10th step to keep file size manageable)
            if step % 10 == 0 or forces["adrenaline"] != 0:
                telemetry.log_step(episode, step, state, forces, action, ramp)

            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            episode_max_pos = max(episode_max_pos, next_state[0])
            
            if next_state[0] >= 0.5:
                won = True
                success_count += 1
                total_steps += step + 1
                win_steps.append(step + 1)
                break

            state = next_state

        max_positions.append(episode_max_pos)

        # Progress logging
        if (episode + 1) % 100 == 0:
            rate = success_count / (episode + 1) * 100
            avg = np.mean(win_steps[-100:]) if win_steps else 0
            ghost_size = len(physics.ghost_memory)
            print(f"  Episode {episode+1:4d}/{TOTAL_EPISODES} | "
                  f"Wins: {success_count:4d} ({rate:5.1f}%) | "
                  f"Avg Steps: {avg:5.1f} | "
                  f"Ghost Memory: {ghost_size:3d} | "
                  f"Adrenaline: {adrenaline_events} | "
                  f"Viscosity: {viscosity_events}")

    env.close()

    # ── Final Results ──
    print("\n" + "=" * 70)
    print("  RESULTS")
    print("=" * 70)
    print(f"  Total Wins:       {success_count}/{TOTAL_EPISODES}")
    print(f"  Win Rate:         {success_count/TOTAL_EPISODES*100:.1f}%")
    if win_steps:
        print(f"  Avg Steps to Win: {np.mean(win_steps):.1f}")
        print(f"  Min Steps to Win: {min(win_steps)}")
    print(f"  Adrenaline Spikes: {adrenaline_events}")
    print(f"  Viscosity Events:  {viscosity_events}")
    print(f"  Max Position:      {max(max_positions):.4f}")
    
    target = 1500
    if success_count >= target:
        print(f"\n  ✅ TARGET HIT: {success_count} ≥ {target}")
    else:
        print(f"\n  ❌ TARGET MISSED: {success_count} < {target}")
    print("=" * 70)

    # Finalize telemetry
    telemetry.finalize({
        "total_wins": success_count,
        "total_episodes": TOTAL_EPISODES,
        "win_rate": success_count / TOTAL_EPISODES,
        "avg_steps": float(np.mean(win_steps)) if win_steps else 0,
        "adrenaline_events": adrenaline_events,
        "viscosity_events": viscosity_events,
    })

    # ── Generate comparison plot ──
    _plot_results(max_positions, win_steps, success_count, config)

    return success_count


def _plot_results(max_positions, win_steps, success_count, config):
    """Generate diagnostic plots showing force dynamics."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [PLOT] matplotlib not available, skipping plots")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f'Niodoo Physics Engine — Mountain Car\n'
        f'blend={config["physics_blend"]}, ghost={config["ghost_gravity"]}, '
        f'rep={config["repulsion_strength"]}, grav={config["gravity_well"]}',
        fontsize=13
    )

    # 1. Max position per episode
    ax1 = axes[0, 0]
    ax1.plot(max_positions, alpha=0.3, linewidth=0.5, color='steelblue')
    window = 50
    if len(max_positions) >= window:
        rolling = np.convolve(max_positions, np.ones(window)/window, mode='valid')
        ax1.plot(range(window-1, len(max_positions)), rolling, color='navy', linewidth=2)
    ax1.axhline(y=0.5, color='green', linestyle='--', alpha=0.7, label='Goal (0.5)')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Max Position')
    ax1.set_title('Max Position Reached')
    ax1.legend()

    # 2. Rolling win rate
    ax2 = axes[0, 1]
    wins_binary = [1 if p >= 0.5 else 0 for p in max_positions]
    if len(wins_binary) >= window:
        rolling_rate = np.convolve(wins_binary, np.ones(window)/window, mode='valid') * 100
        ax2.plot(range(window-1, len(wins_binary)), rolling_rate, color='darkgreen', linewidth=2)
    ax2.axhline(y=75, color='orange', linestyle='--', alpha=0.7, label='75% target')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Win Rate (%)')
    ax2.set_title(f'Rolling {window}-Episode Win Rate')
    ax2.set_ylim(0, 105)
    ax2.legend()

    # 3. Steps to win (histogram)
    ax3 = axes[1, 0]
    if win_steps:
        ax3.hist(win_steps, bins=30, color='coral', edgecolor='darkred', alpha=0.8)
        ax3.axvline(x=np.mean(win_steps), color='red', linestyle='--',
                   label=f'Mean: {np.mean(win_steps):.0f}')
    ax3.set_xlabel('Steps to Win')
    ax3.set_ylabel('Count')
    ax3.set_title('Steps Distribution (Wins Only)')
    ax3.legend()

    # 4. Summary text
    ax4 = axes[1, 1]
    ax4.axis('off')
    summary = (
        f"NIODOO PHYSICS SUMMARY\n"
        f"{'─'*30}\n\n"
        f"Wins: {success_count}/{TOTAL_EPISODES}\n"
        f"Win Rate: {success_count/TOTAL_EPISODES*100:.1f}%\n"
        f"Avg Steps: {np.mean(win_steps):.1f}\n\n"
        f"FORCE MAPPING\n"
        f"{'─'*30}\n"
        f"Gravity Well → Goal attractor\n"
        f"Repulsion    → Anti-boring field\n"
        f"Ghost Vector → High-energy memory\n"
        f"Viscosity    → Sleepwalk detection\n"
        f"Adrenaline   → [REQUEST: SPIKE]\n"
        f"Dynamic Ramp → Token 0→10 ramp-up\n"
    )
    ax4.text(0.05, 0.95, summary, transform=ax4.transAxes,
            fontsize=11, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout()
    plot_path = 'niodoo_mountaincar_results.png'
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"  [PLOT] Saved to {plot_path}")


# =============================================================================
# ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    import sys
    
    # Allow config overrides from command line
    config = CONFIG.copy()
    for arg in sys.argv[1:]:
        if '=' in arg:
            key, val = arg.split('=', 1)
            key = key.lstrip('-').replace('-', '_')
            if key in config:
                config[key] = type(config[key])(val)
    
    wins = run_experiment(config)
