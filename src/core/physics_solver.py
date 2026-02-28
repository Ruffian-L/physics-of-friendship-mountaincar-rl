"""
Mountain Car: Pure Physics-Based Solver
========================================
Uses energy pumping (resonance) — always accelerate in the direction of motion.
No Q-learning, no neural nets, no TDA. Just physics.

The insight: the car can't climb the hill with raw force (push=0.001 < gravity=0.0025).
It must BUILD kinetic energy by oscillating, like pumping a swing.

Optimal policy: push WITH your velocity. Moving right → push right. Moving left → push left.
"""

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

RUN_TIMESTAMP = datetime.now().strftime('%Y-%m-%d_%H%M')
TOTAL_EPISODES = 2000
SAVE_PLOTS = True


def energy_pump_policy(state):
    """
    Pure physics: always push in the direction of current velocity.
    - velocity >= 0 → action 2 (push right)
    - velocity < 0  → action 0 (push left)
    
    This maximizes work done on the car each timestep: W = F · v > 0 always.
    """
    position, velocity = state
    if velocity >= 0:
        return 2  # Push right
    else:
        return 0  # Push left


def energy_aware_policy(state):
    """
    Enhanced physics: energy pumping + gravity-aware momentum management.
    
    Near the bottom of the well, we want to build maximum speed.
    On the slopes, we want to push in the right direction to climb.
    
    The key physics:
    - Potential energy: V(x) ∝ sin(3x) (terrain height)
    - Gravity force: F_g = -0.0025 * cos(3x)
    - At x ≈ -π/6 ≈ -0.5236, cos(3x) = 0 → inflection point
    - Below this: gravity pulls left. Above: gravity pulls right (toward goal).
    
    Strategy: Always align push with velocity (energy pump), but be smarter
    about the inflection regions.
    """
    position, velocity = state
    
    # Core: always push with velocity (energy pumping)
    if velocity >= 0:
        return 2  # Push right
    else:
        return 0  # Push left


def compute_mechanical_energy(state):
    """Compute total mechanical energy: KE + PE."""
    position, velocity = state
    # MountainCar height function: sin(3*position)
    # Using the Gymnasium convention
    height = np.sin(3 * position)
    potential = 0.0025 * height  # gravity * height
    kinetic = 0.5 * velocity ** 2
    return kinetic + potential


def run_experiment(policy_fn, policy_name="Physics"):
    """Run the full experiment with a given policy function."""
    env = gym.make('MountainCar-v0')
    
    success_count = 0
    max_positions = []
    step_counts = []
    energies_over_time = []
    first_success_ep = None
    
    # Track rolling success rate
    window = 100
    rolling_successes = []
    recent_wins = 0

    print(f"\n{'='*60}")
    print(f"🏔️  MOUNTAIN CAR: {policy_name} Policy")
    print(f"   Episodes: {TOTAL_EPISODES}")
    print(f"   Strategy: Energy pumping — always push with velocity")
    print(f"{'='*60}\n")

    for episode in range(TOTAL_EPISODES):
        state, _ = env.reset()
        done = False
        steps = 0
        max_pos = -1.2
        ep_energies = []

        while not done and steps < 500:
            action = policy_fn(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            state = next_state
            steps += 1
            max_pos = max(max_pos, state[0])
            ep_energies.append(compute_mechanical_energy(state))

        max_positions.append(max_pos)
        
        won = state[0] >= 0.5
        if won:
            success_count += 1
            step_counts.append(steps)
            if first_success_ep is None:
                first_success_ep = episode
        
        # Rolling window tracking
        rolling_successes.append(1 if won else 0)
        if len(rolling_successes) > window:
            rolling_successes.pop(0)
        recent_wins = sum(rolling_successes)
        
        energies_over_time.append(np.max(ep_energies) if ep_energies else 0)

        # Progress logging
        if episode < 10 or episode % 100 == 0 or (won and success_count <= 5):
            status = "🚩 SUCCESS" if won else f"MaxPos={max_pos:.3f}"
            rolling_pct = 100 * recent_wins / len(rolling_successes)
            print(f"Ep {episode:4d}: {status:20s} | Steps={steps:3d} | "
                  f"Total={success_count} | Rolling={rolling_pct:.0f}%")

    # === RESULTS ===
    avg_steps = np.mean(step_counts) if step_counts else 0
    success_rate = 100 * success_count / TOTAL_EPISODES
    
    print(f"\n{'='*60}")
    print(f"📊 RESULTS: {policy_name} Policy")
    print(f"{'='*60}")
    print(f"   Total Successes: {success_count}/{TOTAL_EPISODES} ({success_rate:.1f}%)")
    print(f"   Avg Steps to Win: {avg_steps:.0f}")
    print(f"   First Success:    Episode {first_success_ep}")
    print(f"   Max Position Ever: {max(max_positions):.4f}")
    
    if success_count >= 1500:
        print(f"\n   ✅✅✅ TARGET HIT! {success_count} ≥ 1500 wins! ✅✅✅")
    elif success_count >= 1000:
        print(f"\n   ✅ Strong result! {success_count} wins.")
    elif success_count > 0:
        print(f"\n   ⚠️  Some wins but below target. Need tuning.")
    else:
        print(f"\n   ❌ No wins. Something is wrong.")
    print(f"{'='*60}")

    # === PLOTS ===
    if SAVE_PLOTS:
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle(f"Mountain Car: {policy_name} — {success_count}/{TOTAL_EPISODES} Wins "
                     f"({success_rate:.1f}%) [{RUN_TIMESTAMP}]", fontsize=14, fontweight='bold')
        
        # 1. Max position per episode
        ax = axes[0, 0]
        ax.plot(max_positions, 'o', markersize=1, alpha=0.5, color='steelblue')
        ax.axhline(y=0.5, color='red', linestyle='--', linewidth=1.5, label='Goal (0.5)')
        ax.set_title("Max Position per Episode")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Max Position")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Rolling success rate
        ax = axes[0, 1]
        rolling = []
        wins_in_window = 0
        win_list = [1 if mp >= 0.5 else 0 for mp in max_positions]
        for i in range(len(win_list)):
            wins_in_window += win_list[i]
            if i >= window:
                wins_in_window -= win_list[i - window]
            denom = min(i + 1, window)
            rolling.append(100 * wins_in_window / denom)
        ax.plot(rolling, color='green', linewidth=1.5)
        ax.axhline(y=75, color='orange', linestyle='--', alpha=0.5, label='75% target')
        ax.set_title(f"Rolling Success Rate (window={window})")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Success %")
        ax.set_ylim(0, 105)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. Step count distribution (wins only)
        ax = axes[1, 0]
        if step_counts:
            ax.hist(step_counts, bins=50, color='coral', edgecolor='black', alpha=0.7)
            ax.axvline(x=np.mean(step_counts), color='red', linestyle='--', 
                      label=f'Mean={np.mean(step_counts):.0f}')
            ax.set_title(f"Steps to Win (n={len(step_counts)})")
            ax.set_xlabel("Steps")
            ax.set_ylabel("Count")
            ax.legend()
        else:
            ax.text(0.5, 0.5, "No wins", ha='center', va='center', fontsize=16)
            ax.set_title("Steps to Win")
        ax.grid(True, alpha=0.3)
        
        # 4. Max energy per episode
        ax = axes[1, 1]
        ax.plot(energies_over_time, color='purple', alpha=0.5, linewidth=0.5)
        ax.set_title("Max Mechanical Energy per Episode")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Energy")
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        fname = f'physics_results_{RUN_TIMESTAMP}.png'
        plt.savefig(fname, dpi=120)
        plt.savefig('physics_results.png', dpi=120)
        print(f"\n📊 Plot saved: {fname}")
        plt.close()

    env.close()
    return success_count


if __name__ == "__main__":
    print("=" * 60)
    print("  MOUNTAIN CAR: PURE PHYSICS SOLVER")
    print("  Strategy: Energy Pumping (push with velocity)")
    print("=" * 60)
    
    wins = run_experiment(energy_pump_policy, "Energy Pump")
