import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from core.steering import SteeringController
from core.tda import TopologicalBrain
from core.agent import QSMA_Agent
import argparse
from datetime import datetime

# Timestamp for this run — used in all filenames and plots
RUN_TIMESTAMP = datetime.now().strftime('%Y-%m-%d_%H%M')

# --- CONFIGURATION ---
SAVE_PLOTS = True
TOTAL_EPISODES = 2000
TDA_INTERVAL = 5  # Run TDA every N episodes (continuous metacognitive loop)

def plot_phase_space(brain, title):
    if not SAVE_PLOTS or len(brain.buffer) < 10:
        return
    data = np.array(brain.buffer)
    plt.figure(figsize=(10, 6))
    plt.scatter(data[:, 0], data[:, 1], c=data[:, 5], cmap='inferno', alpha=0.3, s=2)
    plt.title(title)
    plt.xlabel("Position (Gravity Well)")
    plt.ylabel("Velocity (Momentum)")
    plt.axvline(x=0.5, color='cyan', linestyle='--', label='Goal')
    plt.axvline(x=-0.5, color='gray', linestyle=':', label='Bottom')
    plt.colorbar(label='Mechanical Energy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    fname = f'{title.replace(" ", "_").replace("[", "").replace("]", "")}.png'
    plt.figtext(0.01, 0.01, f'Run: {RUN_TIMESTAMP}', fontsize=8, alpha=0.5)
    plt.savefig(fname)
    print(f"📊 Plot saved: {fname}")
    plt.close()

def run_experiment():
    parser = argparse.ArgumentParser()
    parser.add_argument('--render', action='store_true', help='Render the environment visually')
    args = parser.parse_args()

    render_mode = 'human' if args.render else None
    env = gym.make('MountainCar-v0', render_mode=render_mode)
    
    ctrl = SteeringController()
    brain = TopologicalBrain(ctrl)
    agent = QSMA_Agent()

    success_count = 0
    max_positions = []
    tda_events = []  # Track when TDA fired and what it did

    print(f"🚀 CONTINUOUS METACOGNITIVE LOOP ({TOTAL_EPISODES} Episodes)")
    print(f"   TDA runs every {TDA_INTERVAL} episodes — recursive self-correction\n")

    for episode in range(TOTAL_EPISODES):
        state, _ = env.reset()
        done = False
        steps = 0
        max_pos = -1.2

        # Sync agent with latest steering params EVERY episode
        agent.sync(ctrl)
        
        # CONFIDENCE SCALING: Beta decays as Q-values mature
        # Early: flux=habit drives exploration. Late: Q=logic dominates.
        beta = max(0.1, 1.5 * (0.995 ** episode))
        agent.params['beta'] = beta

        while not done and steps < 500:
            action = agent.act(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Learn & log 6D
            energy = agent.learn(state, action, reward, next_state)
            s = agent.discretize(state)
            q = agent.q_table[s, action]
            flux = agent.flux[s, action]
            delta = abs(reward + 0.999 * np.max(agent.q_table[agent.discretize(next_state)]) - agent.q_table[s, action])
            brain.add_log([state[0], state[1], q, flux, delta, energy])

            state = next_state
            steps += 1
            max_pos = max(max_pos, state[0])

        max_positions.append(max_pos)

        if state[0] >= 0.5:
            success_count += 1
            print(f"Ep {episode}: 🚩 SUCCESS in {steps} steps! (Total: {success_count})")
        elif max_pos > 0.2:
            print(f"Ep {episode}: Almost... {max_pos:.2f}")
        elif episode % 10 == 0:
            print(f"Ep {episode}: MaxPos={max_pos:.2f}, Beta={beta:.2f}, Decay={ctrl.decay_rate:.3f}, Explr={ctrl.exploration_rate:.3f}")

        # =============================================
        # RECURSIVE TDA: Run every N episodes
        # The agent learns from its OWN behavioral topology
        # =============================================
        if (episode + 1) % TDA_INTERVAL == 0 and len(brain.buffer) >= 100:
            brain.analyze()
            ctrl.normalize()  # Gently relax params back toward baseline
            tda_events.append(episode)
    
    # === RESULTS ===
    print(f"\n{'='*50}")
    print(f"Total Successes: {success_count}/{TOTAL_EPISODES}")
    print(f"TDA Interventions: {len(tda_events)}")
    if success_count > 0:
        print("✅ The Metacognitive Loop successfully broke the habit.")
    else:
        print("❌ Agent did not escape. Try more episodes or tuning.")
    print(f"{'='*50}")

    plot_phase_space(brain, f"Final Phase Space [{RUN_TIMESTAMP}]")

    # Results plot with TDA events marked
    plt.figure(figsize=(14, 7))
    plt.plot(max_positions, 'o-', markersize=2, alpha=0.7, label='Max Position')
    plt.axhline(y=0.5, color='r', linestyle='--', label='Goal (0.5)')
    for ev in tda_events:
        plt.axvline(x=ev, color='orange', alpha=0.15, linewidth=1)
    plt.title(f"Mountain Car: Continuous TDA Loop — {success_count} Successes [{RUN_TIMESTAMP}]")
    plt.xlabel("Episode")
    plt.ylabel("Max Position")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.figtext(0.01, 0.01, f'Run: {RUN_TIMESTAMP} | Episodes: {TOTAL_EPISODES} | TDA Interval: {TDA_INTERVAL} | Successes: {success_count}', fontsize=8, alpha=0.5)
    results_fname = f'experiment_results_{RUN_TIMESTAMP}.png'
    plt.savefig(results_fname)
    plt.savefig('experiment_results.png')  # Also save as latest
    print(f"📊 Results saved to {results_fname}")
    plt.close()

    env.close()

if __name__ == "__main__":
    run_experiment()
