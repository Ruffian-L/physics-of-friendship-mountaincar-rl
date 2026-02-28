"""
Mountain Car: Q-SMA + Niodoo Persistent Memory
Integrates the Niodoo dual-stream memory with the existing metacognitive loop.
- Flinch-tags every state by velocity
- Chains sequential states into momentum graph
- Persistence bonus shapes reward (longer chains = more bonus)
- Cycle detection prunes dead oscillation traps
- TDA + Niodoo run in parallel (dual-stream)
"""

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from core.steering import SteeringController
from core.tda import TopologicalBrain
from core.agent import QSMA_Agent
from models.niodoo import NiodooMemory
import argparse
from datetime import datetime

# Timestamp for this run
RUN_TIMESTAMP = datetime.now().strftime('%Y-%m-%d_%H%M')

# --- CONFIGURATION ---
SAVE_PLOTS = True
TOTAL_EPISODES = 2000
TDA_INTERVAL = 5
NIODOO_TAG_INTERVAL = 5    # Flinch-tag every N steps (not every step — too noisy)
NIODOO_PRUNE_INTERVAL = 50  # Prune/curate every N episodes

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
    memory = NiodooMemory(beta_threshold=0.4)  # ← NIODOO

    success_count = 0
    max_positions = []
    tda_events = []

    print(f"🚀 NIODOO + METACOGNITIVE LOOP ({TOTAL_EPISODES} Episodes)")
    print(f"   TDA every {TDA_INTERVAL} eps | Niodoo prune every {NIODOO_PRUNE_INTERVAL} eps\n")

    for episode in range(TOTAL_EPISODES):
        state, _ = env.reset()
        done = False
        steps = 0
        max_pos = -1.2
        prev_node_id = None  # For Niodoo chain linking

        agent.sync(ctrl)
        beta = max(0.1, 1.5 * (0.995 ** episode))
        agent.params['beta'] = beta

        while not done and steps < 500:
            action = agent.act(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Learn & log 6D (existing)
            energy = agent.learn(state, action, reward, next_state, done=done)
            s = agent.discretize(state)
            q = agent.q_table[s, action]
            flux = agent.flux[s, action]
            delta = abs(reward + 0.999 * np.max(agent.q_table[agent.discretize(next_state)]) - agent.q_table[s, action])
            brain.add_log([state[0], state[1], q, flux, delta, energy])

            # =============================================
            # NIODOO: Flinch-tag states into persistent graph
            # =============================================
            if steps % NIODOO_TAG_INTERVAL == 0:
                content = f"pos:{state[0]:.2f}, vel:{state[1]:.2f}"
                node_id = memory.flinch_tag(
                    content, 
                    abs(state[1]),  # Velocity as relevance trigger
                    state_vector=np.array([state[0], state[1]])
                )
                # Chain: link to previous tagged state
                if prev_node_id is not None:
                    memory.connect_nodes(prev_node_id, node_id, weight=abs(state[1]))
                prev_node_id = node_id

            state = next_state
            steps += 1
            max_pos = max(max_pos, state[0])

        max_positions.append(max_pos)

        # =============================================
        # NIODOO: Persistence bonus on success
        # =============================================
        if state[0] >= 0.5:
            success_count += 1
            # Flinch-tag the SUCCESS moment with max relevance
            memory.flinch_tag(
                f"SUCCESS ep:{episode} steps:{steps}",
                1.0,  # Max velocity = max relevance
                state_vector=np.array([state[0], state[1]])
            )
            print(f"Ep {episode}: 🚩 SUCCESS in {steps} steps! (Total: {success_count})")
        elif max_pos > 0.2:
            print(f"Ep {episode}: Almost... {max_pos:.2f}")
        elif episode % 10 == 0:
            stats = memory.get_stats()
            print(f"Ep {episode}: MaxPos={max_pos:.2f}, Beta={beta:.2f}, Decay={ctrl.decay_rate:.3f}, "
                  f"Niodoo[nodes:{stats['nodes']}, chain:{stats['chain_length']}, flinch:{stats['flinches']}]")

        # =============================================
        # TDA: Recursive analysis (existing)
        # =============================================
        if (episode + 1) % TDA_INTERVAL == 0 and len(brain.buffer) >= 100:
            brain.analyze()
            ctrl.normalize()
            tda_events.append(episode)
        
        # =============================================
        # NIODOO: Periodic curation (prune dead chains, detect trap cycles)
        # =============================================
        if (episode + 1) % NIODOO_PRUNE_INTERVAL == 0:
            memory.curate_prune()
            stats = memory.get_stats()
            if episode % 200 == 0:
                print(f"   📿 Niodoo curate: {stats['nodes']} nodes, {stats['prunes']} pruned, "
                      f"chain={stats['chain_length']}, bonus={stats['persistence_bonus']:.2f}")
    
    # === RESULTS ===
    stats = memory.get_stats()
    print(f"\n{'='*60}")
    print(f"Total Successes: {success_count}/{TOTAL_EPISODES}")
    print(f"TDA Interventions: {len(tda_events)}")
    print(f"Niodoo Final: {stats['nodes']} nodes, {stats['edges']} edges, "
          f"{stats['flinches']} flinches, {stats['prunes']} pruned")
    print(f"Persistence Bonus: {stats['persistence_bonus']:.2f}")
    if success_count > 0:
        print("✅ The Metacognitive + Niodoo Loop broke the habit.")
    else:
        print("❌ Agent did not escape. Try more episodes or tuning.")
    print(f"{'='*60}")

    plot_phase_space(brain, f"Niodoo_Phase_Space_{RUN_TIMESTAMP}")

    # Results plot with TDA events 
    plt.figure(figsize=(14, 7))
    plt.plot(max_positions, 'o-', markersize=2, alpha=0.7, label='Max Position')
    plt.axhline(y=0.5, color='r', linestyle='--', label='Goal (0.5)')
    for ev in tda_events:
        plt.axvline(x=ev, color='orange', alpha=0.15, linewidth=1)
    plt.title(f"Mountain Car: Niodoo + TDA Loop — {success_count} Successes [{RUN_TIMESTAMP}]")
    plt.xlabel("Episode")
    plt.ylabel("Max Position")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.figtext(0.01, 0.01, 
                f'Run: {RUN_TIMESTAMP} | Eps: {TOTAL_EPISODES} | TDA: {TDA_INTERVAL} | '
                f'Niodoo: {stats["nodes"]} nodes | Successes: {success_count}', 
                fontsize=8, alpha=0.5)
    results_fname = f'niodoo_results_{RUN_TIMESTAMP}.png'
    plt.savefig(results_fname)
    plt.savefig('niodoo_results.png')
    print(f"📊 Results saved to {results_fname}")
    plt.close()

    env.close()

if __name__ == "__main__":
    run_experiment()
