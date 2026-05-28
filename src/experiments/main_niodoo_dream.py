"""
Mountain Car: Q-SMA + Niodoo + DREAM CYCLE v5 (Holistic)
Mean-center TD errors (normalize gradient, not landscape).
Gold-only flux etching (energy > 0.15).
All transitions in replay buffer — mean-centering handles the balance.
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

RUN_TIMESTAMP = datetime.now().strftime('%Y-%m-%d_%H%M')

SAVE_PLOTS = True
TOTAL_EPISODES = 2000
TDA_INTERVAL = 5
NIODOO_TAG_INTERVAL = 5
NIODOO_PRUNE_INTERVAL = 50
DREAM_INTERVAL = 5       # Dream every 5 episodes
DREAM_BATCH_SIZE = 512    # Replay batch size

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
    parser.add_argument('--render', action='store_true')
    args = parser.parse_args()

    render_mode = 'human' if args.render else None
    env = gym.make('MountainCar-v0', render_mode=render_mode)
    
    ctrl = SteeringController()
    brain = TopologicalBrain(ctrl)
    agent = QSMA_Agent()
    memory = NiodooMemory(beta_threshold=0.4)

    success_count = 0
    max_positions = []
    tda_events = []
    dream_count = 0

    print(f"🚀 NIODOO + HOLISTIC DREAM v5 ({TOTAL_EPISODES} Episodes)")
    print(f"   Dream: every {DREAM_INTERVAL} eps, batch {DREAM_BATCH_SIZE}")
    print(f"   Mean-center TD errors | Gold flux (energy>0.15) | No global decay\n")

    for episode in range(TOTAL_EPISODES):
        state, _ = env.reset()
        done = False
        steps = 0
        max_pos = -1.2
        prev_node_id = None

        agent.sync(ctrl)
        beta = max(0.1, 1.5 * (0.995 ** episode))
        agent.params['beta'] = beta

        while not done and steps < 500:
            action = agent.act(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Standard learn (untouched contract)
            energy = agent.learn(state, action, reward, next_state, done=done)

            # Store transition for dream replay
            agent.remember(state, action, reward, next_state, energy)

            # Log 6D for TDA
            s = agent.discretize(state)
            ns = agent.discretize(next_state)
            q = agent.q_table[s, action]
            flux = agent.flux[s, action]
            delta = abs(reward + 0.999 * np.max(agent.q_table[ns]) - q)
            brain.add_log([state[0], state[1], q, flux, delta, energy])

            # Niodoo flinch-tag 
            if steps % NIODOO_TAG_INTERVAL == 0:
                content = f"pos:{state[0]:.2f}, vel:{state[1]:.2f}"
                node_id = memory.flinch_tag(content, abs(state[1]), np.array([state[0], state[1]]))
                if prev_node_id is not None:
                    memory.connect_nodes(prev_node_id, node_id, weight=abs(state[1]))
                prev_node_id = node_id

            state = next_state
            steps += 1
            max_pos = max(max_pos, state[0])

        max_positions.append(max_pos)

        ep_success = state[0] >= 0.5
        if ep_success:
            success_count += 1
            memory.flinch_tag(f"SUCCESS ep:{episode} steps:{steps}", 1.0, np.array([state[0], state[1]]))
            print(f"Ep {episode}: 🚩 SUCCESS in {steps} steps! (Total: {success_count})")
        elif max_pos > 0.2:
            print(f"Ep {episode}: Almost... {max_pos:.2f}")
        elif episode % 10 == 0:
            stats = memory.get_stats()
            print(f"Ep {episode}: MaxPos={max_pos:.2f}, Beta={beta:.2f}, "
                  f"Dreams={dream_count}, Niodoo[n:{stats['nodes']}, f:{stats['flinches']}]")

        # Commit episode to success buffer if won
        agent.commit_episode(ep_success)

        # === DREAM CYCLE: Post-episode consolidation ===
        if (episode + 1) % DREAM_INTERVAL == 0:
            dreamed = agent.dream_cycle(batch_size=DREAM_BATCH_SIZE)
            if dreamed:
                dream_count += 1
                if dream_count <= 5 or dream_count % 50 == 0:
                    print(f"   🌙 DREAM #{dream_count}: Consolidated replay — "
                          f"Q range [{agent.q_table.min():.1f}, {agent.q_table.max():.1f}], "
                          f"Flux range [{agent.flux.min():.1f}, {agent.flux.max():.1f}]")

        # TDA
        if (episode + 1) % TDA_INTERVAL == 0 and len(brain.buffer) >= 100:
            brain.analyze()
            ctrl.normalize()
            tda_events.append(episode)
        
        # Niodoo prune
        if (episode + 1) % NIODOO_PRUNE_INTERVAL == 0:
            memory.curate_prune()
    
    # === RESULTS ===
    stats = memory.get_stats()
    print(f"\n{'='*60}")
    print(f"Total Successes: {success_count}/{TOTAL_EPISODES} ({100*success_count/TOTAL_EPISODES:.1f}%)")
    print(f"Dream Cycles: {dream_count}")
    print(f"TDA Interventions: {len(tda_events)}")
    print(f"Niodoo: {stats['nodes']} nodes, {stats['flinches']} flinches, {stats['prunes']} pruned")
    print(f"Persistence Bonus: {memory.persistence_bonus():.2f}")
    if success_count > 0:
        print("✅ The Dream Cycle + Niodoo Loop locked in.")
    else:
        print("❌ Agent did not escape.")
    print(f"{'='*60}")

    plot_phase_space(brain, f"Dream_Phase_Space_{RUN_TIMESTAMP}")

    plt.figure(figsize=(14, 7))
    plt.plot(max_positions, 'o-', markersize=2, alpha=0.7, label='Max Position')
    plt.axhline(y=0.5, color='r', linestyle='--', label='Goal (0.5)')
    for ev in tda_events:
        plt.axvline(x=ev, color='orange', alpha=0.15, linewidth=1)
    plt.title(f"Mountain Car DREAM CYCLE — {success_count} Successes, {dream_count} Dreams [{RUN_TIMESTAMP}]")
    plt.xlabel("Episode")
    plt.ylabel("Max Position")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.figtext(0.01, 0.01, 
                f'Run: {RUN_TIMESTAMP} | Successes: {success_count}/{TOTAL_EPISODES} | '
                f'Dreams: {dream_count} | Niodoo: {stats["nodes"]} nodes',
                fontsize=8, alpha=0.5)
    results_fname = f'dream_results_{RUN_TIMESTAMP}.png'
    plt.savefig(results_fname)
    plt.savefig('dream_results.png')
    print(f"📊 Results saved to {results_fname}")
    plt.close()

    env.close()

if __name__ == "__main__":
    run_experiment()
