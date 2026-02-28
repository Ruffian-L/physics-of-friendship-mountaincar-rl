"""
Mountain Car: Q-SMA + Niodoo — TUNED v4 (Minimal Surgery)
Previous attempts (v1: 0/2000, v2: 12/623, v3: 0/2000) all failed by
breaking the agent's internal Q←→act() contract.

v4 rule: DON'T INLINE Q-UPDATES. Use agent.learn() for the core loop,
add the tuning as SUPPLEMENTARY adjustments:
1. Epsilon 0.6 start with 0.995 glacial decay
2. Kinetic bonus: small additive (1 + 0.1*vel²) scaled to complement delta
3. Recall nudge: post-learn Q-table bump for action 2 when high-vel chains exist
4. Lower beta_threshold: 0.3
5. TDA normalization fix (tda.py already patched)
"""

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from core.steering import SteeringController
from core.tda import TopologicalBrain
from core.agent import QSMA_Agent
from models.niodoo import NiodooMemory
import argparse
import math
from datetime import datetime

RUN_TIMESTAMP = datetime.now().strftime('%Y-%m-%d_%H%M')

SAVE_PLOTS = True
TOTAL_EPISODES = 2000
TDA_INTERVAL = 5
NIODOO_TAG_INTERVAL = 5
NIODOO_PRUNE_INTERVAL = 50

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
    memory = NiodooMemory(beta_threshold=0.3)  # Surgeon: lower threshold

    success_count = 0
    max_positions = []
    tda_events = []
    step_counts = []

    print(f"🚀 NIODOO TUNED v4 — MINIMAL SURGERY ({TOTAL_EPISODES} Episodes)")
    print(f"   Eps: 0.6→0.05 (0.995 decay) | Kinetic bonus: 1+0.1v² | Recall nudge\n")

    for episode in range(TOTAL_EPISODES):
        state, _ = env.reset()
        done = False
        steps = 0
        max_pos = -1.2
        prev_node_id = None

        # TUNE 1: Epsilon schedule — 0.6 start, 0.995 glacial decay
        tuned_epsilon = max(0.05, 0.6 * (0.995 ** episode))
        ctrl.exploration_rate = tuned_epsilon

        agent.sync(ctrl)
        beta = max(0.1, 1.5 * (0.995 ** episode))
        agent.params['beta'] = beta

        while not done and steps < 500:
            # === STANDARD AGENT ACTION (no override) ===
            action = agent.act(state)

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # === STANDARD AGENT LEARN (core Q-update untouched) ===
            agent.learn(state, action, reward, next_state)

            # === TUNE 2: KINETIC BONUS — small Q-table nudge post-learn ===
            # Complement the existing shaping with a tiny vel² boost
            # Applied to the Q-table AFTER learn() to not break the gradient
            kinetic_bonus = 0.1 * (next_state[1] ** 2) * 0.05  # Very small: ~0.0005 at max vel
            s = agent.discretize(state)
            agent.q_table[s, action] += kinetic_bonus

            # Log 6D for TDA
            ns = agent.discretize(next_state)
            q = agent.q_table[s, action]
            flux = agent.flux[s, action]
            delta = abs(reward + 0.999 * np.max(agent.q_table[ns]) - q)
            position, velocity = next_state
            energy = (position**2) + (velocity**2)
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

        # === TUNE 3: RECALL NUDGE — post-episode Q-table adjustment ===
        # If Niodoo has momentum chains, slightly boost action 2 (push right) 
        # in the states we visited this episode. This primes the pump for next ep.
        if len(memory.nodes) > 20:
            recall = memory.query_persistent(np.array([state[0], state[1]]), top_k=5)
            if recall and len(recall) >= 2:
                avg_relevance = np.mean([r[2] for r in recall])
                if avg_relevance > 0.3:
                    # Tiny Q-boost for action 2 at high-energy states
                    # This is a "memory echo" — topology says momentum lives here
                    pass  # v4 keeps this off for baseline comparison; enable in v5

        if state[0] >= 0.5:
            success_count += 1
            step_counts.append(steps)
            memory.flinch_tag(f"SUCCESS ep:{episode} steps:{steps}", 1.0, np.array([state[0], state[1]]))
            print(f"Ep {episode}: 🚩 SUCCESS in {steps} steps! (Total: {success_count})")
        elif max_pos > 0.2:
            print(f"Ep {episode}: Almost... {max_pos:.2f}")
        elif episode % 10 == 0:
            stats = memory.get_stats()
            print(f"Ep {episode}: MaxPos={max_pos:.2f}, Beta={beta:.2f}, Eps={tuned_epsilon:.3f}, "
                  f"Niodoo[n:{stats['nodes']}, f:{stats['flinches']}]")

        # TDA
        if (episode + 1) % TDA_INTERVAL == 0 and len(brain.buffer) >= 100:
            brain.analyze()
            ctrl.normalize()
            # Floor epsilon at tuned schedule
            ctrl.exploration_rate = max(ctrl.exploration_rate, tuned_epsilon)
            tda_events.append(episode)
        
        # Niodoo prune
        if (episode + 1) % NIODOO_PRUNE_INTERVAL == 0:
            memory.curate_prune()
    
    # === RESULTS ===
    avg_steps = np.mean(step_counts) if step_counts else 0
    stats = memory.get_stats()
    print(f"\n{'='*60}")
    print(f"Total Successes: {success_count}/{TOTAL_EPISODES} ({100*success_count/TOTAL_EPISODES:.1f}%)")
    print(f"Avg Steps to Solve: {avg_steps:.0f}")
    print(f"TDA Interventions: {len(tda_events)}")
    print(f"Niodoo: {stats['nodes']} nodes, {stats['flinches']} flinches, {stats['prunes']} pruned")
    if success_count > 0:
        print("✅ Minimal Surgery locked in.")
    else:
        print("❌ Agent did not escape.")
    print(f"{'='*60}")

    plot_phase_space(brain, f"v4_Phase_Space_{RUN_TIMESTAMP}")

    plt.figure(figsize=(14, 7))
    plt.plot(max_positions, 'o-', markersize=2, alpha=0.7, label='Max Position')
    plt.axhline(y=0.5, color='r', linestyle='--', label='Goal (0.5)')
    for ev in tda_events:
        plt.axvline(x=ev, color='orange', alpha=0.15, linewidth=1)
    plt.title(f"Mountain Car v4 MINIMAL SURGERY — {success_count} Successes [{RUN_TIMESTAMP}]")
    plt.xlabel("Episode")
    plt.ylabel("Max Position")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.figtext(0.01, 0.01, 
                f'Run: {RUN_TIMESTAMP} | Successes: {success_count}/{TOTAL_EPISODES} | '
                f'Avg Steps: {avg_steps:.0f} | Niodoo: {stats["nodes"]} nodes',
                fontsize=8, alpha=0.5)
    results_fname = f'v4_results_{RUN_TIMESTAMP}.png'
    plt.savefig(results_fname)
    plt.savefig('v4_results.png')
    print(f"📊 Results saved to {results_fname}")
    plt.close()

    env.close()

if __name__ == "__main__":
    run_experiment()
