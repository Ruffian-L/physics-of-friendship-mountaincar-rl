"""
Splat Memory Experiment: The Agent Grows Its Own Conscience
============================================================
Phase 5: Scar Tissue Protocol (Active Healing)

Stream 1 (Actor): Pure wave-surfing. Q + Flux + Curiosity.
Stream 2 (Watcher): Flux sculptor + Memory Gardener.

The Protocol:
1. Yin (Pain) is Immortal: Decay=1.0. Trauma persists.
2. Compound Trauma: Failures deepen the scar.
3. Active Healing: Only victory can heal the scar.

Comparison: Phase 3 (Daydream) = 628/2000 | Yin-Yang = 604/2000
"""

import gymnasium as gym
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime

from core.steering import SteeringController
from core.tda import TopologicalBrain
from core.agent import QSMA_Agent
from models.niodoo import NiodooMemory
from models.bridge import InstinctSeeder

# --- CONFIGURATION ---
TOTAL_EPISODES = 2000
TDA_INTERVAL = 5
DREAM_INTERVAL = 10
RUN_TIMESTAMP = datetime.now().strftime('%Y-%m-%d_%H%M')


def run_splat_experiment():
    """Run the agent with splat memory reflexes — no external governor."""
    
    env = gym.make('MountainCar-v0')
    
    # The Mind — with splat memory already wired in
    ctrl = SteeringController()
    brain = TopologicalBrain(ctrl)
    agent = QSMA_Agent()
    memory = NiodooMemory(beta_threshold=0.4)
    
    print("=" * 70)
    print("  PHASE 5: SCAR TISSUE PROTOCOL (Active Healing)")
    print("  Pain is immortal. Redemption is the only cure.")
    print("=" * 70)
    
    print(f"  Splat config: pain={agent.splat_memory.pain_threshold}, "
          f"pleasure={agent.splat_memory.pleasure_threshold}, "
          f"reflex_weight={agent.splat_memory.reflex_weight}")
    
    # Tracking
    success_count = 0
    max_positions = []
    first_success_ep = None
    splat_counts = []        # Splat count over time
    reflex_fires_per_ep = [] # How often reflexes fired each episode
    pain_pleasure_ratio = [] # Balance of pain vs pleasure splats
    prev_reflex_fires = 0
    
    for episode in range(TOTAL_EPISODES):
        state, _ = env.reset()
        done = False
        steps = 0
        max_pos = -1.2
        prev_node_id = None
        
        # Sync
        agent.sync(ctrl)
        beta = max(0.1, 1.5 * (0.995 ** episode))
        agent.params['beta'] = beta
        
        while not done and steps < 500:
            # Agent acts — reflexes fire internally via splat_memory
            action = agent.act(state)
            
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Agent learns — crystallizes experiences as splats
            energy = agent.learn(state, action, reward, next_state)
            
            # Store in episode buffer for dream cycle (proper 5-tuple format)
            agent.remember(state, action, reward, next_state, energy)
            
            # Log to TDA brain
            s = agent.discretize(state)
            q = agent.q_table[s, action]
            flux = agent.flux[s, action]
            delta = abs(reward + 0.999 * np.max(agent.q_table[agent.discretize(next_state)]) 
                       - agent.q_table[s, action])
            brain.add_log([state[0], state[1], q, flux, delta, energy])
            
            # Replay buffer
            agent.replay_buffer.append((state.copy(), action, reward, 
                                       next_state.copy(), energy))
            
            # Niodoo memory chain
            content = f"pos:{state[0]:.2f},vel:{state[1]:.3f},a:{action},e:{energy:.3f}"
            node_id = memory.flinch_tag(content, state[1], state)
            if prev_node_id:
                memory.connect_nodes(prev_node_id, node_id, weight=abs(state[1]))
            prev_node_id = node_id
            
            state = next_state
            steps += 1
            max_pos = max(max_pos, state[0])
        
        # Episode done
        max_positions.append(max_pos)
        
        # Track reflex activity this episode
        current_fires = agent.splat_memory.total_reflex_fires
        ep_fires = current_fires - prev_reflex_fires
        prev_reflex_fires = current_fires
        reflex_fires_per_ep.append(ep_fires)
        splat_counts.append(len(agent.splat_memory.splats))
        
        # Store success splat (the microwave moment — you came back right on time)
        if state[0] >= 0.5:
            success_count += 1
            agent.splat_memory.store_experience(
                state, 1, 1.0, 100.0, success=True  # Strong pleasure splat at goal
            )
            
            if first_success_ep is None:
                first_success_ep = episode
                print(f"\n  🎯 FIRST SUCCESS at episode {episode}! (Steps: {steps})")
                print(f"     Splats: {len(agent.splat_memory.splats)} "
                      f"(pain: {agent.splat_memory.pain_splats_created}, "
                      f"pleasure: {agent.splat_memory.pleasure_splats_created})")
            
            agent.commit_episode(True)
            
            if success_count <= 10 or success_count % 100 == 0:
                print(f"  Ep {episode}: 🚩 WIN #{success_count} in {steps} steps "
                      f"(splats: {len(agent.splat_memory.splats)}, "
                      f"reflexes fired: {ep_fires})")
        else:
            agent.commit_episode(False)
        
        # Splat memory maintenance: decay + consolidate
        agent.splat_memory.decay_and_consolidate()
        
        # Progress report
        if (episode + 1) % 200 == 0:
            rate = success_count / (episode + 1) * 100
            stats = agent.splat_memory.get_stats()
            recent_fires = np.mean(reflex_fires_per_ep[-100:]) if reflex_fires_per_ep else 0
            print(f"\n  ── Episode {episode+1}/{TOTAL_EPISODES} ──")
            print(f"     Wins: {success_count} ({rate:.1f}%)")
            print(f"     Splats alive: {stats['count']} "
                  f"(pain: {stats.get('alive_pain', 0)}, "
                  f"pleasure: {stats.get('alive_pleasure', 0)})")
            print(f"     Reflex fires/ep: {recent_fires:.0f} | "
                  f"Most reinforced: {stats.get('most_reinforced', 0)}")
            print(f"     Memory: {len(memory.nodes)} nodes, {memory.flinch_count} flinches")
        
        # TDA Metacognition
        if (episode + 1) % TDA_INTERVAL == 0 and len(brain.buffer) >= 100:
            brain.analyze()
            ctrl.normalize()
        
        # Dream Cycle
        if (episode + 1) % DREAM_INTERVAL == 0:
            agent.dream_cycle()
        
        # Niodoo maintenance
        if (episode + 1) % 50 == 0:
            memory.curate_prune()
    
    env.close()
    
    # === RESULTS ===
    stats = agent.splat_memory.get_stats()
    mem_stats = memory.get_stats()
    w_stats = agent.watcher.get_stats()
    
    print("\n" + "=" * 70)
    print("  RESULTS: SCAR TISSUE PROTOCOL (Phase 5)")
    print("=" * 70)
    print(f"  Total Wins:        {success_count}/{TOTAL_EPISODES} "
          f"({success_count/TOTAL_EPISODES*100:.1f}%)")
    print(f"  First Success:     Episode {first_success_ep}")
    print(f"  Splats Alive:      {stats['count']} "
          f"(pain: {stats.get('alive_pain', 0)}, "
          f"pleasure: {stats.get('alive_pleasure', 0)})")
    print(f"  Total Reflex Fires:{stats['fires']}")
    print(f"  Consolidations:    {stats['consolidations']}")
    print(f"  Scars Healed:      {stats.get('healed', 0)} (Redemption events)")
    print(f"  Most Reinforced:   {stats.get('most_reinforced', 0)} times")
    print(f"  Niodoo Memory:     {mem_stats['nodes']} nodes, {mem_stats['edges']} edges")
    print(f"\n  ── WATCHER (Stream 2) ──")
    print(f"  Observations:      {w_stats['observations']}")
    print(f"  Groove Deepenings: {w_stats['groove_deepenings']}")
    print(f"  Debris Clearings:  {w_stats['debris_clearings']}")
    print(f"  Garden Reinforces: {w_stats['garden_reinforcements']}")
    
    print(f"\n  ── COMPARISON ──")
    print(f"  Raw Q-SMA+TDA:     681/2000 (first win: ep 464)")
    print(f"  Phase 3 (Daydream):628/2000 (first win: ep 487)")
    print(f"  Yin-Yang (Ph 4):   604/2000 (first win: ep 460)")
    print(f"  Phase 5 (Scars):   {success_count}/2000 (first win: ep {first_success_ep})")
    
    if first_success_ep is not None and first_success_ep < 300:
        print(f"  ✅ First success before ep 300")
    if success_count > 681:
        print(f"  ✅ Beat Q-SMA baseline ({success_count} > 681)")
    if success_count > 610:
        print(f"  ✅ Beat Phase 2 Dream ({success_count} > 610)")
    
    print("=" * 70)
    
    _plot_splat_results(max_positions, splat_counts, reflex_fires_per_ep,
                       success_count, first_success_ep)
    
    return success_count


def _plot_splat_results(max_positions, splat_counts, reflex_fires,
                       success_count, first_success_ep):
    """Diagnostic plots for splat memory experiment."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(
        f'Splat Memory Reflex Governor - {success_count}/2000 Wins '
        f'(First: ep {first_success_ep}) [{RUN_TIMESTAMP}]',
        fontsize=14, fontweight='bold'
    )
    window = 50
    
    # 1. Learning Curve
    ax1 = axes[0, 0]
    ax1.plot(max_positions, alpha=0.15, linewidth=0.5, color='steelblue')
    if len(max_positions) >= window:
        rolling = np.convolve(max_positions, np.ones(window)/window, mode='valid')
        ax1.plot(range(window-1, len(max_positions)), rolling, color='navy', linewidth=2)
    ax1.axhline(y=0.5, color='green', linestyle='--', alpha=0.7, label='Goal')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Max Position')
    ax1.set_title('Learning Curve')
    ax1.legend(fontsize=8)
    
    # 2. Splat Count Over Time (memory growth)
    ax2 = axes[0, 1]
    ax2.plot(splat_counts, color='purple', linewidth=1.5)
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Active Splats')
    ax2.set_title('Memory Growth\n(splats alive after decay/consolidation)')
    
    # 3. Reflex Fires Per Episode
    ax3 = axes[1, 0]
    if len(reflex_fires) >= window:
        rolling_fires = np.convolve(reflex_fires, np.ones(window)/window, mode='valid')
        ax3.plot(range(window-1, len(reflex_fires)), rolling_fires, 
                color='orangered', linewidth=2)
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Reflex Fires per Episode')
    ax3.set_title('Reflex Activity\n(how often memories influence decisions)')
    
    # 4. Rolling Win Rate
    ax4 = axes[1, 1]
    wins_binary = [1 if p >= 0.5 else 0 for p in max_positions]
    if len(wins_binary) >= window:
        rolling_rate = np.convolve(wins_binary, np.ones(window)/window, mode='valid') * 100
        ax4.plot(range(window-1, len(wins_binary)), rolling_rate, 
                color='darkgreen', linewidth=2)
    ax4.axhline(y=681/2000*100, color='blue', linestyle='--', alpha=0.5, 
               label=f'Q-SMA baseline ({681/2000*100:.1f}%)')
    ax4.set_xlabel('Episode')
    ax4.set_ylabel('Win Rate (%)')
    ax4.set_title(f'Rolling {window}-Episode Win Rate')
    ax4.set_ylim(0, 105)
    ax4.legend(fontsize=8)
    
    plt.tight_layout()
    fname = f'splat_results_{RUN_TIMESTAMP}.png'
    plt.savefig(fname, dpi=150)
    plt.savefig('splat_results.png', dpi=150)
    print(f"  Results saved to {fname}")
    plt.close()


if __name__ == "__main__":
    run_splat_experiment()
