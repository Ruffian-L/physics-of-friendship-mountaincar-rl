"""
Body + Mind: The Bridge Experiment
====================================
Combines the physics solver (Body/Cerebellum) with Q-SMA+TDA (Mind)
through three bridge layers:

  Phase 0: Physics generates instinct seeds + dream data
  Phase 1 (0-500):    Governor strict, dreams frequent, agent learns basics
  Phase 2 (500-1500): Governor loosens, agent's own dreams dominate, TDA active
  Phase 3 (1500-2000): Governor nearly off, measuring TRUE learning

Key diagnostic: The Handoff Curve
  Governor overrides should DECREASE over time = Mind earning autonomy
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
from models.bridge import InstinctSeeder, DreamTeacher, GovernorGate
from models.niodoo import NiodooMemory

# --- CONFIGURATION ---
TOTAL_EPISODES = 2000
TDA_INTERVAL = 5
DREAM_INTERVAL = 10     # Dream every N episodes
SEED_EPISODES = 100     # Physics episodes for seeding
INSTINCT_WEIGHT = 0.5   # How much physics knowledge to inject (increased for faster convergence)
DREAM_FRACTION = 0.3    # Fraction of dreams from teacher vs own
RUN_TIMESTAMP = datetime.now().strftime('%Y-%m-%d_%H%M')


def run_bridge_experiment():
    """The full Body + Mind experiment."""
    
    env = gym.make('MountainCar-v0')
    
    # === THE MIND ===
    ctrl = SteeringController()
    brain = TopologicalBrain(ctrl)
    agent = QSMA_Agent()
    memory = NiodooMemory(beta_threshold=0.4)
    
    # === THE BRIDGE ===
    governor = GovernorGate(
        initial_threshold=0.8,
        final_threshold=0.1,
        warmup_episodes=200,
        rampdown_episodes=1500
    )
    
    print("=" * 70)
    print("  BODY + MIND: BRIDGE EXPERIMENT")
    print("=" * 70)
    
    # ─── PHASE 0: Generate instinct seeds ───────────────────────────
    print("\n[PHASE 0] Physics Body generating instinct seeds...")
    seeder = InstinctSeeder(num_seed_episodes=SEED_EPISODES, 
                           instinct_weight=INSTINCT_WEIGHT)
    trajectories = seeder.generate_seeds(agent)
    
    # Inject teacher dreams
    teacher = DreamTeacher(trajectories, dream_fraction=DREAM_FRACTION)
    teacher.inject_dreams(agent)
    
    print(f"  Seeds planted. Agent has instinct but must learn to walk.\n")
    
    # ─── TRAINING LOOP ──────────────────────────────────────────────
    success_count = 0
    max_positions = []
    phase_wins = {1: 0, 2: 0, 3: 0}
    phase_episodes = {1: (0, 500), 2: (500, 1500), 3: (1500, 2000)}
    first_success_ep = None
    override_log = []
    
    for episode in range(TOTAL_EPISODES):
        state, _ = env.reset()
        done = False
        steps = 0
        max_pos = -1.2
        ep_overrides = 0
        prev_node_id = None
        
        # Determine current phase
        if episode < 500:
            phase = 1
        elif episode < 1500:
            phase = 2
        else:
            phase = 3
        
        # Sync agent with steering controller
        agent.sync(ctrl)
        
        # Beta decay (from original main.py)
        beta = max(0.1, 1.5 * (0.995 ** episode))
        agent.params['beta'] = beta
        
        while not done and steps < 500:
            # Agent proposes action (the Mind)
            agent_action = agent.act(state)
            
            # Governor gate (the Body's safety net)
            # PHASE 3: Governor OFF — the Mind is on its own
            if phase == 3:
                final_action = agent_action
                was_overridden = False
            else:
                final_action, was_overridden = governor.gate(
                    state, agent_action, agent, episode, steps
                )
            
            if was_overridden:
                ep_overrides += 1
            
            # Execute
            next_state, reward, terminated, truncated, _ = env.step(final_action)
            done = terminated or truncated
            
            # Agent learns from ACTUAL outcome (not proposed action)
            energy = agent.learn(state, final_action, reward, next_state)
            
            # Log to TDA brain (6D: pos, vel, Q, flux, delta, energy)
            s = agent.discretize(state)
            q = agent.q_table[s, final_action]
            flux = agent.flux[s, final_action]
            delta = abs(reward + 0.999 * np.max(agent.q_table[agent.discretize(next_state)]) 
                       - agent.q_table[s, final_action])
            brain.add_log([state[0], state[1], q, flux, delta, energy])
            
            # Store in replay buffer for dreams
            agent.replay_buffer.append((state.copy(), final_action, reward, 
                                       next_state.copy(), energy))
            
            # Niodoo memory chain
            content = f"pos:{state[0]:.2f},vel:{state[1]:.3f},a:{final_action},e:{energy:.3f}"
            node_id = memory.flinch_tag(content, state[1], state)
            if prev_node_id:
                memory.connect_nodes(prev_node_id, node_id, weight=abs(state[1]))
            prev_node_id = node_id
            
            state = next_state
            steps += 1
            max_pos = max(max_pos, state[0])
        
        # Episode done
        max_positions.append(max_pos)
        governor.episode_done()
        override_log.append(ep_overrides)
        
        if state[0] >= 0.5:
            success_count += 1
            phase_wins[phase] = phase_wins.get(phase, 0) + 1
            if first_success_ep is None:
                first_success_ep = episode
                print(f"\n  🎯 FIRST SUCCESS at episode {episode}! (Steps: {steps})")
            
            # Store success in Niodoo memory
            agent.commit_episode(True)
            
            if success_count <= 10 or success_count % 50 == 0:
                print(f"  Ep {episode}: 🚩 WIN #{success_count} in {steps} steps "
                      f"(overrides: {ep_overrides}, phase: {phase})")
        else:
            agent.commit_episode(False)
        
        # Progress report
        if (episode + 1) % 200 == 0:
            rate = success_count / (episode + 1) * 100
            recent_overrides = np.mean(override_log[-100:]) if override_log else 0
            threshold = governor.get_threshold(episode)
            print(f"\n  ── Episode {episode+1}/{TOTAL_EPISODES} ──")
            print(f"     Wins: {success_count} ({rate:.1f}%) | Phase {phase}")
            print(f"     Governor: {recent_overrides:.1f} overrides/ep (threshold: {threshold:.2f})")
            print(f"     Memory: {len(memory.nodes)} nodes, {memory.flinch_count} flinches")
        
        # ── TDA Metacognition ──
        if (episode + 1) % TDA_INTERVAL == 0 and len(brain.buffer) >= 100:
            brain.analyze()
            ctrl.normalize()
        
        # ── Dream Cycle ──
        if (episode + 1) % DREAM_INTERVAL == 0:
            teacher.dream_with_teacher(agent)
        
        # ── Niodoo Memory Maintenance ──
        if (episode + 1) % 50 == 0:
            memory.curate_prune()
    
    env.close()
    
    # === RESULTS ===
    gov_stats = governor.get_stats()
    mem_stats = memory.get_stats()
    
    print("\n" + "=" * 70)
    print("  RESULTS: BODY + MIND BRIDGE")
    print("=" * 70)
    print(f"  Total Wins:        {success_count}/{TOTAL_EPISODES} ({success_count/TOTAL_EPISODES*100:.1f}%)")
    print(f"  First Success:     Episode {first_success_ep}")
    print(f"  Phase 1 (0-500):   {phase_wins[1]} wins (governor strict)")
    print(f"  Phase 2 (500-1500):{phase_wins[2]} wins (governor loosening)")
    print(f"  Phase 3 (1500-2000):{phase_wins[3]} wins (mind independent)")
    print(f"  Governor Overrides: {gov_stats['overrides']} total")
    print(f"  Agreement Rate:    {gov_stats['agreement_rate']:.1f}%")
    print(f"  Niodoo Memory:     {mem_stats['nodes']} nodes, {mem_stats['edges']} edges, "
          f"{mem_stats['flinches']} flinches")
    
    # Comparison
    print(f"\n  ── COMPARISON ──")
    print(f"  Raw Q-SMA+TDA:     681/2000 (first win: ep 464)")
    print(f"  Physics Reflex:    2000/2000 (no learning)")
    print(f"  Body+Mind Bridge:  {success_count}/2000 (first win: ep {first_success_ep})")
    
    if first_success_ep is not None and first_success_ep < 200:
        print(f"  ✅ First success before ep 200 (instinct seed working)")
    
    if success_count > 681:
        print(f"  ✅ Beat Q-SMA baseline ({success_count} > 681)")
    
    # Check handoff
    if len(override_log) > 100:
        early_overrides = np.mean(override_log[:200])
        late_overrides = np.mean(override_log[-200:])
        if late_overrides < early_overrides * 0.5:
            print(f"  ✅ Handoff curve: overrides dropped {early_overrides:.1f}→{late_overrides:.1f}")
        else:
            print(f"  ⚠️  Handoff incomplete: overrides {early_overrides:.1f}→{late_overrides:.1f}")
    
    print("=" * 70)
    
    # === PLOTS ===
    _plot_bridge_results(max_positions, override_log, governor, phase_wins, 
                        success_count, first_success_ep)
    
    return success_count


def _plot_bridge_results(max_positions, override_log, governor, phase_wins,
                        success_count, first_success_ep):
    """Generate diagnostic plots for the bridge experiment."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(
        f'Body + Mind Bridge — {success_count}/2000 Wins '
        f'(First: ep {first_success_ep}) [{RUN_TIMESTAMP}]',
        fontsize=14, fontweight='bold'
    )
    
    # 1. Max position + phase boundaries
    ax1 = axes[0, 0]
    ax1.plot(max_positions, alpha=0.2, linewidth=0.5, color='steelblue')
    window = 50
    if len(max_positions) >= window:
        rolling = np.convolve(max_positions, np.ones(window)/window, mode='valid')
        ax1.plot(range(window-1, len(max_positions)), rolling, color='navy', linewidth=2)
    ax1.axhline(y=0.5, color='green', linestyle='--', alpha=0.7, label='Goal')
    ax1.axvline(x=500, color='orange', linestyle=':', alpha=0.5, label='Phase 1→2')
    ax1.axvline(x=1500, color='red', linestyle=':', alpha=0.5, label='Phase 2→3')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Max Position')
    ax1.set_title('Learning Curve (Max Position)')
    ax1.legend(fontsize=8)
    
    # 2. THE HANDOFF CURVE (the money plot)
    ax2 = axes[0, 1]
    if len(override_log) >= window:
        rolling_overrides = np.convolve(override_log, np.ones(window)/window, mode='valid')
        ax2.plot(range(window-1, len(override_log)), rolling_overrides, 
                color='crimson', linewidth=2)
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Governor Overrides per Episode')
    ax2.set_title('🔑 Governor Handoff Curve\n(should decrease = Mind earning autonomy)')
    ax2.axvline(x=500, color='orange', linestyle=':', alpha=0.5)
    ax2.axvline(x=1500, color='red', linestyle=':', alpha=0.5)
    
    # 3. Rolling win rate
    ax3 = axes[1, 0]
    wins_binary = [1 if p >= 0.5 else 0 for p in max_positions]
    if len(wins_binary) >= window:
        rolling_rate = np.convolve(wins_binary, np.ones(window)/window, mode='valid') * 100
        ax3.plot(range(window-1, len(wins_binary)), rolling_rate, color='darkgreen', linewidth=2)
    ax3.axhline(y=681/2000*100, color='blue', linestyle='--', alpha=0.5, 
               label=f'Q-SMA baseline ({681/2000*100:.1f}%)')
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Win Rate (%)')
    ax3.set_title(f'Rolling {window}-Episode Win Rate')
    ax3.set_ylim(0, 105)
    ax3.legend(fontsize=8)
    
    # 4. Phase comparison
    ax4 = axes[1, 1]
    phases = ['Phase 1\n(Gov Strict)\n0-500', 'Phase 2\n(Loosening)\n500-1500', 
              'Phase 3\n(Independent)\n1500-2000']
    phase_rates = [
        phase_wins[1] / 500 * 100,
        phase_wins[2] / 1000 * 100,
        phase_wins[3] / 500 * 100
    ]
    colors = ['#ff9999', '#99ccff', '#99ff99']
    bars = ax4.bar(phases, phase_rates, color=colors, edgecolor='black', linewidth=0.5)
    ax4.axhline(y=681/2000*100, color='blue', linestyle='--', alpha=0.5, 
               label='Q-SMA baseline')
    for bar, rate in zip(bars, phase_rates):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{rate:.1f}%', ha='center', fontsize=10, fontweight='bold')
    ax4.set_ylabel('Win Rate (%)')
    ax4.set_title('Win Rate by Phase')
    ax4.set_ylim(0, 105)
    ax4.legend(fontsize=8)
    
    plt.tight_layout()
    fname = f'bridge_results_{RUN_TIMESTAMP}.png'
    plt.savefig(fname, dpi=150)
    plt.savefig('bridge_results.png', dpi=150)
    print(f"  📊 Results saved to {fname}")
    plt.close()


if __name__ == "__main__":
    run_bridge_experiment()
