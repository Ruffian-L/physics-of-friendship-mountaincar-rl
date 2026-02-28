"""
episode_scaling.py — Does scale help?
======================================
Runs the full champion config (TDA + Splats + Bridge) at three episode counts:
  2000 (baseline / current champion)
  5000
  10000

Key questions:
  - Does the rolling win rate keep rising or plateau?
  - Does the zig-zag pattern converge?
  - When does the win rate stop improving meaningfully?

Live window opens per run. Overlaid comparison chart at the end.

Usage:
  cd src
  python -m experiments.episode_scaling                    # full 2k/5k/10k
  python -m experiments.episode_scaling --scales 500 1000  # custom smoke test
"""

import argparse
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import gymnasium as gym
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from core.steering import SteeringController
from core.tda import TopologicalBrain
from core.agent import QSMA_Agent
from models.bridge import InstinctSeeder, DreamTeacher, GovernorGate
from models.niodoo import NiodooMemory
from experiments.experiment_utils import (
    LiveDashboard, SeedManager, save_json, save_png,
    rolling_mean, PALETTE, CONFIG_COLORS
)

DEFAULT_SCALES  = [2000, 5000, 10000]
TDA_INTERVAL    = 5
DREAM_INTERVAL  = 10
SEED_EPISODES   = 100
INSTINCT_WEIGHT = 0.5
DREAM_FRACTION  = 0.3


def run_scaling(total_episodes: int, seed: int, dashboard: LiveDashboard) -> dict:
    SeedManager.apply(seed)
    env = gym.make('MountainCar-v0')

    ctrl   = SteeringController()
    brain  = TopologicalBrain(ctrl)
    agent  = QSMA_Agent()
    memory = NiodooMemory(beta_threshold=0.4)

    seeder = InstinctSeeder(num_seed_episodes=SEED_EPISODES,
                            instinct_weight=INSTINCT_WEIGHT)
    trajectories = seeder.generate_seeds(agent)
    teacher = DreamTeacher(trajectories, dream_fraction=DREAM_FRACTION)
    teacher.inject_dreams(agent)
    governor = GovernorGate(
        initial_threshold=0.8, final_threshold=0.1,
        warmup_episodes=200,
        rampdown_episodes=min(1500, int(total_episodes * 0.75))
    )

    success_count = 0
    first_success = None
    win_flags     = []
    rolling_pcts  = []    # cumulative win% at every episode (for asymptote plot)

    for episode in range(total_episodes):
        state, _ = env.reset()
        done  = False
        steps = 0
        max_pos = -1.2
        prev_node_id = None

        phase_boundary = int(total_episodes * 0.75)
        phase = 1 if episode < int(total_episodes * 0.25) else \
                (2 if episode < phase_boundary else 3)

        agent.sync(ctrl)
        agent.params['beta'] = max(0.1, 1.5 * (0.995 ** episode))

        while not done and steps < 500:
            agent_action = agent.act(state)

            if phase < 3:
                final_action, _ = governor.gate(state, agent_action, agent, episode, steps)
            else:
                final_action = agent_action

            next_state, reward, terminated, truncated, _ = env.step(final_action)
            done = terminated or truncated

            energy = agent.learn(state, final_action, reward, next_state)

            s   = agent.discretize(state)
            q   = agent.q_table[s, final_action]
            fl  = agent.flux[s, final_action]
            ns  = agent.discretize(next_state)
            dlt = abs(reward + 0.999 * np.max(agent.q_table[ns]) - agent.q_table[s, final_action])
            brain.add_log([state[0], state[1], q, fl, dlt, energy])

            agent.replay_buffer.append((state.copy(), final_action, reward,
                                        next_state.copy(), energy))
            content = f"pos:{state[0]:.2f},vel:{state[1]:.3f},a:{final_action},e:{energy:.3f}"
            node_id = memory.flinch_tag(content, state[1], state)
            if prev_node_id:
                memory.connect_nodes(prev_node_id, node_id, weight=abs(state[1]))
            prev_node_id = node_id

            state = next_state
            steps += 1
            max_pos = max(max_pos, state[0])

        win = max_pos >= 0.5
        win_flags.append(1 if win else 0)
        if win:
            success_count += 1
            if first_success is None:
                first_success = episode
            agent.commit_episode(True)
        else:
            agent.commit_episode(False)

        rolling_pcts.append(success_count / (episode + 1) * 100)
        governor.episode_done()
        agent.splat_memory.decay_and_consolidate()
        splat_stats = agent.splat_memory.get_stats()
        dashboard.update(episode, max_pos, steps, agent=agent, splat_stats=splat_stats)

        if (episode + 1) % TDA_INTERVAL == 0 and len(brain.buffer) >= 100:
            brain.analyze()
            ctrl.normalize()
        if (episode + 1) % DREAM_INTERVAL == 0:
            teacher.dream_with_teacher(agent)
        if (episode + 1) % 50 == 0:
            memory.curate_prune()

        if (episode + 1) % 500 == 0:
            pct = success_count / (episode + 1) * 100
            print(f'  [scaling:{total_episodes}ep] Ep {episode+1} | '
                  f'Wins: {success_count} ({pct:.1f}%)')

    env.close()
    return {
        'total_episodes': total_episodes,
        'seed': seed,
        'wins': success_count,
        'win_pct': success_count / total_episodes * 100,
        'first_success': first_success,
        'win_flags': win_flags,
        'cumulative_pct': rolling_pcts,
    }


def plot_scaling_comparison(all_results: list):
    """Three-panel: cumulative win%, rolling win%, final bar."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), facecolor=PALETTE['bg'])
    fig.suptitle('Episode Scaling: Does More Training Help?',
                 color=PALETTE['text'], fontsize=14, fontweight='bold')

    scales = sorted(set(r['total_episodes'] for r in all_results))
    colors = [CONFIG_COLORS.get(f'{s//1000}k', '#888888') for s in scales]

    # Panel 1: Cumulative win%
    ax = axes[0]
    ax.set_facecolor(PALETTE['panel'])
    for scale, col in zip(scales, colors):
        r = next(x for x in all_results if x['total_episodes'] == scale)
        ax.plot(r['cumulative_pct'], color=col, linewidth=1.8, label=f'{scale} eps')
    ax.axhline(34.05, color='white', linestyle='--', alpha=0.5, linewidth=1)
    ax.set_xlabel('Episode', color=PALETTE['text'])
    ax.set_ylabel('Cumulative Win %', color=PALETTE['text'])
    ax.set_title('Cumulative Win % over Time\n(converging = asymptote found)',
                 color=PALETTE['text'])
    ax.legend(facecolor=PALETTE['panel'], labelcolor=PALETTE['text'])
    ax.tick_params(colors=PALETTE['text'])
    ax.grid(True, color=PALETTE['grid'])
    for sp in ax.spines.values(): sp.set_color(PALETTE['grid'])

    # Panel 2: Rolling 200-ep win rate
    ax = axes[1]
    ax.set_facecolor(PALETTE['panel'])
    for scale, col in zip(scales, colors):
        r = next(x for x in all_results if x['total_episodes'] == scale)
        flags = r['win_flags']
        if len(flags) >= 200:
            rm = rolling_mean(flags, 200) * 100
            ax.plot(np.arange(199, len(flags)), rm,
                    color=col, linewidth=2, label=f'{scale} eps')
    ax.axhline(34.05, color='white', linestyle='--', alpha=0.5, linewidth=1)
    ax.set_xlabel('Episode', color=PALETTE['text'])
    ax.set_ylabel('Win Rate %', color=PALETTE['text'])
    ax.set_title('Rolling 200-ep Win Rate\n(flat line = asymptote reached)',
                 color=PALETTE['text'])
    ax.set_ylim(0, 100)
    ax.legend(facecolor=PALETTE['panel'], labelcolor=PALETTE['text'])
    ax.tick_params(colors=PALETTE['text'])
    ax.grid(True, color=PALETTE['grid'])
    for sp in ax.spines.values(): sp.set_color(PALETTE['grid'])

    # Panel 3: Final win% bar
    ax = axes[2]
    ax.set_facecolor(PALETTE['panel'])
    labels = [f'{s}' for s in scales]
    win_pcts = [next(x for x in all_results if x['total_episodes'] == s)['win_pct']
                for s in scales]
    bars = ax.bar(labels, win_pcts, color=colors,
                  edgecolor=PALETTE['grid'], linewidth=0.8)
    ax.axhline(34.05, color='white', linestyle='--', alpha=0.5)
    for bar, pct in zip(bars, win_pcts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f'{pct:.1f}%', ha='center', fontsize=10,
                fontweight='bold', color=PALETTE['text'])
    ax.set_ylabel('Final Win %', color=PALETTE['text'])
    ax.set_title('Total Win Rate at End\nof Each Scale', color=PALETTE['text'])
    ax.tick_params(colors=PALETTE['text'])
    ax.grid(True, color=PALETTE['grid'], axis='y')
    for sp in ax.spines.values(): sp.set_color(PALETTE['grid'])

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    save_png(fig, 'episode_scaling_comparison')
    plt.ioff()
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scales', type=int, nargs='+', default=DEFAULT_SCALES)
    parser.add_argument('--seed',   type=int, default=42)
    args = parser.parse_args()

    all_results = []
    colors_list = ['2k', '5k', '10k']

    for i, scale in enumerate(args.scales):
        color_key = colors_list[i] if i < len(colors_list) else f'{scale//1000}k'
        color = CONFIG_COLORS.get(color_key, '#888888')

        print(f'\n{"="*60}')
        print(f'  SCALE: {scale} episodes')
        print(f'{"="*60}')

        dash = LiveDashboard(
            title=f'Episode Scaling: {scale} eps (seed={args.seed})',
            total_episodes=scale,
            config_color=color
        )
        result = run_scaling(scale, args.seed, dash)
        dash.close(name=f'scaling_{scale}ep')
        all_results.append(result)

        print(f'  ✓ {scale} eps: {result["wins"]}/{scale} ({result["win_pct"]:.1f}%) '
              f'first_success={result["first_success"]}')

    # Summary
    print('\n' + '='*60)
    print('  SCALING RESULTS')
    print('='*60)
    print(f'  {"Episodes":<10} {"Win%":>8} {"Wins":>8} {"First Success":>15}')
    print('-'*60)
    for r in all_results:
        print(f'  {r["total_episodes"]:<10} {r["win_pct"]:>7.1f}% {r["wins"]:>8}  '
              f'{str(r["first_success"]):>15}')
    print('='*60)

    save_json({'seed': args.seed, 'results': all_results}, 'episode_scaling')
    plot_scaling_comparison(all_results)


if __name__ == '__main__':
    main()
