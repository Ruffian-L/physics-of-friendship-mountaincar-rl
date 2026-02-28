"""
long_run_ablation.py — Full Matrix at 5k / 10k / 20k Episodes
==============================================================
Runs all 5 ablation configs to 20k (or whatever you set) to answer:

  "Do TDA and Splats become meaningful AFTER the bridge scaffold fades?"

Key differences from ablation_study.py:
  - Governor turns off at 15% of total episodes (not hardcoded 1500)
  - Phase 3 is the MAJORITY of the run — that's the point
  - Niodoo memory is capped at 5000 nodes to prevent RAM explosion at 20k
  - UPDATE_EVERY = 50 (smoother window refresh)
  - Extra "Phase 3 zoom" panel + phase-split win-rate comparison at the end

Usage:
  cd src
  MPLBACKEND=MacOSX ../.venv/bin/python3 -m experiments.long_run_ablation
  MPLBACKEND=MacOSX ../.venv/bin/python3 -m experiments.long_run_ablation --episodes 10000 --seeds 2
  MPLBACKEND=Agg    ../.venv/bin/python3 -m experiments.long_run_ablation --episodes 5000  --seeds 1
"""

import argparse
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import gymnasium as gym
import matplotlib
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

CONFIGS = [
    {'name': 'full',       'use_tda': True,  'use_splats': True,  'use_bridge': True},
    {'name': 'no_tda',     'use_tda': False, 'use_splats': True,  'use_bridge': True},
    {'name': 'no_splats',  'use_tda': True,  'use_splats': False, 'use_bridge': True},
    {'name': 'no_bridge',  'use_tda': True,  'use_splats': True,  'use_bridge': False},
    {'name': 'baseline',   'use_tda': False, 'use_splats': False, 'use_bridge': False},
]

SEED_EPISODES   = 100
INSTINCT_WEIGHT = 0.5
DREAM_FRACTION  = 0.3
TDA_INTERVAL    = 5
DREAM_INTERVAL  = 10
NIODOO_MAX_NODES = 5000   # cap to prevent RAM issues at 20k eps


class CappedNiodoo(NiodooMemory):
    """NiodooMemory with a node cap so 20k run doesn't eat RAM."""
    def __init__(self, max_nodes=5000, **kwargs):
        super().__init__(**kwargs)
        self._max_nodes = max_nodes

    def flinch_tag(self, content, velocity, state_vector=None):
        if len(self.nodes) >= self._max_nodes:
            # Drop oldest node to make room
            oldest = next(iter(self.nodes))
            try:
                self.graph.remove_node(oldest)
            except Exception:
                pass
            del self.nodes[oldest]
        return super().flinch_tag(content, velocity, state_vector)


# ─── Single run ─────────────────────────────────────────────────────────────

def run_config(cfg_name, use_tda, use_splats, use_bridge,
               total_episodes, seed, dashboard):

    SeedManager.apply(seed)
    env = gym.make('MountainCar-v0')

    # Governor phases scale with episode count
    governor_warmup   = int(total_episodes * 0.05)   # 5% warmup
    governor_rampdown = int(total_episodes * 0.15)   # off by 15%

    ctrl  = SteeringController()
    brain = TopologicalBrain(ctrl) if use_tda else None
    agent = QSMA_Agent()
    memory = CappedNiodoo(max_nodes=NIODOO_MAX_NODES, beta_threshold=0.4)

    teacher  = None
    governor = None
    if use_bridge:
        seeder = InstinctSeeder(num_seed_episodes=SEED_EPISODES,
                                instinct_weight=INSTINCT_WEIGHT)
        trajectories = seeder.generate_seeds(agent)
        teacher = DreamTeacher(trajectories, dream_fraction=DREAM_FRACTION)
        teacher.inject_dreams(agent)
        governor = GovernorGate(
            initial_threshold=0.8, final_threshold=0.1,
            warmup_episodes=governor_warmup,
            rampdown_episodes=governor_rampdown
        )

    win_flags     = []
    first_success = None

    for episode in range(total_episodes):
        state, _ = env.reset()
        done  = False
        steps = 0
        max_pos = -1.2
        prev_node_id = None

        # Phase 3 starts at 15% — the rest of the run is scaffold-free
        in_phase3 = episode >= governor_rampdown

        agent.sync(ctrl)
        agent.params['beta'] = max(0.1, 1.5 * (0.995 ** episode))

        while not done and steps < 500:
            agent_action = agent.act(state)

            if use_bridge and governor and not in_phase3:
                final_action, _ = governor.gate(state, agent_action, agent, episode, steps)
            else:
                final_action = agent_action

            next_state, reward, terminated, truncated, _ = env.step(final_action)
            done = terminated or truncated

            energy = agent.learn(state, final_action, reward, next_state)

            if use_tda and brain:
                s   = agent.discretize(state)
                ns  = agent.discretize(next_state)
                q   = agent.q_table[s, final_action]
                fl  = agent.flux[s, final_action]
                dlt = abs(reward + 0.999 * np.max(agent.q_table[ns])
                          - agent.q_table[s, final_action])
                brain.add_log([state[0], state[1], q, fl, dlt, energy])

            agent.replay_buffer.append((state.copy(), final_action, reward,
                                        next_state.copy(), energy))
            content = f"pos:{state[0]:.2f},vel:{state[1]:.3f}"
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
            if first_success is None:
                first_success = episode
            agent.commit_episode(True)
        else:
            agent.commit_episode(False)

        if not use_splats:
            agent.splat_memory.splats.clear()

        if use_bridge and governor:
            governor.episode_done()

        agent.splat_memory.decay_and_consolidate()
        splat_stats = agent.splat_memory.get_stats() if use_splats else None
        dashboard.update(episode, max_pos, steps, agent=agent, splat_stats=splat_stats)

        if use_tda and brain and (episode + 1) % TDA_INTERVAL == 0 and len(brain.buffer) >= 100:
            brain.analyze()
            ctrl.normalize()

        if use_bridge and teacher and (episode + 1) % DREAM_INTERVAL == 0:
            teacher.dream_with_teacher(agent)
        elif (episode + 1) % DREAM_INTERVAL == 0:
            agent.dream_cycle()

        if (episode + 1) % 50 == 0:
            memory.curate_prune()

        if (episode + 1) % 1000 == 0:
            pct = sum(win_flags) / (episode + 1) * 100
            phase = "SCAFFOLD" if not in_phase3 else "FREE LEARNING"
            print(f'  [{cfg_name}] Ep {episode+1}/{total_episodes} | '
                  f'Wins: {sum(win_flags)} ({pct:.1f}%) [{phase}]')

    env.close()
    return {
        'config': cfg_name, 'seed': seed,
        'total_episodes': total_episodes,
        'wins': sum(win_flags),
        'win_pct': sum(win_flags) / total_episodes * 100,
        'first_success': first_success,
        'win_flags': win_flags,
        'scaffold_off_at': governor_rampdown,
    }


# ─── Final comparison plots ──────────────────────────────────────────────────

def plot_long_run_results(all_results, total_episodes, scaffold_off):
    """
    Three panels focused on the post-scaffold period:
    1. Full learning curve (all configs, rolling win rate)
    2. ZOOM: post-scaffold only (scaffold_off → total)
    3. Phase comparison: scaffold vs post-scaffold win rates per config
    """
    fig, axes = plt.subplots(1, 3, figsize=(20, 7), facecolor=PALETTE['bg'])
    fig.suptitle(
        f'Long Run Ablation ({total_episodes:,} eps) — '
        f'Governor off at ep {scaffold_off:,} ({scaffold_off/total_episodes*100:.0f}%)',
        color=PALETTE['text'], fontsize=13, fontweight='bold'
    )

    configs = [c['name'] for c in CONFIGS]
    window  = max(50, total_episodes // 100)   # rolling window scales with run length

    # Panel 1: Full curve
    ax = axes[0]
    ax.set_facecolor(PALETTE['panel'])
    for cfg in configs:
        col   = CONFIG_COLORS.get(cfg, '#888')
        res   = [r for r in all_results if r['config'] == cfg]
        if not res:
            continue
        # Average win flags across seeds
        min_len = min(len(r['win_flags']) for r in res)
        avg_flags = np.mean([r['win_flags'][:min_len] for r in res], axis=0)
        if len(avg_flags) >= window:
            rm = rolling_mean(avg_flags, window) * 100
            ax.plot(np.arange(window - 1, len(avg_flags)), rm,
                    color=col, linewidth=2, label=cfg)
    ax.axvline(scaffold_off, color='white', linestyle='--', alpha=0.5,
               linewidth=1.5, label=f'Scaffold off (ep {scaffold_off})')
    ax.axhline(34.05, color='#9b59b6', linestyle=':', alpha=0.6,
               linewidth=1, label='Orig champion 34%')
    ax.set_xlabel('Episode', color=PALETTE['text'])
    ax.set_ylabel('Win Rate %', color=PALETTE['text'])
    ax.set_title(f'Full Run — Rolling {window}-ep Win Rate', color=PALETTE['text'])
    ax.set_ylim(0, 105)
    ax.legend(facecolor=PALETTE['panel'], labelcolor=PALETTE['text'], fontsize=8)
    ax.tick_params(colors=PALETTE['text'])
    ax.grid(True, color=PALETTE['grid'])
    for sp in ax.spines.values(): sp.set_color(PALETTE['grid'])

    # Panel 2: POST-SCAFFOLD ZOOM
    ax = axes[1]
    ax.set_facecolor(PALETTE['panel'])
    for cfg in configs:
        col = CONFIG_COLORS.get(cfg, '#888')
        res = [r for r in all_results if r['config'] == cfg]
        if not res:
            continue
        min_len = min(len(r['win_flags']) for r in res)
        avg_flags = np.mean([r['win_flags'][:min_len] for r in res], axis=0)
        post = avg_flags[scaffold_off:]
        if len(post) >= window:
            rm = rolling_mean(post, window) * 100
            x  = np.arange(scaffold_off + window - 1, scaffold_off + len(post))
            ax.plot(x, rm, color=col, linewidth=2.5, label=cfg)
    ax.axhline(34.05, color='#9b59b6', linestyle=':', alpha=0.6, linewidth=1)
    ax.set_xlabel('Episode', color=PALETTE['text'])
    ax.set_ylabel('Win Rate %', color=PALETTE['text'])
    ax.set_title('🔍 POST-SCAFFOLD ZOOM\n(After governor turned off — pure learned)',
                 color=PALETTE['text'])
    ax.set_ylim(0, 105)
    ax.legend(facecolor=PALETTE['panel'], labelcolor=PALETTE['text'], fontsize=8)
    ax.tick_params(colors=PALETTE['text'])
    ax.grid(True, color=PALETTE['grid'])
    for sp in ax.spines.values(): sp.set_color(PALETTE['grid'])

    # Panel 3: scaffold vs post-scaffold bar comparison
    ax = axes[2]
    ax.set_facecolor(PALETTE['panel'])
    x_pos = np.arange(len(configs))
    bar_w = 0.35

    scaffold_rates  = []
    post_rates      = []
    for cfg in configs:
        res = [r for r in all_results if r['config'] == cfg]
        if not res:
            scaffold_rates.append(0); post_rates.append(0); continue
        min_len = min(len(r['win_flags']) for r in res)
        avg     = np.mean([r['win_flags'][:min_len] for r in res], axis=0)
        s_end   = min(scaffold_off, min_len)
        scaffold_rates.append(np.mean(avg[:s_end]) * 100  if s_end > 0 else 0)
        post_rates.append(   np.mean(avg[s_end:])  * 100  if min_len > s_end else 0)

    colors = [CONFIG_COLORS.get(c, '#888') for c in configs]
    b1 = ax.bar(x_pos - bar_w/2, scaffold_rates, bar_w,
                color=colors, alpha=0.5, edgecolor='white', linewidth=0.5,
                label='Scaffold period')
    b2 = ax.bar(x_pos + bar_w/2, post_rates, bar_w,
                color=colors, alpha=1.0, edgecolor='white', linewidth=0.5,
                label='Post-scaffold (free)')
    for bar, val in zip(list(b1) + list(b2), scaffold_rates + post_rates):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{val:.0f}%', ha='center', fontsize=8,
                    fontweight='bold', color=PALETTE['text'])
    ax.axhline(34.05, color='#9b59b6', linestyle=':', alpha=0.6, linewidth=1)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(configs, rotation=15, color=PALETTE['text'])
    ax.set_ylabel('Win Rate %', color=PALETTE['text'])
    ax.set_title('Scaffold vs Post-Scaffold\n(same color = same config, faded=early, bright=late)',
                 color=PALETTE['text'])
    ax.set_ylim(0, 110)
    ax.legend(facecolor=PALETTE['panel'], labelcolor=PALETTE['text'], fontsize=8)
    ax.tick_params(colors=PALETTE['text'])
    ax.grid(True, color=PALETTE['grid'], axis='y')
    for sp in ax.spines.values(): sp.set_color(PALETTE['grid'])

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    save_png(fig, f'long_run_{total_episodes}ep_comparison')
    if matplotlib.get_backend().lower() not in ('agg', 'pdf', 'svg', 'ps'):
        plt.show(block=False)
        plt.pause(0.5)
    plt.close(fig)


# ─── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=20000,
                        help='Total episodes per config (default: 20000)')
    parser.add_argument('--seeds',    type=int, default=2,
                        help='Number of seeds (default: 2)')
    parser.add_argument('--configs',  type=str, nargs='+',
                        default=['full', 'no_tda', 'no_splats', 'no_bridge', 'baseline'],
                        help='Configs to run')
    args = parser.parse_args()

    seeds  = SeedManager.get_seeds(args.seeds)
    scaffold_off = int(args.episodes * 0.15)

    # Filter configs
    cfg_list = [c for c in CONFIGS if c['name'] in args.configs]

    print('=' * 65)
    print(f'  LONG RUN ABLATION — {args.episodes:,} episodes')
    print(f'  Governor scaffold OFF at episode {scaffold_off:,} '
          f'({scaffold_off/args.episodes*100:.0f}%)')
    print(f'  Pure learned performance measured: ep {scaffold_off:,} → {args.episodes:,}')
    print(f'  Configs: {[c["name"] for c in cfg_list]}')
    print(f'  Seeds: {seeds}')
    print('=' * 65)

    all_results = []

    for cfg in cfg_list:
        cfg_name = cfg['name']
        color    = CONFIG_COLORS.get(cfg_name, '#888')
        print(f'\n{"=" * 65}')
        print(f'  CONFIG: {cfg_name.upper()}')
        print(f'  TDA={cfg["use_tda"]}  Splats={cfg["use_splats"]}  Bridge={cfg["use_bridge"]}')
        print(f'{"=" * 65}')

        for seed in seeds:
            print(f'\n  ── Seed {seed} ──')

            # Bigger UPDATE_EVERY for 20k runs
            LiveDashboard.UPDATE_EVERY = max(10, args.episodes // 400)

            dash = LiveDashboard(
                title=f'{cfg_name} | {args.episodes:,}ep (seed={seed})',
                total_episodes=args.episodes,
                config_color=color
            )
            result = run_config(
                cfg_name=cfg_name,
                use_tda=cfg['use_tda'],
                use_splats=cfg['use_splats'],
                use_bridge=cfg['use_bridge'],
                total_episodes=args.episodes,
                seed=seed,
                dashboard=dash
            )
            result['scaffold_off'] = scaffold_off
            dash.close(name=f'longrun_{cfg_name}_{args.episodes}ep_seed{seed}')
            all_results.append(result)

            pct = result['win_pct']
            # Post-scaffold win rate
            post_flags = result['win_flags'][scaffold_off:]
            post_pct   = sum(post_flags) / max(1, len(post_flags)) * 100
            print(f'  ✓ {cfg_name} seed={seed}: '
                  f'{result["wins"]}/{args.episodes} ({pct:.1f}%) | '
                  f'Post-scaffold: {post_pct:.1f}% | '
                  f'First win: {result["first_success"]}')

    # Summary table
    print('\n' + '=' * 65)
    print('  LONG RUN RESULTS')
    print('=' * 65)
    print(f'  {"Config":<14} {"Total%":>8} {"Post-Scaffold%":>16} {"First Win":>12}')
    print('-' * 65)
    for cfg in cfg_list:
        cname = cfg['name']
        res   = [r for r in all_results if r['config'] == cname]
        if not res: continue
        total_pct = np.mean([r['win_pct'] for r in res])
        post_pcts = []
        for r in res:
            pf = r['win_flags'][scaffold_off:]
            post_pcts.append(sum(pf) / max(1, len(pf)) * 100)
        post_mean = np.mean(post_pcts)
        firsts = [r['first_success'] for r in res if r['first_success'] is not None]
        avg_f  = int(np.mean(firsts)) if firsts else None
        print(f'  {cname:<14} {total_pct:>7.1f}%  {post_mean:>14.1f}%  {str(avg_f):>12}')
    print('=' * 65)
    print(f'\n  Key question: Does TDA/Splats post-scaffold % differ from no_bridge/baseline?')
    print(f'  Scaffold ends at ep {scaffold_off} — everything after is PURE LEARNING.')

    save_json({
        'total_episodes': args.episodes,
        'scaffold_off':   scaffold_off,
        'seeds': seeds,
        'results': all_results
    }, f'long_run_{args.episodes}ep')

    plot_long_run_results(all_results, args.episodes, scaffold_off)
    print('\nDone! Check results/ for plots and JSON.')


if __name__ == '__main__':
    main()
