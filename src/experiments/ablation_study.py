"""
ablation_study.py — Q-SMA Component Ablation
=============================================
Five configs, all sharing the same base training loop from main_bridge.py:

  full       → TDA + Splats + Bridge (Physics seed + Governor)
  no_tda     → Splats + Bridge only  (TopologicalBrain.analyze skipped)
  no_splats  → TDA + Bridge only     (SplatMemory disabled, dreams unweighted)
  no_bridge  → TDA + Splats only     (No instinct seed, no governor, no teacher)
  baseline   → Q-table + energy shaping only (no TDA/Splats/Bridge)

LIVE WINDOW: opens automatically, updates every 10 episodes.
COMPARISON: shown after ALL configs finish.

Usage:
  cd src
  python -m experiments.ablation_study                     # full run (2000 eps, 3 seeds)
  python -m experiments.ablation_study --episodes 200 --seeds 1   # quick smoke test
"""

import argparse
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import gymnasium as gym

from core.steering import SteeringController
from core.tda import TopologicalBrain
from core.agent import QSMA_Agent
from models.bridge import InstinctSeeder, DreamTeacher, GovernorGate
from models.niodoo import NiodooMemory
from experiments.experiment_utils import (
    LiveDashboard, ComparisonDashboard, SeedManager,
    save_json, CONFIG_COLORS
)

# ─── Config definitions ─────────────────────────────────────────────────────

CONFIGS = [
    {'name': 'full',       'use_tda': True,  'use_splats': True,  'use_bridge': True},
    {'name': 'no_tda',     'use_tda': False, 'use_splats': True,  'use_bridge': True},
    {'name': 'no_splats',  'use_tda': True,  'use_splats': False, 'use_bridge': True},
    {'name': 'no_bridge',  'use_tda': True,  'use_splats': True,  'use_bridge': False},
    {'name': 'baseline',   'use_tda': False, 'use_splats': False, 'use_bridge': False},
]

TDA_INTERVAL   = 5
DREAM_INTERVAL = 10
SEED_EPISODES  = 100
INSTINCT_WEIGHT = 0.5
DREAM_FRACTION  = 0.3


# ─── Single-config runner ───────────────────────────────────────────────────

def run_config(cfg_name: str, use_tda: bool, use_splats: bool, use_bridge: bool,
               total_episodes: int, seed: int, dashboard: LiveDashboard) -> dict:

    SeedManager.apply(seed)
    env = gym.make('MountainCar-v0')

    ctrl  = SteeringController()
    brain = TopologicalBrain(ctrl) if use_tda else None
    agent = QSMA_Agent()
    agent.splat_memory.maturation_episodes = 0 if not use_splats else 50

    # ── Bridge Phase 0: seed + teacher ───────────────────────────────
    teacher = None
    governor = None
    if use_bridge:
        seeder = InstinctSeeder(num_seed_episodes=SEED_EPISODES,
                                instinct_weight=INSTINCT_WEIGHT)
        trajectories = seeder.generate_seeds(agent)
        teacher = DreamTeacher(trajectories, dream_fraction=DREAM_FRACTION)
        teacher.inject_dreams(agent)
        governor = GovernorGate(
            initial_threshold=0.8, final_threshold=0.1,
            warmup_episodes=200, rampdown_episodes=1500
        )

    # ── Training Loop ─────────────────────────────────────────────────
    success_count  = 0
    first_success  = None
    win_flags      = []
    memory = NiodooMemory(beta_threshold=0.4)

    for episode in range(total_episodes):
        state, _ = env.reset()
        done  = False
        steps = 0
        max_pos = -1.2
        prev_node_id = None

        # Phase (used by bridge only)
        if episode < 500:
            phase = 1
        elif episode < 1500:
            phase = 2
        else:
            phase = 3

        agent.sync(ctrl)
        beta = max(0.1, 1.5 * (0.995 ** episode))
        agent.params['beta'] = beta

        while not done and steps < 500:
            agent_action = agent.act(state)

            # Governor (bridge only, phase 1 & 2)
            if use_bridge and governor and phase < 3:
                final_action, _ = governor.gate(state, agent_action, agent, episode, steps)
            else:
                final_action = agent_action

            next_state, reward, terminated, truncated, _ = env.step(final_action)
            done = terminated or truncated

            energy = agent.learn(state, final_action, reward, next_state, done=done)

            # TDA logging
            if use_tda and brain:
                s   = agent.discretize(state)
                q   = agent.q_table[s, final_action]
                fl  = agent.flux[s, final_action]
                dlt = abs(reward + 0.999 * np.max(agent.q_table[agent.discretize(next_state)])
                          - agent.q_table[s, final_action])
                brain.add_log([state[0], state[1], q, fl, dlt, energy])

            # Niodoo
            content = f"pos:{state[0]:.2f},vel:{state[1]:.3f},a:{final_action},e:{energy:.3f}"
            node_id = memory.flinch_tag(content, state[1], state)
            if prev_node_id:
                memory.connect_nodes(prev_node_id, node_id, weight=abs(state[1]))
            prev_node_id = node_id

            agent.replay_buffer.append((state.copy(), final_action, reward,
                                        next_state.copy(), energy))
            state = next_state
            steps += 1
            max_pos = max(max_pos, state[0])

        # Episode end
        win = max_pos >= 0.5
        win_flags.append(1 if win else 0)

        if win:
            success_count += 1
            if first_success is None:
                first_success = episode
            agent.commit_episode(True)
        else:
            agent.commit_episode(False)

        # Splat memory: disable influence if use_splats=False
        if not use_splats:
            agent.splat_memory.splats.clear()

        if use_bridge and governor:
            governor.episode_done()

        agent.splat_memory.decay_and_consolidate()

        splat_stats = agent.splat_memory.get_stats() if use_splats else None
        dashboard.update(episode, max_pos, steps, agent=agent, splat_stats=splat_stats)

        # TDA + dreams
        if use_tda and brain and (episode + 1) % TDA_INTERVAL == 0 and len(brain.buffer) >= 100:
            brain.analyze()
            ctrl.normalize()

        if use_bridge and teacher and (episode + 1) % DREAM_INTERVAL == 0:
            teacher.dream_with_teacher(agent)
        elif (episode + 1) % DREAM_INTERVAL == 0:
            agent.dream_cycle()

        if (episode + 1) % 50 == 0:
            memory.curate_prune()

        if (episode + 1) % 200 == 0:
            pct = success_count / (episode + 1) * 100
            print(f'  [{cfg_name}] Ep {episode+1}/{total_episodes} | '
                  f'Wins: {success_count} ({pct:.1f}%)')

    env.close()

    return {
        'config':        cfg_name,
        'seed':          seed,
        'total_episodes': total_episodes,
        'wins':          success_count,
        'win_pct':       success_count / total_episodes * 100,
        'first_success': first_success,
        'win_flags':     win_flags,
    }


# ─── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=2000)
    parser.add_argument('--seeds',    type=int, default=3)
    args = parser.parse_args()

    seeds = SeedManager.get_seeds(args.seeds)
    comp  = ComparisonDashboard('Ablation Study: Which Components Matter?')

    all_results = []

    for cfg in CONFIGS:
        cfg_name = cfg['name']
        color    = CONFIG_COLORS.get(cfg_name, '#888888')
        print(f'\n{"="*60}')
        print(f'  CONFIG: {cfg_name.upper()}')
        print(f'  TDA={cfg["use_tda"]}  Splats={cfg["use_splats"]}  Bridge={cfg["use_bridge"]}')
        print(f'  Episodes={args.episodes}  Seeds={len(seeds)}')
        print(f'{"="*60}')

        cfg_win_flags = []
        cfg_first_success = None

        for seed in seeds:
            print(f'\n  ── Seed {seed} ──')
            dash = LiveDashboard(
                title=f'Ablation: {cfg_name} (seed={seed})',
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
            dash.close(name=f'ablation_{cfg_name}_seed{seed}')

            all_results.append(result)
            cfg_win_flags.extend(result['win_flags'])
            if result['first_success'] is not None:
                if cfg_first_success is None or result['first_success'] < cfg_first_success:
                    cfg_first_success = result['first_success']

            pct = result['win_pct']
            print(f'  ✓ {cfg_name} seed={seed}: {result["wins"]}/{args.episodes} ({pct:.1f}%) '
                  f'first={result["first_success"]}')

        mean_pct = np.mean([r['win_pct'] for r in all_results if r['config'] == cfg_name])
        std_pct  = np.std ([r['win_pct'] for r in all_results if r['config'] == cfg_name])
        print(f'\n  ── {cfg_name} SUMMARY: {mean_pct:.1f}% ± {std_pct:.1f}% ──')

        comp.add_result(cfg_name, cfg_win_flags, cfg_first_success)

    # Summary table
    print('\n' + '='*60)
    print('  ABLATION RESULTS')
    print('='*60)
    print(f'  {"Config":<14} {"Mean Win%":>10} {"±Std":>8} {"First Success":>15}')
    print('-'*60)
    for cfg in CONFIGS:
        cname = cfg['name']
        rlist = [r for r in all_results if r['config'] == cname]
        wpts  = [r['win_pct'] for r in rlist]
        feps  = [r['first_success'] for r in rlist if r['first_success'] is not None]
        avg_f = int(np.mean(feps)) if feps else None
        print(f'  {cname:<14} {np.mean(wpts):>9.1f}% {np.std(wpts):>7.1f}%  {str(avg_f):>15}')
    print('='*60)

    # Save JSON
    summary = {
        'episodes': args.episodes,
        'seeds': seeds,
        'configs': {
            cfg['name']: {
                'use_tda': cfg['use_tda'],
                'use_splats': cfg['use_splats'],
                'use_bridge': cfg['use_bridge'],
                'results': [r for r in all_results if r['config'] == cfg['name']]
            }
            for cfg in CONFIGS
        }
    }
    save_json(summary, 'ablation')

    # Comparison plot
    comp.show(name='ablation_comparison')


if __name__ == '__main__':
    main()
