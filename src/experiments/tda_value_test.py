"""
tda_value_test.py — Does Ripser H1 Actually Help?
==================================================
The TopologicalBrain uses TWO detection methods:
  A) Density heuristic   — fast, custom, no dependencies
  B) Ripser H1           — real persistent homology (slow, needs ripser package)

Key question: does Method B (Ripser) catch loops that Method A misses,
and does that difference translate to better learning?

Three sub-configs:
  tda_full          → both heuristic + Ripser (current champion)
  tda_heuristic     → heuristic only (Ripser block skipped)
  no_tda            → no TDA at all (reuse baseline)

Also logs: how many times Ripser fired DIFFERENTLY from the heuristic.

Usage:
  cd src
  python -m experiments.tda_value_test                    # 2000 eps, 3 seeds
  python -m experiments.tda_value_test --episodes 200 --seeds 1

Note: if 'ripser' is not installed, tda_full will fall back to heuristic only
automatically (you'll see a warning). Install with:
  pip install ripser persim
"""

import argparse
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import gymnasium as gym

from core.steering import SteeringController
from core.agent import QSMA_Agent
from models.bridge import InstinctSeeder, DreamTeacher, GovernorGate
from models.niodoo import NiodooMemory
from experiments.experiment_utils import (
    LiveDashboard, ComparisonDashboard, SeedManager,
    save_json, CONFIG_COLORS
)

# Patched TopologicalBrain so we can control Ripser independently
try:
    from ripser import ripser as _ripser
    HAS_RIPSER = True
except ImportError:
    HAS_RIPSER = False
    print("WARNING: ripser not installed — tda_full will behave like tda_heuristic")

from collections import deque


class PatchedTopoBrain:
    """
    Identical to core.tda.TopologicalBrain but with use_ripser flag.
    Logs whether Ripser fired differently from the heuristic on each call.
    """

    def __init__(self, controller, use_ripser: bool = True, max_buffer=2000):
        self.controller  = controller
        self.use_ripser  = use_ripser and HAS_RIPSER
        self.buffer      = deque(maxlen=max_buffer)

        # Telemetry
        self.ripser_fires       = 0    # ripser detected something heuristic didn't
        self.heuristic_fires    = 0
        self.heuristic_calls    = 0

    def add_log(self, log):
        self.buffer.append(log)

    def analyze(self):
        if len(self.buffer) < 100:
            return

        recent = list(self.buffer)[-500:]
        data   = np.array(recent)
        positions  = data[:, 0]
        velocities = data[:, 1]

        # ── Heuristic (Method A) ──────────────────────────────────────
        self.heuristic_calls += 1
        center_mask = ((positions > -0.7) & (positions < -0.3) &
                       (np.abs(velocities) < 0.03))
        loop_density = np.sum(center_mask) / len(data)
        heuristic_fired = False

        if loop_density > 0.3:
            energies = data[:, 5]
            n = len(energies)
            x = np.arange(n)
            denom = np.mean(x**2) - np.mean(x)**2 + 1e-10
            energy_slope = (np.mean(x * energies) - np.mean(x) * np.mean(energies)) / denom

            if energy_slope <= 0.001:
                self.controller.apply_decay_spike(loop_density)
                heuristic_fired = True
                self.heuristic_fires += 1

        # ── Ripser H1 (Method B) ─────────────────────────────────────
        ripser_fired = False
        if self.use_ripser:
            try:
                point_cloud = data[:, [0, 1, 4]]
                if len(point_cloud) > 200:
                    idx = np.random.choice(len(point_cloud), 200, replace=False)
                    point_cloud = point_cloud[idx]
                point_cloud = np.clip(point_cloud, -1e6, 1e6)
                stds = point_cloud.std(axis=0)
                if not np.any(stds < 1e-6):
                    point_cloud = (point_cloud - point_cloud.mean(axis=0)) / stds
                    point_cloud = np.clip(point_cloud, -10.0, 10.0)
                    results = _ripser(point_cloud, maxdim=1)
                    dgms = results['dgms']
                    if len(dgms) > 1:
                        h1 = dgms[1]
                        pers = h1[:, 1] - h1[:, 0]
                        valid = np.isfinite(pers)
                        if np.any(valid) and np.max(pers[valid]) > 0.5:
                            if not heuristic_fired:
                                self.controller.apply_decay_spike(np.max(pers[valid]))
                                ripser_fired = True
                                self.ripser_fires += 1
            except Exception:
                pass

        # ── Void detection (always on) ────────────────────────────────
        hist, _, _ = np.histogram2d(positions, velocities, bins=10,
                                    range=[[-1.2, 0.6], [-0.07, 0.07]])
        if np.sum(hist[7:, :]) < 10:
            self.controller.inject_attractor([0.45, 0.04])

    def get_tda_stats(self):
        return {
            'heuristic_calls': self.heuristic_calls,
            'heuristic_fires': self.heuristic_fires,
            'ripser_fires':    self.ripser_fires,
        }


TDA_CONFIGS = [
    {'name': 'tda_full',      'use_tda': True,  'use_ripser': True},
    {'name': 'tda_heuristic', 'use_tda': True,  'use_ripser': False},
    {'name': 'no_tda',        'use_tda': False, 'use_ripser': False},
]

TDA_INTERVAL    = 5
DREAM_INTERVAL  = 10
SEED_EPISODES   = 100
INSTINCT_WEIGHT = 0.5
DREAM_FRACTION  = 0.3


def run_tda_config(cfg_name: str, use_tda: bool, use_ripser: bool,
                   total_episodes: int, seed: int,
                   dashboard: LiveDashboard) -> dict:

    SeedManager.apply(seed)
    env = gym.make('MountainCar-v0')

    ctrl   = SteeringController()
    brain  = PatchedTopoBrain(ctrl, use_ripser=use_ripser) if use_tda else None
    agent  = QSMA_Agent()
    memory = NiodooMemory(beta_threshold=0.4)

    seeder = InstinctSeeder(num_seed_episodes=SEED_EPISODES,
                            instinct_weight=INSTINCT_WEIGHT)
    trajectories = seeder.generate_seeds(agent)
    teacher = DreamTeacher(trajectories, dream_fraction=DREAM_FRACTION)
    teacher.inject_dreams(agent)
    governor = GovernorGate(initial_threshold=0.8, final_threshold=0.1,
                            warmup_episodes=200, rampdown_episodes=1500)

    success_count = 0
    first_success = None
    win_flags     = []

    for episode in range(total_episodes):
        state, _ = env.reset()
        done  = False
        steps = 0
        max_pos = -1.2
        prev_node_id = None

        phase = 1 if episode < 500 else (2 if episode < 1500 else 3)
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

            energy = agent.learn(state, final_action, reward, next_state, done=done)

            if use_tda and brain:
                s   = agent.discretize(state)
                ns  = agent.discretize(next_state)
                q   = agent.q_table[s, final_action]
                fl  = agent.flux[s, final_action]
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

        governor.episode_done()
        agent.splat_memory.decay_and_consolidate()
        splat_stats = agent.splat_memory.get_stats()
        dashboard.update(episode, max_pos, steps, agent=agent, splat_stats=splat_stats)

        if use_tda and brain and (episode + 1) % TDA_INTERVAL == 0 and len(brain.buffer) >= 100:
            brain.analyze()
            ctrl.normalize()

        if (episode + 1) % DREAM_INTERVAL == 0:
            teacher.dream_with_teacher(agent)

        if (episode + 1) % 50 == 0:
            memory.curate_prune()

        if (episode + 1) % 200 == 0:
            pct = success_count / (episode + 1) * 100
            print(f'  [{cfg_name}] Ep {episode+1} | Wins: {success_count} ({pct:.1f}%)')

    env.close()

    tda_stats = brain.get_tda_stats() if brain else {'heuristic_fires': 0, 'ripser_fires': 0}
    return {
        'config': cfg_name,
        'seed': seed,
        'total_episodes': total_episodes,
        'wins': success_count,
        'win_pct': success_count / total_episodes * 100,
        'first_success': first_success,
        'win_flags': win_flags,
        'tda_heuristic_fires': tda_stats.get('heuristic_fires', 0),
        'tda_ripser_fires':    tda_stats.get('ripser_fires', 0),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=2000)
    parser.add_argument('--seeds',    type=int, default=3)
    args = parser.parse_args()

    seeds = SeedManager.get_seeds(args.seeds)
    comp  = ComparisonDashboard('TDA Value Test: Full vs Heuristic vs None')
    all_results = []

    for cfg in TDA_CONFIGS:
        cfg_name = cfg['name']
        color    = CONFIG_COLORS.get(cfg_name, '#888888')

        print(f'\n{"="*60}')
        print(f'  TDA CONFIG: {cfg_name.upper()}')
        print(f'  use_tda={cfg["use_tda"]}  use_ripser={cfg["use_ripser"]}')
        print(f'{"="*60}')

        cfg_win_flags    = []
        cfg_first_success = None

        for seed in seeds:
            print(f'\n  ── Seed {seed} ──')
            dash = LiveDashboard(
                title=f'TDA Test: {cfg_name} (seed={seed})',
                total_episodes=args.episodes,
                config_color=color
            )
            result = run_tda_config(
                cfg_name=cfg_name,
                use_tda=cfg['use_tda'],
                use_ripser=cfg['use_ripser'],
                total_episodes=args.episodes,
                seed=seed,
                dashboard=dash
            )
            dash.close(name=f'tda_{cfg_name}_seed{seed}')

            all_results.append(result)
            cfg_win_flags.extend(result['win_flags'])
            if result['first_success'] is not None:
                if cfg_first_success is None or result['first_success'] < cfg_first_success:
                    cfg_first_success = result['first_success']

            print(f'  ✓ {cfg_name} seed={seed}: {result["wins"]}/{args.episodes} '
                  f'({result["win_pct"]:.1f}%) '
                  f'ripser_fires={result["tda_ripser_fires"]} '
                  f'heuristic_fires={result["tda_heuristic_fires"]}')

        comp.add_result(cfg_name, cfg_win_flags, cfg_first_success)

    # Summary
    print('\n' + '='*60)
    print('  TDA VALUE TEST RESULTS')
    print('='*60)
    print(f'  {"Config":<18} {"Mean Win%":>10} {"Heuristic":>12} {"Ripser-Only":>13}')
    print('-'*60)
    for cfg in TDA_CONFIGS:
        cname = cfg['name']
        rlist = [r for r in all_results if r['config'] == cname]
        wpts  = [r['win_pct'] for r in rlist]
        hfire = int(np.mean([r['tda_heuristic_fires'] for r in rlist]))
        rfire = int(np.mean([r['tda_ripser_fires']    for r in rlist]))
        print(f'  {cname:<18} {np.mean(wpts):>9.1f}%  {hfire:>12}  {rfire:>13}')
    print('='*60)
    if all(r['tda_ripser_fires'] == 0 for r in all_results if r['config'] == 'tda_full'):
        print('  ⚠️  Ripser never fired independently — heuristic is sufficient')
    else:
        print('  ✅ Ripser added unique detections')

    save_json({'episodes': args.episodes, 'seeds': seeds, 'results': all_results}, 'tda_value')
    comp.show(name='tda_comparison')


if __name__ == '__main__':
    main()
